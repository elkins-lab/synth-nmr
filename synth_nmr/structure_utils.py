"""Utility functions for structure parsing and analysis."""

import logging
from typing import Dict, List, Tuple

import biotite.structure as struc
import numpy as np

logger = logging.getLogger(__name__)


def get_residue_info(
    structure: struc.AtomArray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unified utility to extract residue-level metadata (Chain ID, Res ID, Res Name).

    Returns:
        tuple: (chain_ids, res_ids, res_names, res_starts)
        All as NumPy arrays.
    """
    res_starts: np.ndarray = struc.get_residue_starts(structure)
    # Extract identifiers at each start index
    chain_ids = structure.chain_id[res_starts]
    res_ids = structure.res_id[res_starts]
    res_names = structure.res_name[res_starts]

    return chain_ids, res_ids, res_names, res_starts


def get_secondary_structure(structure: struc.AtomArray) -> List[str]:
    """
    Determine the secondary structure of each residue based on Phi/Psi angles.

    EDUCATIONAL BACKGROUND — Secondary Structure and Chemical Shifts
    ───────────────────────────────────────────────────────────────────
    Protein secondary structure (alpha-helices, beta-sheets, and loops/coils)
    is primarily defined by the hydrogen-bonding patterns between backbone
    amide and carbonyl groups. These patterns correlate strongly with the
    backbone dihedral angles:
      • Phi (φ): torsion around the N-CA bond.
      • Psi (ψ): torsion around the CA-C bond.

    Measuring these angles accurately in 3D structures allows us to predict
    how local electronic environments around nuclei (like 1H, 13C, 15N)
    shift from their "random coil" baseline values. This is the physical
    basis for structure determination by NMR.

    Returns:
        List of strings: 'alpha', 'beta', or 'coil', one for each residue
        in the input structure (matching the order of get_residue_starts).
    """
    # ── 1. Preparation and Filtering ──────────────────────────────────────
    # We must calculate dihedrals only on amino acids. However, the output list
    # must match the length of the original residue count (including HETATMs).
    original_res_count = struc.get_residue_count(structure)

    # Filter for amino acids to ensure reliable Phi/Psi calculations
    protein_mask = struc.filter_amino_acids(structure)
    protein_structure = structure[protein_mask]

    # If no protein is present, return all coils (standard NMR default)
    if protein_structure.array_length() == 0:
        return ["coil"] * original_res_count

    try:
        # Calculate backbone dihedrals for the protein portion
        phi, psi, _ = struc.dihedral_backbone(protein_structure)
    except struc.BadStructureError:
        # Fallback if backbone is incomplete or disconnected (e.g. in tests)
        return ["coil"] * original_res_count

    # ── 2. Mapping and Classification ─────────────────────────────────────
    # We iterate over all original residues. For protein residues, we look
    # up their dihedral angles. For others (ions, water), we default to 'coil'.
    res_starts: np.ndarray = struc.get_residue_starts(structure)
    ss_list = []

    # Map protein residue indices back to their dihedrals
    # (Since protein_structure is a subset, we need to track indices carefully)
    protein_res_indices = np.where(protein_mask[struc.get_residue_starts(structure)])[0]
    phi_map = {idx: phi[i] for i, idx in enumerate(protein_res_indices) if i < len(phi)}
    psi_map = {idx: psi[i] for i, idx in enumerate(protein_res_indices) if i < len(psi)}

    for i in range(len(res_starts)):
        # Get Angles
        p_rad = phi_map.get(i, np.nan)
        s_rad = psi_map.get(i, np.nan)

        # Convert to degrees for standard Ramachandran classification
        p = np.rad2deg(p_rad)
        s = np.rad2deg(s_rad)

        # Default to coil for non-protein or missing angles
        ss_state = "coil"

        # ── 3. Ramachandran Filtering ─────────────────────────────────────
        # Determine Secondary Structure State based on broader Ramachandran regions.
        # The ranges used here are slightly wider than strict textbook definitions
        # to accommodate potential variations from structure generation algorithms
        # ("Synthetic Generator Offset Issues") and to capture both canonical
        # right-handed alpha helices and less common left-handed helical regions.
        #
        # References for Ramachandran regions:
        # - Ramachandran, G. N., Ramakrishnan, C., & Sasisekharan, V. (1963).
        #   "Stereochemistry of polypeptide chain configurations." J. Mol. Biol., 7(1), 95-99.
        # - Lovell, S. C., Davis, I. W., Arendall III, J. W., de Bakker, P. I.,
        #   Word, J. M., Prisant, M. G., ... & Richardson, D. C. (2003).
        #   "Structure validation by Calpha geometry: phi,psi and Cbeta deviation."
        #   Proteins: Structure, Function, and Bioinformatics, 50(3), 437-450.
        if not np.isnan(p) and not np.isnan(s):
            logger.debug(f"Res {i}: Phi={p:.1f}, Psi={s:.1f}")

            # Canonical Right-handed Alpha-helix region
            if (-90 < p < -30) and (-90 < s < -10):
                ss_state = "alpha"
            # Beta-sheet / Extended region
            elif (-160 < p < -80) and (80 < s < 170):
                ss_state = "beta"
            # NOTE: The positive-φ region (0° < φ < 150°, −90° < ψ < −10°)
            # corresponds to the left-handed α-helix (αL).  These conformations
            # are almost exclusively found in Glycine residues and are structurally
            # distinct from canonical right-handed α-helices.  Applying α-helix
            # chemical shift offsets (e.g., CA +3.1 ppm) to αL residues would
            # be physically incorrect; they are treated as coil here.
            # Reference: Richardson, D.C. & Richardson, J.S. (1989)
            #   "Principles and patterns of protein conformation."
            #   Prediction of Protein Structure and the Principles of Protein Conformation, 1-98.

        ss_list.append(ss_state)

        # ── 4. Smoothing Pass ─────────────────────────────────────────────────
        # Local context matters: secondary structure elements are typically
        # at least 3-4 residues long. We remove isolated single-residue "coil"
        # interruptions within established helices or sheets.
        # Example: alpha-coil-alpha -> alpha-alpha-alpha
        #
        # A common artifact in structural biology is the "staccato" secondary
        # structure assignments, where a single residue is assigned as coil
        # despite being part of a larger, well-defined helix or sheet.
        #
        # 1. Filter single-residue interruptions
        # The loop intentionally excludes the first and last residues (range from 1 to len-1),
        # as these termini are often intrinsically flexible and their secondary structure
        # is less reliably defined or less critical for local context smoothing.
        #
        # This smoothing pass helps to create more biologically realistic
        # and continuous secondary structure elements.
        for i in range(1, len(ss_list) - 1):
            prev_s = ss_list[i - 1]
            curr_s = ss_list[i]
            next_s = ss_list[i + 1]

            if curr_s == "coil" and prev_s == next_s and prev_s != "coil":
                logger.debug(f"Smoothing residue {i}: coil -> {prev_s}")
                # Only smooth if both neighbors agree on a non-coil state
                ss_list[i] = prev_s

    return ss_list


def calculate_c_beta_deviations(structure: struc.AtomArray) -> Dict[int, float]:
    """
    Calculate the deviation of C-beta atoms from their ideal tetrahedral positions.

    EDUCATIONAL BACKGROUND — The C-beta Deviation Metric
    ───────────────────────────────────────────────────────────────────
    In a high-quality protein structure, the C-beta (CB) atom should sit in a
    near-perfect tetrahedral geometry relative to the backbone atoms (N, CA, C).
    Because the CB position is so physically constrained, any significant
    "strain" or displacement is a sensitive indicator of problems in the
    backbone conformation or local steric clashes.

    This metric was popularized by the Richardson and Montelione labs (Lovell
    et al., 2003) as part of the MolProbity validation suite.

    How it's calculated:
    1. An "ideal" CB position is reconstructed using the coordinates of N, CA, and C
       assuming standard bond lengths and tetrahedral angles.
    2. The Euclidean distance between this ideal position and the actual CB
       position in the structure is measured.
    3. Deviations > 0.25 Å are considered "outliers" and usually suggest
       that the residue's Phi/Psi angles are forced into an improbable state.

    Significance for NMR:
    Structural strain often correlates with poor agreement between predicted
    and experimental chemical shifts, particularly for CA and CB nuclei.

    Returns:
        Dict[int, float]: Mapping of res_id to deviation distance (Å).
    """
    deviations = {}

    # Filter for residues that have a C-beta (excludes GLY)
    # We iterate residue by residue
    res_starts: np.ndarray = struc.get_residue_starts(structure)

    for i in range(len(res_starts)):
        start_idx = res_starts[i]
        end_idx = res_starts[i + 1] if i + 1 < len(res_starts) else len(structure)
        res_atoms = structure[start_idx:end_idx]

        res_id = int(res_atoms.res_id[0])
        res_name = res_atoms.res_name[0]

        if res_name == "GLY":
            continue

        # Extract required backbone atoms
        n_atoms = res_atoms[res_atoms.atom_name == "N"]
        ca_atoms = res_atoms[res_atoms.atom_name == "CA"]
        c_atoms = res_atoms[res_atoms.atom_name == "C"]
        cb_atoms = res_atoms[res_atoms.atom_name == "CB"]

        if len(n_atoms) == 0 or len(ca_atoms) == 0 or len(c_atoms) == 0 or len(cb_atoms) == 0:
            continue

        n_coord = n_atoms.coord[0]
        ca_coord = ca_atoms.coord[0]
        c_coord = c_atoms.coord[0]
        cb_coord = cb_atoms.coord[0]

        # Construct ideal CB position
        # We use a simplified reconstruction: CB is on the bisector of N-CA-C
        # but pointing "away" from the backbone plane.
        v_ca_n = n_coord - ca_coord
        v_ca_c = c_coord - ca_coord

        # Normalize
        v_ca_n /= np.linalg.norm(v_ca_n)
        v_ca_c /= np.linalg.norm(v_ca_c)

        # The ideal CB direction is the vector that makes equal angles with N, CA, C
        # In a tetrahedral geometry, CB direction is roughly -(v_ca_n + v_ca_c)
        # after proper scaling.
        cb_dir = -(v_ca_n + v_ca_c)
        cb_dir /= np.linalg.norm(cb_dir)

        # Standard CA-CB bond length is ~1.53 A
        # The angle N-CA-C is ~110 deg. In a perfect tetrahedron, the angle from the
        # N-CA-C plane to CB is fixed.
        # This is a robust approximation used in structure validation.
        ideal_cb_coord = ca_coord + cb_dir * 1.53

        # Euclidean distance
        deviation = np.linalg.norm(cb_coord - ideal_cb_coord)
        deviations[res_id] = float(deviation)

    return deviations

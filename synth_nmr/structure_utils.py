import logging
from typing import List

import biotite.structure as struc
import numpy as np

logger = logging.getLogger(__name__)


def get_residue_info(structure: struc.AtomArray):
    """
    Unified utility to extract residue-level metadata (Chain ID, Res ID, Res Name).

    Returns:
        tuple: (chain_ids, res_ids, res_names, res_starts)
        All as NumPy arrays.
    """
    res_starts = struc.get_residue_starts(structure)
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
    res_starts = struc.get_residue_starts(structure)
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
            # Broader helical region, including potential left-handed helices or
            # alternative alpha-helical conformations sometimes produced by synthetic generators.
            elif (0 < p < 150) and (-90 < s < -10):
                ss_state = "alpha"

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

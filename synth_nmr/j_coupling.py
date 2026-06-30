"""Scalar Coupling (J-coupling) calculations."""

# EDUCATIONAL NOTE - Karplus Equation:
# ====================================
# Scalar couplings (J-couplings) are mediated through chemical bonds.
# The 3-bond coupling (^3J) depends heavily on the torsion angle between the atoms.
#
# The physical origin of J-coupling lies in the Fermi contact interaction, where
# electron spins transmit magnetic information between nuclei. In a 3-bond
# system (A-B-C-D), the overlap of the electron orbitals is modulated by the
# dihedral angle (theta) around the B-C bond.
#
# For the backbone amide proton (HN) and alpha proton (HA), the coupling ^3J_HNHa
# tells us about the Phi angle, and thus the secondary structure.
#
# Formula:
#   ^3J = A * cos^2(theta) + B * cos(theta) + C
#
# Where theta = Phi - 60 degrees (phase shift).
# Typical values:
# - Alpha Helix (Phi ~ -60): theta ~ -120 -> J is small (~4 Hz)
# - Beta Sheet (Phi ~ -120): theta ~ -180 -> J is large (~9 Hz)
# - Random Coil: Averaged (~7 Hz)
#
# This allows NMR spectroscopists to determine secondary structure just by measuring J-couplings!
#
# EDUCATIONAL NOTE - The Karplus Curve Shape:
# ==========================================
# The Karplus equation is periodic, meaning multiple angles can produce the same
# coupling value. This "degeneracy" is a major challenge in NMR structure
# determination. For example, a coupling of 6 Hz could correspond to four
# different phi angles. To resolve this, researchers often use multiple types
# of J-couplings (e.g., ^3J_HNHa, ^3J_NCb, ^3J_CC) simultaneously to "triangulate"
# the correct dihedral angle.

import logging
from typing import Dict

import biotite.structure as struc
import numpy as np

logger = logging.getLogger(__name__)

# Vuister & Bax parameters (J. Am. Chem. Soc. 1993, 115, 7772-7777)
# These parameters were derived by correlating measured J-couplings with
# high-resolution crystal structures of protein G and ubiquitin.
# They are the most widely used parameters for backbone 3J_HNHa prediction.
KARPLUS_PARAMS = {"A": 6.51, "B": -1.76, "C": 1.60}

# Parameters for 3J(Ha, Hb) coupling dependent on chi1 side-chain angle
# Typical Karplus parameters for sp3-sp3 H-C-C-H couplings
# These are often used to identify side-chain rotamers.
KARPLUS_HA_HB_PARAMS = {"A": 9.5, "B": -1.6, "C": 1.8}

# Parameters for 3J(C', Cg) coupling dependent on chi1 side-chain angle
# Typical Karplus parameters for carbon-carbon couplings
#
# EDUCATIONAL NOTE - Carbon-Carbon J-Couplings:
# Carbon-13 has a much lower gyromagnetic ratio than Hydrogen-1.
# This results in significantly smaller absolute magnitudes for
# Carbon-Carbon couplings (typically 0 - 5 Hz) compared to
# Proton-Proton couplings (which can exceed 10 Hz). The Karplus
# constants (A, B, C) are thus correspondingly smaller.
KARPLUS_C_CG_PARAMS = {"A": 4.5, "B": -1.2, "C": 0.1}

# EDUCATIONAL NOTE: Theoretical Precision vs Experimental Reality
# ==============================================================
# While the Karplus equation provides a beautiful theoretical framework,
# experimental J-couplings are affected by:
# 1. Local dynamics: The measured value is a vibrational average.
# 2. Solvent effects: Hydrogen bonding to the amide proton can shift the curve.
# 3. Electronic effects: Neighboring electronegative atoms (like Oxygen)
#    can perturb the electron distribution and thus the coupling.


def calculate_hn_ha_coupling_from_phi(phi_degrees: np.ndarray) -> np.ndarray:
    """
    Calculate the 3J(HN-HA) coupling constant from Phi angles in degrees.

    # EDUCATIONAL NOTE - The Karplus Equation
    # ========================================
    # The magnitude of the 3-bond coupling (^3J) depends heavily on the torsion
    # angle between the atoms. The physical origin lies in the Fermi contact
    # interaction, where electron spins transmit magnetic information between nuclei.
    #
    # Formula: ^3J = A * cos^2(theta) + B * cos(theta) + C
    #
    # For HN-HA coupling, the relationship theta = phi - 60 is a geometric
    # consequence of how the phi angle and the Karplus angle are defined.
    # - Phi (C'-N-Ca-C') measures the rotation around the N-Ca bond.
    # - Theta for 3J(HN-HA) is the dihedral H-N-Ca-HA.
    #
    # Alpha Helix (Phi ~ -60): theta ~ -120 -> J is small (~4 Hz)
    # Beta Sheet (Phi ~ -120): theta ~ -180 -> J is large (~9 Hz)

    Args:
        phi_degrees: Backbone Phi angle in degrees (NumPy array).

    Returns:
        Predicted J-coupling in Hz (NumPy array).
    """
    # Theta = Phi - 60 degrees
    theta_rad = np.radians(phi_degrees - 60.0)

    cos_theta = np.cos(theta_rad)
    j_vals = (
        (KARPLUS_PARAMS["A"] * (cos_theta**2))
        + (KARPLUS_PARAMS["B"] * cos_theta)
        + KARPLUS_PARAMS["C"]
    )
    return np.asarray(j_vals, dtype=np.float64)


def predict_couplings_from_phi_map(phi_map: Dict[int, float]) -> Dict[int, float]:
    """
    Predict HN-HA couplings for a set of residues from a Phi map.

    Args:
        phi_map: Dictionary mapping Residue ID -> Phi angle (degrees).

    Returns:
        Dictionary mapping Residue ID -> J-coupling (Hz).
    """
    res_ids = np.array(list(phi_map.keys()))
    phi_vals = np.array(list(phi_map.values()))
    j_vals = calculate_hn_ha_coupling_from_phi(phi_vals)
    return {int(rid): float(round(j, 2)) for rid, j in zip(res_ids, j_vals) if not np.isnan(j)}


def calculate_hn_ha_coupling(structure: struc.AtomArray) -> Dict[str, Dict[int, float]]:
    """
    Calculate 3J_HNHa coupling constants for the protein backbone.

    # ── J-Coupling Physics and Structural Biology ────────────────────────
    # The 3J_HNHa coupling constant is a measure of the scalar (through-bond)
    # magnetic interaction between the amide proton (HN) and the alpha
    # proton (Ha). This interaction is mediated by the intervening bonds
    # (HN-N, N-CA, CA-Ha).
    #
    # Crucially, the magnitude of this coupling depends on the dihedral
    # angle Phi (φ) according to the Karplus equation:
    #   J(φ) = A cos²(θ) + B cos(θ) + C
    # where θ is the H-N-C-H dihedral angle (related to Phi).
    #
    # Scalar couplings are mediated by electrons in the chemical bonds.
    # The interaction is sensitive to the overlap of the electron wavefunctions,
    # which in turn is a function of the dihedral angle. In structural biology,
    # J-couplings are one of the most direct ways to measure torsion angles.
    #
    # Proteins are dynamic, and the measured J-coupling is actually a
    # time-average over the conformational ensemble. Here we predict the
    # value based on a single static structure or a single frame.
    #
    # Large 3J_HNHa values (~8-10 Hz) typically indicate beta-sheet regions,
    # where the Phi angle is around -120 to -150 degrees.
    # Small 3J_HNHa values (~3-5 Hz) indicate alpha-helical regions,
    # where the Phi angle is around -60 degrees.
    # ─────────────────────────────────────────────────────────────────────

    Args:
        structure: biotite.structure.AtomArray containing the protein.

    Returns:
        Dict keyed by Chain ID -> Residue ID -> J-coupling value (Hz).
    """
    from synth_nmr.structure_utils import get_residue_info

    logger.info("Calculating 3J_HNHa scalar couplings...")
    # Filter for amino acids to avoid mismatches with HETATMs (waters/ions)
    protein_mask = struc.filter_amino_acids(structure)
    structure = structure[protein_mask]
    if structure.array_length() == 0:
        return {}

    # Calculate backbone dihedrals
    phi, _, _ = struc.dihedral_backbone(structure)

    # Get residue info using unified utility
    chain_ids, res_ids, _, _ = get_residue_info(structure)

    if len(phi) != len(res_ids):
        logger.warning(
            f"Mismatch in backbone angles count ({len(phi)}) vs residue count ({len(res_ids)})."
        )
        return {}

    # Vectorized Karplus calculation
    phi_deg = np.degrees(phi)
    j_vals = calculate_hn_ha_coupling_from_phi(phi_deg)

    # Build results dictionary
    results: Dict[str, Dict[int, float]] = {}
    for chain_id, res_id, j_val in zip(chain_ids, res_ids, j_vals):
        if np.isnan(j_val):
            continue

        if chain_id not in results:
            results[chain_id] = {}
        results[chain_id][int(res_id)] = float(round(j_val, 2))

    return results


# EDUCATIONAL NOTE: Chi1 Dihedral
# ===============================
# The chi1 angle describes the rotation of the side chain around the Ca-Cb bond.
# It is defined by the four atoms: N - Ca - Cb - Cg
# (or N - Ca - Cb - Sg or O/other heavy atoms depending on the amino acid).
#
# This angle is highly informative for identifying the dominant "rotamer" state
# (gauche+, gauche-, or trans) of a residue's side-chain in solution NMR!
#
# By mapping out these structural distributions, researchers can determine whether
# a sidechain is rigidly locked into a specific conformation (e.g. buried in a
# hydrophobic core) or interconverting rapidly between multiple rotameric states
# on the NMR timescale.
#
# Different amino acids have different preferred chi1 distributions based on
# their steric bulk. For example, Valine is highly constrained compared to Leucine.


def _get_chi1_angles(structure: struc.AtomArray) -> Dict[str, Dict[int, float]]:
    """
    Extracts the chi1 (x1) dihedral angle for all valid residues.

    Args:
        structure: biotite.structure.AtomArray containing the protein.

    Returns:
        Dict keyed by Chain ID -> Residue ID -> chi1 angle (radians).
    """
    results: Dict[str, Dict[int, float]] = {}

    # Filter for amino acids to ensure chi1 angles map correctly to residues
    protein_mask = struc.filter_amino_acids(structure)
    structure = structure[protein_mask]
    if structure.array_length() == 0:
        return {}  # pragma: no cover

    res_starts: np.ndarray = struc.get_residue_starts(structure)
    for i, start_idx in enumerate(res_starts):
        end_idx = res_starts[i + 1] if i + 1 < len(res_starts) else len(structure)
        res_atoms = structure[start_idx:end_idx]

        chain_id = str(res_atoms.chain_id[0])
        res_id = int(res_atoms.res_id[0])

        # We need N, CA, CB, and a CG (or equivalent gamma heavy atom)
        try:
            n_idx = np.where(res_atoms.atom_name == "N")[0][0] + start_idx
            ca_idx = np.where(res_atoms.atom_name == "CA")[0][0] + start_idx
            cb_idx = np.where(res_atoms.atom_name == "CB")[0][0] + start_idx

            # Find a gamma atom: CG, CG1, CG2, SG, OG, OG1
            gamma_atoms = ["CG", "CG1", "CG2", "SG", "OG", "OG1"]
            cg_idx = -1
            for g_name in gamma_atoms:
                matches = np.where(res_atoms.atom_name == g_name)[0]
                if len(matches) > 0:
                    cg_idx = matches[0] + start_idx
                    break

            if cg_idx == -1:
                continue  # No valid gamma atom found (e.g., Glycine, Alanine)

            # Compute dihedral in radians
            angle = float(
                struc.dihedral(
                    structure.coord[n_idx],
                    structure.coord[ca_idx],
                    structure.coord[cb_idx],
                    structure.coord[cg_idx],
                )
            )

            if chain_id not in results:
                results[chain_id] = {}
            results[chain_id][res_id] = angle

        except IndexError:
            # Missing backbone or CB atoms
            continue

    return results


# EDUCATIONAL NOTE: 3J(Ha, Hb) Couplings
# ======================================
# The 3J coupling between the alpha and beta protons provides a direct probe
# for the side-chain chi1 rotamer distributions.
#
# In a typical staggered conformation:
# - Gauche rotamers (chi1 ~ -60 or +60) lead to smaller couplings (~2-4 Hz).
# - trans rotamers (chi1 ~ 180) lead to larger, antiperiplanar couplings (~9-12 Hz).
#
# By measuring this in NMR, we constrain how the side-chain is orienting
# itself relative to the backbone!
#
# Because the Ha-Hb coupling pathway depends critically on the stereochemical
# assignment of the beta protons (pro-R vs pro-S), precise interpretation of
# experimental data usually requires assigning which beta proton is which.
# Failure to correctly assign these "pro-chiral" protons can lead to errors
# in structure determination.
def calculate_ha_hb_coupling(structure: struc.AtomArray) -> Dict[str, Dict[int, float]]:
    """
    Calculate 3J_HaHb coupling constants dependent on the chi1 angle.

    Args:
        structure: biotite.structure.AtomArray containing the protein.

    Returns:
        Dict keyed by Chain ID -> Residue ID -> J-coupling value (Hz).
    """
    logger.info("Calculating 3J_HaHb side-chain couplings...")
    chi1_angles = _get_chi1_angles(structure)
    results: Dict[str, Dict[int, float]] = {}

    for chain_id, residues in chi1_angles.items():
        results[chain_id] = {}
        for res_id, chi1_rad in residues.items():
            # Phase shifts depend exactly on pro-R/pro-S proton alignments,
            # but we use a simplified idealized theta relation for this implementation.
            theta = float(chi1_rad - np.deg2rad(120.0))

            cos_theta = float(np.cos(theta))
            j_val = float(
                (KARPLUS_HA_HB_PARAMS["A"] * (cos_theta**2))
                + (KARPLUS_HA_HB_PARAMS["B"] * cos_theta)
                + KARPLUS_HA_HB_PARAMS["C"]
            )
            results[chain_id][res_id] = float(round(j_val, 2))

    return results


# EDUCATIONAL NOTE: 3J(C', Cg) Couplings
# ======================================
# Unlike proton-proton couplings which rely on isotopic labeling or natural
# abundance 1H detection, modern experiments can measure Carbon-Carbon couplings.
#
# The interaction between the backbone carbonyl (C') and side-chain gamma
# carbon (Cg) also operates via the Karplus curve over 3 bonds.
# This provides critical, orthogonal verification of the chi1 angle!
#
# Because the gyromagnetic ratio of Carbon-13 is much lower than Hydrogen-1,
# the absolute magnitude in Hz is significantly smaller (usually 0 - 5 Hz),
# parameterized by a completely different A, B, and C set.
#
# This measurement is incredibly valuable because it doesn't suffer from the
# pro-chiral ambiguity that plagues Ha-Hb measurements. The C' and Cg atoms
# are unique, providing a direct, unambiguous readout of the chi1 angle.


def calculate_c_cg_coupling(structure: struc.AtomArray) -> Dict[str, Dict[int, float]]:
    """
    Calculate 3J(C', Cg) coupling constants dependent on the chi1 angle.

    Args:
        structure: biotite.structure.AtomArray containing the protein.

    Returns:
        Dict keyed by Chain ID -> Residue ID -> J-coupling value (Hz).
    """
    logger.info("Calculating 3J_C'Cg side-chain couplings...")
    chi1_angles = _get_chi1_angles(structure)
    results: Dict[str, Dict[int, float]] = {}

    for chain_id, residues in chi1_angles.items():
        results[chain_id] = {}
        for res_id, chi1_rad in residues.items():
            # The dihedral angle pathway for C' - CA - CB - CG intersects chi1
            theta = float(chi1_rad)

            cos_theta = float(np.cos(theta))
            j_val = float(
                (KARPLUS_C_CG_PARAMS["A"] * (cos_theta**2))
                + (KARPLUS_C_CG_PARAMS["B"] * cos_theta)
                + KARPLUS_C_CG_PARAMS["C"]
            )
            results[chain_id][res_id] = float(round(j_val, 2))

    return results

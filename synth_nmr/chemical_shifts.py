import logging
import os
import subprocess
import tempfile
from typing import Any, Dict

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np

from synth_nmr.structure_utils import get_secondary_structure

logger = logging.getLogger(__name__)
try:
    from numba import njit
except ImportError:

    def njit(func: Any = None, **kwargs: Any) -> Any:
        if func is None:
            return lambda f: f
        return func


# --- Random Coil Chemical Shifts (Wishart et al.) ---
# EDUCATIONAL NOTE - Random Coil Shifts:
# ======================================
# "Random Coil" refers to a protein state with no fixed secondary structure (a flexible chain).
# The chemical shift of an atom in a random coil depends primarily on its amino acid type.
#
# These values serve as the "baseline" or "zero point" for structure prediction.
# Any deviation from these values (Secondary Shift) indicates structural formation:
# - Alpha Helix formation moves C-alpha downfield (higher ppm) and N upfield (lower ppm).
# - Beta Sheet formation moves C-alpha upfield (lower ppm) and N downfield (higher ppm).
#
# Reference: Wishart, D.S. et al. (1995) J. Biomol. NMR.
# Referenced to DSS at 25C.
# Values for: HA, CA, CB, C, N, HN (Amide H)
# Units: ppm
RANDOM_COIL_SHIFTS: Dict[str, Dict[str, float]] = {
    "ALA": {"HA": 4.32, "CA": 52.5, "CB": 19.1, "C": 177.8, "N": 123.8, "H": 8.24},
    "ARG": {"HA": 4.34, "CA": 56.0, "CB": 30.9, "C": 176.3, "N": 121.3, "H": 8.23},
    "ASN": {"HA": 4.75, "CA": 53.1, "CB": 38.9, "C": 175.2, "N": 118.7, "H": 8.75},
    "ASP": {"HA": 4.66, "CA": 54.2, "CB": 41.1, "C": 176.3, "N": 120.4, "H": 8.34},
    "CYS": {"HA": 4.69, "CA": 58.2, "CB": 28.0, "C": 174.6, "N": 118.8, "H": 8.32},
    "CYX": {"HA": 4.69, "CA": 58.2, "CB": 28.0, "C": 174.6, "N": 118.8, "H": 8.32},
    "GLN": {"HA": 4.32, "CA": 56.0, "CB": 29.4, "C": 176.0, "N": 120.4, "H": 8.25},
    "GLU": {"HA": 4.29, "CA": 56.6, "CB": 29.9, "C": 176.6, "N": 120.2, "H": 8.35},
    "GLY": {"HA": 3.96, "CA": 45.1, "CB": 0.0, "C": 174.9, "N": 108.8, "H": 8.33},
    "HIS": {"HA": 4.63, "CA": 55.0, "CB": 29.0, "C": 174.1, "N": 118.2, "H": 8.42},
    "HID": {"HA": 4.63, "CA": 55.0, "CB": 29.0, "C": 174.1, "N": 118.2, "H": 8.42},
    "HIE": {"HA": 4.63, "CA": 55.0, "CB": 29.0, "C": 174.1, "N": 118.2, "H": 8.42},
    "HIP": {"HA": 4.63, "CA": 55.0, "CB": 29.0, "C": 174.1, "N": 118.2, "H": 8.42},
    "ILE": {"HA": 4.17, "CA": 61.1, "CB": 38.8, "C": 176.4, "N": 121.4, "H": 8.00},
    "LEU": {"HA": 4.34, "CA": 55.1, "CB": 42.4, "C": 177.6, "N": 121.8, "H": 8.16},
    "LYS": {"HA": 4.32, "CA": 56.2, "CB": 33.1, "C": 176.6, "N": 120.4, "H": 8.29},
    "MET": {"HA": 4.48, "CA": 55.4, "CB": 32.6, "C": 176.3, "N": 119.6, "H": 8.28},
    "PHE": {"HA": 4.62, "CA": 57.7, "CB": 39.6, "C": 175.8, "N": 120.3, "H": 8.12},
    "PRO": {"HA": 4.42, "CA": 63.3, "CB": 32.1, "C": 177.3, "N": 0.0, "H": 0.0},  # No Amide N/H
    "SER": {"HA": 4.47, "CA": 58.3, "CB": 63.8, "C": 174.6, "N": 115.7, "H": 8.31},
    "THR": {"HA": 4.35, "CA": 61.8, "CB": 69.8, "C": 174.7, "N": 113.6, "H": 8.15},
    "TRP": {"HA": 4.66, "CA": 57.5, "CB": 29.6, "C": 176.1, "N": 121.3, "H": 8.25},
    "TYR": {"HA": 4.55, "CA": 57.9, "CB": 38.8, "C": 175.9, "N": 120.3, "H": 8.12},
    "VAL": {"HA": 4.12, "CA": 62.2, "CB": 32.9, "C": 176.3, "N": 119.9, "H": 8.03},
}

# Module-level variable for noise scale, allowing easy monkeypatching in tests.
# Only used when add_noise=True is passed to predict_empirical_shifts.
_NOISE_SCALE = 0.15

# --- Ring Current B-factors ---
# These empirical scaling constants convert the dimensionless geometric factor
# (1 − 3cos²θ) / r³  into ppm.  They are derived by fitting to experimental
# ring-current-shifted NMR data for aromatic rings in proteins.
# Reference: Case, D.A. (1995) Curr. Opin. Struct. Biol. 5, 272–276.
#            Haigh, C.W. & Mallion, R.B. (1980) Prog. NMR Spectrosc. 13, 303–344.
# Units: ppm·Å³
RC_B_FACTOR_H: float = 11.0  # Proton ring-current scaling (literature range 8–15 ppm·Å³)
RC_B_FACTOR_C: float = 2.0  # Carbon ring-current scaling (ca. 5–6× smaller than proton)


# --- Secondary Structure Offsets (SPARTA+) ---
# EDUCATIONAL NOTE - Secondary Chemical Shifts:
# =============================================
# The local magnetic field experienced by a nucleus is heavily influenced by the
# geometry of the protein backbone (Phi/Psi angles).
#
# SPARTA+ (Shift Prediction from Analogy in Residue-type and Torsion Angle):
# It predicts chemical shifts by finding homologous structures with similar geometry.
#
# Our implementation uses simple statistical offsets instead of database mining,
# but follows the same principle: Geometry determines Shift.
#
# Reference State: DSS (4,4-dimethyl-4-silapentane-1-sulfonic acid)
# This is the "Zero" for proton/carbon NMR, much like sea level for altitude.
# Using a standard reference ensures shifts are comparable across different labs.
#
# Approximate mean offsets for Helical and Sheet conformations relative to random coil
# Based on general statistics (e.g. Spera & Bax 1991)
# Format: {metric: {Helix: val, Sheet: val}}
SECONDARY_SHIFTS: Dict[str, Dict[str, float]] = {
    # C-alpha: Shifted downfield (positive) in Helix, upfield (negative) in Sheet
    "CA": {"alpha": 3.1, "beta": -1.5},
    # C-beta: Opposite trend to C-alpha
    "CB": {"alpha": -0.5, "beta": 2.2},
    # Carbonyl Carbon: Follows C-alpha trend
    "C": {"alpha": 2.2, "beta": -1.6},
    # H-alpha: Shifted upfield (negative) in Helix, downfield (positive) in Sheet
    "HA": {"alpha": -0.4, "beta": 0.5},
    # Amide N: Complex, but generally upfield in Helix
    "N": {"alpha": -1.5, "beta": 1.2},
    "H": {"alpha": -0.2, "beta": 0.3},
}

# --- Ring Current Intensity Factors ---
# EDUCATIONAL NOTE - Ring Current Physics:
# ========================================
# Aromatic rings (Benchmark: Benzene) have delocalized pi-electrons that circulate
# when exposed to a magnetic field, creating an opposing induced magnetic field.
#
# - Regions ABOVE/BELOW the ring are SHIELDED (Field opposes external field -> Lower ppm).
# - Regions in the PLANE of the ring are DESHIELDED (Field adds to external field -> Higher ppm).
#
# Model: Point Dipole approximation.
# Shift = Intensity * (1 - 3*cos^2(theta)) / r^3
#
# References for further reading:
# 1. Haigh, C. W., & Mallion, R. B. (1980). "Ring current theories in nuclear magnetic resonance".
#    Progress in Nuclear Magnetic Resonance Spectroscopy, 13(4), 303-344.
# 2. Pople, J. A. (1956). "Proton magnetic shielding in aromatic compounds".
#    The Journal of Chemical Physics, 24(5), 1111.
# 3. Case, D. A. (1995). "Chemical shifts in proteins".
#    Current Opinion in Structural Biology, 5(2), 272-276.
#
# Intensities are relative to Benzene.
RING_INTENSITIES = {
    "PHE": 1.2,  # Benzene ring (Standard)
    "TYR": 1.2,  # Phenol ring (Similar to Benzene)
    "TRP": 1.3,  # Indole (Stronger system)
    "HIS": 0.5,  # Imidazole (Weaker, depends on protonation)
    "HID": 0.5,
    "HIE": 0.5,
    "HIP": 0.5,
}


def _get_random_coil_shifts(res_name: str) -> Dict[str, float]:
    """
    Extract baseline Random Coil shifts for a given residue.

    EDUCATIONAL NOTE - Random Coil Shifts:
    ======================================
    "Random Coil" refers to a protein state with no fixed secondary structure (a flexible chain).
    The chemical shift of an atom in a random coil depends primarily on its amino acid type.

    These values serve as the "baseline" or "zero point" for structure prediction.
    Any deviation from these values (Secondary Shift) indicates structural formation:
    - Alpha Helix formation moves C-alpha downfield (higher ppm) and N upfield (lower ppm).
    - Beta Sheet formation moves C-alpha upfield (lower ppm) and N downfield (higher ppm).

    Reference: Wishart, D.S. et al. (1995) J. Biomol. NMR.
    Referenced to DSS at 25C.
    Values for: HA, CA, CB, C, N, HN (Amide H)
    Units: ppm
    """
    return RANDOM_COIL_SHIFTS.get(res_name, {})


def _apply_secondary_structure_offsets(atom_type: str, ss_state: str, base_val: float) -> float:
    """
    Apply SPARTA+ style statistical offsets based on helical or sheet geometries.

    EDUCATIONAL NOTE - Secondary Chemical Shifts:
    =============================================
    The local magnetic field experienced by a nucleus is heavily influenced by the
    geometry of the protein backbone (Phi/Psi angles).

    SPARTA+ (Shift Prediction from Analogy in Residue-type and Torsion Angle):
    It predicts chemical shifts by finding homologous structures with similar geometry.

    Our implementation uses simple statistical offsets instead of database mining,
    but follows the same principle: Geometry determines Shift.

    Reference State: DSS (4,4-dimethyl-4-silapentane-1-sulfonic acid)
    This is the "Zero" for proton/carbon NMR, much like sea level for altitude.
    Using a standard reference ensures shifts are comparable across different labs.

    Approximate mean offsets for Helical and Sheet conformations relative to random coil
    Based on general statistics (e.g. Spera & Bax 1991)
    """
    offset = SECONDARY_SHIFTS.get(atom_type, {}).get(ss_state, 0.0)
    return base_val + offset


def predict_chemical_shifts(structure: struc.AtomArray) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Predict chemical shifts using the highest-accuracy available model.
    Attempt to use SHIFTX2 first, as it provides the best baseline accuracy.
    If SHIFTX2 is not available in the system PATH, or if prediction fails,
    falls back to the SPARTA+ empirical prediction model.

    Note: The `NeuralShiftPredictor` (available in `synth_nmr.neural_shifts`) is
    strictly experimental and serves as an educational example of how Protein
    Language Models (PLMs) could be applied to NMR prediction. It currently requires
    extensive retraining with geometric attention mechanisms to match or exceed
    classical empirical force fields.
    """
    shiftx_predictor = ShiftX2Predictor()

    if shiftx_predictor.is_available():
        try:
            shifts = shiftx_predictor.predict(structure)
            if shifts:
                logger.info("Successfully predicted chemical shifts using SHIFTX2.")
                return shifts
            else:
                logger.warning(
                    "SHIFTX2 returned empty predictions. Falling back to empirical SPARTA+ model."
                )
        except Exception as e:
            logger.warning(
                f"SHIFTX2 prediction failed: {e}. Falling back to empirical SPARTA+ model."
            )
    else:
        logger.warning(
            "SHIFTX2 executable not found. Falling back to empirical SPARTA+ model. "
            "To use SHIFTX2, ensure it is in your PATH or set the SHIFTX2_DIR environment variable. "
            "For installation, see http://www.shiftx2.ca/download.html or use SBGrid ('sbgrid-cli install shiftx2')."
        )
    return predict_empirical_shifts(structure)


def predict_empirical_shifts(
    structure: struc.AtomArray,
    add_noise: bool = False,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Predict chemical shifts based on secondary structure and ring currents.

    This function combines random coil shifts, secondary structure-based offsets (SPARTA+-like),
    and ring current effects from aromatic residues to predict protein chemical shifts.

    EDUCATIONAL NOTE - Prediction Algorithm:
    ========================================
    1. Calculate Backbone Dihedrals (Phi/Psi) for every residue.
    2. Classify Secondary Structure:
       - Alpha: Phi ~ -60, Psi ~ -45
       - Beta:  Phi ~ -120, Psi ~ 120
       - Coil:  Everything else
    3. Calculate Shift:
       Shift = Random_Coil + Structure_Offset + Noise

    LIMITATIONS:
    - Ring Current Effects (for protons): While an O(N^2) geometry check was previously
      a concern, these effects are now included for protons near aromatic rings
      (Phe, Tyr, Trp, His) using a point-dipole approximation. This is crucial for
      protons in close proximity to aromatic systems. Carbon atoms are not currently
      included for ring current effects.
    - H-Bonding: Hydrogen bonds affect Amide H shifts significantly. We omit this for simplicity.
    - Sequence History: Real shifts depend on (i-1) and (i+1) neighbor types. We omit this for simplicity.

    Args:
        structure: A biotite.structure.AtomArray containing the protein. Must not be empty.

    Returns:
        A nested dictionary of predicted shifts: {chain_id: {res_id: {atom_name: value}}}

    Raises:
        TypeError: If the input is not a biotite.structure.AtomArray.
        ValueError: If the input structure is empty.

    LIMITATIONS:
    - Noise: By default this function is *deterministic* (add_noise=False).
      Pass add_noise=True to add small Gaussian noise (~0.15 ppm) to each
      predicted shift, simulating experimental measurement error.  This is
      useful for generating synthetic training data but should NOT be used
      for validation or structure refinement, where reproducibility matters.
    """
    logger.info("Predicting chemical shifts (SPARTA+ model with ring currents)...")

    # 1. Input Validation
    if not isinstance(structure, struc.AtomArray):
        raise TypeError("Input 'structure' must be a biotite.structure.AtomArray.")
    if structure.array_length() == 0:
        logger.warning("Input 'structure' is empty. Cannot predict chemical shifts.")
        return {}

    # Filter for amino acids to avoid mismatches with HETATMs (waters/ions)
    protein_mask = struc.filter_amino_acids(structure)
    structure = structure[protein_mask]
    if structure.array_length() == 0:
        return {}

    try:
        # 2. Get Secondary Structure and Aromatic Ring Info
        ss_list = get_secondary_structure(structure)
        rings = _get_aromatic_rings(structure)
        if rings.size > 0:
            logger.debug(f"Found {rings.shape[0]} aromatic rings for ring current calculation.")

        # 3. Iterate through residues and calculate shifts
        results: Dict[str, Dict[int, Dict[str, float]]] = {}
        for i, start_idx in enumerate(struc.get_residue_starts(structure)):
            end_idx = (
                struc.get_residue_starts(structure)[i + 1]
                if i + 1 < len(struc.get_residue_starts(structure))
                else None
            )
            res_atoms = structure[start_idx:end_idx]

            res_id = res_atoms.res_id[0]
            res_name = res_atoms.res_name[0]
            chain_id = res_atoms.chain_id[0]

            rc_shifts = _get_random_coil_shifts(res_name)
            if not rc_shifts:
                logger.debug(f"Skipping non-standard residue: {res_name} {res_id}")
                continue

            ss_state = ss_list[i] if i < len(ss_list) else "coil"
            logger.debug(f"Processing ResID {res_id} ({res_name}), SS: {ss_state}")

            atom_shifts = {}

            for atom_type, base_val in rc_shifts.items():
                if base_val == 0:  # Skip atoms with no defined random coil shift (e.g., Proline H)
                    continue

                # Add secondary structure offset to the random coil baseline
                val = _apply_secondary_structure_offsets(atom_type, ss_state, base_val)

                # Add ring current shift
                # EDUCATIONAL NOTE - Ring Currents for Carbons:
                # While most prominent for protons due to their proximity to the ring plane,
                # ring currents also affect Carbon shifts (CA, CB). The magnitude is
                # typically smaller in ppm than for protons.
                is_proton = atom_type.startswith("H")
                is_carbon = atom_type in ["CA", "CB"]

                if rings.size > 0 and (is_proton or is_carbon):
                    try:
                        target_atom = res_atoms[res_atoms.atom_name == atom_type][0]
                        # Use named module constants for B-factors (see RC_B_FACTOR_H/C).
                        b_factor = RC_B_FACTOR_H if is_proton else RC_B_FACTOR_C
                        rc_shift = _calculate_ring_current_shift(target_atom.coord, rings, b_factor)
                        val += rc_shift
                    except IndexError:
                        # Atom not found in this specific residue, skip ring current
                        logger.debug(
                            f"Atom {atom_type} not found in residue {res_id} for ring current calculation."
                        )
                        pass

                # Optionally add Gaussian noise to simulate experimental scatter.
                # Disabled by default to ensure reproducible predictions.
                # Enable with add_noise=True when generating synthetic training data.
                if add_noise and base_val != 0:
                    val += np.random.normal(0, _NOISE_SCALE)

                atom_shifts[atom_type] = round(val, 3)

            if chain_id not in results:
                results[chain_id] = {}
            results[chain_id][res_id] = atom_shifts

        logger.info(
            f"Successfully predicted chemical shifts for {struc.get_residue_count(structure)} residues."
        )
        return results

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during chemical shift prediction: {e}", exc_info=True
        )
        raise


def calculate_csi(
    shifts: Dict[str, Dict[int, Dict[str, float]]], structure: struc.AtomArray
) -> Dict[str, Dict[int, float]]:
    """
    Calculate the Chemical Shift Index (CSI) for C-alpha atoms.

    The CSI is the deviation of an observed chemical shift from its random coil value.
    It is a reliable indicator of secondary structure.
    - Positive Delta(CA) > 0.7 ppm suggests a Helical conformation.
    - Negative Delta(CA) < -0.7 ppm suggests a Sheet conformation.

    Args:
        shifts: A dictionary of chemical shifts, as produced by `predict_chemical_shifts`.
        structure: A biotite.structure.AtomArray, required for mapping residue IDs to names.

    Returns:
        A dictionary containing the C-alpha CSI for each residue: {chain_id: {res_id: delta_ppm}}

    Raises:
        TypeError: If inputs are not of the correct type.
        ValueError: If inputs are empty.
    """
    logger.info("Calculating Chemical Shift Index (CSI) for C-alpha atoms...")

    # 1. Input Validation
    if not isinstance(shifts, dict):
        raise TypeError("Input 'shifts' must be a dictionary.")
    if not shifts:
        logger.warning("Input 'shifts' dictionary is empty. Returning no CSI data.")
        return {}
    if not isinstance(structure, struc.AtomArray):
        raise TypeError("Input 'structure' must be a biotite.structure.AtomArray.")
    if structure.array_length() == 0:
        logger.warning("Input 'structure' is empty. Cannot calculate CSI.")
        return {}

    try:
        # 2. Create a mapping from residue ID to residue name for quick lookup
        res_names = {}
        for idx in struc.get_residue_starts(structure):
            res = structure[idx]
            res_names[res.res_id] = res.res_name
        if not res_names:
            logger.warning("Could not create a residue map from the provided structure.")
            return {}

        # 3. Calculate CSI
        csi_data: Dict[str, Dict[int, float]] = {}
        for chain_id, chain_shifts in shifts.items():
            if not isinstance(chain_shifts, dict):
                continue
            csi_data[chain_id] = {}

            for res_id, atom_shifts in chain_shifts.items():
                if not isinstance(atom_shifts, dict):
                    continue

                res_name = res_names.get(res_id)
                if not res_name:
                    logger.debug(
                        f"Residue ID {res_id} from shifts not found in structure. Skipping."
                    )
                    continue

                if "CA" in atom_shifts and res_name in RANDOM_COIL_SHIFTS:
                    measured = atom_shifts["CA"]
                    random_coil_val = RANDOM_COIL_SHIFTS[res_name].get("CA")

                    if random_coil_val is not None:
                        delta = measured - random_coil_val
                        csi_data[chain_id][res_id] = round(delta, 3)
                        logger.debug(
                            f"CSI for {res_name} {res_id}: Measured={measured}, RC={random_coil_val}, Delta={delta}"
                        )
                    else:
                        logger.debug(
                            f"No random coil 'CA' value for {res_name}. Skipping CSI calculation for this residue."
                        )

        logger.info("CSI calculation complete.")
        return csi_data

    except Exception as e:
        logger.error(f"An unexpected error occurred during CSI calculation: {e}", exc_info=True)
        raise


def _get_aromatic_rings(structure: struc.AtomArray) -> np.ndarray:
    """
    Identify aromatic rings and calculate their centers and normal vectors.
    """
    rings = []

    # Iterate residues
    res_starts = struc.get_residue_starts(structure)
    for idx in res_starts:
        res = structure[idx]
        res_name = res.res_name

        if res_name in RING_INTENSITIES:
            # Extract ring atoms to calculate geometry
            # Simplified definition of ring atoms
            res_slice = structure[structure.res_id == res.res_id]

            if res_name in ["PHE", "TYR"]:
                # 6-membered ring: CG, CD1, CD2, CE1, CE2, CZ
                ring_atoms = res_slice[
                    np.isin(res_slice.atom_name, ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"])
                ]
            elif res_name == "TRP":
                # Indole is 9 atoms, effective center near CD2/CE2 bond.
                # Simplified: averaging all ring atoms
                ring_names = ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]
                ring_atoms = res_slice[np.isin(res_slice.atom_name, ring_names)]
            elif res_name in ["HIS", "HID", "HIE", "HIP"]:
                # 5-membered ring: CG, ND1, CD2, CE1, NE2
                ring_atoms = res_slice[
                    np.isin(res_slice.atom_name, ["CG", "ND1", "CD2", "CE1", "NE2"])
                ]
            else:
                continue

            if len(ring_atoms) >= 3:
                # Geometric Center
                center = np.mean(ring_atoms.coord, axis=0)

                # Normal Vector (Cross product of two vectors in the ring)
                # v1: Center -> Atom 0
                # v2: Center -> Atom 1
                # Normal = v1 x v2
                v1 = ring_atoms[0].coord - center
                v2 = ring_atoms[1].coord - center
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                    intensity = RING_INTENSITIES[res_name]
                    rings.append((center, normal, intensity))

    if not rings:
        return np.empty((0, 7), dtype=np.float64)

    # Convert list of tuples (center, normal, intensity) to (N, 7) array
    ring_array = np.zeros((len(rings), 7), dtype=np.float64)
    for i, (c, n, intensity) in enumerate(rings):
        ring_array[i, 0:3] = c
        ring_array[i, 3:6] = n
        ring_array[i, 6] = intensity

    return ring_array


@njit
def _calculate_ring_current_shift(
    proton_coord: np.ndarray, rings: np.ndarray, b_factor: float = 11.0
) -> float:
    """
    Calculate total ring current shift for a proton from all rings.
    'rings' is a numpy array of shape (N, 7): [cx, cy, cz, nx, ny, nz, intensity]
    Formula: delta = Intensity * B_factor * (1 - 3*cos^2(theta)) / r^3
    """
    total_shift = 0.0
    # B_FACTOR is an empirical scaling constant, typically derived from fitting
    # experimental data or theoretical calculations for a reference aromatic system.
    # It converts the dimensionless geometric factor and intensity into ppm.
    # Typical values range from 8 to 15 ppm*A^3.

    for j in range(rings.shape[0]):
        center = rings[j, 0:3]
        normal = rings[j, 3:6]
        intensity = rings[j, 6]

        # Vector from ring center to proton
        v = (proton_coord - center).astype(np.float64)
        r = np.sqrt(np.sum(v**2))

        if r < 1.0:
            continue  # Too close/clashing, ignore singularity

        # Cos(theta) = dot(v, n) / (|v|*|n|) -> |n|=1
        costheta = np.sum(v * normal) / r

        # Geometric Factor G(r, theta) = (1 - 3*cos^2(theta)) / r^3
        # If theta = 0 (above ring), cos=1 -> (1-3)/r^3 = -2/r^3 (Shielding)
        # If theta = 90 (in plane), cos=0 -> (1-0)/r^3 =  1/r^3 (Deshielding)
        geom_factor = (1.0 - 3.0 * costheta**2) / (r**3)

        shift = intensity * b_factor * geom_factor

        total_shift += shift

    return total_shift


class ShiftX2Predictor:
    """
    Wrapper for the external ShiftX2 predictor.

    Requires the 'shiftx2.py' (or 'shiftx2') executable to be in the system PATH,
    in the directory specified by SHIFTX2_DIR, or in typical installation locations.

    ShiftX2 can be installed via SBGrid: 'sbgrid-cli install shiftx2'
    It can also be downloaded from here: http://www.shiftx2.ca/download.html
    """

    def __init__(self, executable: str = "shiftx2.py"):
        self.executable = self._resolve_path(executable)

    def _resolve_path(self, executable: str) -> str:
        """Resolve the full path to the ShiftX2 executable."""
        from shutil import which

        # 1. Check if the provided name is already found by 'which'
        # (this handles absolute paths, relative paths, and PATH search)
        found = which(executable)
        if found:
            return found

        # 2. Check SHIFTX2_DIR environment variable
        shiftx2_dir = os.environ.get("SHIFTX2_DIR")
        if shiftx2_dir:
            for name in [executable, "shiftx2.py", "shiftx2"]:
                if not os.path.isabs(name):
                    path = os.path.join(shiftx2_dir, name)
                    found = which(path)
                    if found:
                        return found

        # 3. Check typical locations
        home = os.path.expanduser("~")
        typical_locations = [
            os.path.join(home, "shiftx2", "shiftx2.py"),
            "/opt/shiftx2/shiftx2.py",
            "/usr/local/bin/shiftx2.py",
            "/usr/local/bin/shiftx2",
        ]
        for loc in typical_locations:
            found = which(loc)
            if found:
                return found

        return executable

    def is_available(self) -> bool:
        """Check if ShiftX2 executable is available."""
        from shutil import which

        return which(self.executable) is not None

    def predict(self, structure: struc.AtomArray) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Run ShiftX2 on a Biotite AtomArray.

        Args:
            structure: Biotite AtomArray.

        Returns:
            Predicted shifts in the standard dictionary format.
        """
        if not self.is_available():
            raise RuntimeError(
                f"ShiftX2 executable '{self.executable}' not found. "
                "Please install it (e.g., via SBGrid), add it to your PATH, or set SHIFTX2_DIR."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = os.path.join(tmpdir, "input.pdb")
            # Some versions of shiftx2 ignore -o and append .cs to the input file
            expected_out_path = pdb_path + ".cs"

            # 1. Write structure to temporary PDB
            pdb_file = pdb.PDBFile()
            pdb_file.set_structure(structure)
            pdb_file.write(pdb_path)

            # 2. Execute ShiftX2
            # Command: shiftx2.py -i <input.pdb>
            try:
                subprocess.run(
                    [self.executable, "-i", pdb_path], check=True, capture_output=True, text=True
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"ShiftX2 failed: {e.stderr}")
                raise RuntimeError(f"ShiftX2 execution failed: {e.stderr}")

            logger.debug(f"Tmpdir contents: {os.listdir(tmpdir)}")
            logger.debug(f"Expecting out path: {expected_out_path}")

            # 3. Parse ShiftX2 output (Simplified CSV/Tabular parsing)
            return self._parse_output(expected_out_path)

    def _parse_output(self, file_path: str) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Parse ShiftX2's default output format.
        Expects a CSV with columns: NUM, RES, ATOMNAME, SHIFT
        or extended columns: NUM, RES, ATOMNAME, SHIFT, CHAIN

        Supports multi-chain proteins by reading the CHAIN column when present.
        Falls back to chain 'A' when the column is absent (single-chain output).
        """
        shifts: Dict[str, Dict[int, Dict[str, float]]] = {}

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ShiftX2 output file not found: {file_path}")

        with open(file_path) as f:
            lines = f.readlines()

        # Skip header lines (usually starts with 'NUM' or similar)
        header_found = False
        chain_col: int = -1  # Index of CHAIN column, -1 means not present
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "NUM" in line and "RES" in line:
                header_found = True
                # Detect whether a CHAIN column is present
                header_parts = [p.strip().upper() for p in line.split(",")]
                if "CHAIN" in header_parts:
                    chain_col = header_parts.index("CHAIN")
                continue
            if not header_found:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                try:
                    res_id = int(parts[0])
                    atom_name = parts[2]
                    val = float(parts[3])

                    # Determine chain: use CHAIN column if present, else default 'A'
                    chain_id = (
                        parts[chain_col].strip()
                        if chain_col >= 0 and chain_col < len(parts)
                        else "A"
                    )

                    if chain_id not in shifts:
                        shifts[chain_id] = {}
                    if res_id not in shifts[chain_id]:
                        shifts[chain_id][res_id] = {}
                    shifts[chain_id][res_id][atom_name] = val
                except ValueError:
                    continue

        return shifts

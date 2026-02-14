
import numpy as np
import biotite.structure as struc
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# --- Optional Numba JIT Support ---
try:
    from numba import njit
except ImportError:
    def njit(func=None, **kwargs):
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
    "GLY": {"HA": 3.96, "CA": 45.1, "CB": 0.0,  "C": 174.9, "N": 108.8, "H": 8.33},
    "HIS": {"HA": 4.63, "CA": 55.0, "CB": 29.0, "C": 174.1, "N": 118.2, "H": 8.42},
    "HID": {"HA": 4.63, "CA": 55.0, "CB": 29.0, "C": 174.1, "N": 118.2, "H": 8.42},
    "HIE": {"HA": 4.63, "CA": 55.0, "CB": 29.0, "C": 174.1, "N": 118.2, "H": 8.42},
    "HIP": {"HA": 4.63, "CA": 55.0, "CB": 29.0, "C": 174.1, "N": 118.2, "H": 8.42},
    "ILE": {"HA": 4.17, "CA": 61.1, "CB": 38.8, "C": 176.4, "N": 121.4, "H": 8.00},
    "LEU": {"HA": 4.34, "CA": 55.1, "CB": 42.4, "C": 177.6, "N": 121.8, "H": 8.16},
    "LYS": {"HA": 4.32, "CA": 56.2, "CB": 33.1, "C": 176.6, "N": 120.4, "H": 8.29},
    "MET": {"HA": 4.48, "CA": 55.4, "CB": 32.6, "C": 176.3, "N": 119.6, "H": 8.28},
    "PHE": {"HA": 4.62, "CA": 57.7, "CB": 39.6, "C": 175.8, "N": 120.3, "H": 8.12},
    "PRO": {"HA": 4.42, "CA": 63.3, "CB": 32.1, "C": 177.3, "N": 0.0,   "H": 0.0},  # No Amide N/H
    "SER": {"HA": 4.47, "CA": 58.3, "CB": 63.8, "C": 174.6, "N": 115.7, "H": 8.31},
    "THR": {"HA": 4.35, "CA": 61.8, "CB": 69.8, "C": 174.7, "N": 113.6, "H": 8.15},
    "TRP": {"HA": 4.66, "CA": 57.5, "CB": 29.6, "C": 176.1, "N": 121.3, "H": 8.25},
    "TYR": {"HA": 4.55, "CA": 57.9, "CB": 38.8, "C": 175.9, "N": 120.3, "H": 8.12},
    "VAL": {"HA": 4.12, "CA": 62.2, "CB": 32.9, "C": 176.3, "N": 119.9, "H": 8.03},
}

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
    "CA": {"alpha": 3.1,  "beta": -1.5},
    # C-beta: Opposite trend to C-alpha
    "CB": {"alpha": -0.5, "beta": 2.2},
    # Carbonyl Carbon: Follows C-alpha trend
    "C":  {"alpha": 2.2,  "beta": -1.6},
    # H-alpha: Shifted upfield (negative) in Helix, downfield (positive) in Sheet
    "HA": {"alpha": -0.4, "beta": 0.5},
    # Amide N: Complex, but generally upfield in Helix
    "N":  {"alpha": -1.5, "beta": 1.2},
    "H":  {"alpha": -0.2, "beta": 0.3},
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


from synth_nmr.structure_utils import get_secondary_structure

def predict_chemical_shifts(structure: struc.AtomArray) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Predict chemical shifts based on secondary structure (Phi/Psi).
    
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
    - Ring Current Effects: Aromatic rings (Phe, Tyr, Trp) create strong magnetic
      fields that shift nearby protons. We omit this for simplicity ($O(N^2)$ geometry check).
    - H-Bonding: Hydrogen bonds affect Amide H shifts significantly. We omit this.
    - Sequence History: Real shifts depend on (i-1) and (i+1) neighbor types. We omit this.
    
    Args:
        structure: AtomArray containing the protein
        
    Returns:
        shifts: Dict[chain_id, Dict[res_id, Dict[atom_name, value]]]
    """
    logger.info("Predicting Chemical Shifts (SPARTA+ + Ring Currents)...")
    
    # Use shared utility for SS classification
    ss_list = get_secondary_structure(structure)
    
    # Identify aromatic rings once for the whole structure
    rings = _get_aromatic_rings(structure)
    if rings.size > 0:
        logger.debug(f"DEBUG: Found {rings.shape[0]} aromatic rings for shift calculation.")
    
    # We need to iterate over residues
    res_starts = struc.get_residue_starts(structure)
    
    results = {} # Keyed by Chain -> ResID -> Atom -> Value
    
    for i, start_idx in enumerate(res_starts):
        # Identify residue
        res_atoms = structure[start_idx : res_starts[i+1] if i+1 < len(res_starts) else None]
        res_name = res_atoms.res_name[0]
        chain_id = res_atoms.chain_id[0]
        res_id = res_atoms.res_id[0]
        
        if res_name not in RANDOM_COIL_SHIFTS:
            continue
            
        ss_state = ss_list[i] if i < len(ss_list) else "coil"
        logger.debug(f"DEBUG: Res {i} {res_name} -> {ss_state}")
        
        # Calculate Shifts
        rc = RANDOM_COIL_SHIFTS[res_name]
        atom_shifts = {}
        
        for atom_type, base_val in rc.items():
            offset = SECONDARY_SHIFTS.get(atom_type, {}).get(ss_state, 0.0)
            
            # Add small random noise for "realism" (0.1 - 0.3 ppm)
            # Experimental assignments always have error/variation
            noise = np.random.normal(0, 0.15) if base_val != 0 else 0
            
            if base_val != 0:
                # 1. Base + Secondary Shift
                val = base_val + offset + noise
                
                # 2. Add Tertiary Ring Current Effects
                # Only affects protons (H, HA, HB...) and sometimes Carbon.
                # Primarily Protons are interesting for NOESY/Structure.
                # Use .size check for numpy array
                if rings.size > 0 and ("H" in atom_type or atom_type == "H"):
                    # Get atom coordinate
                    try:
                        target_atom = res_atoms[res_atoms.atom_name == atom_type][0]
                        rc_shift = _calculate_ring_current_shift(target_atom.coord, rings)
                        val += rc_shift
                    except IndexError:
                        pass 
# Atom not found in structure (e.g. sometimes amide H is missing)
                
                atom_shifts[atom_type] = round(val, 3)
        
        if chain_id not in results:
            results[chain_id] = {}
        results[chain_id][res_id] = atom_shifts
        
    return results

def calculate_csi(shifts: Dict[str, Dict[int, Dict[str, float]]], structure: struc.AtomArray) -> Dict[str, Dict[int, float]]:
    """
    Calculate Chemical Shift Index (CSI) deviations (Observed - RandomCoil).
    
    This metric is used to predict secondary structure:
    - Positive Delta(CA) (> 0.7 ppm) -> HELIX
    - Negative Delta(CA) (< -0.7 ppm) -> SHEET
    
    returns: {chain_id: {res_id: delta_ppm}}
    """
    csi_data = {}
    
    # Map residue names for lookup
    # Need 3-letter codes
    res_names = {}
    
    # Iterate through structure to build map: ResID -> ResName
    # Using residue starts to handle multi-atom residues correctly
    res_starts = struc.get_residue_starts(structure)
    for idx in res_starts:
        res = structure[idx]
        res_names[res.res_id] = res.res_name
        
    for chain_id, chain_shifts in shifts.items():
        csi_data[chain_id] = {}
        for res_id, atom_shifts in chain_shifts.items():
            if res_id not in res_names:
                continue
                
            res_name = res_names[res_id]
            
            # CSI usually uses C-alpha or C-beta
            # We will use C-alpha (CA) as the primary index
            if "CA" in atom_shifts and res_name in RANDOM_COIL_SHIFTS:
                measured = atom_shifts["CA"]
                random = RANDOM_COIL_SHIFTS[res_name]["CA"]
                delta = measured - random
                csi_data[chain_id][res_id] = delta
                
    return csi_data

def _get_aromatic_rings(structure):
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
                ring_atoms = res_slice[np.isin(res_slice.atom_name, ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"])]
            elif res_name == "TRP":
                # Indole is 9 atoms, effective center near CD2/CE2 bond.
                # Simplified: averaging all ring atoms
                ring_names = ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]
                ring_atoms = res_slice[np.isin(res_slice.atom_name, ring_names)]
            elif res_name in ["HIS", "HID", "HIE", "HIP"]:
                # 5-membered ring: CG, ND1, CD2, CE1, NE2
                ring_atoms = res_slice[np.isin(res_slice.atom_name, ["CG", "ND1", "CD2", "CE1", "NE2"])]
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
def _calculate_ring_current_shift(proton_coord, rings):
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
    B_FACTOR = 11.0 # Empirical scaling factor (ppm * A^3)
    
    for j in range(rings.shape[0]):
        center = rings[j, 0:3]
        normal = rings[j, 3:6]
        intensity = rings[j, 6]
        
        # Vector from ring center to proton
        v = (proton_coord - center).astype(np.float64)
        r = np.sqrt(np.sum(v**2))
        
        if r < 1.0: 
            continue # Too close/clashing, ignore singularity
            
        # Cos(theta) = dot(v, n) / (|v|*|n|) -> |n|=1
        costheta = np.sum(v * normal) / r
        
        # Geometric Factor G(r, theta) = (1 - 3*cos^2(theta)) / r^3
        # If theta = 0 (above ring), cos=1 -> (1-3)/r^3 = -2/r^3 (Shielding)
        # If theta = 90 (in plane), cos=0 -> (1-0)/r^3 =  1/r^3 (Deshielding)
        geom_factor = (1.0 - 3.0 * costheta**2) / (r**3)
        
        shift = intensity * B_FACTOR * geom_factor
        
        total_shift += shift
        
    return total_shift

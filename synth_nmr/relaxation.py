
import numpy as np
import biotite.structure as struc
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# --- Optional Numba JIT Support ---
try:
    from numba import njit
except ImportError:
    def njit(func=None, **kwargs):
        if func is None:
            return lambda f: f
        return func

# --- Physical Constants for NMR Relaxation ---
# SI Units used for internal calculation
MU_0 = 4 * np.pi * 1e-7      # Vacuum permeability derived (T*m/A)
H_PLANCK = 6.62607015e-34    # Planck constant (J*s)
H_BAR = H_PLANCK / (2 * np.pi)

GAMMA_H = 267.522e6          # Proton gyromagnetic ratio (rad s^-1 T^-1)
GAMMA_N = -27.126e6          # Nitrogen-15 gyromagnetic ratio (rad s^-1 T^-1)

R_NH = 1.02e-10              # NH Bond length (meters) - standard value
CSA_N = -160e-6              # Polimorphic 15N CSA (unitless, ppm) -160 to -170 typical

@njit
def spectral_density(omega: float, tau_m: float, s2: float, tau_f: float = 0.0) -> float:
    """
    Calculate Spectral Density J(w) using Lipari-Szabo Model-Free formalism.
    
    EDUCATIONAL NOTE - BPP Theory & Spectral Density:
    -------------------------------------------------
    In NMR, relaxation is caused by local magnetic fields that "flicker" due to 
    molecular tumbling. Spectral Density J(w) is essentially a measure of the 
    POWER of these fluctuations at a specific frequency (w).
    
    - BPP Theory (Bloembergen-Purcell-Pound) shows that relaxation is most 
      efficient when the frequency of motion (1/tau_m) matches the Larmor 
      frequency of the nuclei.
    - If a protein tumbles too fast (small tau_m), the fluctuations are too 
      high-frequency to cause efficient relaxation (Extreme Narrowing Limit).
    - If it tumbles too slow (large tau_m), the energy is concentrated at 
      low frequencies, leading to fast T2 relaxation and broad lines.
    
    Model-Free Analysis (Lipari-Szabo) separates this into Global Tumbling (tau_m) 
    and fast Local Motion (tau_f), weighted by the Order Parameter (S^2).
    
    Formula:
    J(w) = (2/5) * [ S^2 * tm / (1 + (w*tm)^2) + (1-S^2) * te / (1 + (w*te)^2) ]
    
    where te (tau_e) is the effective internal correlation time: 1/te = 1/tm + 1/tf
    Usually for simple MF, we assume fast motion tf << tm.
    
    Args:
        omega: Frequency (rad/s)
        tau_m: Global rotational correlation time (seconds)
        s2: Generalized order parameter (0.0 to 1.0)
        tau_f: Fast internal correlation time (seconds). If 0 (default), internal motion is considered infinitely fast/negligible, and the function simplifies to a one-time scale model-free.
    """
    # Simple Model Free (assuming tf is very small/negligible or incorporated)
    # If tau_f is provided, calculate effective time tau_e
    
    # Term 1: Global tumbling
    j_global = (s2 * tau_m) / (1 + (omega * tau_m)**2)
    
    # Term 2: Fast internal motion
    # Effective correlation time 1/tau_e = 1/tau_m + 1/tau_f
    # If tau_f is 0, this term vanishes in standard simplified approximation
    # or acts as a very fast motion limit.
    j_fast = 0.0
    if tau_f > 0:
        tau_e = (tau_m * tau_f) / (tau_m + tau_f)
        j_fast = ((1 - s2) * tau_e) / (1 + (omega * tau_e)**2)
        
    return 0.4 * (j_global + j_fast) 

from synth_nmr.structure_utils import get_secondary_structure

def predict_order_parameters(structure: struc.AtomArray) -> Dict[int, float]:
    """
    Predict Generalized Order Parameters (S2) based on secondary structure.
    
    EDUCATIONAL NOTE - Lipari-Szabo Model Free:
    ===========================================
    The Order Parameter (S2) describes the amplitude of internal motion:
    - S2 = 1.0: Completely rigid (no internal motion relative to tumbling).
    - S2 = 0.0: Completely disordered (isotropic internal motion).
    
    Typical values in proteins:
    - Alpha Helices / Beta Sheets: S2 ~ 0.85 (Very rigid H-bond network)
    - Loops / Turns: S2 ~ 0.60 - 0.70 (Flexible)
    - Termini (N/C): S2 ~ 0.40 - 0.50 (Fraying)
    """
    res_starts = struc.get_residue_starts(structure)
    res_ids = np.unique(structure.res_id)
    if len(res_ids) == 0:
        return {}
        
    ss_list = get_secondary_structure(structure)
        
    start_res = res_ids[0]
    end_res = res_ids[-1]
    
    # Heuristic Max SASA per residue (Angstrom^2) for normalization
    MAX_SASA = 150.0
    
    # Calculate SASA for "Packing Awareness"
    sasa_per_residue = {}
    try:
        # Map non-standard residues to standard ones for SASA calculation
        # This prevents "atom not found" or missing radii errors
        temp_struc = structure.copy()
        
        # Histidine Tautomers
        temp_struc.res_name[np.isin(temp_struc.res_name, ["HIE", "HID", "HIP"])] = "HIS"
        
        # Phosphorylated Residues
        temp_struc.res_name[temp_struc.res_name == "SEP"] = "SER"
        temp_struc.res_name[temp_struc.res_name == "TPO"] = "THR"
        temp_struc.res_name[temp_struc.res_name == "PTR"] = "TYR"
        
        # Filter out extra atoms (P, O1P, etc.) that Biotite doesn't have radii for
        ptm_atom_names = ["P", "O1P", "O2P", "O3P"]
        ptm_mask = np.isin(temp_struc.atom_name, ptm_atom_names)
        if np.any(ptm_mask):
             temp_struc = temp_struc[~ptm_mask]

        # CRITICAL FIX for Metal Ions (ZN, etc.):
        # Biotite's sasa function with ProtOr radii set fails for non-amino-acid residues
        # like 'ZN' because they are not in the lookup table.
        # We simply exclude them from the calculation. While this slightly reduces accuracy
        # (ignoring burial by ions), it prevents the entire SASA calculation from crashing.
        # We assume any residue not in our standard/modified list is a cofactor/ion.
        # Actually, simpler: Just filter by hetero flag if ions are HETATM.
        # But let's be explicit about filtering out known ions to be safe.
        
        # Keep only amino acids (standard + converted PTMs)
        # We check against a known set of 3-letter codes?
        # Or just remove ZN, MG, CA, NA, CL.
        ion_res_names = ["ZN", "MG", "CA", "NA", "CL", "K", "FE", "CU", "MN"]
        ion_mask = np.isin(temp_struc.res_name, ion_res_names)
        if np.any(ion_mask):
             temp_struc = temp_struc[~ion_mask]

        # vdw_radii: Simple lookup. Biotite has defaults but good to be explicit or use default.
        # atom_sasa: Array of same length as structure
        # probe_radius=1.4 standard for water
        filtered_sasa = struc.sasa(temp_struc, probe_radius=1.4)
        
        # Handle NaNs
        if np.any(np.isnan(filtered_sasa)):
             filtered_sasa = np.nan_to_num(filtered_sasa, nan=50.0)

        # Aggregate SASA by residue (robust to atom count changes)
        curr_res_id = -99999
        current_sum = 0.0
        
        for i, atom in enumerate(temp_struc):
             if atom.res_id != curr_res_id:
                 if curr_res_id != -99999:
                     sasa_per_residue[curr_res_id] = current_sum
                 curr_res_id = atom.res_id
                 current_sum = 0.0
             current_sum += filtered_sasa[i]
        # Last residue
        if curr_res_id != -99999:
             sasa_per_residue[curr_res_id] = current_sum
            
    except Exception as e:
        logger.warning(f"SASA calculation failed ({e}). All residues will be treated as fully exposed (rel_sasa=1.0), which typically leads to lower S2 values.")
        
    s2_map = {}
    
    # Identify termini residues (by ID)
    
    for i, start_idx in enumerate(res_starts):
        # Identify residue ID
        rid = structure.res_id[start_idx]
        
        # Get SASA from map (default to MAX_SASA/Exposed if failed/missing)
        res_sasa = sasa_per_residue.get(rid, MAX_SASA)
        
        # Relative SASA (0.0 = Buried, 1.0 = Exposed)
        rel_sasa = min(res_sasa / MAX_SASA, 1.0)
        
        ss = ss_list[i] if i < len(ss_list) else "coil"
        
        # Base S2 from Secondary Structure
        if ss in ["alpha", "beta"]:
            base_s2 = 0.85
        else:
            base_s2 = 0.70 # Increased base slightly, so exposed loops drop to ~0.50
            
        # Termini effects override secondary structure
        # Fraying usually affects first/last 2-3 residues
        if rid <= start_res + 1 or rid >= end_res - 1:
            base_s2 = 0.50
            
        # Modulate by SASA
        # Buried (rel_sasa=0) -> Bonus rigidity (+0.1)
        # Exposed (rel_sasa=1) -> Penalty flexibility (-0.1)
        # Formula: S2 = Base + 0.1 - 0.2 * rel_sasa
        # Let's align with plan: 0.85 -> 0.65 range.
        
        s2 = base_s2 + 0.05 * (1.0 - rel_sasa) - 0.15 * rel_sasa
        
        # Add realistic noise
        s2 += np.random.normal(0, 0.02)
        s2 = np.clip(s2, 0.01, 0.98)
        
        s2_map[rid] = s2
        
    return s2_map

def calculate_relaxation_rates(
    structure: struc.AtomArray,
    field_mhz: float = 600.0,
    tau_m_ns: float = 10.0,
    s2_map: Dict[int, float] = None
) -> Dict[int, Dict[str, float]]:
    """
    Calculate R1, R2, and Heteronuclear NOE for all backbone Amides (N-H).
    
    Args:
        structure: The protein structure (must have hydrogens).
        field_mhz: Proton Larmor frequency in MHz (e.g. 600).
        tau_m_ns: Global tumbling time in ns (default 10.0).
        s2_map: Optional dictionary of {res_id: S2}. If None, predicted from structure.
        
    Returns:
        Dictionary keyed by residue ID:
        { res_id: {'R1': float, 'R2': float, 'NOE': float, 'S2': float} }
    """
    logger.info(f"Calculating Relaxation Rates (Field={field_mhz}MHz, tm={tau_m_ns}ns)...")
    
    # Calculate S2 profile if not provided
    if s2_map is None:
        s2_map = predict_order_parameters(structure)
    
    # Convert inputs to SI units
    tau_m = tau_m_ns * 1e-9
    
    # Larmor Frequencies (rad/s)
    omega_h = 2 * np.pi * field_mhz * 1e6
    
    # Calculate B0 from proton freq
    b0 = omega_h / GAMMA_H
    
    omega_n = GAMMA_N * b0 # Negative val
    
    logger.debug(f"B0 Field: {b0:.2f} T")
    logger.debug(f"wH: {omega_h:.2e} rad/s, wN: {omega_n:.2e} rad/s")
    
    logger.debug(f"wH: {omega_h:.2e} rad/s, wN: {omega_n:.2e} rad/s")
    
    # EDUCATIONAL NOTE - Dipolar Integration Constant (d):
    # ====================================================
    # The dominant relaxation mechanism for 15N is the Dipole-Dipole interaction
    # with the directly attached Amide Proton (H).
    # d = (μ0 * ħ * γH * γN) / (4π * r^3)
    #
    # Where:
    # - μ0: Vacuum permeability
    # - r: N-H bond length (approx 1.02 Å)
    # - γH, γN: Gyromagnetic ratios
    # 
    # This constant represents the strength of the magnetic interaction distance dependence (r^-3).
    # In relaxation rate equations (R1, R2), it appears squared (d^2), leading to the famous r^-6 dependence.
    
    dd_const = (MU_0 / (4 * np.pi)) * H_BAR * GAMMA_H * GAMMA_N * (R_NH**-3)
    d_sq = dd_const**2
    
    # EDUCATIONAL NOTE - Chemical Shift Anisotropy (CSA) Constant (c):
    # ================================================================
    # The second major relaxation mechanism is CSA. The electron cloud around the 15N nucleus
    # is not spherical, so as the protein tumbles, the local magnetic field fluctuates.
    # c = (Δσ * ωN) / √3
    #
    # Where:
    # - Δσ (CSA_N): The anisotropy parameter (-160 ppm typical for Beta Sheet / Helix average).
    # - ωN: The Larmor frequency of Nitrogen (field dependent!).
    #
    # Note: Because 'c' depends on ωN (and thus B0), CSA relaxation increases quadratically
    # with magnetic field strength. At high fields (>800 MHz), CSA becomes dominant over Dipolar.
    
    csa_const = (CSA_N * omega_n) / np.sqrt(3)
    c_sq = csa_const**2
    
    results = {}
    
    # Iterate over residues that have an N-H pair
    res_ids = np.unique(structure.res_id)
    
    for rid in res_ids:
        # Check if N and H exist
        res_mask = structure.res_id == rid
        res_atoms = structure[res_mask]
        
        has_n = "N" in res_atoms.atom_name
        has_h = "H" in res_atoms.atom_name
        res_name = res_atoms.res_name[0]
        
        if not (has_n and has_h):
            continue
            
        if res_name == "PRO":
            continue
            
        # Get S2
        s2 = s2_map.get(rid, 0.85) # Fallback to 0.85 if missing from map
        
        # Frequencies for J(w)
        # R1 depends on: J(wH-wN), J(wN), J(wH+wN)
        # R2 depends on: J(0), J(wH-wN), J(wN), J(wH), J(wH+wN)
        # NOE depends on: J(wH+wN), J(wH-wN) ? -> Actually NOE = 1 + ...
        
        j_0 = spectral_density(0, tau_m, s2)
        j_wn = spectral_density(omega_n, tau_m, s2)
        j_wh = spectral_density(omega_h, tau_m, s2)
        j_diff = spectral_density(omega_h - omega_n, tau_m, s2)
        j_sum = spectral_density(omega_h + omega_n, tau_m, s2)
        
        # Calculate Rates
        # References for relaxation rate equations:
        # 1. Palmer, A. G. (2001). "NMR relaxation in biology: experimental aspects and model-free analysis."
        #    Encyclopedia of Nuclear Magnetic Resonance, Vol. 9, pp. 277-293. (Often cited as standard for Model-Free)
        # 2. Cavanagh, J., Fairbrother, W. J., Palmer, A. G., & Skelton, N. J. (1996).
        #    "Protein NMR Spectroscopy: Principles and Practice." Academic Press. (Chapter 6 for relaxation)
        
        # R1 (Longitudinal Relaxation Rate)
        r1_val = d_sq * (j_diff + 3*j_wn + 6*j_sum) + c_sq * j_wn
        
        # R2 (Transverse Relaxation Rate)
        r2_val = 0.5 * d_sq * (4*j_0 + j_diff + 3*j_wn + 6*j_wh + 6*j_sum) + \
                 (1.0/6.0) * c_sq * (4*j_0 + 3*j_wn)
                 
        # NOE (Heteronuclear Steady-State NOE)
        noe_val = 1.0 + (GAMMA_H / GAMMA_N) * d_sq * (6*j_sum - j_diff) * (1.0 / r1_val)
        
        results[rid] = {
            'R1': r1_val,
            'R2': r2_val,
            'NOE': noe_val,
            'S2': s2
        }
        
    return results

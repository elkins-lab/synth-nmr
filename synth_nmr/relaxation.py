import logging
from typing import Any, Dict, Optional

import biotite.structure as struc
import numpy as np

from synth_nmr.structure_utils import get_secondary_structure

logger = logging.getLogger(__name__)

# --- Optional Numba JIT Support ---
try:
    from numba import njit
except ImportError:

    def njit(func: Any = None, **kwargs: Any) -> Any:
        if func is None:
            return lambda f: f
        return func


# --- Physical Constants for NMR Relaxation ---
# SI Units used for internal calculation
from synth_core.constants import (
    GAMMA_15N,
    GAMMA_1H,
    REDUCED_PLANCK_CONSTANT,
    VACUUM_PERMEABILITY,
)

R_NH = 1.02e-10  # NH Bond length (meters) - standard value
# 15N Chemical Shift Anisotropy (Δσ).
# The 15N CSA varies across residues (~−100 ppm in loops to ~−200 ppm in helices).
# The commonly used fixed value of −160 ppm is a backbone-average approximation.
# For residue-specific accuracy see: Loth, K. et al. (2005) J. Am. Chem. Soc. 127, 6062–6068.
CSA_N = -160e-6  # Mean backbone 15N CSA (dimensionless, fraction; corresponds to -160 ppm)


# --- Spectral Density Mapping and Theory ────────────────────────────
# The goal of NMR relaxation analysis is often to determine the
# "Spectral Density" at specific frequencies. J(w) tells us how much
# motion exists at frequency 'w'.
#
# In the high-field limit, we often use "Reduced Spectral Density Mapping":
# 1. J(0.87*wH): High frequency component, primarily from R1 and NOE.
# 2. J(wN): Medium frequency, from R1.
# 3. J(0): Low frequency (static), from R2.
#
# The "Model-Free" approach (Lipari & Szabo) avoids assuming a specific
# geometric model for the motion (like "wobbling in a cone"). Instead,
# it assumes that global and internal motions are statistically
# independent. The total correlation function C(t) is the product:
#   C(t) = C_global(t) * C_internal(t)
#
# C_global(t) = exp(-t / tau_m)
# C_internal(t) = S2 + (1 - S2) * exp(-t / tau_f)
#
# This leads to the classic formula for J(w) used in this module.
#
# By analyzing these relaxation rates at multiple magnetic field strengths,
# one can gain a more complete picture of the molecular dynamics, as different
# motions become visible at different field-dependent frequencies.
# ─────────────────────────────────────────────────────────────────────


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
        omega: Frequency (rad/s). Must be numeric.
        tau_m: Global rotational correlation time (seconds). Must be numeric and positive.
        s2: Generalized order parameter (0.0 to 1.0). Must be numeric and within this range.
        tau_f: Fast internal correlation time (seconds). Must be numeric and non-negative.
               If 0 (default), internal motion is considered infinitely fast/negligible,
               and the function simplifies to a one-time scale model-free.
    """
    # Simple Model Free (assuming tf is very small/negligible or incorporated)
    # If tau_f is provided, calculate effective time tau_e

    # Term 1: Global tumbling
    j_global = (s2 * tau_m) / (1 + (omega * tau_m) ** 2)

    # Term 2: Fast internal motion
    # Effective correlation time 1/tau_e = 1/tau_m + 1/tau_f
    # If tau_f is 0, this term vanishes in standard simplified approximation
    # or acts as a very fast motion limit.
    j_fast = 0.0
    if tau_f > 0:
        tau_e = (tau_m * tau_f) / (tau_m + tau_f)
        j_fast = ((1 - s2) * tau_e) / (1 + (omega * tau_e) ** 2)

    return 0.4 * (j_global + j_fast)


def _predict_s2_from_sasa(rel_sasa: float, base_s2: float) -> float:
    """
    Modulate the base Order Parameter (S2) depending on Solvent Accessible Surface Area (SASA).

    EDUCATIONAL NOTE - SASA and S2:
    ===============================
    Protein interiors are tightly packed. If a residue is fully buried (rel_sasa=0),
    it is geometrically restricted from moving, getting a bonus to its S2 rigidity (+0.05).
    Conversely, if it's fully exposed to solvent (rel_sasa=1.0), it has more
    freedom of motion, receiving a penalty flexibility (-0.15).
    """
    return base_s2 + 0.05 * (1.0 - rel_sasa) - 0.15 * rel_sasa


def _apply_termini_effects(res_id: int, start_res: int, end_res: int, current_s2: float) -> float:
    """
    Apply chain-fraying flexibility penalties to the terminal residues.

    EDUCATIONAL NOTE - Termini Dynamics:
    ====================================
    Regardless of secondary structure assigned by backbone dihedral angles, the
    first and last few residues of a polypeptide chain lack the full network of
    hydrogen bonds and packing constraints (Fraying usually affects first/last 2-3 residues).
    They consistently exhibit high amplitude, fast timescale motions (lower S2) due to "fraying".
    Termini effects override secondary structure.
    """
    if res_id <= start_res + 1 or res_id >= end_res - 1:
        return 0.50
    return current_s2


def _calculate_dipolar_constant(r_nh: float) -> float:
    """
    Calculate the squared Dipolar integration constant (d^2) for N-H relaxation.

    EDUCATIONAL NOTE - Dipolar Integration Constant (d):
    ====================================================
    The dominant relaxation mechanism for 15N is the Dipole-Dipole interaction
    with the directly attached Amide Proton (H).
    d = (μ0 * ħ * γH * γN) / (4π * r^3)

    Where:
    - μ0: Vacuum permeability
    - r: N-H bond length (approx 1.02 Å)
    - γH, γN: Gyromagnetic ratios

    This constant represents the strength of the magnetic interaction distance dependence (r^-3).
    In relaxation rate equations (R1, R2), it appears squared (d^2), leading to the famous r^-6 dependence.
    """
    dd_const = (VACUUM_PERMEABILITY / (4 * np.pi)) * REDUCED_PLANCK_CONSTANT * GAMMA_1H * GAMMA_15N * (r_nh**-3)
    return dd_const**2


def _calculate_csa_constant(csa_n: float, omega_n: float) -> float:
    """
    Calculate the squared Chemical Shift Anisotropy (CSA) constant (c^2) for 15N.

    EDUCATIONAL NOTE - Chemical Shift Anisotropy (CSA) Constant (c):
    ================================================================
    The second major relaxation mechanism is CSA. The electron cloud around the 15N nucleus
    is not spherical, so as the protein tumbles, the local magnetic field fluctuates.
    c = (Δσ * ωN) / √3

    Where:
    - Δσ (CSA_N): The anisotropy parameter (-160 ppm typical for Beta Sheet / Helix average).
    - ωN: The Larmor frequency of Nitrogen (field dependent!).

    Note: Because 'c' depends on ωN (and thus B0), CSA relaxation increases quadratically
    with magnetic field strength. At high fields (>800 MHz), CSA becomes dominant over Dipolar.
    """
    csa_const = (csa_n * omega_n) / np.sqrt(3)
    return float(csa_const**2)


def predict_order_parameters(structure: struc.AtomArray) -> Dict[int, float]:
    """
    Predict Generalized Order Parameters (S2) based on secondary structure,
    termini effects, and solvent accessible surface area (SASA).

    EDUCATIONAL NOTE - Lipari-Szabo Model Free:
    ===========================================
    The Order Parameter (S2) describes the amplitude of internal motion:
    - S2 = 1.0: Completely rigid (no internal motion relative to tumbling).
    - S2 = 0.0: Completely disordered (isotropic internal motion).

    Typical values in proteins:
    - Alpha Helices / Beta Sheets: S2 ~ 0.85 (Very rigid H-bond network)
    - Loops / Turns: S2 ~ 0.60 - 0.70 (Flexible)
    - Termini (N/C): S2 ~ 0.40 - 0.50 (Fraying)

    Args:
        structure: The biotite.structure.AtomArray containing the protein.

    Returns:
        A dictionary mapping each residue ID to its predicted S2 value.

    Raises:
        TypeError: If the input is not a biotite.structure.AtomArray.
    """
    from synth_nmr.structure_utils import get_residue_info

    logger.info("Predicting Generalized Order Parameters (S2)...")

    if not isinstance(structure, struc.AtomArray):
        raise TypeError("Input 'structure' must be a biotite.structure.AtomArray.")
    if structure.array_length() == 0:
        return {}

    try:
        # ── Physical Basis of Order Parameters (S2) ─────────────────────────
        # In the Lipari-Szabo Model-Free formalism, the spectral density is
        # decomposed into global tumbling (tau_m) and internal motions.
        # S2 is the spatial part of the internal correlation function.
        #
        # A value of 0.85 indicates that the N-H bond vector is highly
        # constrained, sampling only a small portion of the unit sphere.
        # This is typical for well-defined secondary structural elements
        # like alpha-helices and beta-sheets, where hydrogen bonding
        # stabilizes the backbone.
        # ─────────────────────────────────────────────────────────────────────

        # Check unique residues for test compatibility (some tests mock np.unique)
        if len(np.unique(structure.res_id)) == 0:
            return {}

        chain_ids, res_ids, res_names, res_starts = get_residue_info(structure)
        if len(res_ids) == 0:
            return {}

        # ── Correlation Times and Molecular Motion ──────────────────────────
        # Protein relaxation is driven by the rotational diffusion (tumbling).
        # We model this using the Global Correlation Time (tau_m).
        # However, residues also have internal motion (fast fluctuations).
        # We use a heuristic 'internal' correlation time (tau_f) which is
        # inversely proportional to the Order Parameter (S2).
        #
        # - Rigid residues (High S2): Slower internal motion, dominant global tumbling.
        # - Flexible residues (Low S2): Significant fast internal motion (ps-ns range).
        # ─────────────────────────────────────────────────────────────────────

        ss_list = get_secondary_structure(structure)

        # Heuristic Max SASA per residue (Å²) used to compute relative solvent
        # exposure: rel_sasa = total_residue_sasa / MAX_SASA, clamped to [0, 1].
        #
        # APPROXIMATION NOTICE — single constant for all residue types:
        # The maximum possible SASA varies substantially with residue size:
        #   Gly  ~  75 Å²,  Ala ~  92 Å²,  Val ~ 142 Å²,
        #   Leu ~ 168 Å²,   Phe ~ 197 Å²,  Trp ~ 240 Å²
        # (Miller et al., 1987, J. Mol. Biol. 196, 641-656)
        #
        # Using 150 Å² as a single backbone-average therefore introduces a
        # systematic bias: small residues (Gly, Ala) score as "more buried"
        # than they physically are (→ S² over-estimated), while large aromatic
        # residues (Trp, Phe) score as "more exposed" (→ S² under-estimated).
        # For applications requiring accurate per-residue dynamics, replace
        # this constant with a residue-type lookup table derived from the
        # Miller et al. (1987) extended-chain reference SASA values.
        MAX_SASA = 150.0

        # Calculate SASA for "Packing Awareness"
        sasa_per_residue_array = np.full(len(res_ids), MAX_SASA)
        try:
            temp_struc = structure.copy()
            temp_struc.res_name[np.isin(temp_struc.res_name, ["HIE", "HID", "HIP"])] = "HIS"
            temp_struc.res_name[temp_struc.res_name == "SEP"] = "SER"
            temp_struc.res_name[temp_struc.res_name == "TPO"] = "THR"
            temp_struc.res_name[temp_struc.res_name == "PTR"] = "TYR"

            ptm_atom_names = ["P", "O1P", "O2P", "O3P"]
            ptm_mask = np.isin(temp_struc.atom_name, ptm_atom_names)
            if np.any(ptm_mask):
                temp_struc = temp_struc[~ptm_mask]

            ion_res_names = ["ZN", "MG", "CA", "NA", "CL", "K", "FE", "CU", "MN"]
            ion_mask = np.isin(temp_struc.res_name, ion_res_names)
            if np.any(ion_mask):
                temp_struc = temp_struc[~ion_mask]

            atom_sasa = struc.sasa(temp_struc, probe_radius=1.4)
            atom_sasa = np.nan_to_num(atom_sasa, nan=50.0)

            # Vectorized aggregation of SASA per residue
            temp_res_starts = struc.get_residue_starts(temp_struc)
            for i in range(len(temp_res_starts)):
                start = temp_res_starts[i]
                end = temp_res_starts[i + 1] if i + 1 < len(temp_res_starts) else len(temp_struc)
                rid = temp_struc.res_id[start]
                idx = np.where(res_ids == rid)[0]
                if len(idx) > 0:
                    sasa_per_residue_array[idx[0]] = np.sum(atom_sasa[start:end])

        except Exception as e:
            logger.warning(f"SASA calculation failed ({e}). Using default exposed values.")

        start_res = res_ids[0]
        end_res = res_ids[-1]

        results = {}
        for i in range(len(res_ids)):
            rid = int(res_ids[i])
            ss = ss_list[i] if i < len(ss_list) else "coil"
            res_sasa = sasa_per_residue_array[i]

            # Relative SASA (0.0 = Buried, 1.0 = Exposed)
            rel_sasa = min(res_sasa / MAX_SASA, 1.0)

            # Base S2 from Secondary Structure
            base_s2 = 0.85 if ss in ["alpha", "beta"] else 0.70

            # Termini effects (Must call helper for test mocking)
            base_s2 = _apply_termini_effects(rid, start_res, end_res, base_s2)

            # Modulate by SASA (Must call helper for test mocking)
            s2 = _predict_s2_from_sasa(rel_sasa, base_s2)

            # Clamp to physically valid range [0, 1]
            s2 = float(np.clip(s2, 0.01, 0.98))
            results[rid] = s2

        logger.info(f"Successfully predicted S2 for {len(results)} residues.")
        return results

    except Exception:
        logger.error("An unexpected error occurred during S2 prediction", exc_info=True)
        raise


def calculate_relaxation_rates(
    structure: struc.AtomArray,
    field_mhz: float = 600.0,
    tau_m_ns: float = 10.0,
    s2_map: Optional[Dict[int, float]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Calculate R1, R2, and Heteronuclear NOE for all backbone Amides (N-H).

    Args:
        structure: biotite.structure.AtomArray containing the protein.
        field_mhz: Proton Larmor frequency in MHz (e.g., 600.0).
        tau_m_ns: Global rotational correlation time in nanoseconds.
        s2_map: Optional dictionary of pre-calculated Order Parameters (S2).

    Returns:
        Dict keyed by Residue ID -> {R1, R2, NOE, S2}.
    """
    from synth_nmr.structure_utils import get_residue_info

    logger.info(f"Starting Relaxation Rates calculation (Field={field_mhz}MHz, tm={tau_m_ns}ns)...")

    if not isinstance(structure, struc.AtomArray):
        raise TypeError("Input 'structure' must be a biotite.structure.AtomArray.")
    if structure.array_length() == 0:
        logger.warning("Input 'structure' is empty. Returning no relaxation rates.")
        return {}
    if not isinstance(field_mhz, (int, float)) or field_mhz <= 0:
        raise ValueError("Parameter 'field_mhz' must be a positive numeric value.")
    if not isinstance(tau_m_ns, (int, float)) or tau_m_ns <= 0:
        raise ValueError("Parameter 'tau_m_ns' must be a positive numeric value.")
    if s2_map is not None and not isinstance(s2_map, dict):
        raise TypeError("Parameter 's2_map' must be a dictionary or None.")

    # Calculate S2 profile if not provided
    if s2_map is None:
        s2_map = predict_order_parameters(structure)

    # Physical constants and frequencies
    tau_m = tau_m_ns * 1e-9
    omega_h = 2 * np.pi * field_mhz * 1e6
    b0 = omega_h / GAMMA_1H
    omega_n = GAMMA_15N * b0
    d_sq = _calculate_dipolar_constant(R_NH)
    c_sq = _calculate_csa_constant(CSA_N, omega_n)

    # Get residue info
    _, res_ids, res_names, _ = get_residue_info(structure)

    # Map S2 to array matching res_ids
    s2_arr = np.array([s2_map.get(int(rid), 0.85) for rid in res_ids])

    # Find residues with both N and H atoms (backbone amides)
    n_mask = (structure.atom_name == "N") & struc.filter_amino_acids(structure)
    h_mask = (structure.atom_name == "H") & struc.filter_amino_acids(structure)

    n_res_ids = structure.res_id[n_mask]
    h_res_ids = structure.res_id[h_mask]

    has_n = np.isin(res_ids, n_res_ids)
    has_h = np.isin(res_ids, h_res_ids)

    # Valid amides: have N, H and NOT Proline
    is_pro = res_names == "PRO"
    valid_mask = has_n & has_h & (~is_pro)

    # Filter arrays to valid amides only
    active_res_ids = res_ids[valid_mask]
    active_s2 = s2_arr[valid_mask]

    # Heuristic tau_f for extended model-free
    # ─────────────────────────────────────────────────────────────────────
    # The fast-limit model-free (tau_f = 0) is the standard single-field
    # approach: spectral density = (2/5) * S² * τ_m / (1 + (ωτ_m)²).
    # This avoids the unphysical assumption that tau_f scales inversely
    # with S².  When tau_f is unknown, the fast-limit is the correct
    # conservative choice (Lipari & Szabo, 1982).
    tau_f_vals = np.zeros(len(active_res_ids), dtype=np.float64)

    results = {}
    for i in range(len(active_res_ids)):
        rid = active_res_ids[i]
        s2 = active_s2[i]
        tau_f = tau_f_vals[i]

        j0 = spectral_density(0, tau_m, s2, tau_f)
        jwn = spectral_density(np.abs(omega_n), tau_m, s2, tau_f)
        jwh = spectral_density(omega_h, tau_m, s2, tau_f)
        # Since omega_n is negative, omega_h - omega_n is the SUM of magnitudes
        # and omega_h + omega_n is the DIFFERENCE of magnitudes.
        j_sum = spectral_density(omega_h - omega_n, tau_m, s2, tau_f)
        j_diff = spectral_density(omega_h + omega_n, tau_m, s2, tau_f)

        r1 = 0.25 * d_sq * (j_diff + 3 * jwn + 6 * j_sum) + c_sq * jwn
        r2 = 0.125 * d_sq * (4 * j0 + j_diff + 3 * jwn + 6 * jwh + 6 * j_sum) + (
            1.0 / 6.0
        ) * c_sq * (4 * j0 + 3 * jwn)

        if r1 != 0:
            noe = 1.0 + (GAMMA_1H / GAMMA_15N) * 0.25 * d_sq * (6 * j_sum - j_diff) * (1.0 / r1)
        else:
            noe = np.nan
            logger.warning(f"R1 value for residue {rid} is zero, NOE cannot be calculated.")

        results[int(rid)] = {
            "R1": float(r1),
            "R2": float(r2),
            "NOE": float(noe),
            "S2": float(s2),
        }

    logger.info(f"Successfully calculated relaxation rates for {len(results)} residues.")
    return results

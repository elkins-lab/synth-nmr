"""
Validation and comparison utilities for chemical shift predictions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import biotite.structure as struc
import numpy as np

logger = logging.getLogger(__name__)


def compare_chemical_shifts(
    predicted: Dict[str, Dict[int, Dict[str, float]]],
    reference: Dict[str, Dict[int, Dict[str, float]]],
    atom_types: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare two sets of chemical shifts and calculate validation metrics.

    This function aligns predicted and reference chemical shifts by chain and
    residue ID, then calculates the Root Mean Square Error (RMSE) and
    Pearson correlation coefficient (R) for each specified atom type.

    Educational Note:
    - RMSE provides a measure of the absolute accuracy of the predictions in ppm.
    - Pearson R measures the linear correlation, sensitive to how well the
      relative shifts (the "spread") are captured, even if there is a global offset.

    Args:
        predicted: Predicted shifts {chain_id: {res_id: {atom: val}}}
        reference: Reference shifts (e.g., experimental or ShiftX2)
        atom_types: List of atoms to compare.

    Returns:
        A dictionary of metrics per atom type: {atom: {"rmse": float, "pearson": float}}
    """
    if atom_types is None:
        atom_types = ["CA", "HA", "N", "H"]
    stats = {}

    for atom in atom_types:
        y_pred = []
        y_ref = []

        # Iterate through chains and residues in predicted to find matches in reference
        for chain_id, chain_data in predicted.items():
            if chain_id not in reference:
                continue

            for res_id, atoms in chain_data.items():
                if res_id not in reference[chain_id]:
                    continue

                if atom in atoms and atom in reference[chain_id][res_id]:
                    y_pred.append(atoms[atom])
                    y_ref.append(reference[chain_id][res_id][atom])

        if len(y_pred) < 2:
            logger.warning(f"Insufficient data to compare atom type: {atom}")
            continue

        y_pred_np = np.array(y_pred)
        y_ref_np = np.array(y_ref)

        rmse = np.sqrt(np.mean((y_pred_np - y_ref_np) ** 2))
        pearson = np.corrcoef(y_pred_np, y_ref_np)[0, 1]

        stats[atom] = {
            "rmse": round(float(rmse), 3),
            "pearson": round(float(pearson), 3),
            "count": len(y_pred),
        }

    return stats


def calculate_rpf_scores(
    predicted_noes: List[Dict], experimental_restraints: List[Dict], distance_cutoff: float = 5.0
) -> Dict[str, float]:
    """
    Calculate Recall, Precision, and F-measure (RPF) scores for NOE validation.

    EDUCATIONAL BACKGROUND — The RPF Validation Framework
    ───────────────────────────────────────────────────────────────────
    The RPF framework (Montelione et al., 2005) is the gold standard for
    measuring the consistency between a structural model and experimental
    NOE data.

    1. Recall (R):
       What fraction of the EXPERIMENTAL restraints are satisfied by the
       model? If R is low, the model is failing to explain the data.
       Formula: R = satisfied_restraints / total_experimental_restraints

    2. Precision (P):
       What fraction of the PREDICTED NOEs (based on the structure's coordinates)
       are actually observed in the experimental data? If P is low, the
       model contains short-range contacts that the data says shouldn't be there.
       Formula: P = supported_predictions / total_predicted_noes

    3. F-measure (F):
       The harmonic mean of R and P. This provides a single, balanced metric
       of structural quality. High F (> 0.7) indicates a model that is both
       accurate and complete.

    Args:
        predicted_noes: NOE list from nmr.calculate_synthetic_noes.
        experimental_restraints: Parsed experimental upper bounds.
        distance_cutoff: Distance (Å) to consider a pair "close".

    Returns:
        Dict: {"recall": float, "precision": float, "f_measure": float}
    """
    # 1. Align and check Recall
    satisfied_count = 0
    total_exp = len(experimental_restraints)

    if total_exp == 0:
        return {"recall": 0.0, "precision": 0.0, "f_measure": 0.0}

    # For fast lookup, create a key (res1-atom1-res2-atom2)
    # Note: Pairs are undirected, so we normalize the order
    def get_pair_key(r: Dict) -> Tuple:
        p1 = f"{r['seq_1']}-{r['atom_1']}"
        p2 = f"{r['seq_2']}-{r['atom_2']}"
        return tuple(sorted([p1, p2]))

    # Predicted map for satisfaction check
    pred_map = {get_pair_key(r): r["distance"] for r in predicted_noes}

    for exp in experimental_restraints:
        key = get_pair_key(exp)
        if key in pred_map:
            # Satisfaction: Model distance must be <= experimental upper bound
            # (plus a small tolerance for thermal noise)
            if pred_map[key] <= exp["dist"] + 0.5:
                satisfied_count += 1

    recall = satisfied_count / total_exp

    # 2. Precision
    # How many of our predicted short-range NOEs are in the experimental list?
    supported_count = 0
    total_pred = len(predicted_noes)

    if total_pred == 0:
        return {"recall": recall, "precision": 0.0, "f_measure": 0.0}

    exp_keys = {get_pair_key(r) for r in experimental_restraints}

    for pred in predicted_noes:
        if get_pair_key(pred) in exp_keys:
            supported_count += 1

    precision = supported_count / total_pred

    # 3. F-measure
    if (recall + precision) > 0:
        f_measure = (2 * recall * precision) / (recall + precision)
    else:
        f_measure = 0.0

    return {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "f_measure": round(f_measure, 3),
    }


def calculate_dp_score(rpf_scores: Dict[str, float]) -> float:
    """
    Calculate the Discriminating Power (DP) score.

    EDUCATIONAL BACKGROUND — The DP-Score
    ───────────────────────────────────────────────────────────────────
    The DP-score (Huang et al., 2012) normalizes the F-measure to provide
    a standardized metric of structural quality.

    It compares the F-measure of the model against the F-measure expected
    for a "random coil" or poorly folded structure.

    - DP > 0.7: High-quality, native-like fold.
    - DP < 0.5: Likely an incorrect fold or highly disordered.

    Formula: DP = (F_model - F_random) / (1 - F_random)
    (Note: For this project, we use a simplified baseline of F_random = 0.1)
    """
    f = rpf_scores["f_measure"]
    f_random = 0.1  # Statistical baseline for random/denatured proteins
    dp = (f - f_random) / (1.0 - f_random)
    return max(0.0, round(float(dp), 3))


def calculate_cs_r_factor(
    predicted: Dict[str, Dict[int, Dict[str, float]]],
    reference: Dict[str, Dict[int, Dict[str, float]]],
    atom: str = "CA",
    res_name_map: Optional[Dict[int, str]] = None,
) -> float:
    """
    Calculate the Chemical Shift R-factor (Rcs).

    EDUCATIONAL BACKGROUND — Chemical Shift R-factors
    ───────────────────────────────────────────────────────────────────
    Inspired by X-ray Crystallography, the Rcs factor measures the
    normalised agreement between predicted and experimental shifts.

    Unlike RMSE (which is in absolute ppm), the R-factor is dimensionless,
    making it comparable across different atom types (CA vs. N vs. H).

    Correct formula (Wishart 1995 / SPARTA+ convention):
        Rcs = Σ |δ_calc − δ_exp|  /  Σ |δ_exp − δ_rc|

    where δ_rc is the per-residue *random-coil* baseline.  The denominator
    is the total secondary shift amplitude in the reference data — normalising
    by this quantity ensures Rcs ~ 1 for a random model and Rcs ~ 0 for a
    perfect model.

    NOTE: Correct normalisation requires knowing each residue's amino acid
    type.  Supply ``res_name_map`` ({res_id: three_letter_code}) for full
    accuracy.  When omitted the function falls back to using the mean
    random-coil value across all 20 standard amino acids for the requested
    atom type, which is a reasonable approximation for CA (mean ~ 56 ppm).

    Args:
        predicted:    {chain_id: {res_id: {atom_name: value}}}
        reference:    {chain_id: {res_id: {atom_name: value}}}
        atom:         Atom type to compare (default "CA").
        res_name_map: Optional {res_id: three_letter_code} for exact RC lookup.

    Returns:
        Dimensionless R-factor.  Returns 0.0 if there is no overlapping data.
    """
    from synth_nmr.chemical_shifts import RANDOM_COIL_SHIFTS

    # Pre-compute fallback: median random-coil value across all standard residues
    _rc_vals = [v[atom] for v in RANDOM_COIL_SHIFTS.values() if atom in v and v[atom] != 0.0]
    _rc_fallback = float(np.median(_rc_vals)) if _rc_vals else 0.0

    diffs: List[float] = []
    baseline: List[float] = []

    for chain_id, chain_data in predicted.items():
        if chain_id not in reference:
            continue
        for res_id, atoms in chain_data.items():
            if (
                res_id in reference[chain_id]
                and atom in atoms
                and atom in reference[chain_id][res_id]
            ):
                exp_val = reference[chain_id][res_id][atom]
                calc_val = atoms[atom]

                # Numerator: |predicted − experimental|
                diffs.append(abs(calc_val - exp_val))

                # Denominator: |experimental − random_coil|
                # Use per-residue RC value when res_name_map is provided.
                if res_name_map is not None:
                    res_name = res_name_map.get(res_id)
                    rc_val = (
                        RANDOM_COIL_SHIFTS.get(res_name, {}).get(atom, _rc_fallback)
                        if res_name
                        else _rc_fallback
                    )
                else:
                    rc_val = _rc_fallback

                baseline.append(abs(exp_val - rc_val))

    if not diffs:
        return 0.0

    denom = np.sum(baseline)
    if denom == 0.0:
        # Defensive: if all exp values equal their RC (perfectly random coil),
        # the denominator is zero and Rcs is undefined — return 0.
        return 0.0  # pragma: no cover

    r_cs = np.sum(diffs) / denom
    return round(float(r_cs), 4)


def calculate_rdc_q_factor(predicted: Dict[int, float], experimental: Dict[int, float]) -> float:
    """
    Calculate the RDC Q-factor (Cornilescu Q).

    EDUCATIONAL BACKGROUND — The RDC Q-factor
    ───────────────────────────────────────────────────────────────────
    The Q-factor is the standard metric for Residual Dipolar Coupling (RDC)
    validation. It measures the agreement between calculated and
    observed couplings, normalized by the magnitude of the observed data.

    Formula: Q = sqrt( sum( (D_calc - D_exp)^2 ) / sum( D_exp^2 ) )

    Interpretation:
    - Q < 0.2:  Excellent agreement (high-resolution structure).
    - 0.2 < Q < 0.5: Reasonable agreement.
    - Q > 0.5:  Poor agreement or incorrect alignment tensor.

    Args:
        predicted: Map of {residue_id: RDC_value}
        experimental: Map of {residue_id: RDC_value}

    Returns:
        float: The Q-factor.
    """
    common_res = set(predicted.keys()) & set(experimental.keys())
    if not common_res:
        logger.warning("No overlapping residues found for RDC Q-factor calculation.")
        return 1.0

    y_pred = np.array([predicted[r] for r in common_res])
    y_exp = np.array([experimental[r] for r in common_res])

    num = np.sum((y_pred - y_exp) ** 2)
    den = np.sum(y_exp**2)

    if den == 0:
        return 1.0

    q = np.sqrt(num / den)
    return round(float(q), 4)


def validate_against_bmrb(
    bmrb_id: int, structure: struc.AtomArray, predictor: Any = None
) -> Dict[str, Dict[str, float]]:
    """
    Automated validation of a structure against a BMRB entry.

    This high-level function:
    1. Downloads experimental data from BMRB.
    2. Predicts observables for the provided structure.
    3. Calculates accuracy metrics (RMSE, R-factor, RPF).

    Args:
        bmrb_id: BMRB accession ID.
        structure: biotite.structure.AtomArray.
        predictor: Optional custom shift predictor.

    Returns:
        Dict: Validation metrics.
    """
    from synth_nmr.chemical_shifts import predict_chemical_shifts
    from synth_nmr.data_pipeline import download_bmrb_file, parse_bmrb_shifts

    # 1. Get Experimental Data
    bmrb_path = download_bmrb_file(bmrb_id)
    if not bmrb_path:
        raise RuntimeError(f"Could not retrieve BMRB entry {bmrb_id}")

    exp_shifts = parse_bmrb_shifts(bmrb_path)

    # 2. Prediction
    if predictor:
        pred_shifts = predictor.predict(structure)
    else:
        pred_shifts = predict_chemical_shifts(structure)

    # 3. Validation
    # Align and compare (assuming chain A for simplified automated check)
    ref_dict = {"A": exp_shifts}
    stats = compare_chemical_shifts(pred_shifts, ref_dict)

    # Add R-factor for CA
    r_cs = calculate_cs_r_factor(pred_shifts, ref_dict, atom="CA")
    if "CA" in stats:
        stats["CA"]["r_factor"] = r_cs

    return stats


def print_validation_report(stats: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted validation report.
    """
    print("\n" + "=" * 40)
    print(f"{'Atom':<6} | {'RMSE (ppm)':<10} | {'Pearson R':<10} | {'Count':<6}")
    print("-" * 40)
    for atom, metrics in stats.items():
        print(
            f"{atom:<6} | {metrics['rmse']:<10.3f} | {metrics['pearson']:<10.3f} | {metrics['count']:<6}"
        )
    print("=" * 40 + "\n")

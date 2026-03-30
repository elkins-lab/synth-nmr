"""
Validation and comparison utilities for chemical shift predictions.
"""

import logging
from typing import Dict, List, Optional

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
    def get_pair_key(r):
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
) -> float:
    """
    Calculate the Chemical Shift R-factor (Rcs).

    EDUCATIONAL BACKGROUND — Chemical Shift R-factors
    ───────────────────────────────────────────────────────────────────
    Inspired by X-ray Crystallography, the Rcs factor measures the
    normalized agreement between predicted and experimental shifts.

    Unlike RMSE, which is in absolute units (ppm), the R-factor is
    dimensionless, making it easier to compare the quality of different
    atom types (e.g., comparing CA shifts to N shifts).

    Formula: Rcs = sum(|calc - exp|) / sum(|exp - random_coil|)

    A low Rcs (< 0.1) indicates excellent structural agreement.
    """
    # Simplified version: normalize by the range of observed shifts
    # In practice, one would subtract random coil shifts (Wishart et al.)
    diffs = []
    baseline = []

    for chain_id, chain_data in predicted.items():
        if chain_id not in reference:
            continue
        for res_id, atoms in chain_data.items():
            if (
                res_id in reference[chain_id]
                and atom in atoms
                and atom in reference[chain_id][res_id]
            ):
                diffs.append(abs(atoms[atom] - reference[chain_id][res_id][atom]))
                baseline.append(abs(reference[chain_id][res_id][atom]))

    if not diffs:
        return 0.0

    r_cs = np.sum(diffs) / np.sum(baseline)
    return round(float(r_cs), 4)


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

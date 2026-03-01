"""
Validation and comparison utilities for chemical shift predictions.
"""

import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def compare_chemical_shifts(
    predicted: Dict[str, Dict[int, Dict[str, float]]],
    reference: Dict[str, Dict[int, Dict[str, float]]],
    atom_types: List[str] = ["CA", "HA", "N", "H"],
) -> Dict[str, Dict[str, float]]:
    """
    Compare two sets of chemical shifts and calculate validation metrics.

    Args:
        predicted: Predicted shifts {chain_id: {res_id: {atom: val}}}
        reference: Reference shifts (e.g., experimental or ShiftX2)
        atom_types: List of atoms to compare.

    Returns:
        A dictionary of metrics per atom type: {atom: {"rmse": float, "pearson": float}}
    """
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

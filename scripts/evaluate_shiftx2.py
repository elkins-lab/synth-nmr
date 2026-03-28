"""
evaluate_shiftx2.py

Evaluates and compares Chemical Shift Predictions:
This script runs the `synth-nmr` Empirical (SPARTA-based) predictor and the
experimental Neural Predictor against the established `SHIFTX2` baseline on a
given PDB structure. It aligns the backbone and sidechain predictions, calculates
MAE, RMSE, and Pearson correlation, and produces a comparative scatter plot.
"""

import argparse
import os
import sys

# Add parent directory to sys.path to ensure we use local synth_nmr package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import biotite.structure.io.pdb as pdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from synth_nmr.chemical_shifts import ShiftX2Predictor, predict_empirical_shifts

try:
    from synth_nmr.neural_shifts import NeuralShiftPredictor

    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False


def load_structure(pdb_path):
    """
    Loads a PDB file using Biotite and extracts the first structural model.

    Args:
        pdb_path (str): Local path to the input PDB file.

    Returns:
        biotite.structure.AtomArray: The parsed 3D structural model.
    """
    if not os.path.exists(pdb_path):
        print(f"Error: PDB file not found at {pdb_path}")
        sys.exit(1)

    pdb_file = pdb.PDBFile.read(pdb_path)
    return pdb_file.get_structure(model=1)


def align_predictions(synth_preds, shiftx_preds):
    """
    Aligns predictions from both methods for direct element-wise comparison.
    Extracts predictions for CA, CB, C, N, H, and HA atoms, ensuring that
    only atoms predicted by both tools are compared.

    Args:
        synth_preds (dict): Prediction dictionary from `synth-nmr`.
        shiftx_preds (dict): Prediction dictionary from SHIFTX2.

    Returns:
        dict: Mapping of atom types to matched tuples, e.g.
              { "CA": ([synth_val_1, ...], [shiftx_val_1, ...]) }
    """
    aligned_data = {
        "CA": ([], []),
        "CB": ([], []),
        "C": ([], []),
        "N": ([], []),
        "H": ([], []),
        "HA": ([], []),
    }

    # Flatten synth_preds if chain IDs are unpredictable.
    flat_synth = {}
    for _chain_id, res_dict in synth_preds.items():
        for res_id, shifts in res_dict.items():
            flat_synth[int(res_id)] = shifts

    # Flatten shiftx_preds
    flat_shiftx = {}
    for _chain_id, res_dict in shiftx_preds.items():
        for res_id, shifts in res_dict.items():
            flat_shiftx[int(res_id)] = shifts  # Just in case

    for res_id, synth_atom_shifts in flat_synth.items():
        if res_id not in flat_shiftx:
            continue

        shiftx_atom_shifts = flat_shiftx[res_id]

        for atom_type in aligned_data.keys():
            if atom_type in synth_atom_shifts and atom_type in shiftx_atom_shifts:
                aligned_data[atom_type][0].append(synth_atom_shifts[atom_type])
                aligned_data[atom_type][1].append(shiftx_atom_shifts[atom_type])

    return aligned_data


def calculate_metrics(aligned_data):
    """
    Calculates Mean Absolute Error (MAE), Root Mean Square Error (RMSE),
    and Pearson correlation coefficient for each aligned atom type.

    Args:
        aligned_data (dict): Output from `align_predictions`.

    Returns:
        dict: Dictionary of computed metrics for each atom type.
    """
    metrics = {}
    for atom_type, (synth_vals, shiftx_vals) in aligned_data.items():
        if not synth_vals:
            continue

        synth_arr = np.array(synth_vals)
        shiftx_arr = np.array(shiftx_vals)

        mae = np.mean(np.abs(synth_arr - shiftx_arr))
        rmse = np.sqrt(np.mean((synth_arr - shiftx_arr) ** 2))

        # Pearson correlation
        if len(synth_arr) > 1 and np.std(synth_arr) > 0 and np.std(shiftx_arr) > 0:
            corr, _ = pearsonr(synth_arr, shiftx_arr)
        else:
            corr = float("nan")

        metrics[atom_type] = {"Count": len(synth_arr), "MAE": mae, "RMSE": rmse, "Pearson_r": corr}
    return metrics


def plot_comparison(aligned_data_dict, output_file="shift_comparison.png"):
    """
    Generates a scatter plot comparing multiple predictors.
    aligned_data_dict is { "Empirical": aligned_data_emp, "Neural": aligned_data_neural }
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Chemical Shift Prediction Comparison to SHIFTX2")

    axes_flat = axes.flatten()

    # Assume the atom types are the same for all methods
    first_method_data = list(aligned_data_dict.values())[0]

    colors = ["blue", "green", "orange"]

    for idx, atom_type in enumerate(first_method_data.keys()):
        if idx >= len(axes_flat):
            break

        ax = axes_flat[idx]
        has_data = False

        for method_idx, (method_name, aligned_data) in enumerate(aligned_data_dict.items()):
            synth_vals, shiftx_vals = aligned_data[atom_type]
            if not synth_vals:
                continue
            has_data = True

            c = colors[method_idx % len(colors)]
            ax.scatter(shiftx_vals, synth_vals, alpha=0.5, s=20, label=method_name, c=c)

            # Print correlation on plot if it's the first plot of the subgroup or space allows
            if len(synth_vals) > 1:
                try:
                    corr, _ = pearsonr(synth_vals, shiftx_vals)
                    # Offset text depending on method
                    y_pos = 0.95 - (method_idx * 0.05)
                    ax.text(
                        0.05,
                        y_pos,
                        f"{method_name} $r$ = {corr:.2f}",
                        transform=ax.transAxes,
                        verticalalignment="top",
                        color=c,
                        fontweight="bold",
                    )
                except Exception:
                    pass

        if not has_data:
            ax.set_title(f"{atom_type} (No Data)")
            ax.axis("off")
            continue

        # Add y=x line
        all_vals = []
        for d in aligned_data_dict.values():
            all_vals.extend(d[atom_type][0])
            all_vals.extend(d[atom_type][1])

        if all_vals:
            min_val = min(all_vals)
            max_val = max(all_vals)
            ax.plot([min_val, max_val], [min_val, max_val], "r--", label="y=x", alpha=0.7)

        ax.set_title(f"{atom_type} Chemical Shifts")
        ax.set_xlabel("SHIFTX2 Predicted (ppm)")
        ax.set_ylabel("synth-nmr Predicted (ppm)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")


def print_metrics_table(method_name, metrics):
    """
    Prints the calculated metrics as a formatted table to stdout.

    Args:
        method_name (str): Label for the model being evaluated (e.g., Empirical, Neural).
        metrics (dict): The output of `calculate_metrics`.
    """
    print(f"\n--- {method_name} vs SHIFTX2 ---")
    print(f"{'Atom':<6} | {'Count':<6} | {'MAE':<8} | {'RMSE':<8} | {'Pearson r'}")
    print("-" * 50)

    for atom_type, m in metrics.items():
        if m["Count"] == 0:
            print(f"{atom_type:<6} | {0:<6d} | {'N/A':<8} | {'N/A':<8} | N/A")
            continue
        corr_str = f"{m['Pearson_r']:.3f}" if not np.isnan(m["Pearson_r"]) else "NaN"
        print(
            f"{atom_type:<6} | {m['Count']:<6d} | {m['MAE']:<8.3f} | {m['RMSE']:<8.3f} | {corr_str}"
        )


def main():
    """
    Main entry point for the evaluation script. Parses CLI arguments, loads
    target PDB structure, runs the predictions mapping for Empirical, Neural,
    and SHIFTX2 models, computes the statistical differences, and finally
    generates the visual comparison plot.
    """
    parser = argparse.ArgumentParser(description="Evaluate synth-nmr predictions against SHIFTX2")
    parser.add_argument("pdb_file", help="Path to input PDB file")
    parser.add_argument(
        "--shiftx2-exe", default="shiftx2.py", help="Path/command for SHIFTX2 executable"
    )
    parser.add_argument(
        "--plot", default="shift_comparison.png", help="Output path for the comparison plot"
    )

    args = parser.parse_args()

    print(f"Loading structure from {repr(args.pdb_file)}...")
    structure = load_structure(args.pdb_file)

    print("Running SHIFTX2 prediction...")
    shiftx_predictor = ShiftX2Predictor(executable=args.shiftx2_exe)
    if not shiftx_predictor.is_available():
        print(
            f"Error: SHIFTX2 executable '{args.shiftx2_exe}' not found in PATH or not executable."
        )
        sys.exit(1)

    try:
        shiftx_preds = shiftx_predictor.predict(structure)
    except Exception as e:
        print(f"Error running SHIFTX2: {e}")
        sys.exit(1)

    aligned_data_dict = {}

    print("\nRunning synth-nmr Empirical prediction...")
    emp_preds = predict_empirical_shifts(structure)
    aligned_emp = align_predictions(emp_preds, shiftx_preds)
    aligned_data_dict["Empirical"] = aligned_emp

    metrics_emp = calculate_metrics(aligned_emp)
    print_metrics_table("Empirical", metrics_emp)

    if HAS_NEURAL:
        print("\nRunning synth-nmr Neural prediction...")
        neural_preds = NeuralShiftPredictor().predict(structure)
        aligned_neu = align_predictions(neural_preds, shiftx_preds)
        aligned_data_dict["Neural"] = aligned_neu

        metrics_neu = calculate_metrics(aligned_neu)
        print_metrics_table("Neural", metrics_neu)
    else:
        print("\nSkipping Neural prediction (model not available).")

    try:
        plot_comparison(aligned_data_dict, args.plot)
    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

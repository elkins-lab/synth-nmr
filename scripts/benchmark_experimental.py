"""
A script to evaluate the newly trained Neural Shift Predictor against the empirical
baseline (SPARTA+) using the experimental BMRB dataset.
"""

import logging

import numpy as np

from synth_nmr.chemical_shifts import predict_chemical_shifts
from synth_nmr.data_pipeline import load_matched_dataset
from synth_nmr.neural_shifts import NUCLEUS_ORDER, NeuralShiftPredictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    print("Loading Experimental Benchmark Dataset...")
    dataset_pairs = load_matched_dataset(data_dir="data")

    print("Loading Trained Neural Model...")
    # Load the model we just trained into the standard checkpoint path
    neural_model = NeuralShiftPredictor()

    empirical_errors = {nuc: [] for nuc in NUCLEUS_ORDER}
    neural_errors = {nuc: [] for nuc in NUCLEUS_ORDER}

    valid_count = 0

    for struct, exp_shifts in dataset_pairs:
        # Run predictions on the whole structure
        emp_preds = predict_chemical_shifts(struct)
        neu_preds = neural_model.predict(struct)

        for res_idx in range(len(struct.res_id)):
            res_id = int(struct.res_id[res_idx])
            chain_id = struct.chain_id[res_idx]

            # Skip if we already processed this residue (atom iterates)
            if res_idx > 0 and struct.res_id[res_idx] == struct.res_id[res_idx - 1]:
                continue

            exp = exp_shifts.get(res_id, {})
            if not exp:
                continue

            emp = emp_preds.get(chain_id, {}).get(res_id, {})
            neu = neu_preds.get(chain_id, {}).get(res_id, {})

            for nuc in NUCLEUS_ORDER:
                if nuc in exp and nuc in emp and nuc in neu:
                    empirical_errors[nuc].append(emp[nuc] - exp[nuc])
                    neural_errors[nuc].append(neu[nuc] - exp[nuc])
                    valid_count += 1

    print(f"\nEvaluated on {valid_count} specific atom assignments.")
    print("=" * 55)
    print(f"{'Nucleus':<10} | {'Empirical RMSE':<15} | {'Neural RMSE':<15}")
    print("=" * 55)

    emp_all_err = []
    neu_all_err = []

    for nuc in NUCLEUS_ORDER:
        emp_err = np.array(empirical_errors[nuc])
        neu_err = np.array(neural_errors[nuc])

        emp_all_err.extend(emp_err)
        neu_all_err.extend(neu_err)

        emp_rmse = np.sqrt(np.mean(emp_err**2)) if len(emp_err) > 0 else 0.0
        neu_rmse = np.sqrt(np.mean(neu_err**2)) if len(neu_err) > 0 else 0.0

        print(f"{nuc:<10} | {emp_rmse:<15.3f} | {neu_rmse:<15.3f}")

    overall_emp = np.sqrt(np.mean(np.array(emp_all_err) ** 2))
    overall_neu = np.sqrt(np.mean(np.array(neu_all_err) ** 2))

    print("-" * 55)
    print(f"{'OVERALL':<10} | {overall_emp:<15.3f} | {overall_neu:<15.3f}")
    print("=" * 55)


if __name__ == "__main__":
    main()

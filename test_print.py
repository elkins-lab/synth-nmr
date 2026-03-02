from scripts.evaluate_shiftx2 import *;
s = load_structure('/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/1UBQ.pdb')
pb=predict_empirical_shifts(s)
px=ShiftX2Predictor(executable='/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/shiftx2.py').predict(s)
aligned=align_predictions(pb, px)
metrics=calculate_metrics(aligned)
print("Metrics dictionary:")
print(metrics)
print("Running loop...")
for atom_type, m in metrics.items():
    corr_str = f"{m['Pearson_r']:.3f}" if not np.isnan(m['Pearson_r']) else "NaN"
    print(f"{atom_type:<6} | {m['Count']:<6d} | {m['MAE']:<8.3f} | {m['RMSE']:<8.3f} | {corr_str}")
print("Loop finished.")

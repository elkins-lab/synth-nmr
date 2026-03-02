from scripts.evaluate_shiftx2 import *;
s = load_structure('/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/1UBQ.pdb')
pb=predict_empirical_shifts(s)

shiftx_predictor = ShiftX2Predictor(executable='/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/shiftx2.py')
shiftx_preds = shiftx_predictor.predict(s)

print("Synth chain count:", len(pb), "Res count A:", len(pb.get('A', {})))
print("ShiftX2 chain count:", len(shiftx_preds), "Res count A:", len(shiftx_preds.get('A', {})))

aligned_data = align_predictions(pb, shiftx_preds)
print("Aligned CA elements:", len(aligned_data['CA'][0]))

metrics = calculate_metrics(aligned_data)
print("Calculated metrics for", len(metrics), "atom types")

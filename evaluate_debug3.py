from scripts.evaluate_shiftx2 import *;
s = load_structure('/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/1UBQ.pdb')
pb=predict_empirical_shifts(s)
pn=NeuralShiftPredictor().predict(s)
px=ShiftX2Predictor(executable='/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/shiftx2.py').predict(s)

aligned_emp = align_predictions(pb, px)
aligned_neu = align_predictions(pn, px)

metrics_emp = calculate_metrics(aligned_emp)
metrics_neu = calculate_metrics(aligned_neu)

print("Empirical aligned subset sizes:", [len(v[0]) for v in aligned_emp.values()])
print("Neural aligned subset sizes:", [len(v[0]) for v in aligned_neu.values()])

print("Empirical metrics count for CA:", metrics_emp.get("CA", {}).get("Count"))
print("Neural metrics count for CA:", metrics_neu.get("CA", {}).get("Count"))


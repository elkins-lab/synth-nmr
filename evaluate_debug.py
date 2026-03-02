from scripts.evaluate_shiftx2 import *;
s = load_structure('/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/1UBQ.pdb')
pb=predict_empirical_shifts(s)
px=ShiftX2Predictor(executable='/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/shiftx2.py').predict(s)

print("--- DEBUG PB ---")
print(list(pb.keys()))
if 'A' in pb: print(list(pb['A'].keys())[:5])

print("--- DEBUG PX ---")
print(list(px.keys()))
if 'A' in px: print(list(px['A'].keys())[:5])

aligned = align_predictions(pb, px)
print("Aligned:", {k: len(v[0]) for k,v in aligned.items()})

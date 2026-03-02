from scripts.evaluate_shiftx2 import *;
s = load_structure('/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/1UBQ.pdb')

print("Running SHIFTX2 prediction...")
shiftx_predictor = ShiftX2Predictor(executable='/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/shiftx2.py')
shiftx_preds = shiftx_predictor.predict(s)
print("ShiftX2 count:", len(shiftx_preds.get('A', {})))

import biotite.structure.io.pdb as pdb
from synth_nmr.chemical_shifts import predict_empirical_shifts, ShiftX2Predictor
from scripts.evaluate_shiftx2 import align_predictions

s = pdb.PDBFile.read('/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/1UBQ.pdb').get_structure(model=1)
synth_preds = predict_empirical_shifts(s)
shiftx_preds = ShiftX2Predictor(executable='/Users/georgeelkins/nmr/shiftx2/shiftx2-mac/shiftx2.py').predict(s)

aligned = align_predictions(synth_preds, shiftx_preds)
print("Aligned:", {k: len(v[0]) for k, v in aligned.items()})

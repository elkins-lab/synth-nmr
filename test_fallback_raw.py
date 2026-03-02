import logging
import numpy as np
import biotite.structure as struc

# Enable logging to console precisely so we can see it
logging.basicConfig(level=logging.WARNING)

from synth_nmr.chemical_shifts import predict_chemical_shifts

def mock_predict(*args, **kwargs):
    raise RuntimeError("Standalone missing torch mock")

# We monkey-patch the class before we use the function
import synth_nmr.neural_shifts
synth_nmr.neural_shifts.NeuralShiftPredictor.predict = mock_predict

structure = struc.AtomArray(1)
structure.res_name = np.array(["ALA"])
structure.res_id = np.array([1])
structure.chain_id = np.array(["A"])
structure.atom_name = np.array(["CA"])
structure.coord = np.array([[0,0,0]])

print("--- Calling predict_chemical_shifts ---")
shifts = predict_chemical_shifts(structure)
print("--- Result ---")
print(shifts)
if "A" in shifts and 1 in shifts["A"]:
    print("Fallback successful!")
else:
    print("Fallback FAILED!")

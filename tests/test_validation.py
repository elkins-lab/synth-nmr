"""
Validation tests for comparing synth-nmr output against published results.
"""

import pytest
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from io import StringIO
import requests
from synth_nmr.validation import compare_chemical_shifts
from synth_nmr.chemical_shifts import ShiftX2Predictor, predict_chemical_shifts
from synth_nmr.j_coupling import calculate_hn_ha_coupling

@pytest.fixture
def ubiquitin_structure():
    """
    Downloads the 1D3Z PDB file from RCSB and returns the first model
    filtered for amino acids.
    """
    PDB_ID = "1D3Z"
    RCSB_URL = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
    
    # Download PDB file
    response = requests.get(RCSB_URL)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    pdb_string = response.text
    
    # Parse PDB file
    pdb_file = pdb.PDBFile.read(StringIO(pdb_string))
    # get_structure with a model number returns an AtomArray
    structure = pdb.get_structure(pdb_file, model=1)
    
    # Filter for protein and remove alternate location identifiers
    structure = structure[struc.filter_amino_acids(structure)]
    
    return structure

def test_j_coupling_validation_against_1d3z(ubiquitin_structure):
    """
    Validates calculated 3J(HN-Ha) couplings against experimental
    values published for ubiquitin (PDB: 1D3Z, BMRB: 5943).
    """
    experimental_j_couplings = {
        2: 5.44, 3: 8.65, 4: 7.74, 5: 8.52, 6: 7.77, 7: 5.67, 8: 6.99,
        9: 7.07, 10: 7.06, 11: 8.01, 12: 8.44, 13: 8.87, 14: 9.77, 15: 8.35,
        16: 8.61, 17: 6.94, 21: 5.16, 22: 4.08, 23: 5.68, 24: 6.72, 25: 5.92,
        26: 7.21, 27: 7.79, 28: 7.37, 29: 7.42, 30: 8.48, 31: 3.51, 32: 5.48,
        33: 5.86, 34: 4.38, 35: 5.82, 36: 8.23, 40: 6.5, 41: 8.24, 42: 8.79,
        43: 8.45, 44: 8.61, 45: 8.5, 46: 8.04, 47: 7.45, 48: 8.2, 49: 6.76,
        50: 3.99, 51: 4.29, 52: 6.13, 53: 6.09, 54: 7.82, 55: 6.37, 56: 4.09,
        57: 5.46, 58: 4.88, 59: 7.34, 60: 8.32, 61: 9.38, 62: 5.51, 63: 6.84,
        64: 5.0, 65: 7.87, 66: 8.8, 67: 8.71, 68: 7.27, 69: 8.84, 70: 8.73,
        71: 8.25
    }
    
    structure = ubiquitin_structure
    predicted_couplings_all_chains = calculate_hn_ha_coupling(structure)
    predicted_couplings = predicted_couplings_all_chains.get('A', {})
    
    experimental_vals, predicted_vals = [], []
    for res_id, exp_j in experimental_j_couplings.items():
        if res_id in predicted_couplings:
            experimental_vals.append(exp_j)
            predicted_vals.append(predicted_couplings[res_id])
    
    assert len(predicted_vals) > 50
    rmsd = np.sqrt(np.mean((np.array(predicted_vals) - np.array(experimental_vals))**2))
    assert rmsd < 2.2, f"RMSD for J-couplings is {rmsd:.2f} Hz, which is too high."

def test_compare_chemical_shifts():
    """Test the shift comparison logic with synthetic data."""
    # Data format: {chain: {res: {atom: val}}}
    pred = {
        "A": {
            1: {"CA": 55.0, "HA": 4.5},
            2: {"CA": 60.0, "HA": 4.0}
        }
    }
    
    # Reference with slight deviations
    ref = {
        "A": {
            1: {"CA": 55.5, "HA": 4.6},
            2: {"CA": 59.5, "HA": 3.9}
        }
    }
    
    stats = compare_chemical_shifts(pred, ref, atom_types=["CA", "HA"])
    
    assert "CA" in stats
    assert "HA" in stats
    assert stats["CA"]["count"] == 2
    # RMSE for CA: sqrt(((55-55.5)^2 + (60-59.5)^2)/2) = sqrt((0.25+0.25)/2) = 0.5
    assert np.isclose(stats["CA"]["rmse"], 0.5)

def test_shiftx2_predictor_availability():
    """Test the availability check (should be False in this environment)."""
    predictor = ShiftX2Predictor("non_existent_executable")
    assert not predictor.is_available()

def test_shiftx2_parse_output(tmp_path):
    """Test parsing logic of ShiftX2 wrapper."""
    # Create a mock ShiftX2 output file
    mock_file = tmp_path / "test.cs"
    mock_content = """NUM,RES,ATOMNAME,SHIFT
1,MET,CA,54.321
1,MET,HA,4.123
2,GLY,CA,45.678
"""
    mock_file.write_text(mock_content)
    
    predictor = ShiftX2Predictor()
    shifts = predictor._parse_output(str(mock_file))
    
    assert shifts["A"][1]["CA"] == 54.321
    assert shifts["A"][1]["HA"] == 4.123
    assert shifts["A"][2]["CA"] == 45.678

def test_integration_workflow_mocked(monkeypatch):
    """
    Simulate a full validation workflow using a mock ShiftX2.
    """
    # 1. Mock ShiftX2Predictor.predict to return fixed shifts
    def mock_predict(self, structure):
        return {
            "A": {
                i: {"CA": 50.0 + i} for i in range(1, 6)
            }
        }
    
    monkeypatch.setattr(ShiftX2Predictor, "predict", mock_predict)
    
    # 2. Setup a dummy structure
    atom1 = struc.Atom([0,0,0], res_id=1, res_name="ALA", atom_name="CA", chain_id="A")
    atom2 = struc.Atom([1,1,1], res_id=2, res_name="ALA", atom_name="CA", chain_id="A")
    structure = struc.array([atom1, atom2])
    
    # 3. Predict internally
    # Note: predict_chemical_shifts needs more than 2 atoms usually for SS, 
    # but for this mock test we just want to see if the comparison runs.
    # We'll mock the internal predictor too for stability in this test
    monkeypatch.setattr("synth_nmr.chemical_shifts._NOISE_SCALE", 0.0)
    
    internal_shifts = {
        "A": {
            1: {"CA": 51.5},
            2: {"CA": 52.5}
        }
    }
    
    # 4. Use "ShiftX2"
    sx2 = ShiftX2Predictor()
    sx2_shifts = sx2.predict(structure)
    
    # 5. Compare
    stats = compare_chemical_shifts(internal_shifts, sx2_shifts, atom_types=["CA"])
    
    assert stats["CA"]["count"] == 2
    # Internal (51.5, 52.5) vs SX2 (51.0, 52.0) -> Error 0.5 each -> RMSE 0.5
    assert np.isclose(stats["CA"]["rmse"], 0.5)


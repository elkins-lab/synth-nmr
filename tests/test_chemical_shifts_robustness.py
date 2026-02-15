"""
Tests for the robustness and validation of the chemical_shifts module.
"""

import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import os
from synth_nmr.chemical_shifts import predict_chemical_shifts, calculate_csi

# Get the directory of the current test file
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@pytest.fixture
def sample_structure():
    """Fixture to load a sample PDB file."""
    pdb_path = os.path.join(TEST_DATA_DIR, "test.pdb")
    return strucio.load_structure(pdb_path)

# --- Tests for predict_chemical_shifts ---

def test_predict_shifts_invalid_input():
    """Test predict_chemical_shifts with invalid input types."""
    with pytest.raises(TypeError):
        predict_chemical_shifts("not_a_structure")
    with pytest.raises(TypeError):
        predict_chemical_shifts(None)

def test_predict_shifts_empty_structure():
    """Test predict_chemical_shifts with an empty AtomArray."""
    empty_structure = struc.AtomArray(0)
    # Should return an empty dict and log a warning
    assert predict_chemical_shifts(empty_structure) == {}

def test_predict_shifts_normal_case(sample_structure, caplog):
    """Test predict_chemical_shifts with a valid structure."""
    import logging
    caplog.set_level(logging.INFO)
    
    shifts = predict_chemical_shifts(sample_structure)
    
    assert "Predicting chemical shifts" in caplog.text
    assert isinstance(shifts, dict)
    assert "A" in shifts  # Chain A
    
    # The test PDB has two residues, GLY (1) and ALA (2)
    assert 1 in shifts["A"]
    assert 2 in shifts["A"]
    
    # Check content of ALA residue shifts
    ala_shifts = shifts["A"][2]
    assert "CA" in ala_shifts
    assert "H" in ala_shifts
    assert isinstance(ala_shifts["CA"], float)

# --- Tests for calculate_csi ---

def test_csi_invalid_input(sample_structure):
    """Test calculate_csi with invalid input types."""
    # Valid shifts, invalid structure
    shifts = {"A": {1: {"CA": 55.0}}}
    with pytest.raises(TypeError):
        calculate_csi(shifts, "not_a_structure")
    with pytest.raises(TypeError):
        calculate_csi(shifts, None)
        
    # Invalid shifts, valid structure
    with pytest.raises(TypeError):
        calculate_csi("not_a_dict", sample_structure)
    with pytest.raises(TypeError):
        calculate_csi(None, sample_structure)

def test_csi_empty_inputs(sample_structure):
    """Test calculate_csi with empty inputs."""
    empty_shifts = {}
    empty_structure = struc.AtomArray(0)
    
    # Empty shifts dict
    assert calculate_csi(empty_shifts, sample_structure) == {}
    # Empty structure
    shifts = {"A": {1: {"CA": 55.0}}}
    assert calculate_csi(shifts, empty_structure) == {}

def test_csi_normal_case(sample_structure, caplog):
    """Test a full cycle of shift prediction and CSI calculation."""
    import logging
    caplog.set_level(logging.INFO)

    # 1. Predict shifts
    shifts = predict_chemical_shifts(sample_structure)
    assert shifts, "Shift prediction should return a non-empty dictionary."

    # 2. Calculate CSI
    csi_data = calculate_csi(shifts, sample_structure)
    
    assert "Calculating Chemical Shift Index" in caplog.text
    assert isinstance(csi_data, dict)
    assert "A" in csi_data
    
    # Check that we have CSI values for our residues
    assert 1 in csi_data["A"] # GLY
    assert 2 in csi_data["A"] # ALA
    
    # CSI value should be a float
    assert isinstance(csi_data["A"][2], float)

def test_csi_mismatched_inputs(sample_structure):
    """Test CSI calculation where shifts data doesn't match the structure."""
    # Shifts dictionary contains a residue not in the structure
    shifts = {
        "A": {
            1: {"CA": 50.0},
            99: {"CA": 55.0} # Residue 99 is not in sample_structure
        }
    }
    csi_data = calculate_csi(shifts, sample_structure)
    
    assert "A" in csi_data
    assert 1 in csi_data["A"]
    assert 99 not in csi_data["A"] # Should be skipped

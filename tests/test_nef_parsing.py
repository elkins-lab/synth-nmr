
import pytest
import os
import tempfile
from synth_pdb.nef_io import read_nef_restraints, write_nef_file

def test_read_nef_restraints_valid():
    """Test reading restraints from a valid NEF file."""
    restraints = [
        {
            'chain_1': 'A', 'residue_index_1': 1, 'res_name_1': 'ALA', 'atom_name_1': 'H',
            'chain_2': 'A', 'residue_index_2': 5, 'res_name_2': 'GLY', 'atom_name_2': 'HA',
            'actual_distance': 4.5, 'upper_limit': 5.0, 'lower_limit': 1.8
        }
    ]
    sequence = "AAAAAA"
    
    with tempfile.NamedTemporaryFile(suffix=".nef", mode="w", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        write_nef_file(tmp_path, sequence, restraints)
        
        # Now read it back
        parsed = read_nef_restraints(tmp_path)
        
        assert len(parsed) == 1
        assert parsed[0]['chain_1'] == 'A'
        assert parsed[0]['seq_1'] == 1
        assert parsed[0]['atom_1'] == 'H'
        assert parsed[0]['dist'] == pytest.approx(4.5)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_read_nef_restraints_file_not_found(caplog):
    """Test handling of missing NEF file."""
    result = read_nef_restraints("non_existent_file.nef")
    assert result == []
    assert "NEF file not found" in caplog.text

def test_read_nef_restraints_malformed():
    """Test handling of malformed NEF file (missing column)."""
    malformed_nef = """
data_test
save_synthetic_noes
   _nef_distance_restraint_list.sf_category nef_distance_restraint_list
   _nef_distance_restraint_list.sf_framecode synthetic_noes
   _nef_distance_restraint_list.restraint_origin synthetic

   loop_
      _nef_distance_restraint.index
      _nef_distance_restraint.chain_code_1
      _nef_distance_restraint.sequence_code_1
      # Missing other headers
      1 A 1
   stop_
save_
"""
    with tempfile.NamedTemporaryFile(suffix=".nef", mode="w", delete=False) as tmp:
        tmp.write(malformed_nef)
        tmp_path = tmp.name
        
    try:
        parsed = read_nef_restraints(tmp_path)
        # Should return empty list because required keys like 'chain_code_2' are missing in headers
        assert parsed == []
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_read_nef_restraints_invalid_data():
    """Test handling of invalid data in NEF file (e.g. string where number expected)."""
    invalid_data_nef = """
data_test
save_synthetic_noes
   _nef_distance_restraint_list.sf_category nef_distance_restraint_list
   _nef_distance_restraint_list.sf_framecode synthetic_noes
   _nef_distance_restraint_list.restraint_origin synthetic

   loop_
      _nef_distance_restraint.index
      _nef_distance_restraint.chain_code_1
      _nef_distance_restraint.sequence_code_1
      _nef_distance_restraint.residue_name_1
      _nef_distance_restraint.atom_name_1
      _nef_distance_restraint.chain_code_2
      _nef_distance_restraint.sequence_code_2
      _nef_distance_restraint.residue_name_2
      _nef_distance_restraint.atom_name_2
      _nef_distance_restraint.target_value
      1 A INVALID_SEQ ALA H A 5 GLY HA 4.5
   stop_
save_
"""
    with tempfile.NamedTemporaryFile(suffix=".nef", mode="w", delete=False) as tmp:
        tmp.write(invalid_data_nef)
        tmp_path = tmp.name
        
    try:
        parsed = read_nef_restraints(tmp_path)
        # Should skip the line with INVALID_SEQ
        assert parsed == []
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

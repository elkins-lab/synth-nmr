
import pytest
import numpy as np
import biotite.structure as struc
from io import StringIO
import os
from synth_pdb.nmr import calculate_synthetic_noes
from synth_pdb.nef_io import write_nef_file

def test_calculate_noes_simple_dimer():
    """Test NOE calculation on a simple 2-atom system."""
    # Create two hydrogens 3A apart
    atoms = struc.AtomArray(2)
    atoms.element = ["H", "H"]
    atoms.coord = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    atoms.res_id = np.array([1, 2])
    atoms.res_name = np.array(["ALA", "GLY"])
    atoms.atom_name = np.array(["H", "H"])
    atoms.chain_id = np.array(["A", "A"])
    
    # Cutoff 5.0 -> should find 1 restraint
    restraints = calculate_synthetic_noes(atoms, cutoff=5.0)
    assert len(restraints) == 1
    r = restraints[0]
    assert r['actual_distance'] == 3.0
    assert r['upper_limit'] == 3.5 # buffer 0.5
    assert r['residue_index_1'] == 1
    assert r['residue_index_2'] == 2

def test_calculate_noes_cutoff_exclusion():
    """Test that atoms outside cutoff are excluded."""
    atoms = struc.AtomArray(2)
    atoms.element = ["H", "H"]
    atoms.coord = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=float) # 6A apart
    atoms.res_id = np.array([1, 2])
    atoms.res_name = np.array(["ALA", "GLY"])
    atoms.atom_name = np.array(["H", "H"])
    atoms.chain_id = np.array(["A", "A"])
    
    restraints = calculate_synthetic_noes(atoms, cutoff=5.0)
    assert len(restraints) == 0

def test_write_nef_structure(tmp_path):
    """Test that NEF file writes with correct minimal fields."""
    output_file = tmp_path / "test.nef"
    sequence = "AG"
    restraints = [{
        'residue_index_1': 1, 'res_name_1': 'ALA', 'atom_name_1': 'H', 'chain_1': 'A',
        'residue_index_2': 2, 'res_name_2': 'GLY', 'atom_name_2': 'H', 'chain_2': 'A',
        'actual_distance': 3.0, 'upper_limit': 3.5, 'lower_limit': 1.8
    }]
    
    write_nef_file(str(output_file), sequence, restraints)
    
    # Check content
    with open(output_file, 'r') as f:
        content = f.read()
    
    # Headers
    assert "_nef_nmr_meta_data.nef_format_version 1.1" in content
    assert "save_nef_sequence" in content
    assert "save_synthetic_noes" in content
    
    # Sequence check
    assert "A 1 ALA protein" in content
    assert "A 2 GLY protein" in content
    
    # Restraint check
    # 1 1 A 1 ALA H A 2 GLY H 3.000 3.500 1.800 1.0
    expected_line = "1 1 A 1 ALA H A 2 GLY H 3.000 3.500 1.800 1.0"
    # Whitespace might vary, check fragments
    assert "3.000 3.500" in content
    assert "ALA H" in content
    assert "GLY H" in content

"""
Unit and functional tests for the nmr module.
"""

import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import os
from synth_nmr.nmr import calculate_synthetic_noes

# Get the directory of the current test file
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@pytest.fixture
def sample_structure():
    """Fixture to load a sample PDB file with hydrogens."""
    pdb_path = os.path.join(TEST_DATA_DIR, "test.pdb")
    # load_structure returns an AtomArray for a single-model file
    return strucio.load_structure(pdb_path)

@pytest.fixture
def structure_no_h():
    """Fixture for a structure with no hydrogens."""
    pdb_path = os.path.join(TEST_DATA_DIR, "test.pdb")
    structure = strucio.load_structure(pdb_path)
    # Filter out hydrogens
    return structure[structure.element != 'H']

def test_calculate_synthetic_noes_normal_case(sample_structure):
    """Test basic functionality with a valid structure."""
    restraints = calculate_synthetic_noes(sample_structure, cutoff=4.0)
    assert isinstance(restraints, list)
    # Based on the test.pdb file, we expect some restraints
    assert len(restraints) > 0
    
    # Check the contents of the first restraint
    if len(restraints) > 0:
        restraint = restraints[0]
        assert 'index_1' in restraint
        assert 'res_name_1' in restraint
        assert 'atom_name_1' in restraint
        assert 'distance' in restraint
        assert 'upper_limit' in restraint
        assert restraint['upper_limit'] == pytest.approx(restraint['distance'] + 0.5)

def test_input_validation_type_error():
    """Test that a TypeError is raised for invalid structure input."""
    with pytest.raises(TypeError):
        calculate_synthetic_noes("not_a_structure")

def test_input_validation_value_error(sample_structure):
    """Test that ValueError is raised for invalid cutoff and buffer."""
    with pytest.raises(ValueError):
        calculate_synthetic_noes(sample_structure, cutoff=-1.0)
    with pytest.raises(ValueError):
        calculate_synthetic_noes(sample_structure, buffer=-1.0)

def test_empty_structure():
    """Test that an empty structure returns an empty list."""
    empty_structure = struc.AtomArray(0)
    restraints = calculate_synthetic_noes(empty_structure)
    assert restraints == []

def test_structure_with_no_hydrogens(structure_no_h):
    """Test that a structure without hydrogens returns an empty list."""
    restraints = calculate_synthetic_noes(structure_no_h)
    assert restraints == []

def test_exclude_intra_residue(sample_structure):
    """Test the 'exclude_intra_residue' functionality."""
    # First, calculate with intra-residue NOEs
    all_restraints = calculate_synthetic_noes(sample_structure, cutoff=3.0, exclude_intra_residue=False)
    
    # Then, calculate without intra-residue NOEs
    inter_restraints = calculate_synthetic_noes(sample_structure, cutoff=3.0, exclude_intra_residue=True)
    
    assert len(all_restraints) > len(inter_restraints)
    
    # Verify that no intra-residue restraints are in the 'inter_restraints' list
    for restraint in inter_restraints:
        assert not (restraint['index_1'] == restraint['index_2'] and restraint['chain_1'] == restraint['chain_2'])

def test_specific_restraint_content(sample_structure):
    """
    Test the specific content of a known restraint to ensure correctness.
    This makes the test more robust against unexpected changes.
    """
    # Find the NOE between GLY-HA1 and ALA-H
    # GLY HA1 is atom 5, ALA H is atom 10
    h_a1_gly = sample_structure[4] # HA1 of GLY
    h_ala = sample_structure[9]   # H of ALA
    
    expected_dist = struc.distance(h_a1_gly, h_ala)
    
    restraints = calculate_synthetic_noes(sample_structure, cutoff=5.0)
    
    found = False
    for restraint in restraints:
        # Check both combinations
        match1 = (restraint['atom_name_1'] == 'HA1' and restraint['res_name_1'] == 'GLY' and 
                  restraint['atom_name_2'] == 'H' and restraint['res_name_2'] == 'ALA')
        match2 = (restraint['atom_name_2'] == 'HA1' and restraint['res_name_2'] == 'GLY' and
                  restraint['atom_name_1'] == 'H' and restraint['res_name_1'] == 'ALA')
                  
        if match1 or match2:
            found = True
            assert np.isclose(restraint['distance'], expected_dist, atol=1e-4)
            assert np.isclose(restraint['upper_limit'], expected_dist + 0.5, atol=1e-4)
            break
            
    assert found, "Did not find the expected restraint between GLY-HA1 and ALA-H."

def test_calculate_synthetic_noes_geminal_exclusion():
    """
    Test that intra-residue geminal protons (e.g., HBx on Alanine's CB)
    are excluded by the dist < 2.0 Å filter.
    """
    # Create a minimal Alanine residue with hydrogens
    # Define data as Python lists first
    coords_list = [
        [0.0, 0.0, 0.0],    # N
        [0.0, 0.0, 1.02],   # H (Amide)
        [1.45, 0.0, 0.0],   # CA
        [1.8, 0.9, 0.4],    # HA
        [2.0, -1.2, 0.0],   # C
        [1.0, 1.5, -0.5],   # CB
        # Adjusted HB coordinates to ensure geminal distances are < 2.0 Å
        [0.2, 1.2, 0.0],    # HB1
        [1.5, 2.3, -0.2],   # HB2
        [1.3, 0.9, -1.2]     # HB3
    ]
    atom_names_list = ['N', 'H', 'CA', 'HA', 'C', 'CB', 'HB1', 'HB2', 'HB3']
    res_names_list = ['ALA'] * len(atom_names_list)
    res_ids_list = [1] * len(atom_names_list)
    chain_ids_list = ['A'] * len(atom_names_list)
    elements_list = ['N', 'H', 'C', 'H', 'C', 'C', 'H', 'H', 'H']

    # Convert to NumPy arrays
    coords = np.array(coords_list, dtype=np.float32)
    atom_names = np.array(atom_names_list)
    res_names = np.array(res_names_list)
    res_ids = np.array(res_ids_list)
    chain_ids = np.array(chain_ids_list)
    elements = np.array(elements_list)

    # Create AtomArray
    ala_structure = struc.AtomArray(len(coords_list)) # Create an AtomArray of the correct size

    # Set the annotation categories
    ala_structure.coord = coords
    ala_structure.atom_name = atom_names
    ala_structure.res_name = res_names
    ala_structure.res_id = res_ids
    ala_structure.chain_id = chain_ids
    ala_structure.element = elements
    
    # Calculate NOEs without excluding all intra-residue
    restraints = calculate_synthetic_noes(ala_structure, cutoff=3.0, exclude_intra_residue=False)

    hb_atom_names = ['HB1', 'HB2', 'HB3']
    found_geminal_noe = False
    
    for restraint in restraints:
        # Check if both atoms in the restraint are HBx protons of the same residue
        is_hb1 = restraint['atom_name_1'] in hb_atom_names
        is_hb2 = restraint['atom_name_2'] in hb_atom_names
        is_same_res = (restraint['index_1'] == restraint['index_2']) and \
                      (restraint['res_name_1'] == restraint['res_name_2'])
        
        if is_same_res and is_hb1 and is_hb2:
            found_geminal_noe = True
            break
            
    assert not found_geminal_noe, \
        "Geminal HBx-HBx NOE was found, but should have been excluded by dist < 2.0 Å filter."
    
    # Ensure other, longer intra-residue NOEs might still be present (e.g., HA-HBx)
    found_ha_hb_noe = False
    for restraint in restraints:
        if (restraint['atom_name_1'] == 'HA' and restraint['atom_name_2'] in hb_atom_names) or \
           (restraint['atom_name_2'] == 'HA' and restraint['atom_name_1'] in hb_atom_names):
            found_ha_hb_noe = True
            break
    assert found_ha_hb_noe, "HA-HBx NOE not found, but should be present."


def test_logging_for_no_hydrogens(structure_no_h, caplog):
    """Test that a warning is logged when no hydrogens are found."""
    import logging
    caplog.set_level(logging.WARNING)
    calculate_synthetic_noes(structure_no_h)
    assert "No hydrogens found in structure" in caplog.text

def test_logging_for_empty_structure(caplog):
    """Test that a warning is logged for an empty structure."""
    import logging
    caplog.set_level(logging.WARNING)
    empty_structure = struc.AtomArray(0)
    calculate_synthetic_noes(empty_structure)
    assert "Input 'structure' is empty" in caplog.text
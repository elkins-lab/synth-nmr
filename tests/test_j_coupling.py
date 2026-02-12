import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.j_coupling import calculate_hn_ha_coupling

@pytest.fixture
def mock_structure(mocker):
    """Mock structure providing controlled Phi angles."""
    # 3 Residues
    # 0: N-term (NaN)
    # 1: Helix (-60)
    # 2: Sheet (-120)
    
    phi = np.array([np.nan, np.deg2rad(-60.0), np.deg2rad(-120.0)])
    psi = np.array([0.0, 0.0, 0.0]) # Irrelevant
    omega = np.array([0.0, 0.0, 0.0])
    
    mocker.patch("biotite.structure.dihedral_backbone", return_value=(phi, psi, omega))
    mocker.patch("biotite.structure.get_residue_starts", return_value=np.array([0, 1, 2]))
    
    structure = struc.AtomArray(3)
    structure.res_id = np.array([1, 2, 3])
    structure.chain_id = np.array(["A", "A", "A"])
    structure.res_name = np.array(["ALA", "ALA", "ALA"]) # Needs res_name to not crash
    
    return structure

def test_helix_coupling(mock_structure):
    """Test Alpha Helix small J-coupling (~4 Hz)."""
    couplings = calculate_hn_ha_coupling(mock_structure)
    
    # Res 2 (Index 1) is Helix
    j_helix = couplings["A"][2]
    # Expected: ~4.1 Hz
    assert 3.5 < j_helix < 5.0

def test_sheet_coupling(mock_structure):
    """Test Beta Sheet large J-coupling (~9-10 Hz)."""
    couplings = calculate_hn_ha_coupling(mock_structure)
    
    # Res 3 (Index 2) is Sheet
    j_sheet = couplings["A"][3]
    # Expected: ~9.9 Hz
    assert 9.0 < j_sheet < 11.0

def test_n_terminus_skipped(mock_structure):
    """Test that N-terminus (NaN phi) returns no coupling."""
    couplings = calculate_hn_ha_coupling(mock_structure)
    
    # Res 1 should be missing or handled
    assert 1 not in couplings["A"]

def test_structure_iteration(mocker):
    """Integrity check for real structure iteration logic."""
    # Using a real (tiny) structure just to check loop logic without mocks
    structure = struc.AtomArray(1)
    # Cannot calculate dihedrals on size 1, will error or return empty arrays depending on biotite version
    # So we stick to mocking dihedrals but ensure logic handles mismatch
    
    phi = np.array([1.0]) # 1 angle
    psi = np.array([1.0])
    omega = np.array([1.0])
    mocker.patch("biotite.structure.dihedral_backbone", return_value=(phi, psi, omega))
    mocker.patch("biotite.structure.get_residue_starts", return_value=np.array([0, 5])) # 2 residues? Mismatch
    
    # Logic in code: if len(phi) != len(res_starts): return {}
    res = calculate_hn_ha_coupling(structure)
    assert res == {}

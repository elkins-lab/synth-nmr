"""
Tests for the robustness and validation of the rdc module.
"""

import os

import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import pytest

from synth_nmr.rdc import calculate_rdcs

# Get the directory of the current test file
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def sample_structure():
    """Fixture to load a sample PDB file (Gly-Ala dipeptide)."""
    pdb_path = os.path.join(TEST_DATA_DIR, "test.pdb")
    return strucio.load_structure(pdb_path)


@pytest.fixture
def structure_no_n_atoms():
    """Fixture for a structure missing 'N' atoms."""
    atom1 = struc.Atom(
        coord=[0, 0, 0], atom_name="CA", element="C", res_id=1, res_name="GLY", chain_id="A"
    )
    atom2 = struc.Atom(
        coord=[0, 0, 1], atom_name="H", element="H", res_id=1, res_name="GLY", chain_id="A"
    )
    return struc.array([atom1, atom2])


@pytest.fixture
def structure_no_h_atoms():
    """Fixture for a structure missing 'H' atoms."""
    atom1 = struc.Atom(
        coord=[0, 0, 0], atom_name="N", element="N", res_id=1, res_name="GLY", chain_id="A"
    )
    atom2 = struc.Atom(
        coord=[0, 0, 1], atom_name="CA", element="C", res_id=1, res_name="GLY", chain_id="A"
    )
    return struc.array([atom1, atom2])


# --- Tests for calculate_rdcs ---


def test_rdcs_invalid_input_structure():
    """Test calculate_rdcs with invalid structure input types."""
    with pytest.raises(TypeError, match="Input 'structure' must be a biotite.structure.AtomArray."):
        calculate_rdcs("not_a_structure", Da=10.0, R=0.5)
    with pytest.raises(TypeError, match="Input 'structure' must be a biotite.structure.AtomArray."):
        calculate_rdcs(None, Da=10.0, R=0.5)


def test_rdcs_empty_structure():
    """Test calculate_rdcs with an empty AtomArray."""
    empty_structure = struc.AtomArray(0)
    rdcs = calculate_rdcs(empty_structure, Da=10.0, R=0.5)
    assert rdcs == {}


def test_rdcs_invalid_Da_input():
    """Test calculate_rdcs with invalid Da values."""
    dummy_structure = struc.AtomArray(1)
    dummy_structure.add_annotation("atom_name", dtype="U4")
    dummy_structure.add_annotation("element", dtype="U1")
    dummy_structure.atom_name[:] = "N"
    dummy_structure.element[:] = "N"

    # Da=0.0 should now log a warning, not raise an error
    with pytest.raises(ValueError, match="Parameter 'Da' must be a numeric value."):
        calculate_rdcs(dummy_structure, Da="not_a_float", R=0.5)
    with pytest.raises(ValueError, match="Parameter 'Da' must be a numeric value."):
        calculate_rdcs(dummy_structure, Da=None, R=0.5)


def test_rdcs_invalid_R_input():
    """Test calculate_rdcs with invalid R values."""
    dummy_structure = struc.AtomArray(1)
    dummy_structure.add_annotation("atom_name", dtype="U4")
    dummy_structure.add_annotation("element", dtype="U1")
    dummy_structure.atom_name[:] = "N"
    dummy_structure.element[:] = "N"

    with pytest.raises(ValueError, match="Parameter 'R' must be a numeric value between 0 and 2/3"):
        calculate_rdcs(dummy_structure, Da=10.0, R=-0.1)
    with pytest.raises(ValueError, match="Parameter 'R' must be a numeric value between 0 and 2/3"):
        calculate_rdcs(dummy_structure, Da=10.0, R=1.0)  # 1.0 > 2/3
    with pytest.raises(ValueError, match="Parameter 'R' must be a numeric value between 0 and 2/3"):
        calculate_rdcs(dummy_structure, Da=10.0, R="not_a_float")
    with pytest.raises(ValueError, match="Parameter 'R' must be a numeric value between 0 and 2/3"):
        calculate_rdcs(dummy_structure, Da=10.0, R=None)


def test_rdcs_structure_no_n_or_h(structure_no_n_atoms, structure_no_h_atoms):
    """Test calculate_rdcs with structures missing N or H atoms."""
    # Structure missing N atoms
    rdcs_no_n = calculate_rdcs(structure_no_n_atoms, Da=10.0, R=0.5)
    assert rdcs_no_n == {}

    # Structure missing H atoms
    rdcs_no_h = calculate_rdcs(structure_no_h_atoms, Da=10.0, R=0.5)
    assert rdcs_no_h == {}


def test_rdcs_normal_case(sample_structure):
    """Test calculate_rdcs with a valid sample structure (Gly-Ala)."""
    Da = 10.0
    R = 0.5
    rdcs = calculate_rdcs(sample_structure, Da=Da, R=R)

    assert isinstance(rdcs, dict)
    # The sample_structure is a Gly-Ala dipeptide.
    # GLY (res_id 1) should have an RDC. ALA (res_id 2) should have an RDC.
    # We don't have ideal N-H vectors for precise assertion, but can check presence.
    assert 1 in rdcs
    assert 2 in rdcs
    assert isinstance(rdcs[1], (float, np.floating))
    assert isinstance(rdcs[2], (float, np.floating))
    # Ensure values are within a reasonable range (e.g., -50 to 50 Hz for typical Da)
    assert -50.0 < rdcs[1] < 50.0
    assert -50.0 < rdcs[2] < 50.0

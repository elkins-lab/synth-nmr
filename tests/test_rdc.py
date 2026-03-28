import biotite.structure as struc
import numpy as np
import pytest

# This import will fail until the rdc module is created
from synth_nmr.rdc import calculate_rdcs


def test_calculate_rdcs_single_vector_aligned_z():
    """
    Test RDC calculation for a single N-H vector perfectly aligned
    with the Z-axis of the alignment tensor's principal axis frame.
    """
    # Create a simple structure with one residue and two atoms (N and H)
    n_atom = struc.Atom(
        coord=[0.0, 0.0, 0.0], atom_name="N", element="N", res_id=1, res_name="GLY", chain_id="A"
    )
    h_atom = struc.Atom(
        coord=[0.0, 0.0, 1.02],  # Standard N-H bond length
        atom_name="H",
        element="H",
        res_id=1,
        res_name="GLY",
        chain_id="A",
    )
    structure = struc.array([n_atom, h_atom])
    # Add a bond between N and H for completeness (not strictly required by the function)
    structure.bonds = struc.BondList(structure.array_length(), np.array([[0, 1, 1]]))

    # Define alignment tensor parameters
    # Da: Axial component of the alignment tensor in Hz
    # R: Rhombicity of the alignment tensor (unitless)
    Da = 10.0
    R = 0.5

    # --- Theoretical Calculation ---
    # The N-H vector is (0, 0, 1.02), which is perfectly aligned with the Z-axis.
    # In the Principal Axis System (PAS) of the tensor, the polar angle theta
    # (the angle with the Z-axis) is 0. The azimuthal angle phi is undefined but irrelevant.
    # The RDC formula is:
    # RDC = Da * [ (3*cos^2(theta) - 1) + 1.5 * R * sin^2(theta) * cos(2*phi) ]
    #
    # With theta = 0:
    # cos(theta) = 1
    # sin(theta) = 0
    # The rhombicity term (containing sin^2(theta)) becomes zero.
    # RDC = Da * (3 * 1^2 - 1) = Da * (3 - 1) = 2 * Da
    expected_rdc = 2 * Da  # 2 * 10.0 = 20.0

    # Calculate RDCs using the function to be implemented
    rdcs = calculate_rdcs(structure, Da=Da, R=R)

    # --- Assertions ---
    # Check that the result is a dictionary with the correct residue ID
    assert isinstance(rdcs, dict)
    assert 1 in rdcs
    # Check that the value is a float
    assert isinstance(rdcs[1], (float, np.floating))
    # Check that the calculated value matches the theoretical expectation
    assert rdcs[1] == pytest.approx(expected_rdc, abs=1e-4)


def test_calculate_rdcs_single_vector_aligned_x():
    """
    Test RDC calculation for a single N-H vector perfectly aligned
    with the X-axis of the PAS.
    """
    n_atom = struc.Atom(
        [0, 0, 0], atom_name="N", element="N", res_id=1, res_name="GLY", chain_id="A"
    )
    h_atom = struc.Atom(
        [1.02, 0, 0], atom_name="H", element="H", res_id=1, res_name="GLY", chain_id="A"
    )
    structure = struc.array([n_atom, h_atom])
    Da = 10.0
    R = 0.5

    # --- Theoretical Calculation ---
    # Vector is on the XY plane, so theta = 90 degrees. cos(theta) = 0.
    # Vector is on the X axis, so phi = 0 degrees. cos(2*phi) = 1.
    # RDC = Da * [ (3*0^2 - 1) + 1.5 * R * sin^2(90) * cos(0) ]
    # RDC = Da * [ -1 + 1.5 * R * 1 * 1 ] = Da * (-1 + 1.5 * R)
    expected_rdc = Da * (-1 + 1.5 * R)  # 10 * (-1 + 1.5*0.5) = -2.5

    rdcs = calculate_rdcs(structure, Da=Da, R=R)
    assert rdcs[1] == pytest.approx(expected_rdc, abs=1e-4)


def test_calculate_rdcs_single_vector_aligned_y():
    """
    Test RDC calculation for a single N-H vector perfectly aligned
    with the Y-axis of the PAS.
    """
    n_atom = struc.Atom(
        [0, 0, 0], atom_name="N", element="N", res_id=1, res_name="GLY", chain_id="A"
    )
    h_atom = struc.Atom(
        [0, 1.02, 0], atom_name="H", element="H", res_id=1, res_name="GLY", chain_id="A"
    )
    structure = struc.array([n_atom, h_atom])
    Da = 10.0
    R = 0.5

    # --- Theoretical Calculation ---
    # Vector is on the XY plane, so theta = 90 degrees. cos(theta) = 0.
    # Vector is on the Y axis, so phi = 90 degrees. cos(2*phi) = -1.
    # RDC = Da * [ (3*0^2 - 1) + 1.5 * R * sin^2(90) * cos(180) ]
    # RDC = Da * [ -1 + 1.5 * R * 1 * -1 ] = Da * (-1 - 1.5 * R)
    expected_rdc = Da * (-1 - 1.5 * R)  # 10 * (-1 - 1.5*0.5) = -17.5

    rdcs = calculate_rdcs(structure, Da=Da, R=R)
    assert rdcs[1] == pytest.approx(expected_rdc, abs=1e-4)


def test_calculate_rdcs_proline_and_missing_h():
    """
    Test that residues like Proline (no amide H) and residues where the
    amide H is missing are handled correctly (i.e., skipped).
    """
    # Residue 1: GLY with N and H
    atom1 = struc.Atom(
        [0, 0, 0], atom_name="N", element="N", res_id=1, res_name="GLY", chain_id="A"
    )
    atom2 = struc.Atom(
        [0, 0, 1], atom_name="H", element="H", res_id=1, res_name="GLY", chain_id="A"
    )
    # Residue 2: PRO with just N
    atom3 = struc.Atom(
        [1, 0, 0], atom_name="N", element="N", res_id=2, res_name="PRO", chain_id="A"
    )
    # Residue 3: ALA with just N (missing H)
    atom4 = struc.Atom(
        [2, 0, 0], atom_name="N", element="N", res_id=3, res_name="ALA", chain_id="A"
    )

    structure = struc.array([atom1, atom2, atom3, atom4])
    Da = 10.0
    R = 0.5

    rdcs = calculate_rdcs(structure, Da=Da, R=R)

    # RDC should be calculated for residue 1
    assert 1 in rdcs
    # RDC should NOT be calculated for Proline (res 2) or the residue missing H (res 3)
    assert 2 not in rdcs
    assert 3 not in rdcs
    # The final dictionary should contain exactly one entry
    assert len(rdcs) == 1


def test_calculate_rdcs_zero_da(caplog):
    n_atom = struc.Atom([0, 0, 0], atom_name="N", element="N", res_id=1, chain_id="A")
    h_atom = struc.Atom([0, 0, 1], atom_name="H", element="H", res_id=1, chain_id="A")
    structure = struc.array([n_atom, h_atom])
    with caplog.at_level("WARNING"):
        calculate_rdcs(structure, Da=0.0, R=0.5)
    assert "Parameter 'Da' is zero" in caplog.text


def test_calculate_rdcs_zero_length_vector(caplog):
    n_atom = struc.Atom([0, 0, 0], atom_name="N", element="N", res_id=1, chain_id="A")
    # Same coordinate as N to produce zero-length vector
    h_atom = struc.Atom([0, 0, 0], atom_name="H", element="H", res_id=1, chain_id="A")
    structure = struc.array([n_atom, h_atom])
    with caplog.at_level("WARNING"):
        rdcs = calculate_rdcs(structure, Da=10.0, R=0.5)
    assert "has a zero-length N-H vector" in caplog.text
    assert 1 not in rdcs


def test_calculate_rdcs_empty_h(caplog):
    n_atom = struc.Atom([0, 0, 0], atom_name="N", element="N", res_id=1, chain_id="A")
    h_atom = struc.Atom(
        [0, 0, 1], atom_name="H", element="H", res_id=2, chain_id="A"
    )  # Diff res_id
    structure = struc.array([n_atom, h_atom])
    with caplog.at_level("WARNING"):
        calculate_rdcs(structure, Da=10.0, R=0.5)
    assert "No RDCs were calculated" in caplog.text


def test_calculate_rdcs_exception(mocker):
    structure = struc.AtomArray(1)

    # Mocking something that causes an unexpected exception in calculate_rdcs
    mocker.patch(
        "biotite.structure.AtomArray.__getitem__", side_effect=Exception("Mock Unexpected Error")
    )

    with pytest.raises(Exception, match="Mock Unexpected Error"):
        calculate_rdcs(structure, Da=10.0, R=0.5)

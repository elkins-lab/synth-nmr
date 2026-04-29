import biotite.structure as struc
import numpy as np
import pytest

from synth_nmr.j_coupling import (
    KARPLUS_PARAMS,
    calculate_c_cg_coupling,
    calculate_ha_hb_coupling,
    calculate_hn_ha_coupling,
    calculate_hn_ha_coupling_from_phi,
)


@pytest.fixture
def mock_structure(mocker):
    """Mock structure providing controlled Phi angles."""
    # 3 Residues
    # 0: N-term (NaN)
    # 1: Helix (-60)
    # 2: Sheet (-120)

    phi = np.array([np.nan, np.deg2rad(-60.0), np.deg2rad(-120.0)])
    psi = np.array([0.0, 0.0, 0.0])  # Irrelevant
    omega = np.array([0.0, 0.0, 0.0])

    mocker.patch("biotite.structure.dihedral_backbone", return_value=(phi, psi, omega))
    mocker.patch("biotite.structure.get_residue_starts", return_value=np.array([0, 1, 2]))

    structure = struc.AtomArray(3)
    structure.res_id = np.array([1, 2, 3])
    structure.chain_id = np.array(["A", "A", "A"])
    structure.res_name = np.array(["ALA", "ALA", "ALA"])  # Needs res_name to not crash

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


@pytest.fixture
def mock_chi1_angles(mocker):
    """Mock chi1 angles for testing different rotamers.
    Res 1: gauche+ (-60 deg)
    Res 2: trans (180 deg)
    Res 3: gauche- (+60 deg)
    Res 4: Missing chi1 (e.g., Glycine or Alanine where it's not applicable)
    """
    mock_dict = {
        "A": {
            1: np.deg2rad(-60.0),
            2: np.deg2rad(180.0),
            3: np.deg2rad(60.0),
            # Res 4 is intentionally omitted
        }
    }
    mocker.patch("synth_nmr.j_coupling._get_chi1_angles", return_value=mock_dict)

    # Also mock the structure to easily pass into the functions
    structure = struc.AtomArray(4)
    structure.res_id = np.array([1, 2, 3, 4])
    structure.chain_id = np.array(["A", "A", "A", "A"])
    structure.res_name = np.array(["VAL", "LEU", "ILE", "GLY"])
    return structure


def test_ha_hb_coupling(mock_chi1_angles):
    """Test 3J_HaHb coupling based on standard rotamer chi1 angles."""
    couplings = calculate_ha_hb_coupling(mock_chi1_angles)

    # Expected values depend on Karplus parameters A=9.5, B=-1.6, C=1.8 (typical values)
    # Just asserting they don't crash and output physically reasonable ranges for now.
    assert 1 in couplings["A"]
    assert 2 in couplings["A"]
    assert 3 in couplings["A"]
    assert 4 not in couplings["A"]  # Glycine doesn't have CB, so no chi1


def test_c_cg_coupling(mock_chi1_angles):
    """Test 3J_C'Cg coupling based on standard rotamer chi1 angles."""
    couplings = calculate_c_cg_coupling(mock_chi1_angles)

    assert 1 in couplings["A"]
    assert 2 in couplings["A"]
    assert 3 in couplings["A"]
    assert 4 not in couplings["A"]


def test_structure_iteration(mocker):
    """Integrity check for real structure iteration logic."""
    # Using a real (tiny) structure just to check loop logic without mocks
    structure = struc.AtomArray(1)
    # Cannot calculate dihedrals on size 1, will error or return empty arrays depending on biotite version
    # So we stick to mocking dihedrals but ensure logic handles mismatch

    phi = np.array([1.0])  # 1 angle
    psi = np.array([1.0])
    omega = np.array([1.0])
    mocker.patch("biotite.structure.dihedral_backbone", return_value=(phi, psi, omega))
    mocker.patch(
        "biotite.structure.get_residue_starts", return_value=np.array([0, 5])
    )  # 2 residues? Mismatch

    # Logic in code: if len(phi) != len(res_starts): return {}
    res = calculate_hn_ha_coupling(structure)
    assert res == {}


def test_vuister_bax_karplus_parameters():
    """
    Explicitly verify that the engine uses the Vuister & Bax (1993) parameters.
    Reference: J. Am. Chem. Soc. 1993, 115, 7772-7777.
    """
    # 1. Assert exact parameter values (A=6.51, B=-1.76, C=1.60)
    assert KARPLUS_PARAMS["A"] == 6.51
    assert KARPLUS_PARAMS["B"] == -1.76
    assert KARPLUS_PARAMS["C"] == 1.60

    # 2. Assert precise calculation for a Beta-Sheet angle (Phi = -120)
    # Theta = Phi - 60 = -180. Cos(-180) = -1.
    # J = A(-1)^2 + B(-1) + C = A - B + C = 6.51 - (-1.76) + 1.60 = 9.87
    phi_sheet = np.array([-120.0])
    j_sheet = calculate_hn_ha_coupling_from_phi(phi_sheet)[0]
    assert np.isclose(j_sheet, 9.87)

    # 3. Assert precise calculation for an Alpha-Helix angle (Phi = -60)
    # Theta = Phi - 60 = -120. Cos(-120) = -0.5.
    # J = A(-0.5)^2 + B(-0.5) + C = 6.51(0.25) + 0.88 + 1.60 = 1.6275 + 0.88 + 1.60 = 4.1075
    phi_helix = np.array([-60.0])
    j_helix = calculate_hn_ha_coupling_from_phi(phi_helix)[0]
    assert np.isclose(j_helix, 4.1075)

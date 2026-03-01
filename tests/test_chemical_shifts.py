import pytest
import numpy as np
import biotite.structure as struc
from synth_nmr.chemical_shifts import predict_chemical_shifts, calculate_csi, RANDOM_COIL_SHIFTS

# NOTE: generate_pdb_content is not part of synth_nmr, tests using it will be adapted
# from synth_nmr.generator import generate_pdb_content
# from synth_nmr.pdb_utils import extract_atomic_content, assemble_pdb_content


@pytest.fixture
def mock_alpha_structure(mocker):
    """Mock structure that returns Alpha Helix angles when measured."""
    # Create a dummy structure
    structure = struc.AtomArray(10)
    structure.res_id = np.array([1] * 5 + [2] * 5)  # 2 residues
    structure.res_name = np.array(["ALA"] * 10)
    structure.chain_id = np.array(["A"] * 10)

    # Mock dihedral_backbone function
    # return (phi, psi, omega)
    # Alpha: -57, -47
    phi = np.array([np.nan, -60.0, -60.0, -60.0, -60.0])  # 5 residues/angles
    psi = np.array([-50.0, -50.0, -50.0, -50.0, np.nan])
    omega = np.array([180.0] * 5)

    mocker.patch(
        "biotite.structure.dihedral_backbone",
        return_value=(np.radians(phi), np.radians(psi), np.radians(omega)),
    )
    mocker.patch("biotite.structure.get_residue_starts", return_value=np.array([0, 1]))  # Indices

    return structure


def test_helix_trends(mocker):
    """Test using mocked angles."""
    # Setup Mock
    phi = np.array([np.nan, -60.0, -60.0, -60.0, -60.0])
    psi = np.array([-50.0, -50.0, -50.0, -50.0, np.nan])
    omega = np.array([180.0] * 5)

    mocker.patch(
        "biotite.structure.dihedral_backbone",
        return_value=(np.radians(phi), np.radians(psi), np.radians(omega)),
    )

    # Needs a real-ish structure to iterate over
    # Create a minimal one: 5 Residues of ALA
    # We need atom arrays to iterate
    # Create dummy structure
    structure = struc.AtomArray(1)  # size doesn't matter if we mock get_residue_starts

    # Mock resid iteration
    # 5 residues
    mocker.patch("biotite.structure.get_residue_starts", return_value=np.array([0, 1, 2, 3, 4]))

    # We need structure slices to have .res_name, .chain_id, .res_id
    # We can mock the __getitem__ or just make structure valid-ish
    # Easier: make real structure of length 5 res
    structure = struc.AtomArray(5)  # 1 atom per res
    structure.res_name = np.array(["ALA"] * 5)
    structure.chain_id = np.array(["A"] * 5)
    structure.res_id = np.array([1, 2, 3, 4, 5])

    # Run
    shifts = predict_chemical_shifts(structure)

    # Residue 2 (Index 2 in Py, ID 3) should have Alpha angles (-60, -50)
    # Index 2: Phi[2]=-60, Psi[2]=-50. Matches Alpha criteria.
    res3 = shifts["A"][3]
    rc = RANDOM_COIL_SHIFTS["ALA"]

    # Should contain secondary structure offset
    # CA offset +3.1
    # 52.5 + 3.1 = 55.6
    assert res3["CA"] > rc["CA"] + 1.0


def test_sheet_trends(mocker):
    """Test using mocked angles for Beta Sheet."""
    # Beta: -120, 120
    phi = np.array([np.nan, -130.0, -130.0, -130.0, -130.0])
    psi = np.array([130.0, 130.0, 130.0, 130.0, np.nan])
    omega = np.array([180.0] * 5)

    mocker.patch(
        "biotite.structure.dihedral_backbone",
        return_value=(np.radians(phi), np.radians(psi), np.radians(omega)),
    )
    mocker.patch("biotite.structure.get_residue_starts", return_value=np.array([0, 1, 2, 3, 4]))

    structure = struc.AtomArray(5)
    structure.res_name = np.array(["VAL"] * 5)
    structure.chain_id = np.array(["A"] * 5)
    structure.res_id = np.array([1, 2, 3, 4, 5])

    shifts = predict_chemical_shifts(structure)

    # Residue 3
    res3 = shifts["A"][3]
    rc = RANDOM_COIL_SHIFTS["VAL"]

    # CA offset -1.5
    assert res3["CA"] < rc["CA"] - 0.5


def test_structure_returns_dict(mock_alpha_structure):
    """Test that function returns dictionary with expected keys."""
    shifts = predict_chemical_shifts(mock_alpha_structure)
    assert isinstance(shifts, dict)
    assert "A" in shifts  # Chain A
    assert 1 in shifts["A"]  # Residue 1

    # Check atoms exist
    res1 = shifts["A"][1]
    for atom in ["N", "CA", "CB"]:  # ALA has CB
        assert atom in res1


def test_glycine_shifts():
    """Test Glycine has no CB."""
    # Create a structure with Glycine
    structure = struc.AtomArray(3)
    structure.res_name = np.array(["GLY"] * 3)
    structure.res_id = np.array([1, 2, 3])
    structure.chain_id = np.array(["A"] * 3)
    structure.atom_name = np.array(["CA", "CA", "CA"])

    shifts = predict_chemical_shifts(structure)
    res2_gly = shifts["A"][2]

    assert "CA" in res2_gly
    assert "CB" not in res2_gly
    assert "HA" in res2_gly


def test_proline_shifts():
    """Test Proline has no Amide N/H."""
    structure = struc.AtomArray(3)
    structure.res_name = np.array(["ALA", "PRO", "ALA"])
    structure.res_id = np.array([1, 2, 3])
    structure.chain_id = np.array(["A"] * 3)
    structure.atom_name = np.array(["CA", "CA", "CA"])

    shifts = predict_chemical_shifts(structure)
    res2_pro = shifts["A"][2]

    assert "N" not in res2_pro
    assert "H" not in res2_pro
    assert "CA" in res2_pro


def test_csi_calculation():
    """Test that CSI correctly subtracts Random Coil values."""
    # 1. Setup Mock Structure (needed for ResName lookup)
    structure = struc.AtomArray(1)
    structure.res_id = np.array([10])
    structure.res_name = np.array(["ALA"])
    structure.atom_name = np.array(["CA"])
    structure.chain_id = np.array(["A"])

    # 2. Setup Shifts (Simulate a Helix)
    # ALA Random Coil CA = 52.5
    # Helix Offset = +3.1 -> Predicted = 55.6
    shifts = {"A": {10: {"CA": 56.5}}}  # +4.0 deviation

    # 3. Calculate
    csi = calculate_csi(shifts, structure)

    # 4. Verify
    # Delta = 56.5 - 52.5 = 4.0
    assert "A" in csi
    assert 10 in csi["A"]
    delta = csi["A"][10]
    assert pytest.approx(delta, 0.1) == 4.0


def test_ring_current_shift_simple():
    """Test the ring current calculation math (Manual test)."""
    from synth_nmr.chemical_shifts import _calculate_ring_current_shift

    # Ring at origin, normal in Z
    # cx, cy, cz, nx, ny, nz, intensity
    ring = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])

    # Proton at (0, 0, 2) -> distance 2.0 along normal
    # theta = 0, cos(0) = 1
    # B-factor for 3D is 1/(r^3) * (1 - 3*cos^2(theta)) roughly
    # Actually the implementation uses a specific formula.
    # delta = Intensity * (1 - 3*cos^2(theta)) / (r^3) * C_scale

    proton_coord = np.array([0.0, 0.0, 2.0])
    shift = _calculate_ring_current_shift(proton_coord, ring)

    # At (0,0,2), theta = 0, cos^2 = 1. (1 - 3*1) = -2.
    # Delta should be negative (upfield shift).
    assert shift < 0

    # Proton at (2, 0, 0) -> distance 2.0 in the plane
    # theta = 90, cos(90) = 0. (1 - 3*0) = 1.
    # Delta should be positive (downfield shift).
    proton_coord_plane = np.array([2.0, 0.0, 0.0])
    shift_plane = _calculate_ring_current_shift(proton_coord_plane, ring)
    assert shift_plane > 0


def test_ring_current_singularity():
    """Test the r < 1.0 singularity check."""
    from synth_nmr.chemical_shifts import _calculate_ring_current_shift

    ring = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
    # Proton very close (e.g. 0.5A)
    proton_coord = np.array([0.0, 0.0, 0.5])
    shift = _calculate_ring_current_shift(proton_coord, ring)
    # Should be skipped (return 0.0 for this ring)
    assert shift == 0.0


def test_aromatic_ring_identification():
    """Test that PHE aromatic rings are correctly identified."""
    from synth_nmr.chemical_shifts import _get_aromatic_rings

    # Create a PHE residue
    structure = struc.AtomArray(7)
    structure.res_name = np.array(["ALA"] + ["PHE"] * 6)
    structure.res_id = np.array([1] + [2] * 6)
    structure.chain_id = np.array(["A"] * 7)
    structure.atom_name = np.array(["CA", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"])
    structure.coord = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 1, 0], [2, -1, 0], [3, 1, 0], [3, -1, 0], [4, 0, 0]]
    )

    rings = _get_aromatic_rings(structure)

    # One ring for PHE
    assert len(rings) == 1
    # cx, cy, cz, nx, ny, nz, intensity
    assert rings.shape == (1, 7)
    assert rings[0, 6] == 1.2  # PHE intensity


def test_aromatic_ring_identification_his():
    """Test that HIS aromatic rings are correctly identified."""
    from synth_nmr.chemical_shifts import _get_aromatic_rings

    structure = struc.AtomArray(6)
    structure.res_name = np.array(["ALA"] + ["HIS"] * 5)
    structure.res_id = np.array([1] + [2] * 5)
    structure.chain_id = np.array(["A"] * 6)
    structure.atom_name = np.array(["CA", "CG", "ND1", "CD2", "CE1", "NE2"])
    structure.coord = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [2, -1, 0], [3, 1, 0], [3, -1, 0]])
    rings = _get_aromatic_rings(structure)
    assert len(rings) == 1
    assert rings[0, 6] == 0.5  # HIS intensity


def test_integration_ring_current_shifts():
    """Test that aromatic residues affect nearby shifts."""
    # Create a structure where a proton is close to a ring
    structure = struc.AtomArray(7)
    structure.res_name = np.array(["ALA"] + ["PHE"] * 6)
    structure.res_id = np.array([1] + [2] * 6)
    structure.chain_id = np.array(["A"] * 7)
    structure.atom_name = np.array(["HA", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"])
    structure.coord[0] = np.array([0.0, 0.0, 2.0])  # ALA HA

    # Predict with rings
    shifts = predict_chemical_shifts(structure)

    assert 1 in shifts["A"]
    assert "HA" in shifts["A"][1]


def test_predict_unknown_residue():
    """Test that unknown residues are skipped gracefully."""
    structure = struc.AtomArray(1)
    structure.res_name = np.array(["XXX"])
    structure.res_id = np.array([1])
    structure.chain_id = np.array(["A"])
    structure.atom_name = np.array(["CA"])
    structure.coord = np.array([[0, 0, 0]])

    shifts = predict_chemical_shifts(structure)
    assert shifts == {} or "A" not in shifts


def test_predict_missing_atom():
    """Test handling of missing atoms (e.g. HA missing)."""
    # Generate AL-PHE to have a ring, then remove HA from ALA
    structure = struc.AtomArray(6)
    structure.res_name = np.array(["PHE"] * 6)
    structure.res_id = np.array([2] * 6)
    structure.chain_id = np.array(["A"] * 6)
    structure.atom_name = np.array(["CG", "CD1", "CD2", "CE1", "CE2", "CZ"])

    shifts = predict_chemical_shifts(structure)
    # Even if missing, it currently returns theoretical base shift
    # but the goal is to trigger the IndexError branch in ring current logic
    assert 1 not in shifts["A"]


def test_get_secondary_structure_coil():
    """Test getting coil for unknown angles."""
    structure = struc.AtomArray(5)  # One residue
    structure.res_name = np.array(["ALA"] * 5)
    structure.atom_name = np.array(["N", "CA", "C", "CA", "C"])  # Dummy
    # We need a way to make Biotite calculate specific angles or just mock it
    # Easier to mock structure.get_residue_starts and the dihedrals if we really want to test the logic
    # But get_secondary_structure is used in predict_chemical_shifts.


def test_ppii_shifts(mocker):
    """Test PPII secondary structure shifts."""
    # PPII: -100 < phi < -30 AND 100 < psi < 180
    # Beta: -160 < phi < -40 AND 90 < psi < 180
    # Use phi = -35 to be in PPII but NOT Beta
    phi = np.array([np.nan, -35.0, -35.0, -35.0, -35.0])
    psi = np.array([145.0, 145.0, 145.0, 145.0, np.nan])
    omega = np.array([180.0] * 5)

    mocker.patch(
        "biotite.structure.dihedral_backbone",
        return_value=(np.radians(phi), np.radians(psi), np.radians(omega)),
    )
    mocker.patch("biotite.structure.get_residue_starts", return_value=np.array([0, 1, 2, 3, 4]))

    structure = struc.AtomArray(5)
    structure.res_name = np.array(["PRO"] * 5)
    structure.chain_id = np.array(["A"] * 5)
    structure.res_id = np.array([1, 2, 3, 4, 5])

    shifts = predict_chemical_shifts(structure)
    assert 3 in shifts["A"]


def test_csi_calculation_missing_res_id():
    """Test calculate_csi skips residue IDs not in structure."""
    structure = struc.AtomArray(1)
    structure.res_id = np.array([1])
    structure.res_name = np.array(["ALA"])
    structure.atom_name = np.array(["CA"])
    structure.chain_id = np.array(["A"])

    # Shifts for a non-existent residue ID 99
    shifts = {"A": {99: {"CA": 55.0}}}

    csi = calculate_csi(shifts, structure)
    assert csi["A"] == {}


def test_aromatic_ring_identification_trp():
    """Test that TRP aromatic rings are correctly identified."""
    from synth_nmr.chemical_shifts import _get_aromatic_rings

    structure = struc.AtomArray(10)
    structure.res_name = np.array(["ALA"] + ["TRP"] * 9)
    structure.res_id = np.array([1] + [2] * 9)
    structure.chain_id = np.array(["A"] * 10)
    structure.atom_name = np.array(
        ["CA", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]
    )
    structure.coord = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 1, 0],
            [2, -1, 0],
            [3, 1, 0],
            [3, -1, 0],
            [4, 0, 0],
            [5, 1, 0],
            [5, -1, 0],
            [6, 0, 0],
        ]
    )
    rings = _get_aromatic_rings(structure)
    assert len(rings) == 1
    assert rings[0, 6] == 1.3  # TRP intensity


def test_aromatic_ring_identification_tyr():
    """Test that TYR aromatic rings are correctly identified."""
    from synth_nmr.chemical_shifts import _get_aromatic_rings

    structure = struc.AtomArray(7)
    structure.res_name = np.array(["ALA"] + ["TYR"] * 6)
    structure.res_id = np.array([1] + [2] * 6)
    structure.chain_id = np.array(["A"] * 7)
    structure.atom_name = np.array(["CA", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"])
    structure.coord = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 1, 0], [2, -1, 0], [3, 1, 0], [3, -1, 0], [4, 0, 0]]
    )
    rings = _get_aromatic_rings(structure)
    assert len(rings) == 1
    assert rings[0, 6] == 1.2  # TYR intensity


def test_numba_fallback(mocker):
    """Test that the njit decorator falls back to a regular function when numba is not installed."""
    mocker.patch.dict("sys.modules", {"numba": None})
    import importlib
    import synth_nmr.chemical_shifts

    importlib.reload(synth_nmr.chemical_shifts)
    from synth_nmr.chemical_shifts import _calculate_ring_current_shift

    ring = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
    proton_coord = np.array([0.0, 0.0, 2.0])
    shift = _calculate_ring_current_shift(proton_coord, ring)
    assert shift < 0
    importlib.reload(synth_nmr.chemical_shifts)


def test_get_aromatic_rings_no_rings():
    """Test _get_aromatic_rings with a structure that has no aromatic residues."""
    from synth_nmr.chemical_shifts import _get_aromatic_rings

    structure = struc.AtomArray(5)
    structure.res_name = np.array(["ALA"] * 5)
    rings = _get_aromatic_rings(structure)
    assert rings.shape == (0, 7)


def test_numba_fallback_with_args(mocker):
    """Test the njit decorator fallback when called with arguments."""
    mocker.patch.dict("sys.modules", {"numba": None})
    import importlib
    import synth_nmr.chemical_shifts

    importlib.reload(synth_nmr.chemical_shifts)
    from synth_nmr.chemical_shifts import njit

    @njit(fastmath=True)
    def my_func(x):
        return x + 1

    assert my_func(1) == 2
    importlib.reload(synth_nmr.chemical_shifts)


def test_predict_chemical_shifts_index_error():
    """Test predict_chemical_shifts with a missing atom to trigger IndexError."""
    from synth_nmr.chemical_shifts import predict_chemical_shifts

    # Structure with a PHE ring but no HA on the first residue
    structure = struc.AtomArray(7)
    structure.res_name = np.array(["ALA"] + ["PHE"] * 6)
    structure.res_id = np.array([1] + [2] * 6)
    structure.chain_id = np.array(["A"] * 7)
    structure.atom_name = np.array(["CA", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"])
    shifts = predict_chemical_shifts(structure)

    # The 'HA' key will be present, but its value should be close to the random coil value
    # because the ring current shift was not applied.
    assert "HA" in shifts["A"][1]
    rc_ha = RANDOM_COIL_SHIFTS["ALA"]["HA"]
    assert pytest.approx(shifts["A"][1]["HA"], 0.3) == rc_ha


def test_get_aromatic_rings_else_continue():
    """Test _get_aromatic_rings with a non-aromatic residue to cover else: continue."""
    from synth_nmr.chemical_shifts import _get_aromatic_rings

    structure = struc.AtomArray(5)
    structure.res_name = np.array(["GLY"] * 5)
    rings = _get_aromatic_rings(structure)
    assert rings.shape == (0, 7)


def test_predict_chemical_shifts_fallback(mocker):
    """Test that predict_chemical_shifts falls back to empirical if NeuralShiftPredictor fails."""
    import synth_nmr.chemical_shifts
    from synth_nmr.chemical_shifts import predict_chemical_shifts

    # Spy on the empirical fallback function
    spy_empirical = mocker.spy(synth_nmr.chemical_shifts, "predict_empirical_shifts")

    # Force the neural predictor to fail during prediction
    mocker.patch(
        "synth_nmr.neural_shifts.NeuralShiftPredictor.predict",
        side_effect=RuntimeError("Mocked missing torch"),
    )

    structure = struc.AtomArray(1)
    structure.res_name = np.array(["ALA"])
    structure.res_id = np.array([1])
    structure.chain_id = np.array(["A"])
    structure.atom_name = np.array(["CA"])
    structure.coord = np.array([[0, 0, 0]])

    shifts = predict_chemical_shifts(structure)

    assert "A" in shifts
    assert 1 in shifts["A"]
    spy_empirical.assert_called_once()


def test_shiftx2_is_available_mocked(mocker):
    """Test ShiftX2 `is_available` true and false branches."""
    from synth_nmr.chemical_shifts import ShiftX2Predictor
    predictor = ShiftX2Predictor()

    mocker.patch("shutil.which", return_value="/fake/path/to/shiftx2")
    assert predictor.is_available() is True

    mocker.patch("shutil.which", return_value=None)
    assert predictor.is_available() is False


def test_shiftx2_predict_not_available(mocker):
    """Test ShiftX2 predict raises error when executable is missing."""
    from synth_nmr.chemical_shifts import ShiftX2Predictor
    predictor = ShiftX2Predictor()
    mocker.patch.object(predictor, "is_available", return_value=False)

    structure = struc.AtomArray(1)
    with pytest.raises(RuntimeError, match="ShiftX2 executable.*not found"):
        predictor.predict(structure)


def test_shiftx2_predict_subprocess_error(mocker, tmp_path):
    """Test ShiftX2 predict raises error when subprocess fails."""
    import subprocess
    from synth_nmr.chemical_shifts import ShiftX2Predictor
    
    predictor = ShiftX2Predictor()
    mocker.patch.object(predictor, "is_available", return_value=True)
    mocker.patch('tempfile.TemporaryDirectory', return_value=tmp_path)
    
    # Fully mock PDBFile to bypass all biotite.structure.io.pdb logic
    mock_pdb_file = mocker.patch("synth_nmr.chemical_shifts.pdb.PDBFile")
    mock_instance = mock_pdb_file.return_value
    mock_instance.set_structure.return_value = None
    mock_instance.write.return_value = None

    mocker.patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "shiftx2", stderr="Fake Error")
    )

    structure = struc.AtomArray(1)  # Invalid structure is fine now because PDBFile is mocked
    with pytest.raises(RuntimeError, match="ShiftX2 execution failed: Fake Error"):
        predictor.predict(structure)


def test_shiftx2_parse_output_missing_file():
    """Test _parse_output with non-existent file."""
    from synth_nmr.chemical_shifts import ShiftX2Predictor
    predictor = ShiftX2Predictor()
    
    with pytest.raises(FileNotFoundError, match="ShiftX2 output file not found: missing.file"):
        predictor._parse_output("missing.file")


def test_shiftx2_parse_output_bad_format(tmp_path):
    """Test _parse_output gracefully ignores poorly formatted lines."""
    from synth_nmr.chemical_shifts import ShiftX2Predictor
    predictor = ShiftX2Predictor()
    
    fake_csv = tmp_path / "fake_output.cs"
    fake_csv.write_text("NUM, RES, ATOMNAME, SHIFT\n1, ALA, CA, NOT_A_FLOAT\n2, GLY, N")
    
    # Should catch ValueError and `continue`, resulting in empty dict
    shifts = predictor._parse_output(str(fake_csv))
    assert shifts == {"A": {}}

import pytest
import numpy as np
import biotite.structure as struc
from synth_nmr.chemical_shifts import predict_empirical_shifts, calculate_csi

def test_predict_empirical_shifts_wrong_type():
    with pytest.raises(TypeError, match="must be a biotite.structure.AtomArray"):
        predict_empirical_shifts("not an array")

def test_predict_empirical_shifts_empty_array():
    empty_arr = struc.AtomArray(0)
    assert predict_empirical_shifts(empty_arr) == {}

def test_calculate_csi_wrong_shifts_type():
    structure = struc.AtomArray(1)
    with pytest.raises(TypeError, match="Input 'shifts' must be a dictionary."):
        calculate_csi("not a dict", structure)

def test_calculate_csi_empty_shifts():
    structure = struc.AtomArray(1)
    assert calculate_csi({}, structure) == {}

def test_calculate_csi_wrong_structure_type():
    with pytest.raises(TypeError, match="must be a biotite.structure.AtomArray"):
        calculate_csi({"A": {1: {"CA": 50.0}}}, "not an array")

def test_calculate_csi_empty_structure():
    empty_arr = struc.AtomArray(0)
    assert calculate_csi({"A": {1: {"CA": 50.0}}}, empty_arr) == {}
    
def test_calculate_csi_chain_not_dict():
    structure = struc.AtomArray(1)
    structure.res_id = np.array([1])
    structure.res_name = np.array(["ALA"])
    
    # "A" maps to a string instead of dict
    shifts = {"A": "not a dict"}
    assert calculate_csi(shifts, structure) == {}

def test_calculate_csi_atom_not_dict():
    structure = struc.AtomArray(1)
    structure.res_id = np.array([1])
    structure.res_name = np.array(["ALA"])
    
    # Residue 1 maps to string instead of dict
    shifts = {"A": {1: "not a dict"}}
    assert calculate_csi(shifts, structure) == {"A": {}}

def test_calculate_csi_exception_raising(mocker):
    structure = struc.AtomArray(1)
    structure.res_id = np.array([1])
    structure.res_name = np.array(["ALA"])
    shifts = {"A": {1: {"CA": 50.0}}}
    
    mocker.patch("biotite.structure.get_residue_starts", side_effect=Exception("Mock CSI Error"))
    with pytest.raises(Exception, match="Mock CSI Error"):
        calculate_csi(shifts, structure)

def test_predict_empirical_shifts_exception_raising(mocker):
    structure = struc.AtomArray(1)
    structure.res_id = np.array([1])
    structure.res_name = np.array(["ALA"])
    
    mocker.patch("biotite.structure.get_residue_starts", side_effect=Exception("Mock Prediction Error"))
    with pytest.raises(Exception, match="Mock Prediction Error"):
        predict_empirical_shifts(structure)

def test_predict_chemical_shifts_shiftx2_empty(mocker):
    from synth_nmr.chemical_shifts import predict_chemical_shifts
    
    mocker.patch("synth_nmr.chemical_shifts.ShiftX2Predictor.is_available", return_value=True)
    mocker.patch("synth_nmr.chemical_shifts.ShiftX2Predictor.predict", return_value={})
    
    spy_empirical = mocker.patch("synth_nmr.chemical_shifts.predict_empirical_shifts", return_value={"mock": "fallback"})
    
    structure = struc.AtomArray(1)
    structure.res_name = np.array(["ALA"])
    
    res = predict_chemical_shifts(structure)
    assert res == {"mock": "fallback"}
    spy_empirical.assert_called_once()

def test_predict_chemical_shifts_shiftx2_exception(mocker):
    from synth_nmr.chemical_shifts import predict_chemical_shifts
    
    mocker.patch("synth_nmr.chemical_shifts.ShiftX2Predictor.is_available", return_value=True)
    mocker.patch("synth_nmr.chemical_shifts.ShiftX2Predictor.predict", side_effect=Exception("Crash"))
    
    spy_empirical = mocker.patch("synth_nmr.chemical_shifts.predict_empirical_shifts", return_value={"mock": "fallback_crash"})
    
    structure = struc.AtomArray(1)
    structure.res_name = np.array(["ALA"])
    
    res = predict_chemical_shifts(structure)
    assert res == {"mock": "fallback_crash"}
    spy_empirical.assert_called_once()
    
def test_predict_chemical_shifts_shiftx2_success(mocker):
    from synth_nmr.chemical_shifts import predict_chemical_shifts
    structure = struc.AtomArray(1)
    structure.res_name = np.array(["ALA"])
    structure.res_id = np.array([1])
    
    mocker.patch("synth_nmr.chemical_shifts.ShiftX2Predictor.is_available", return_value=True)
    mocker.patch("synth_nmr.chemical_shifts.ShiftX2Predictor.predict", return_value={"A": {1: {"CA": 50.0}}})
    
    shifts = predict_chemical_shifts(structure)
    assert shifts == {"A": {1: {"CA": 50.0}}}

def test_calculate_csi_empty_structure(mocker):
    from synth_nmr.chemical_shifts import calculate_csi
    structure = struc.AtomArray(1)  # Length 1 is truthy!
    shifts = {"A": {1: {"CA": 55.0}}}
    mocker.patch("biotite.structure.get_residue_starts", return_value=np.array([]))
    csi = calculate_csi(shifts, structure)
    assert csi == {}

def test_calculate_csi_missing_random_coil(mocker):
    from synth_nmr.chemical_shifts import calculate_csi, RANDOM_COIL_SHIFTS
    structure = struc.AtomArray(1)
    structure.res_name = np.array(["ALA"])
    structure.res_id = np.array([1])
    structure.chain_id = np.array(["A"])
    shifts = {"A": {1: {"CA": 55.0}}}
    
    # Mock RANDOM_COIL_SHIFTS so "ALA" has NO "CA" shift, but is in the dict.
    mocker.patch.dict(RANDOM_COIL_SHIFTS, {"ALA": {"CB": 19.3}})
    csi = calculate_csi(shifts, structure)
    assert csi == {"A": {}}

def test_get_aromatic_rings_unsupported(mocker):
    from synth_nmr.chemical_shifts import _get_aromatic_rings, RING_INTENSITIES
    structure = struc.AtomArray(1)
    structure.res_name = np.array(["XYZ"])
    structure.res_id = np.array([1])
    structure.atom_name = np.array(["CA"])
    
    # Temporarily add XYZ to RING_INTENSITIES but skip specific structural handling
    mocker.patch.dict(RING_INTENSITIES, {"XYZ": 1.0})
    rings = _get_aromatic_rings(structure)
    assert len(rings) == 0

def test_shiftx2_predictor_predict_success(mocker):
    from synth_nmr.chemical_shifts import ShiftX2Predictor
    
    structure = struc.AtomArray(1)
    structure.coord = np.array([[0,0,0]])
    predictor = ShiftX2Predictor()
    
    # Mock subprocess run to simulate successful SHIFTX2 call
    mocker.patch("synth_nmr.chemical_shifts.ShiftX2Predictor.is_available", return_value=True)
    mocker.patch("biotite.structure.io.pdb.PDBFile.set_structure")
    mocker.patch("subprocess.run")
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("synth_nmr.chemical_shifts.ShiftX2Predictor._parse_output", return_value={"A": {1: {"CA": 50.0}}})
    mocker.patch("os.listdir", return_value=["test.cs"])
    
    shifts = predictor.predict(structure)
    assert shifts == {"A": {1: {"CA": 50.0}}}

def test_shiftx2_parse_output(tmp_path):
    from synth_nmr.chemical_shifts import ShiftX2Predictor
    csv_file = tmp_path / "output.csv"
    csv_file.write_text(
        "Random gibberish before header\n"
        "\n"
        "NUM, RES, ATOMNAME, SHIFT\n"
        "1, ALA, CA, 50.5\n"
        "1, ALA, CB, error\n"
        "2, GLY, CA, 45.0\n"
    )
    predictor = ShiftX2Predictor()
    shifts = predictor._parse_output(str(csv_file))
    assert shifts["A"][1]["CA"] == 50.5
    assert "CB" not in shifts["A"][1]
    assert shifts["A"][2]["CA"] == 45.0

from unittest.mock import MagicMock, patch

import pytest

from synth_nmr.data_pipeline import (
    download_bmrb_file,
    download_pdb_file,
    ensure_data_dir_exists,
    load_matched_dataset,
    parse_bmrb_j_couplings,
    parse_bmrb_shifts,
)


def test_ensure_data_dir_exists(tmp_path):
    test_dir = tmp_path / "test_data_dir"
    ensure_data_dir_exists(str(test_dir))
    assert test_dir.exists()
    assert test_dir.is_dir()
    # Test when dir already exists
    ensure_data_dir_exists(str(test_dir))
    assert test_dir.exists()


@patch("urllib.request.urlretrieve")
def test_download_bmrb_file(mock_urlretrieve, tmp_path):
    test_dir = tmp_path / "data"
    test_dir.mkdir()

    # Not existing -> downloads
    res = download_bmrb_file(17769, str(test_dir))
    assert res == str(test_dir / "bmr17769.str")
    mock_urlretrieve.assert_called_once()

    # Create the file so it exists
    with open(res, "w") as f:
        f.write("dummy")

    # Exists -> doesn't download
    mock_urlretrieve.reset_mock()
    res2 = download_bmrb_file(17769, str(test_dir))
    assert res2 == res
    mock_urlretrieve.assert_not_called()


@patch("urllib.request.urlretrieve", side_effect=Exception("Network error"))
def test_download_bmrb_file_error(mock_urlretrieve, tmp_path):
    res = download_bmrb_file(99999, str(tmp_path))
    assert res is None


@patch("urllib.request.urlretrieve")
def test_download_pdb_file(mock_urlretrieve, tmp_path):
    test_dir = tmp_path / "data"
    test_dir.mkdir()

    # Not existing -> downloads
    res = download_pdb_file("1D3Z", str(test_dir))
    assert res == str(test_dir / "1D3Z.pdb")
    mock_urlretrieve.assert_called_once()

    # Create the file so it exists
    with open(res, "w") as f:
        f.write("dummy")

    # Exists -> doesn't download
    mock_urlretrieve.reset_mock()
    res2 = download_pdb_file("1D3Z", str(test_dir))
    assert res2 == res
    mock_urlretrieve.assert_not_called()


@patch("urllib.request.urlretrieve", side_effect=Exception("Network error"))
def test_download_pdb_file_error(mock_urlretrieve, tmp_path):
    res = download_pdb_file("ERROR", str(tmp_path))
    assert res is None


def test_parse_bmrb_shifts(tmp_path):
    bmrb_content = """
data_17769
save_shift_list
   loop_
      _Atom_chem_shift.ID
      _Atom_chem_shift.Seq_ID
      _Atom_chem_shift.Comp_index_ID
      _Atom_chem_shift.Atom_ID
      _Atom_chem_shift.Atom_type
      _Atom_chem_shift.Val
      1    1    MET    H     H      8.2
      2    1    MET    CA    C      55.3
      3    1    MET    HA2   H      4.2
      4    1    MET    HA3   H      4.4
      5    2    GLN    N     N      120.1
   stop_
save_
"""
    bmrb_file = tmp_path / "test.str"
    bmrb_file.write_text(bmrb_content)

    shifts = parse_bmrb_shifts(str(bmrb_file))

    assert 1 in shifts
    assert 2 in shifts

    assert shifts[1]["H"] == 8.2
    assert shifts[1]["CA"] == 55.3
    # HA2 and HA3 should average to (4.2+4.4)/2 = 4.3
    assert shifts[1]["HA"] == pytest.approx(4.3)

    assert shifts[2]["N"] == 120.1


def test_parse_bmrb_shifts_io_error():
    shifts = parse_bmrb_shifts("non_existent_file.str")
    assert shifts == {}


@patch("synth_nmr.data_pipeline.parse_bmrb_shifts")
@patch("biotite.structure.io.pdb.PDBFile")
@patch("synth_nmr.data_pipeline.download_pdb_file")
@patch("synth_nmr.data_pipeline.download_bmrb_file")
@patch("synth_nmr.data_pipeline.TRAINING_PAIRS", [("TEST_PDB", 12345)])
def test_load_matched_dataset(
    mock_download_bmrb, mock_download_pdb, mock_pdb_file, mock_parse_bmrb, tmp_path
):
    mock_download_pdb.return_value = "dummy.pdb"
    mock_download_bmrb.return_value = "dummy.str"

    mock_struct = MagicMock()
    mock_struct.__len__.return_value = 10
    mock_struct.__getitem__.return_value = mock_struct

    mock_pdb_instance = MagicMock()
    mock_pdb_instance.get_structure.return_value = mock_struct
    mock_pdb_file.read.return_value = mock_pdb_instance

    mock_parse_bmrb.return_value = {1: {"CA": 50.0}}

    with patch("biotite.structure.filter_amino_acids", return_value=[True] * 10):
        dataset = load_matched_dataset(str(tmp_path))

    assert len(dataset) == 1
    assert dataset[0][1] == {1: {"CA": 50.0}}


@patch("synth_nmr.data_pipeline.download_pdb_file")
@patch("synth_nmr.data_pipeline.download_bmrb_file")
@patch("synth_nmr.data_pipeline.TRAINING_PAIRS", [("TEST_PDB", 12345)])
def test_load_matched_dataset_download_fail(mock_download_bmrb, mock_download_pdb, tmp_path):
    mock_download_pdb.return_value = None
    mock_download_bmrb.return_value = None

    dataset = load_matched_dataset(str(tmp_path))
    assert len(dataset) == 0


@patch("synth_nmr.data_pipeline.download_pdb_file")
@patch("synth_nmr.data_pipeline.download_bmrb_file")
@patch("synth_nmr.data_pipeline.TRAINING_PAIRS", [("TEST_PDB", 12345)])
def test_load_matched_dataset_parse_fail(mock_download_bmrb, mock_download_pdb, tmp_path):
    mock_download_pdb.return_value = "dummy.pdb"
    mock_download_bmrb.return_value = "dummy.str"

    # exception on pdB read
    with patch("biotite.structure.io.pdb.PDBFile.read", side_effect=Exception("Read Error")):
        dataset = load_matched_dataset(str(tmp_path))

    assert len(dataset) == 0


def test_parse_bmrb_shifts_invalid_data(tmp_path):
    from synth_nmr.data_pipeline import parse_bmrb_shifts

    csv_path = tmp_path / "test.csv"

    # Create a mock CSV that has the columns but bad data that triggers the ValueError exception block inside the data parser
    csv_path.write_text(
        """loop_
_Atom_chem_shift.ID
_Atom_chem_shift.Comp_index_ID
_Atom_chem_shift.Seq_ID
_Atom_chem_shift.Atom_ID
_Atom_chem_shift.Atom_type
_Atom_chem_shift.Val
_Atom_chem_shift.Value_err
1 ALA 1 H H INVALID_FLOAT 0.0
2 ALA 1 CA C 50.0 0.0
3 ALA 1 CB C 15.0 0.0
4 ALA 1 C C 170.0 0.0
stop_
"""
    )

    shifts = parse_bmrb_shifts(str(csv_path))

    # Verify the shifts map successfully grabbed the CA, CB, and C and skipped the invalid H
    assert 1 in shifts
    assert "CA" in shifts[1]
    assert "CB" in shifts[1]
    assert "C" in shifts[1]
    assert "H" not in shifts[1]


def test_parse_bmrb_j_couplings(tmp_path):
    bmrb_content = """
data_17769
save_coupling_list
   loop_
      _Coupling_constant.ID
      _Coupling_constant.Seq_ID_1
      _Coupling_constant.Comp_index_ID_1
      _Coupling_constant.Atom_ID_1
      _Coupling_constant.Seq_ID_2
      _Coupling_constant.Comp_index_ID_2
      _Coupling_constant.Atom_ID_2
      _Coupling_constant.Code
      _Coupling_constant.Val
      1    1    MET    H     1    MET    HA     3JHNHA    7.5
      2    2    GLN    CA    2    GLN    CB     3JHAHB    4.2
   stop_
save_
"""
    bmrb_file = tmp_path / "test_j.str"
    bmrb_file.write_text(bmrb_content)

    couplings = parse_bmrb_j_couplings(str(bmrb_file))

    assert 1 in couplings
    assert "3JHNHA" in couplings[1]
    assert couplings[1]["3JHNHA"] == 7.5

    assert 2 in couplings
    assert "3JHAHB" in couplings[2]
    assert couplings[2]["3JHAHB"] == 4.2


def test_parse_bmrb_j_couplings_io_error():
    from synth_nmr.data_pipeline import parse_bmrb_j_couplings

    couplings = parse_bmrb_j_couplings("non_existent_j.str")
    assert couplings == {}


def test_parse_bmrb_shifts_atom_types(tmp_path):
    from synth_nmr.data_pipeline import parse_bmrb_shifts

    bmrb_content = """
loop_
_Atom_chem_shift.ID
_Atom_chem_shift.Seq_ID
_Atom_chem_shift.Atom_ID
_Atom_chem_shift.Atom_type
_Atom_chem_shift.Val
1 5 N N 115.0
2 5 H H 8.5
stop_
"""
    bmrb_file = tmp_path / "test_atoms.str"
    bmrb_file.write_text(bmrb_content)

    shifts = parse_bmrb_shifts(str(bmrb_file))
    assert 5 in shifts
    assert shifts[5]["N"] == 115.0
    assert shifts[5]["H"] == 8.5


def test_parse_bmrb_restraints(tmp_path):
    from synth_nmr.data_pipeline import parse_bmrb_restraints

    bmrb_content = """
data_restraints
save_dist_constraints
   loop_
      _Gen_dist_constraint.ID
      _Gen_dist_constraint.Seq_ID_1
      _Gen_dist_constraint.Atom_ID_1
      _Gen_dist_constraint.Seq_ID_2
      _Gen_dist_constraint.Atom_ID_2
      _Gen_dist_constraint.Target_value
      1    1    H     10    HA    5.0
      2    5    N     15    H     4.5
   stop_
save_
"""
    bmrb_file = tmp_path / "test_restraints.str"
    bmrb_file.write_text(bmrb_content)

    restraints = parse_bmrb_restraints(str(bmrb_file))

    assert len(restraints) == 2
    assert restraints[0]["seq_1"] == 1
    assert restraints[0]["atom_1"] == "H"
    assert restraints[0]["seq_2"] == 10
    assert restraints[0]["atom_2"] == "HA"
    assert restraints[0]["dist"] == 5.0

    assert restraints[1]["seq_1"] == 5
    assert restraints[1]["atom_1"] == "N"
    assert restraints[1]["seq_2"] == 15
    assert restraints[1]["atom_2"] == "H"
    assert restraints[1]["dist"] == 4.5


def test_parse_bmrb_restraints_io_error():
    from synth_nmr.data_pipeline import parse_bmrb_restraints

    restraints = parse_bmrb_restraints("non_existent_restraints.str")
    assert restraints == []

import pytest
from unittest.mock import patch, MagicMock
from synth_nmr.data_pipeline import (
    ensure_data_dir_exists,
    download_bmrb_file,
    download_pdb_file,
    parse_bmrb_shifts,
    load_matched_dataset,
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

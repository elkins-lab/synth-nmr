import pytest
import sys
import io
import biotite.structure as struc
from unittest.mock import patch, MagicMock
from synth_nmr.synth_nmr_cli import main, process_commands, interactive_mode
import synth_nmr.synth_nmr_cli as cli_module


@pytest.fixture
def mock_pdb_file(tmp_path):
    # Create a dummy PDB file for testing
    pdb_content = (
        "ATOM      1  N   ALA A   1      -0.525   1.362   0.000  1.00  0.00           N  \n"
        "ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      3  C   ALA A   1       1.520   0.000   0.000  1.00  0.00           C  \n"
        "ATOM      4  O   ALA A   1       2.115   1.066   0.000  1.00  0.00           O  \n"
        "ATOM      5  CB  ALA A   1      -0.525  -0.741  -1.229  1.00  0.00           C  \n"
        "ATOM      6  H   ALA A   1      -1.465   1.362   0.000  1.00  0.00           H  \n"
    )
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text(pdb_content)
    return str(pdb_path)


@patch("sys.stdout", new_callable=io.StringIO)
def test_process_commands_read(mock_stdout, mock_pdb_file):
    cli_module.structure = None
    process_commands(["read", "pdb", mock_pdb_file])
    output = mock_stdout.getvalue()
    assert f"Read PDB file: {mock_pdb_file}" in output
    assert cli_module.structure is not None


@patch("sys.stdout", new_callable=io.StringIO)
def test_process_commands_calculate_rdc(mock_stdout, mock_pdb_file):
    cli_module.structure = None
    process_commands(["read", "pdb", mock_pdb_file, "calculate", "rdc", "10.0", "0.5"])
    output = mock_stdout.getvalue()
    assert "ResID:" in output


@patch("sys.stdout", new_callable=io.StringIO)
def test_process_commands_calculate_rdc_bad_params(mock_stdout, mock_pdb_file):
    cli_module.structure = None
    process_commands(["read", "pdb", mock_pdb_file, "calculate", "rdc", "10x", "0.5"])
    assert "Error: Invalid value for Da" in mock_stdout.getvalue()
    mock_stdout.truncate(0)
    mock_stdout.seek(0)
    process_commands(["read", "pdb", mock_pdb_file, "calculate", "rdc", "10.0", "10x"])
    assert "Error: Invalid value for R" in mock_stdout.getvalue()


@patch("sys.stdout", new_callable=io.StringIO)
def test_process_commands_calculate_rdc_no_structure(mock_stdout):
    cli_module.structure = None
    process_commands(["calculate", "rdc"])
    assert "Error: No PDB file loaded" in mock_stdout.getvalue()


@patch("sys.stdout", new_callable=io.StringIO)
def test_process_commands_predict_shifts(mock_stdout, mock_pdb_file):
    cli_module.structure = None
    process_commands(["read", "pdb", mock_pdb_file, "predict", "shifts"])
    output = mock_stdout.getvalue()
    assert "Chain:" in output


@patch("sys.stdout", new_callable=io.StringIO)
def test_process_commands_predict_shifts_no_structure(mock_stdout):
    cli_module.structure = None
    process_commands(["predict", "shifts"])
    assert "Error: No PDB file loaded" in mock_stdout.getvalue()


@patch("synth_nmr.synth_nmr_cli.calculate_hn_ha_coupling")
@patch("synth_nmr.synth_nmr_cli.calculate_ha_hb_coupling")
@patch("synth_nmr.synth_nmr_cli.calculate_c_cg_coupling")
@patch("sys.stdout", new_callable=io.StringIO)
def test_process_commands_calculate_j_coupling(
    mock_stdout, mock_ccg, mock_hahb, mock_calc_j, mock_pdb_file
):
    cli_module.structure = None
    mock_calc_j.return_value = {"A": {1: 8.5}}
    mock_hahb.return_value = {"A": {1: 3.5}}
    mock_ccg.return_value = {"A": {1: 1.2}}
    process_commands(["read", "pdb", mock_pdb_file, "calculate", "j-coupling"])
    output = mock_stdout.getvalue()
    assert "3J_HNHa = 8.500 Hz" in output
    assert "3J_HaHb = 3.500 Hz" in output
    assert "3J_C'Cg = 1.200 Hz" in output


@patch("sys.stdout", new_callable=io.StringIO)
def test_process_commands_calculate_j_coupling_no_structure(mock_stdout):
    cli_module.structure = None
    process_commands(["calculate", "j-coupling"])
    assert "Error: No PDB file loaded" in mock_stdout.getvalue()


@patch("sys.stdout", new_callable=io.StringIO)
def test_process_commands_unknown(mock_stdout):
    process_commands(["unknown_command"])
    assert "Error: Unknown command" in mock_stdout.getvalue()


@patch("sys.stdout", new_callable=io.StringIO)
def test_main_process_commands(mock_stdout):
    with patch.object(sys, "argv", ["synth_nmr_cli.py", "unknown_cmd"]):
        main()
        assert "Error: Unknown command" in mock_stdout.getvalue()


@patch("sys.stdout", new_callable=io.StringIO)
@patch("sys.stdin", new_callable=io.StringIO)
def test_main_interactive(mock_stdin, mock_stdout):
    mock_stdin.write("help\nexit\n")
    mock_stdin.seek(0)
    with patch.object(sys, "argv", ["synth_nmr_cli.py"]):
        main()
        output = mock_stdout.getvalue()
        assert "Welcome to the synth-nmr CLI!" in output
        assert "Commands:" in output
        assert "SynthNMR>" in output


@patch("sys.stdout", new_callable=io.StringIO)
@patch("sys.stdin", new_callable=io.StringIO)
def test_interactive_mode_commands(mock_stdin, mock_stdout, mock_pdb_file):
    mock_stdin.write(
        f"read pdb {mock_pdb_file}\ncalculate rdc\npredict shifts\ncalculate j-coupling\nexit\nexit\nexit\n"
    )
    mock_stdin.seek(0)
    interactive_mode()
    output = mock_stdout.getvalue()
    assert "Read PDB file" in output
    assert "ResID:" in output


@patch("sys.stdout", new_callable=io.StringIO)
@patch("sys.stdin", new_callable=io.StringIO)
def test_interactive_mode_bad_params(mock_stdin, mock_stdout, mock_pdb_file):
    cli_module.structure = None
    mock_stdin.write(
        f"read pdb {mock_pdb_file}\ncalculate rdc 10x 0.5\ncalculate rdc 10.0 10x\nexit\nexit\nexit\n"
    )
    mock_stdin.seek(0)
    interactive_mode()
    output = mock_stdout.getvalue()
    assert "Error: Invalid value for Da" in output
    assert "Error: Invalid value for R" in output


@patch("sys.stdout", new_callable=io.StringIO)
@patch("sys.stdin", new_callable=io.StringIO)
def test_interactive_mode_bad_read(mock_stdin, mock_stdout):
    mock_stdin.write("read pdb\nexit\n")
    mock_stdin.seek(0)
    interactive_mode()
    assert "Usage: read pdb <filename>" in mock_stdout.getvalue()


@patch("sys.stdout", new_callable=io.StringIO)
@patch("sys.stdin", new_callable=io.StringIO)
def test_interactive_mode_no_structure(mock_stdin, mock_stdout):
    cli_module.structure = None
    mock_stdin.write("calculate rdc\npredict shifts\ncalculate j-coupling\nexit\nexit\nexit\n")
    mock_stdin.seek(0)
    interactive_mode()
    output = mock_stdout.getvalue()
    assert "Error: No PDB file loaded" in output


def test_interactive_mode_exit(capsys, mocker):
    mocker.patch("sys.stdin.readline", side_effect=["exit\n"])
    interactive_mode()

    out = capsys.readouterr().out
    assert (
        "Entering interactive mode" not in out
    )  # Entering interactive mode is printed by process_commands, not interactive_mode


def test_interactive_mode_help(capsys, mocker):
    mocker.patch("sys.stdin.readline", side_effect=["help\n", "exit\n"])
    interactive_mode()

    out = capsys.readouterr().out
    assert "Commands:" in out


def test_interactive_mode_read_missing_filename(capsys, mocker):
    mocker.patch("sys.stdin.readline", side_effect=["read pdb\n", "exit\n"])
    interactive_mode()

    out = capsys.readouterr().out
    assert "Usage: read pdb <filename>" in out


def test_interactive_mode_read_file_not_found(capsys, mocker):
    mocker.patch("sys.stdin.readline", side_effect=["read pdb does_not_exist.pdb\n", "exit\n"])
    interactive_mode()

    out = capsys.readouterr().out
    assert "Error: File not found: does_not_exist.pdb" in out


def test_interactive_mode_read_exception(capsys, mocker):
    mocker.patch("biotite.structure.io.pdb.PDBFile.read", side_effect=Exception("Mock Read Error"))
    mocker.patch("sys.stdin.readline", side_effect=["read pdb bad_file.pdb\n", "exit\n"])
    interactive_mode()

    out = capsys.readouterr().out
    assert "Error: Failed to read PDB file: Mock Read Error" in out


def test_interactive_mode_calculate_rdc_no_structure(capsys, mocker):
    mocker.patch("sys.stdin.readline", side_effect=["calculate rdc\n", "exit\n"])
    interactive_mode()

    out = capsys.readouterr().out
    assert "Error: No PDB file loaded" in out


def test_interactive_mode_calculate_rdc_bad_da(capsys, mocker):
    # Mock structure load
    mock_struct = MagicMock()
    mocker.patch("biotite.structure.io.pdb.PDBFile.read")
    mocker.patch("biotite.structure.io.pdb.PDBFile.get_structure", return_value=mock_struct)

    mocker.patch(
        "sys.stdin.readline",
        side_effect=["read pdb fake.pdb\n", "calculate rdc bad_da\n", "exit\n"],
    )
    interactive_mode()

    out = capsys.readouterr().out
    assert "Error: Invalid value for Da" in out


def test_interactive_mode_calculate_rdc_bad_r(capsys, mocker):
    mock_struct = MagicMock()
    mocker.patch("biotite.structure.io.pdb.PDBFile.read")
    mocker.patch("biotite.structure.io.pdb.PDBFile.get_structure", return_value=mock_struct)

    mocker.patch(
        "sys.stdin.readline",
        side_effect=["read pdb fake.pdb\n", "calculate rdc 10.0 bad_r\n", "exit\n"],
    )
    interactive_mode()

    out = capsys.readouterr().out
    assert "Error: Invalid value for R" in out


def test_interactive_mode_predict_shifts_no_structure(capsys, mocker):
    cli_module.structure = None
    mocker.patch("sys.stdin.readline", side_effect=["predict shifts\n", "exit\n"])
    interactive_mode()

    out = capsys.readouterr().out
    assert "Error: No PDB file loaded" in out


def test_interactive_mode_calculate_jcoupling_no_structure(capsys, mocker):
    cli_module.structure = None
    mocker.patch("sys.stdin.readline", side_effect=["calculate j-coupling\n", "exit\n"])
    interactive_mode()

    out = capsys.readouterr().out
    assert "Error: No PDB file loaded" in out


def test_interactive_mode_unknown_command(capsys, mocker):
    mocker.patch("sys.stdin.readline", side_effect=["fake_command\n", "exit\n"])
    interactive_mode()

    out = capsys.readouterr().out
    assert "Error: Unknown command: fake_command" in out


def test_interactive_mode_keyboard_interrupt(capsys, mocker):
    mocker.patch("sys.stdin.readline", side_effect=KeyboardInterrupt())
    interactive_mode()


def test_interactive_mode_unexpected_exception(capsys, mocker):
    mocker.patch("sys.stdin.readline", side_effect=[Exception("Mock Unexpected Error"), "exit\n"])
    interactive_mode()

    out = capsys.readouterr().out
    assert "An error occurred: Mock Unexpected Error" in out


def test_interactive_mode_calculate_rdc_valid(capsys, mocker):
    mock_struct = MagicMock()
    mocker.patch("biotite.structure.io.pdb.PDBFile.read")
    mocker.patch("biotite.structure.io.pdb.PDBFile.get_structure", return_value=mock_struct)

    # Calculate requires a mocked version of calculate_rdcs since it runs the physics engine
    mocker.patch("synth_nmr.synth_nmr_cli.calculate_rdcs", return_value={1: 10.0})

    mocker.patch(
        "sys.stdin.readline",
        side_effect=["read pdb fake.pdb\n", "calculate rdc 10.0 0.5\n", "exit\n"],
    )
    interactive_mode()

    out = capsys.readouterr().out
    assert "ResID: 1, RDC: 10.0" in out


def test_interactive_mode_predict_shifts_valid(capsys, mocker):
    mock_struct = MagicMock()
    mocker.patch("biotite.structure.io.pdb.PDBFile.read")
    mocker.patch("biotite.structure.io.pdb.PDBFile.get_structure", return_value=mock_struct)

    mocker.patch(
        "synth_nmr.synth_nmr_cli.predict_chemical_shifts", return_value={"A": {1: {"CA": 50.0}}}
    )

    mocker.patch(
        "sys.stdin.readline", side_effect=["read pdb fake.pdb\n", "predict shifts\n", "exit\n"]
    )
    interactive_mode()

    out = capsys.readouterr().out
    assert "Chain: A, ResID: 1" in out
    assert "CA: 50.0" in out


def test_interactive_mode_jcoupling_valid(capsys, mocker):
    mock_struct = MagicMock()
    mocker.patch("biotite.structure.io.pdb.PDBFile.read")
    mocker.patch("biotite.structure.io.pdb.PDBFile.get_structure", return_value=mock_struct)

    mocker.patch("synth_nmr.synth_nmr_cli.calculate_hn_ha_coupling", return_value={"A": {1: 8.5}})
    mocker.patch("synth_nmr.synth_nmr_cli.calculate_ha_hb_coupling", return_value={})
    mocker.patch("synth_nmr.synth_nmr_cli.calculate_c_cg_coupling", return_value={})

    mocker.patch(
        "sys.stdin.readline",
        side_effect=["read pdb fake.pdb\n", "calculate j-coupling\n", "exit\n"],
    )
    interactive_mode()

    out = capsys.readouterr().out
    assert "Chain A ResID    1  3J_HNHa = 8.500 Hz" in out


def test_process_commands_ensemble_j_coupling(capsys, mocker):
    """Test the ensemble j-coupling command."""
    # Mock ensemble and frames
    mock_frame = MagicMock(spec=struc.AtomArray)
    mock_ensemble = cli_module.TrajectoryEnsemble(frames=[mock_frame, mock_frame])
    cli_module.ensemble = mock_ensemble
    
    # Mock calculation return
    mocker.patch("synth_nmr.synth_nmr_cli.calculate_hn_ha_coupling", return_value={"A": {1: 7.0}})
    
    process_commands(["ensemble", "j-coupling"])
    
    out = capsys.readouterr().out
    assert "Chain A ResID    1  3J_HNHa = 7.000 Hz" in out


def test_process_commands_invalid():
    from synth_nmr.synth_nmr_cli import process_commands
    import sys
    from io import StringIO

    # redirect stdout to intercept print output rather than mocking since process_commands
    # lacks dependency injection for it
    old_stdout = sys.stdout
    sys.stdout = my_stdout = StringIO()

    # Pass unknown argument format
    process_commands(["--unknown", "arg"])

    sys.stdout = old_stdout
    out = my_stdout.getvalue()
    assert "Error: Unknown command: --unknown" in out


def test_process_commands_empty():
    from synth_nmr.synth_nmr_cli import process_commands

    # empty list should just return cleanly
    process_commands([])


def test_process_commands_invalid_da(capsys):
    from synth_nmr.synth_nmr_cli import process_commands
    import biotite.structure as struc
    import numpy as np
    from synth_nmr import synth_nmr_cli as cli_module

    structure = struc.AtomArray(1)
    structure.res_id = np.array([1])
    cli_module.structure = structure

    process_commands(["calculate", "rdc", "invalid_da"])
    out = capsys.readouterr().out
    assert "Error: Invalid value for Da" in out


def test_process_commands_invalid_r(capsys):
    from synth_nmr.synth_nmr_cli import process_commands
    import biotite.structure as struc
    import numpy as np
    from synth_nmr import synth_nmr_cli as cli_module

    structure = struc.AtomArray(1)
    structure.res_id = np.array([1])
    cli_module.structure = structure

    process_commands(["calculate", "rdc", "10.0", "invalid_r"])
    out = capsys.readouterr().out
    assert "Error: Invalid value for R" in out


def test_interactive_mode_eof(capsys, mocker):
    from synth_nmr.synth_nmr_cli import interactive_mode

    mocker.patch("sys.stdin.readline", side_effect=["", EOFError()])
    interactive_mode()
    out = capsys.readouterr().out
    assert "Welcome to the synth-nmr CLI" in out


def test_process_commands_invalid_read(capsys, mocker):
    from synth_nmr.synth_nmr_cli import process_commands

    mocker.patch("biotite.structure.io.pdb.PDBFile.read", side_effect=FileNotFoundError("fake.pdb"))
    process_commands(["read", "pdb", "fake.pdb"])
    out = capsys.readouterr().out
    assert "Error: File not found" in out


def test_process_commands_invalid_read_exception(capsys, mocker):
    from synth_nmr.synth_nmr_cli import process_commands

    mocker.patch("biotite.structure.io.pdb.PDBFile.read", side_effect=Exception("Read Error"))
    process_commands(["read", "pdb", "fake.pdb"])
    out = capsys.readouterr().out
    assert "Error: Failed to read PDB file" in out


def test_interactive_mode_invalid_read(capsys, mocker):
    from synth_nmr.synth_nmr_cli import interactive_mode

    mocker.patch("sys.stdin.readline", side_effect=["read pdb fake.pdb\n", "exit\n"])
    mocker.patch("biotite.structure.io.pdb.PDBFile.read", side_effect=Exception("Mock Read Error"))
    interactive_mode()
    out = capsys.readouterr().out
    assert "Error: Failed to read PDB file" in out

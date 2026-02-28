import pytest
import sys
import io
import os
from unittest.mock import patch, MagicMock
from synth_nmr.synth_nmr_cli import main, process_commands, interactive_mode
import synth_nmr.synth_nmr_cli as cli_module

@pytest.fixture
def mock_pdb_file(tmp_path):
    # Create a dummy PDB file for testing
    pdb_content = "ATOM      1  N   ALA A   1      -0.525   1.362   0.000  1.00  0.00           N  \n" \
                  "ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n" \
                  "ATOM      3  C   ALA A   1       1.520   0.000   0.000  1.00  0.00           C  \n" \
                  "ATOM      4  O   ALA A   1       2.115   1.066   0.000  1.00  0.00           O  \n" \
                  "ATOM      5  CB  ALA A   1      -0.525  -0.741  -1.229  1.00  0.00           C  \n" \
                  "ATOM      6  H   ALA A   1      -1.465   1.362   0.000  1.00  0.00           H  \n"
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text(pdb_content)
    return str(pdb_path)

@patch('sys.stdout', new_callable=io.StringIO)
def test_process_commands_read(mock_stdout, mock_pdb_file):
    cli_module.structure = None
    process_commands(["read", "pdb", mock_pdb_file])
    output = mock_stdout.getvalue()
    assert f"Read PDB file: {mock_pdb_file}" in output
    assert cli_module.structure is not None

@patch('sys.stdout', new_callable=io.StringIO)
def test_process_commands_invalid_read(mock_stdout):
    process_commands(["read", "pdb", "nonexistent.pdb"])
    assert "Error: File not found" in mock_stdout.getvalue()

@patch('sys.stdout', new_callable=io.StringIO)
def test_process_commands_calculate_rdc(mock_stdout, mock_pdb_file):
    cli_module.structure = None
    process_commands(["read", "pdb", mock_pdb_file, "calculate", "rdc", "10.0", "0.5"])
    output = mock_stdout.getvalue()
    assert "ResID:" in output

@patch('sys.stdout', new_callable=io.StringIO)
def test_process_commands_calculate_rdc_bad_params(mock_stdout, mock_pdb_file):
    cli_module.structure = None
    process_commands(["read", "pdb", mock_pdb_file, "calculate", "rdc", "10x", "0.5"])
    assert "Error: Invalid value for Da" in mock_stdout.getvalue()
    mock_stdout.truncate(0)
    mock_stdout.seek(0)
    process_commands(["read", "pdb", mock_pdb_file, "calculate", "rdc", "10.0", "10x"])
    assert "Error: Invalid value for R" in mock_stdout.getvalue()

@patch('sys.stdout', new_callable=io.StringIO)
def test_process_commands_calculate_rdc_no_structure(mock_stdout):
    cli_module.structure = None
    process_commands(["calculate", "rdc"])
    assert "Error: No PDB file loaded" in mock_stdout.getvalue()

@patch('sys.stdout', new_callable=io.StringIO)
def test_process_commands_predict_shifts(mock_stdout, mock_pdb_file):
    cli_module.structure = None
    process_commands(["read", "pdb", mock_pdb_file, "predict", "shifts"])
    output = mock_stdout.getvalue()
    assert "Chain:" in output

@patch('sys.stdout', new_callable=io.StringIO)
def test_process_commands_predict_shifts_no_structure(mock_stdout):
    cli_module.structure = None
    process_commands(["predict", "shifts"])
    assert "Error: No PDB file loaded" in mock_stdout.getvalue()

@patch('sys.stdout', new_callable=io.StringIO)
def test_process_commands_calculate_j_coupling(mock_stdout, mock_pdb_file):
    cli_module.structure = None
    process_commands(["read", "pdb", mock_pdb_file, "calculate", "j-coupling"])
    output = mock_stdout.getvalue()
    # PDB only has 1 residue, no coupling can be calculated
    # But it shouldn't crash
    assert "Read PDB file" in output

@patch('sys.stdout', new_callable=io.StringIO)
def test_process_commands_calculate_j_coupling_no_structure(mock_stdout):
    cli_module.structure = None
    process_commands(["calculate", "j-coupling"])
    assert "Error: No PDB file loaded" in mock_stdout.getvalue()

@patch('sys.stdout', new_callable=io.StringIO)
def test_process_commands_unknown(mock_stdout):
    process_commands(["unknown_command"])
    assert "Error: Unknown command" in mock_stdout.getvalue()

@patch('sys.stdout', new_callable=io.StringIO)
def test_main_process_commands(mock_stdout):
    with patch.object(sys, 'argv', ['synth_nmr_cli.py', 'unknown_cmd']):
        main()
        assert "Error: Unknown command" in mock_stdout.getvalue()

@patch('sys.stdout', new_callable=io.StringIO)
@patch('sys.stdin', new_callable=io.StringIO)
def test_main_interactive(mock_stdin, mock_stdout):
    mock_stdin.write("help\nexit\n")
    mock_stdin.seek(0)
    with patch.object(sys, 'argv', ['synth_nmr_cli.py']):
        main()
        output = mock_stdout.getvalue()
        assert "Welcome to the synth-nmr CLI!" in output
        assert "Commands:" in output
        assert "SynthNMR>" in output

@patch('sys.stdout', new_callable=io.StringIO)
@patch('sys.stdin', new_callable=io.StringIO)
def test_interactive_mode_commands(mock_stdin, mock_stdout, mock_pdb_file):
    mock_stdin.write(f"read pdb {mock_pdb_file}\ncalculate rdc\npredict shifts\ncalculate j-coupling\nexit\n")
    mock_stdin.seek(0)
    interactive_mode()
    output = mock_stdout.getvalue()
    assert "Read PDB file" in output
    assert "ResID:" in output

@patch('sys.stdout', new_callable=io.StringIO)
@patch('sys.stdin', new_callable=io.StringIO)
def test_interactive_mode_bad_params(mock_stdin, mock_stdout, mock_pdb_file):
    cli_module.structure = None
    mock_stdin.write(f"read pdb {mock_pdb_file}\ncalculate rdc 10x 0.5\ncalculate rdc 10.0 10x\nexit\n")
    mock_stdin.seek(0)
    interactive_mode()
    output = mock_stdout.getvalue()
    assert "Error: Invalid value for Da" in output
    assert "Error: Invalid value for R" in output

@patch('sys.stdout', new_callable=io.StringIO)
@patch('sys.stdin', new_callable=io.StringIO)
def test_interactive_mode_bad_read(mock_stdin, mock_stdout):
    mock_stdin.write("read pdb\nexit\n")
    mock_stdin.seek(0)
    interactive_mode()
    assert "Usage: read pdb <filename>" in mock_stdout.getvalue()

@patch('sys.stdout', new_callable=io.StringIO)
@patch('sys.stdin', new_callable=io.StringIO)
def test_interactive_mode_no_structure(mock_stdin, mock_stdout):
    cli_module.structure = None
    mock_stdin.write("calculate rdc\npredict shifts\ncalculate j-coupling\nexit\n")
    mock_stdin.seek(0)
    interactive_mode()
    output = mock_stdout.getvalue()
    assert "Error: No PDB file loaded" in output

@patch('sys.stdout', new_callable=io.StringIO)
@patch('sys.stdin', new_callable=io.StringIO)
def test_interactive_mode_unknown_command(mock_stdin, mock_stdout):
    mock_stdin.write("foo_bar\nexit\n")
    mock_stdin.seek(0)
    interactive_mode()
    output = mock_stdout.getvalue()
    assert "Error: Unknown command: foo_bar" in output

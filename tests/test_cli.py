import subprocess
import os

def run_cli_commands(commands):
    """Run a series of commands in the synth-nmr-cli and return the output."""
    process = subprocess.Popen(
        ['python', '-m', 'synth_nmr.synth_nmr_cli'] + commands,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.getcwd()
    )
    stdout, stderr = process.communicate()
    return stdout, stderr

def test_cli_read_pdb():
    """Test the 'read pdb' command in the CLI."""
    pdb_file_path = os.path.abspath('tests/data/test.pdb')
    commands = ['read', 'pdb', pdb_file_path]
    stdout, stderr = run_cli_commands(commands)
    assert f"Read PDB file: {pdb_file_path}" in stdout
    assert "Error" not in stderr

def test_cli_calculate_rdc():
    """Test the 'calculate rdc' command in the CLI."""
    pdb_file_path = os.path.abspath('tests/data/test.pdb')
    commands = ['read', 'pdb', pdb_file_path, 'calculate', 'rdc']
    stdout, stderr = run_cli_commands(commands)
    assert "ResID" in stdout
    assert "RDC" in stdout
    assert "Error" not in stderr

def test_cli_predict_shifts():
    """Test the 'predict shifts' command in the CLI."""
    pdb_file_path = os.path.abspath('tests/data/test.pdb')
    commands = ['read', 'pdb', pdb_file_path, 'predict', 'shifts']
    stdout, stderr = run_cli_commands(commands)
    assert "Chain" in stdout
    assert "ResID" in stdout
    assert "Error" not in stderr

def test_cli_calculate_j_coupling():
    """Test the 'calculate j-coupling' command in the CLI."""
    pdb_file_path = os.path.abspath('tests/data/test.pdb')
    commands = ['read', 'pdb', pdb_file_path, 'calculate', 'j-coupling']
    stdout, stderr = run_cli_commands(commands)
    assert "Chain" in stdout
    assert "ResID" in stdout
    assert "J-coupling" in stdout
    assert "Error" not in stderr

def test_cli_no_pdb_error():
    """Test that an error is raised if no PDB is read."""
    commands = ['calculate', 'rdc']
    stdout, stderr = run_cli_commands(commands)
    assert "Error: No PDB file loaded" in stdout

def test_cli_bad_da_error():
    """Test that an error is raised with a bad Da value."""
    pdb_file_path = os.path.abspath('tests/data/test.pdb')
    commands = ['read', 'pdb', pdb_file_path, 'calculate', 'rdc', 'bad_value']
    stdout, stderr = run_cli_commands(commands)
    assert "Error: Invalid value for Da" in stdout

def test_cli_bad_r_error():
    """Test that an error is raised with a bad R value."""
    pdb_file_path = os.path.abspath('tests/data/test.pdb')
    commands = ['read', 'pdb', pdb_file_path, 'calculate', 'rdc', '10.0', 'bad_value']
    stdout, stderr = run_cli_commands(commands)
    assert "Error: Invalid value for R" in stdout

def test_cli_nonexistent_pdb():
    """Test that an error is raised with a nonexistent PDB file."""
    commands = ['read', 'pdb', 'nonexistent.pdb']
    stdout, stderr = run_cli_commands(commands)
    assert "Error: File not found" in stdout

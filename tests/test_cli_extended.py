import os
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

import biotite.structure as struc

from synth_nmr import synth_nmr_cli as cli
from synth_nmr.synth_nmr_cli import handle_interactive_command, process_commands


class TestCLIExtended(unittest.TestCase):
    def setUp(self):
        cli.structure = None
        cli.ensemble = None
        # Use existing test PDB from the repository
        self.pdb_path = os.path.abspath("tests/data/test.pdb")

    def tearDown(self):
        if os.path.exists("test_export.nef"):
            os.remove("test_export.nef")
        if os.path.exists("test_export.csv"):
            os.remove("test_export.csv")

    def test_cli_export_commands(self):
        """Test export commands coverage."""
        args = [
            "read",
            "pdb",
            self.pdb_path,
            "export",
            "nef",
            "test_export.nef",
            "export",
            "shifts",
            "test_export.csv",
        ]
        with patch("sys.stdout", new=StringIO()) as fake_out, patch(
            "synth_nmr.synth_nmr_cli.calculate_synthetic_noes", return_value=[]
        ):
            process_commands(args)
            output = fake_out.getvalue()
            assert "Exported data to test_export.nef" in output
            assert "Exported chemical shifts to test_export.csv" in output

        assert os.path.exists("test_export.nef")
        assert os.path.exists("test_export.csv")

    def test_cli_validate_noes_mocked(self):
        """Test validate noes subcommand with mocking."""
        cli.structure = MagicMock(spec=struc.AtomArray)

        with patch("synth_nmr.synth_nmr_cli.download_bmrb_file", return_value="mock.str"), patch(
            "synth_nmr.synth_nmr_cli.parse_bmrb_restraints", return_value={}
        ), patch("synth_nmr.synth_nmr_cli.calculate_synthetic_noes", return_value=[]), patch(
            "synth_nmr.synth_nmr_cli.calculate_rpf_scores",
            return_value={"recall": 0.9, "precision": 0.8, "f_measure": 0.85},
        ), patch("synth_nmr.synth_nmr_cli.calculate_dp_score", return_value=0.7), patch(
            "sys.stdout", new=StringIO()
        ) as fake_out:
            handle_interactive_command("validate noes 12345")
            output = fake_out.getvalue()
            assert "NOE Validation (RPF) against BMRB 12345:" in output
            assert "Recall:    0.900" in output

    def test_cli_validate_rdc(self):
        """Test validate rdc subcommand."""
        cli.structure = MagicMock(spec=struc.AtomArray)
        rdc_file = "test_exp_rdcs.csv"
        with open(rdc_file, "w") as f:
            f.write("1,10.0\n2,20.0\n")

        try:
            with patch(
                "synth_nmr.synth_nmr_cli.calculate_rdcs", return_value={1: 11.0, 2: 21.0}
            ), patch("sys.stdout", new=StringIO()) as fake_out:
                handle_interactive_command(f"validate rdc {rdc_file}")
                output = fake_out.getvalue()
                assert "RDC Validation (Cornilescu Q-factor)" in output
                assert "Q-factor:" in output
        finally:
            if os.path.exists(rdc_file):
                os.remove(rdc_file)

    def test_cli_export_errors(self):
        """Test export commands error handling."""
        # No PDB loaded
        cli.structure = None
        with patch("sys.stdout", new=StringIO()) as fake_out:
            handle_interactive_command("export nef test.nef")
            assert "Error: No PDB file loaded" in fake_out.getvalue()

        # Unknown export subcommand
        cli.structure = MagicMock()
        with patch("sys.stdout", new=StringIO()) as fake_out:
            handle_interactive_command("export unknown test.txt")
            assert "Error: Unknown export subcommand" in fake_out.getvalue()

    def test_cli_help(self):
        """Test help command."""
        with patch("sys.stdout", new=StringIO()) as fake_out:
            handle_interactive_command("help")
            output = fake_out.getvalue()
            assert "Commands:" in output
            assert "read pdb" in output
            assert "export nef" in output

    def test_cli_main_interactive_trigger(self):
        """Test main entry point triggers interactive mode when no args."""
        with patch("sys.argv", ["synth_nmr_cli"]), patch(
            "synth_nmr.synth_nmr_cli.interactive_mode"
        ) as mock_interactive:
            cli.main()
            mock_interactive.assert_called_once()

    def test_cli_main_batch_trigger(self):
        """Test main entry point triggers process_commands when args provided."""
        with patch("sys.argv", ["synth_nmr_cli", "read", "pdb", "file.pdb"]), patch(
            "synth_nmr.synth_nmr_cli.process_commands"
        ) as mock_process:
            cli.main()
            mock_process.assert_called_once_with(["read", "pdb", "file.pdb"])


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import biotite.structure as struc

from synth_nmr.validation import calculate_rdc_q_factor, validate_against_bmrb


class TestValidationExtended(unittest.TestCase):
    def test_rdc_q_factor_perfect(self):
        """Test Q-factor with perfect agreement."""
        pred = {1: 10.0, 2: -5.0, 3: 15.0}
        exp = {1: 10.0, 2: -5.0, 3: 15.0}
        q = calculate_rdc_q_factor(pred, exp)
        self.assertEqual(q, 0.0)

    def test_rdc_q_factor_noisy(self):
        """Test Q-factor with noisy data."""
        # Simple case: exp = [10], pred = [12]
        # num = (12-10)^2 = 4
        # den = 10^2 = 100
        # Q = sqrt(4/100) = 0.2
        pred = {1: 12.0}
        exp = {1: 10.0}
        q = calculate_rdc_q_factor(pred, exp)
        self.assertAlmostEqual(q, 0.2)

    def test_rdc_q_factor_no_overlap(self):
        """Test Q-factor with no overlapping residues."""
        pred = {1: 10.0}
        exp = {2: 10.0}
        q = calculate_rdc_q_factor(pred, exp)
        self.assertEqual(q, 1.0)

    @patch("synth_nmr.data_pipeline.download_bmrb_file")
    @patch("synth_nmr.data_pipeline.parse_bmrb_shifts")
    @patch("synth_nmr.chemical_shifts.predict_chemical_shifts")
    def test_validate_against_bmrb(self, mock_predict, mock_parse, mock_download):
        """Test high-level BMRB validation wrapper."""
        mock_download.return_value = "fake.str"
        # BMRB shifts: {res_id: {atom: val}}
        mock_parse.return_value = {1: {"CA": 50.0}, 2: {"CA": 60.0}}
        # Predicted: {chain_id: {res_id: {atom: val}}}
        mock_predict.return_value = {"A": {1: {"CA": 51.0}, 2: {"CA": 61.0}}}

        structure = struc.AtomArray(2)
        structure.chain_id[:] = "A"
        structure.res_id[:] = [1, 2]
        structure.res_name[:] = ["ALA", "ALA"]
        structure.atom_name[:] = ["CA", "CA"]
        structure.element[:] = ["C", "C"]

        stats = validate_against_bmrb(12345, structure)

        self.assertIn("CA", stats)
        self.assertEqual(stats["CA"]["rmse"], 1.0)
        self.assertIn("r_factor", stats["CA"])


if __name__ == "__main__":
    unittest.main()

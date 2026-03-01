import unittest
from unittest.mock import patch
import numpy as np
import biotite.structure as struc
from synth_nmr.relaxation import (
    spectral_density,
    predict_order_parameters,
    calculate_relaxation_rates,
)
import importlib
import sys


class TestRelaxation(unittest.TestCase):
    def test_spectral_density(self):
        """Test the spectral_density function with known values."""
        # Test case 1: Rigid body (S2=1.0)
        j = spectral_density(omega=5e8, tau_m=10e-9, s2=1.0, tau_f=0.0)
        self.assertAlmostEqual(j, 1.538e-10, delta=1e-12)

        # Test case 2: Flexible (S2=0.5) with fast internal motion
        j_flex = spectral_density(omega=5e8, tau_m=10e-9, s2=0.5, tau_f=100e-12)
        self.assertAlmostEqual(j_flex, 9.667e-11, delta=1e-12)

    def test_predict_order_parameters(self):
        """Test the S2 prediction."""
        # Create a simple two-residue structure
        res_count = 2
        structure = struc.AtomArray(res_count * 4)
        structure.atom_name = np.array(["N", "CA", "C", "O"] * res_count)
        structure.res_id = np.repeat(np.arange(1, res_count + 1), 4)
        structure.res_name = np.repeat(["ALA"], res_count * 4)
        structure.chain_id = np.repeat(["A"], res_count * 4)

        s2_map = predict_order_parameters(structure)
        self.assertEqual(len(s2_map), res_count)
        # Check that the S2 values are within a reasonable range
        for s2 in s2_map.values():
            self.assertTrue(0.01 <= s2 <= 0.98)

    def test_calculate_relaxation_rates(self):
        """Test the relaxation rate calculation."""
        # Create a simple alpha helix structure
        res_count = 10
        # Create a simple mock structure with N and H atoms
        helix = struc.AtomArray(res_count * 2)
        helix.atom_name = np.array(["N", "H"] * res_count)
        helix.res_id = np.repeat(np.arange(1, res_count + 1), 2)
        helix.res_name[:] = "ALA"
        helix.chain_id[:] = "A"

        # Assign some dummy coordinates
        for i in range(res_count):
            helix.coord[2 * i] = [i * 3.8, 0, 0]  # N
            helix.coord[2 * i + 1] = [i * 3.8, 1.02, 0]  # H

        rates = calculate_relaxation_rates(helix)
        self.assertEqual(len(rates), res_count)
        for res_id, res_rates in rates.items():
            self.assertIn("R1", res_rates)
            self.assertIn("R2", res_rates)
            self.assertIn("NOE", res_rates)
            self.assertIn("S2", res_rates)

    def test_empty_structure_order_parameters(self):
        """Test S2 prediction with an empty structure."""
        structure = struc.AtomArray(0)
        s2_map = predict_order_parameters(structure)
        self.assertEqual(s2_map, {})

    def test_proline_and_missing_atoms_relaxation(self):
        """Test that proline and residues missing N or H are skipped."""
        structure = struc.AtomArray(4)
        structure.atom_name = ["N", "CA", "N", "CA"]
        structure.res_id = [1, 1, 2, 2]
        structure.res_name = ["PRO", "PRO", "ALA", "ALA"]
        structure.chain_id = ["A", "A", "A", "A"]
        rates = calculate_relaxation_rates(structure)
        self.assertEqual(len(rates), 0)

    def test_sasa_failure_order_parameters(self):
        """Test S2 prediction when SASA calculation fails."""
        with patch("biotite.structure.sasa", side_effect=Exception("SASA fail")):
            # Create a simple two-residue structure
            res_count = 2
            structure = struc.AtomArray(res_count * 4)
            structure.atom_name = np.array(["N", "CA", "C", "O"] * res_count)
            structure.res_id = np.repeat(np.arange(1, res_count + 1), 4)
            structure.res_name = np.repeat(["ALA"], res_count * 4)
            structure.chain_id = np.repeat(["A"], res_count * 4)
            s2_map = predict_order_parameters(structure)
            self.assertEqual(len(s2_map), res_count)

    def test_s2_map_fallback_relaxation(self):
        """Test the S2 map fallback in relaxation calculation."""
        res_count = 2
        helix = struc.AtomArray(res_count * 2)
        helix.atom_name = np.array(["N", "H"] * res_count)
        helix.res_id = np.repeat(np.arange(1, res_count + 1), 2)
        helix.res_name[:] = "ALA"
        helix.chain_id[:] = "A"
        rates = calculate_relaxation_rates(helix, s2_map={1: 0.5})
        self.assertEqual(len(rates), res_count)
        self.assertAlmostEqual(rates[1]["S2"], 0.5)
        self.assertAlmostEqual(rates[2]["S2"], 0.85)

    def test_spectral_density_fast_motion(self):
        """Test spectral density with significant fast internal motion."""
        # This test is specifically to ensure the 'if tau_f > 0' block is covered.
        omega = 5e8
        tau_m = 10e-9
        s2 = 0.5
        tau_f = 100e-12

        tau_e = (tau_m * tau_f) / (tau_m + tau_f)
        j_global = (s2 * tau_m) / (1 + (omega * tau_m) ** 2)
        j_fast = ((1 - s2) * tau_e) / (1 + (omega * tau_e) ** 2)

        expected_j = 0.4 * (j_global + j_fast)

        j = spectral_density(omega=omega, tau_m=tau_m, s2=s2, tau_f=tau_f)
        self.assertAlmostEqual(j, expected_j, delta=1e-12)

    def test_predict_order_parameters_with_ptm_and_ion(self):
        """Test S2 prediction with PTMs (SEP) and ions (ZN)."""
        structure = struc.AtomArray(6)
        structure.atom_name = ["N", "CA", "P", "O1P", "ZN", "CA"]
        structure.res_id = [1, 1, 1, 1, 2, 2]
        structure.res_name = ["SEP", "SEP", "SEP", "SEP", "ZN", "ZN"]
        structure.chain_id = ["A", "A", "A", "A", "A", "A"]
        s2_map = predict_order_parameters(structure)
        self.assertEqual(len(s2_map), 2)

    def test_sasa_failure_in_predict_order_parameters(self):
        """Test S2 prediction when biotite.structure.sasa raises an exception."""
        with patch("biotite.structure.sasa", side_effect=Exception("SASA calculation failed")):
            structure = struc.AtomArray(4)
            structure.atom_name = ["N", "CA", "C", "O"]
            structure.res_id = [1, 1, 1, 1]
            structure.res_name = ["ALA", "ALA", "ALA", "ALA"]
            structure.chain_id = ["A", "A", "A", "A"]
            s2_map = predict_order_parameters(structure)
            self.assertEqual(len(s2_map), 1)

    def test_s2_map_fallback_in_calculate_relaxation_rates(self):
        """Test the S2 map fallback for a specific residue."""
        structure = struc.AtomArray(4)
        structure.atom_name = ["N", "H", "N", "H"]
        structure.res_id = [1, 1, 2, 2]
        structure.res_name = ["ALA", "ALA", "GLY", "GLY"]
        structure.chain_id = ["A", "A", "A", "A"]

        # Provide S2 for residue 1, but not for residue 2
        rates = calculate_relaxation_rates(structure, s2_map={1: 0.5})

        self.assertIn(1, rates)
        self.assertIn(2, rates)
        self.assertAlmostEqual(rates[1]["S2"], 0.5)
        # Check that residue 2 falls back to the default of 0.85
        self.assertAlmostEqual(rates[2]["S2"], 0.85)

    def test_numba_fallback(self):
        """Test that the njit decorator falls back to a regular function when numba is not installed."""
        # Ensure that the module is loaded before we try to reload it

        with patch.dict("sys.modules", {"numba": None}):
            # Reload the module to trigger the fallback
            importlib.reload(sys.modules["synth_nmr.relaxation"])
            from synth_nmr.relaxation import spectral_density

            # Test case from test_relaxation.py
            j = spectral_density(omega=5e8, tau_m=10e-9, s2=1.0, tau_f=0.0)
            self.assertAlmostEqual(j, 1.538e-10, delta=1e-12)

        # Reload the module again to restore the original state
        importlib.reload(sys.modules["synth_nmr.relaxation"])

    def test_numba_fallback_with_args(self):
        """Test the njit decorator fallback when called with arguments."""
        with patch.dict("sys.modules", {"numba": None}):
            importlib.reload(sys.modules["synth_nmr.relaxation"])
            from synth_nmr.relaxation import njit

            @njit(fastmath=True)
            def my_func(x):
                return x + 1

            self.assertEqual(my_func(1), 2)

        importlib.reload(sys.modules["synth_nmr.relaxation"])


if __name__ == "__main__":
    unittest.main()

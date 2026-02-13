import unittest
import numpy as np
from synth_nmr.relaxation import spectral_density, predict_order_parameters, calculate_relaxation_rates
import biotite.structure as struc
from biotite.structure.io import pdbx

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
            helix.coord[2*i] = [i * 3.8, 0, 0] # N
            helix.coord[2*i+1] = [i * 3.8, 1.02, 0] # H

        rates = calculate_relaxation_rates(helix)
        self.assertEqual(len(rates), res_count)
        for res_id, res_rates in rates.items():
            self.assertIn('R1', res_rates)
            self.assertIn('R2', res_rates)
            self.assertIn('NOE', res_rates)
            self.assertIn('S2', res_rates)

if __name__ == '__main__':
    unittest.main()

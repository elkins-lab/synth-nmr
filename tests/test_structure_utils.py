import unittest
import numpy as np
import biotite.structure as struc
from synth_nmr.structure_utils import get_secondary_structure

class TestStructureUtils(unittest.TestCase):
    def test_get_secondary_structure_bad_structure(self):
        """Test that BadStructureError is handled gracefully."""
        # A single CA atom is not a valid backbone
        structure = struc.AtomArray(1)
        structure.atom_name = np.array(["CA"])
        structure.res_id = np.array([1])
        structure.res_name = np.array(["ALA"])
        ss = get_secondary_structure(structure)
        self.assertEqual(ss, ["coil"])

    def test_get_secondary_structure_hetatm(self):
        """Test that HETATMs that cause shorter phi/psi arrays are handled."""
        structure = struc.AtomArray(2)
        structure.atom_name = np.array(["CA", "ZN"])
        structure.res_id = np.array([1, 2])
        structure.res_name = np.array(["ALA", "ZN"])
        structure.hetero = np.array([False, True])
        ss = get_secondary_structure(structure)
        self.assertEqual(len(ss), 2)
        self.assertIn("coil", ss)

    def test_get_secondary_structure_left_handed_helix(self):
        """Test the classification of left-handed alpha helices."""
        # Create a mock structure with left-handed alpha helix geometry
        structure = struc.AtomArray(3)
        structure.atom_name = np.array(["N", "CA", "C"])
        structure.res_id = np.array([1, 1, 1])
        structure.res_name = np.array(["ALA", "ALA", "ALA"])

        # Mock the dihedral angles to represent a left-handed helix
        phi = np.deg2rad(np.array([60.0]))
        psi = np.deg2rad(np.array([-40.0]))
        omega = np.deg2rad(np.array([180.0]))
        
        with unittest.mock.patch('biotite.structure.dihedral_backbone', return_value=(phi, psi, omega)):
            ss = get_secondary_structure(structure)
            self.assertEqual(ss, ["alpha"])

    def test_smoothing(self):
        """Test the secondary structure smoothing logic."""
        # Create a mock structure with an alpha-coil-alpha sequence
        structure = struc.AtomArray(9)
        structure.atom_name = np.array(["N", "CA", "C"] * 3)
        structure.res_id = np.array([1,1,1,2,2,2,3,3,3])
        structure.res_name = np.array(["ALA"] * 9)

        # Mock the dihedral angles for alpha, coil, alpha
        phi = np.deg2rad(np.array([-60.0, 0.0, -60.0]))
        psi = np.deg2rad(np.array([-40.0, 0.0, -40.0]))
        omega = np.deg2rad(np.array([180.0] * 3))

        with unittest.mock.patch('biotite.structure.dihedral_backbone', return_value=(phi, psi, omega)):
            ss = get_secondary_structure(structure)
            self.assertEqual(ss, ["alpha", "alpha", "alpha"])

if __name__ == '__main__':
    unittest.main()

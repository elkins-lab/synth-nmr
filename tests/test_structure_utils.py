import unittest

import biotite.structure as struc
import numpy as np

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

        with unittest.mock.patch(
            "biotite.structure.dihedral_backbone", return_value=(phi, psi, omega)
        ):
            ss = get_secondary_structure(structure)
            self.assertEqual(ss, ["alpha"])

    def test_smoothing(self):
        """Test the secondary structure smoothing logic."""
        # Create a mock structure with an alpha-coil-alpha sequence
        structure = struc.AtomArray(9)
        structure.atom_name = np.array(["N", "CA", "C"] * 3)
        structure.res_id = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        structure.res_name = np.array(["ALA"] * 9)

        # Mock the dihedral angles for alpha, coil, alpha
        phi = np.deg2rad(np.array([-60.0, 0.0, -60.0]))
        psi = np.deg2rad(np.array([-40.0, 0.0, -40.0]))
        omega = np.deg2rad(np.array([180.0] * 3))

        with unittest.mock.patch(
            "biotite.structure.dihedral_backbone", return_value=(phi, psi, omega)
        ):
            ss = get_secondary_structure(structure)
            self.assertEqual(ss, ["alpha", "alpha", "alpha"])

    def test_get_secondary_structure_known_motifs(self):
        """
        Test the classification of known secondary structure motifs
        (alpha-helix, beta-strand, coil) based on ideal dihedral angles.
        """
        # --- Alpha Helix Motif ---
        alpha_structure = struc.AtomArray(9)  # 3 residues (N, CA, C per residue)
        alpha_structure.atom_name = np.array(["N", "CA", "C"] * 3)
        alpha_structure.res_id = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        alpha_structure.res_name = np.array(["ALA"] * 9)
        alpha_structure.coord = np.zeros((9, 3))  # Coords not important for this mock

        # Ideal alpha-helical Phi/Psi
        alpha_phi = np.deg2rad(np.array([-60.0, -60.0, -60.0]))
        alpha_psi = np.deg2rad(np.array([-45.0, -45.0, -45.0]))
        alpha_omega = np.deg2rad(np.array([180.0] * 3))  # Trans-peptide bond

        with unittest.mock.patch(
            "biotite.structure.dihedral_backbone", return_value=(alpha_phi, alpha_psi, alpha_omega)
        ):
            ss_alpha = get_secondary_structure(alpha_structure)
            self.assertEqual(ss_alpha, ["alpha", "alpha", "alpha"])

        # --- Beta Strand Motif ---
        beta_structure = struc.AtomArray(9)
        beta_structure.atom_name = np.array(["N", "CA", "C"] * 3)
        beta_structure.res_id = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        beta_structure.res_name = np.array(["ALA"] * 9)
        beta_structure.coord = np.zeros((9, 3))

        # Ideal beta-strand Phi/Psi
        beta_phi = np.deg2rad(np.array([-120.0, -120.0, -120.0]))
        beta_psi = np.deg2rad(np.array([120.0, 120.0, 120.0]))
        beta_omega = np.deg2rad(np.array([180.0] * 3))

        with unittest.mock.patch(
            "biotite.structure.dihedral_backbone", return_value=(beta_phi, beta_psi, beta_omega)
        ):
            ss_beta = get_secondary_structure(beta_structure)
            self.assertEqual(ss_beta, ["beta", "beta", "beta"])

        # --- Random Coil Motif ---
        coil_structure = struc.AtomArray(9)
        coil_structure.atom_name = np.array(["N", "CA", "C"] * 3)
        coil_structure.res_id = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        coil_structure.res_name = np.array(["ALA"] * 9)
        coil_structure.coord = np.zeros((9, 3))

        # Non-ideal Phi/Psi for coil
        coil_phi = np.deg2rad(np.array([-75.0, 20.0, -10.0]))
        coil_psi = np.deg2rad(np.array([150.0, -50.0, 80.0]))
        coil_omega = np.deg2rad(np.array([180.0] * 3))

        with unittest.mock.patch(
            "biotite.structure.dihedral_backbone", return_value=(coil_phi, coil_psi, coil_omega)
        ):
            get_secondary_structure(coil_structure)
            # Depending on the phi_deg ranges in get_secondary_structure, these might be "coil"
            # The current ranges are -80 < phi_deg < -40 for alpha, -160 < phi_deg < -80 for beta
            # So, -75 should be alpha, 20 should be coil, -10 should be coil.
            # The get_secondary_structure has smoothing, so might change.
            # Let's mock a sequence that clearly falls into 'coil' for all.
            # The easiest way is to set angles outside alpha/beta ranges.

            # Recalculate based on known motif ranges
            # -80 < phi_deg < -40: alpha
            # -160 < phi_deg < -80: beta
            # Else: coil

            # So if phi = 0 or 20, it is 'coil'
            # If phi = -75, it is 'alpha'
            # If phi = -10, it is 'coil'

            # Let's make sure the phi values are clearly 'coil'
            coil_phi_adjusted = np.deg2rad(
                np.array([170.0, 170.0, 170.0])
            )  # All outside alpha/beta ranges
            coil_psi_adjusted = np.deg2rad(np.array([0.0, 0.0, 0.0]))

            with unittest.mock.patch(
                "biotite.structure.dihedral_backbone",
                return_value=(coil_phi_adjusted, coil_psi_adjusted, coil_omega),
            ):
                ss_coil_adjusted = get_secondary_structure(coil_structure)
                self.assertEqual(ss_coil_adjusted, ["coil", "coil", "coil"])

    def test_get_secondary_structure_no_protein(self):
        """Test get_secondary_structure when no protein is present."""
        structure = struc.AtomArray(1)
        structure.atom_name = np.array(["HOH"])
        structure.res_id = np.array([1])
        structure.res_name = np.array(["HOH"])
        structure.hetero = np.array([True])
        ss = get_secondary_structure(structure)
        self.assertEqual(ss, ["coil"])

    def test_calculate_c_beta_deviations_edge_cases(self):
        """Test calculate_c_beta_deviations with GLY and missing atoms."""
        from synth_nmr.structure_utils import calculate_c_beta_deviations

        # GLY residue
        atoms = []
        atoms.append(
            struc.Atom(coord=[0, 0, 0], atom_name="CA", res_id=1, res_name="GLY", chain_id="A")
        )
        structure = struc.array(atoms)
        deviations = calculate_c_beta_deviations(structure)
        self.assertEqual(deviations, {})

        # Missing backbone atoms
        atoms = []
        atoms.append(
            struc.Atom(coord=[0, 0, 0], atom_name="CA", res_id=2, res_name="ALA", chain_id="A")
        )
        atoms.append(
            struc.Atom(coord=[1, 1, 1], atom_name="CB", res_id=2, res_name="ALA", chain_id="A")
        )
        # Missing N and C
        structure = struc.array(atoms)
        deviations = calculate_c_beta_deviations(structure)
        self.assertEqual(deviations, {})

import unittest
import os


class TestDocumentationIntegrity(unittest.TestCase):
    """
    Safeguard to ensure educational notes are not accidentally removed.

    These tests scan the source code for specific educational content that
    must be preserved to maintain the pedagogical value of the project.
    """

    def setUp(self):
        # Define paths relative to this test file
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.nmr_path = os.path.join(self.base_dir, "synth_nmr", "nmr.py")
        self.chemical_shifts_path = os.path.join(self.base_dir, "synth_nmr", "chemical_shifts.py")
        self.relaxation_path = os.path.join(self.base_dir, "synth_nmr", "relaxation.py")
        self.j_coupling_path = os.path.join(self.base_dir, "synth_nmr", "j_coupling.py")
        self.nef_io_path = os.path.join(self.base_dir, "synth_nmr", "nef_io.py")
        self.dataset_path = os.path.join(self.base_dir, "synth_nmr", "dataset.py")
        self.data_path = os.path.join(self.base_dir, "synth_nmr", "data.py")

    def _check_file_contains(self, filepath, substrings):
        """Helper to assert file contains list of substrings."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = " ".join(f.read().split())

        for substring in substrings:
            normalized_substring = " ".join(substring.split())
            self.assertIn(
                normalized_substring,
                content,
                f"Missing educational note in {os.path.basename(filepath)}: '{substring[:50]}...'",
            )

    def test_nmr_educational_notes(self):
        """Ensure nmr.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - The Physics of NOEs",
            "intensity of the NOE signal is proportional to the inverse 6th power",
            "practical limit for detection is usually 5.0 - 6.0 Angstroms",
            "Upper Distance Bounds",
        ]
        self._check_file_contains(self.nmr_path, required_notes)

    def test_chemical_shifts_educational_notes(self):
        """Ensure chemical_shifts.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - Random Coil Shifts:",
            "Random Coil",
            "flexible chain",
            "EDUCATIONAL NOTE - Secondary Chemical Shifts:",
            "deviation from these values (Secondary Shift)",
            "EDUCATIONAL NOTE - Ring Current Physics:",
            "Aromatic rings",
            "delocalized pi-electrons",
            "Shielding",
            "Deshielding",
            "EDUCATIONAL NOTE - Prediction Algorithm:",
            "Shift = Random_Coil + Structure_Offset + Noise",
        ]
        self._check_file_contains(self.chemical_shifts_path, required_notes)

    def test_relaxation_educational_notes(self):
        """Ensure relaxation.py retains key educational blocks."""
        required_notes = [
            "EDUCATIONAL NOTE - Lipari-Szabo Model Free:",
            "Order Parameter (S2)",
            "amplitude of internal motion",
            "EDUCATIONAL NOTE - Dipolar Integration Constant (d):",
            "EDUCATIONAL NOTE - Chemical Shift Anisotropy (CSA) Constant (c):",
            "EDUCATIONAL NOTE - BPP Theory & Spectral Density:",
            "Extreme Narrowing Limit",
        ]
        self._check_file_contains(self.relaxation_path, required_notes)

    def test_j_coupling_educational_notes(self):
        """Ensure j_coupling.py retains Karplus and new sidechain notes."""
        required_notes = [
            "EDUCATIONAL NOTE - Karplus Equation:",
            "J = A * cos^2(theta) + B * cos(theta) + C",
            "depends heavily on the torsion angle",
            "EDUCATIONAL NOTE: Chi1 Dihedral",
            'identifying the dominant "rotamer" state',
            "EDUCATIONAL NOTE: 3J(Ha, Hb) Couplings",
            "trans rotamers (chi1 ~ 180) lead to larger, antiperiplanar couplings",
            "EDUCATIONAL NOTE: 3J(C', Cg) Couplings",
            "Unlike proton-proton couplings which rely on isotopic labeling",
        ]
        self._check_file_contains(self.j_coupling_path, required_notes)

    def test_data_educational_notes(self):
        """Ensure data.py retains geometric and amino acid notes."""
        required_notes = [
            "EDUCATIONAL NOTE - Engh & Huber Parameters (The Gold Standard):",
            "Gold Standard",
            'EDUCATIONAL NOTE - Proline Sterics (The "Proline Effect"):',
            "structure breaker",
            'EDUCATIONAL NOTE - The "Mirror Image" World:',
            "EDUCATIONAL NOTE - Backbone Dependency:",
            "EDUCATIONAL NOTE - Rotamers for Non-Branched Residues:",
            "EDUCATIONAL NOTE - Aromatic Residues (PHE, TYR, TRP):",
            "EDUCATIONAL NOTE - Electrostatics vs Sterics:",
        ]
        self._check_file_contains(self.data_path, required_notes)

    def test_nef_io_educational_notes(self):
        """Ensure nef_io.py retains format notes."""
        required_notes = [
            "EDUCATIONAL NOTE - NEF Chemical Shift Format:",
            "NMR-STAR syntax",
        ]
        self._check_file_contains(self.nef_io_path, required_notes)

    # def test_dataset_educational_notes(self):
    #    """Ensure dataset.py retains key educational blocks."""
    #    required_notes = [
    #        "EDUCATIONAL NOTE - The Balanced Dataset Problem:",
    #        "Alpha-Helix Trap",
    #        "Halls of Mirrors",
    #        "Data Factory Overview:",
    # ]
    # self._check_file_contains(self.dataset_path, required_notes)

    def test_comment_ratio(self):
        """Verify that the codebase maintains a high ratio of educational comments."""
        # A simple check to ensure that the j_coupling.py file is at least roughly 30% comments
        # which acts as a proxy for "the code is the textbook"
        with open(self.j_coupling_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)
        comment_lines = sum(
            1 for line in lines if line.strip().startswith("#") or '"""' in line or "'''" in line
        )

        ratio = comment_lines / total_lines if total_lines > 0 else 0

        # Expecting at least 25% of the file to be comments/docstrings
        self.assertGreater(
            ratio,
            0.25,
            f"j_coupling.py comment ratio is critically low: {ratio*100:.1f}%. Please add more educational explanations.",
        )

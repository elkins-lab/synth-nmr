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
        self.nmr_path = os.path.join(self.base_dir, 'synth_nmr', 'nmr.py')
        self.chemical_shifts_path = os.path.join(self.base_dir, 'synth_nmr', 'chemical_shifts.py')
        self.relaxation_path = os.path.join(self.base_dir, 'synth_nmr', 'relaxation.py')
        self.j_coupling_path = os.path.join(self.base_dir, 'synth_nmr', 'j_coupling.py')
        self.nef_io_path = os.path.join(self.base_dir, 'synth_nmr', 'nef_io.py')
        self.dataset_path = os.path.join(self.base_dir, 'synth_nmr', 'dataset.py')
        self.data_path = os.path.join(self.base_dir, 'synth_nmr', 'data.py')
        self.coupling_path = os.path.join(self.base_dir, 'synth_nmr', 'coupling.py')

    def _check_file_contains(self, filepath, substrings):
        """Helper to assert file contains list of substrings."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = " ".join(f.read().split())
            
        for substring in substrings:
            normalized_substring = " ".join(substring.split())
            self.assertIn(
                normalized_substring, 
                content, 
                f"Missing educational note in {os.path.basename(filepath)}: '{substring[:50]}...'"
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
            "Random Coil", "flexible chain",
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
        """Ensure j_coupling.py retains Karplus note."""
        required_notes = [
            "EDUCATIONAL NOTE - Karplus Equation:",
            "J = A cos^2(theta) + B cos(theta) + C",
        ]
        self._check_file_contains(self.j_coupling_path, required_notes)

    def test_data_educational_notes(self):
        """Ensure data.py retains geometric and amino acid notes."""
        required_notes = [
            "EDUCATIONAL NOTE - Engh & Huber Parameters (The Gold Standard):",
            "Gold Standard",
            "EDUCATIONAL NOTE - Proline Sterics (The \"Proline Effect\"):",
            "structure breaker",
            "EDUCATIONAL NOTE - The \"Mirror Image\" World:",
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

    #def test_dataset_educational_notes(self):
    #    """Ensure dataset.py retains key educational blocks."""
    #    required_notes = [
    #        "EDUCATIONAL NOTE - The Balanced Dataset Problem:",
    #        "Alpha-Helix Trap",
    #        "Halls of Mirrors",
    #        "Data Factory Overview:",
    #]
    #self._check_file_contains(self.dataset_path, required_notes)

    def test_coupling_educational_notes(self):
        """Ensure coupling.py retains Karplus note."""
        required_notes = [
            "Educational Note - The Karplus Equation",
            "depends strongly on the dihedral angle",
            "A * cos^2(theta) + B * cos(theta) + C",
        ]
        self._check_file_contains(self.coupling_path, required_notes)

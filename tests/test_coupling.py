
import pytest
import numpy as np
from synth_pdb.coupling import calculate_hn_ha_coupling, predict_couplings_from_structure

class TestJCoupling:
    """
    TDD Test Suite for J-Coupling Prediction.
    
    Verifies the implementation of the Karplus Equation for 3J(HN-HA) couplings.
    """

    def test_ideal_alpha_helix(self):
        """
        Test J-coupling for an ideal Alpha Helix.
        Phi approx -57 degrees.
        Expected J: Small (< 6.0 Hz).
        """
        phi = -57.0
        j_val = calculate_hn_ha_coupling(phi)
        
        # Textbook expectation: ~3.9 - 5.0 Hz
        assert 2.0 < j_val < 6.0, f"Helix coupling {j_val} out of expected range (2-6 Hz)"

    def test_ideal_beta_sheet(self):
        """
        Test J-coupling for an ideal Beta Sheet.
        Phi approx -139 degrees (parallel) / -119 (antiparallel).
        Expected J: Large (> 8.0 Hz).
        """
        # Beta sheet average ~ -120 to -140
        phi = -120.0
        j_val = calculate_hn_ha_coupling(phi)
        
        # Textbook expectation: ~8.0 - 10.0 Hz
        assert j_val > 7.5, f"Beta sheet coupling {j_val} too small (expected > 7.5 Hz)"

    def test_periodicity(self):
        """Karplus equation should be periodic (360 degrees)."""
        j1 = calculate_hn_ha_coupling(-60.0)
        j2 = calculate_hn_ha_coupling(300.0) # -60 + 360
        assert np.isclose(j1, j2), "Karplus function is not periodic!"

    def test_predict_from_dummy_structure(self):
        """Test bulk prediction function with a dummy inputs."""
        # Mock geometric data (N-residue list of phi angles in degrees)
        # Note: Phi is usually defined for residues 2..N-1 (N-terminus has no Phi).
        # We assume the input is a list of angles where index i corresponds to residue i+1?
        # Or a dictionary? Ideally a dictionary mapping Residue ID -> Phi.
        
        phis = {
            2: -57.0,  # Helix
            3: -120.0, # Sheet
            4: np.nan  # Undefined (e.g. Proline or missing)
        }
        
        couplings = predict_couplings_from_structure(phis)
        
        assert 2.0 < couplings[2] < 6.0
        assert couplings[3] > 7.5
        assert np.isnan(couplings[4])

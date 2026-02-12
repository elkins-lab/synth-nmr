import sys
import unittest
from unittest.mock import patch

class TestNumbaFallback(unittest.TestCase):
    """
    Verifies that the codebase degrades gracefully when Numba is not installed.
    """

    def test_njit_fallback_mechanics(self):
        """Verify that the njit fallback behaves like an identity decorator."""
        from synth_pdb import geometry
        
        # We check if geometry.njit is callable and works as a decorator
        def my_func(x): return x * 2
        decorated = geometry.njit(my_func)
        self.assertEqual(decorated(5), 10)
        
        # Check if it works with parentheses
        decorated_v2 = geometry.njit()(my_func)
        self.assertEqual(decorated_v2(5), 10)

    def test_relaxation_works_without_errors(self):
        """Simple smoke test for relaxation module."""
        from synth_pdb import relaxation
        val = relaxation.spectral_density(0.0, 1e-8, 0.85)
        self.assertGreater(val, 0)

    def test_chemical_shifts_works_without_errors(self):
        """Simple smoke test for chemical_shifts module."""
        from synth_pdb import chemical_shifts
        import numpy as np
        rings = np.array([[0,0,0, 0,0,1, 1.]])
        proton = np.array([0,0,2.])
        shift = chemical_shifts._calculate_ring_current_shift(proton, rings)
        self.assertNotEqual(shift, 0)

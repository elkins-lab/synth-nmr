import importlib
import sys
import unittest
from unittest.mock import patch

import biotite.structure as struc
import numpy as np

from synth_nmr.relaxation import (
    calculate_relaxation_rates,
    predict_order_parameters,
    spectral_density,
)


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
        for _res_id, res_rates in rates.items():
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


def test_spectral_density_tau_f():
    # Test path where tau_f > 0
    from synth_nmr.relaxation import spectral_density

    j1 = spectral_density(600e6, 10e-9, 0.8, tau_f=0.0)
    j2 = spectral_density(600e6, 10e-9, 0.8, tau_f=1e-12)
    assert j2 != j1
    assert j2 > 0.0


def test_njit_fallback(mocker):
    # Test the exception branch where numba is not installed
    import importlib
    import sys

    # Force reload of relaxation module with numba missing
    mocker.patch.dict(sys.modules, {"numba": None})
    import synth_nmr.relaxation

    importlib.reload(synth_nmr.relaxation)

    # Use the mocked njit decorator
    njit_func = synth_nmr.relaxation.njit

    @njit_func
    def dummy_func():
        return 1

    assert dummy_func() == 1

    # Also test `njit` with no args
    assert njit_func()("foo") == "foo"

    # Restore the module to working order so subsequent tests aren't using the mock
    sys.modules.pop("numba", None)
    importlib.reload(synth_nmr.relaxation)


def test_calculate_internal_correlation_time_zero_s2():
    # Placeholder for test removed since _calculate_internal_correlation_time doesn't exist
    pass


def test_calculate_relaxation_rates_zero_s2():
    # Test when calculate_relaxation_rates is given s2=0
    import biotite.structure as struc
    import numpy as np

    from synth_nmr.relaxation import calculate_relaxation_rates

    structure = struc.AtomArray(2)
    structure.res_id = np.array([1, 1])
    structure.res_name = np.array(["ALA", "ALA"])
    structure.atom_name = np.array(["N", "H"])

    rates = calculate_relaxation_rates(structure, 600.0, 10e-9, s2_map={1: 0.0})
    assert 1 in rates
    assert "R1" in rates[1]
    assert "R2" in rates[1]
    assert "NOE" in rates[1]


def test_predict_order_parameters_no_res_ids(mocker):
    import biotite.structure as struc
    import numpy as np

    from synth_nmr.relaxation import predict_order_parameters

    structure = struc.AtomArray(1)
    structure.res_id = np.array([1])
    mocker.patch("numpy.unique", return_value=np.array([]))

    res = predict_order_parameters(structure)
    assert res == {}


def test_predict_order_parameters_typeerror():
    import pytest

    from synth_nmr.relaxation import predict_order_parameters

    with pytest.raises(TypeError):
        predict_order_parameters("Not an AtomArray")


def test_predict_order_parameters_alpha(mocker):
    import biotite.structure as struc
    import numpy as np

    from synth_nmr.relaxation import predict_order_parameters

    structure = struc.AtomArray(2)
    structure.res_id = np.array([1, 2])
    structure.coord = np.random.rand(2, 3)
    structure.res_name = np.array(["ALA", "GLY"])
    structure.atom_name = np.array(["CA", "CA"])
    structure.element = np.array(["C", "C"])

    mocker.patch("synth_nmr.relaxation.get_secondary_structure", return_value=["alpha", "coil"])
    mocker.patch("biotite.structure.get_residue_starts", return_value=np.array([0, 1]))
    # Mock SASA to return fully buried so S2 goes UP
    mocker.patch("synth_nmr.relaxation.struc.sasa", return_value=np.array([0.0, 0.0]))
    mocker.patch("synth_nmr.relaxation._apply_termini_effects", side_effect=lambda a, b, c, x: x)

    res = predict_order_parameters(structure)
    # the alpha base is 0.85, fully buried gets +0.05 -> 0.90
    assert res[1] >= 0.85


def test_calculate_relaxation_rates_pro():
    import biotite.structure as struc
    import numpy as np

    from synth_nmr.relaxation import calculate_relaxation_rates

    structure = struc.AtomArray(2)
    structure.res_id = np.array([1, 1])
    structure.res_name = np.array(["PRO", "PRO"])
    structure.atom_name = np.array(["N", "H"])

    res = calculate_relaxation_rates(structure, s2_map={1: 0.85})
    assert 1 not in res


def test_predict_relaxation_from_structure_no_sasa_fallback(mocker):
    import biotite.structure as struc
    import numpy as np

    from synth_nmr.relaxation import predict_order_parameters

    structure = struc.AtomArray(3)
    structure.res_id = np.array([1, 1, 1])
    structure.res_name = np.array(["ALA", "ALA", "ALA"])
    structure.atom_name = np.array(["N", "CA", "C"])
    structure.coord = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    structure.element = np.array(["N", "C", "C"])

    mocker.patch("biotite.structure.sasa", side_effect=Exception("Mock SASA Error"))

    predictions = predict_order_parameters(structure)
    assert 1 in predictions


def test_predict_s2_from_sasa_invalid_value():
    from synth_nmr.relaxation import _predict_s2_from_sasa

    # Though it doesn't throw, we're ensuring the math handles bounds
    s2 = _predict_s2_from_sasa(2.0, 0.8)
    assert s2 != 0.8


def test_predict_relaxation_multichain(mocker):
    import biotite.structure as struc
    import numpy as np

    from synth_nmr.relaxation import calculate_relaxation_rates

    structure = struc.AtomArray(4)
    structure.res_id = np.array([1, 1, 2, 2])
    structure.res_name = np.array(["ALA", "ALA", "GLY", "GLY"])
    structure.atom_name = np.array(["N", "H", "N", "H"])
    structure.chain_id = np.array(["A", "A", "B", "B"])
    structure.coord = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])

    predictions = calculate_relaxation_rates(structure)
    assert 1 in predictions
    assert 2 in predictions


def test_print_relaxation_report_single_chain(capsys):
    pass  # No print function exists for relaxation


def test_predict_relaxation_from_structure_model_rigid():
    # No predict_relaxation_from_structure exists, so removing
    pass


def test_predict_relaxation_from_structure_model_invalid():
    pass


def test_spectral_density_tau_f_branch():
    from synth_nmr.relaxation import spectral_density

    j = spectral_density(600e6, 10e-9, 0.8, tau_f=1e-12)
    assert j > 0.0


def test_predict_order_parameters_empty_structure():
    import biotite.structure as struc

    from synth_nmr.relaxation import predict_order_parameters

    structure = struc.AtomArray(0)
    assert predict_order_parameters(structure) == {}


def test_calculate_relaxation_rates_type_errors():
    import biotite.structure as struc
    import numpy as np
    import pytest

    from synth_nmr.relaxation import calculate_relaxation_rates

    structure = struc.AtomArray(1)
    structure.res_id = np.array([1])
    structure.res_name = np.array(["ALA"])

    with pytest.raises(TypeError, match="must be a biotite.structure.AtomArray"):
        calculate_relaxation_rates("not_a_structure")

    with pytest.raises(ValueError, match="must be a positive numeric value"):
        calculate_relaxation_rates(structure, field_mhz=-10.0)

    with pytest.raises(ValueError, match="must be a positive numeric value"):
        calculate_relaxation_rates(structure, tau_m_ns=0.0)

    with pytest.raises(TypeError, match="must be a dictionary or None"):
        calculate_relaxation_rates(structure, s2_map="not_a_dict")


def test_calculate_relaxation_rates_empty_structure(caplog):
    import biotite.structure as struc

    from synth_nmr.relaxation import calculate_relaxation_rates

    structure = struc.AtomArray(0)
    rates = calculate_relaxation_rates(structure)
    assert rates == {}
    assert "is empty. Returning no relaxation rates" in caplog.text


def test_calculate_relaxation_rates_divide_by_zero_r1(mocker, caplog):
    import biotite.structure as struc
    import numpy as np

    from synth_nmr.relaxation import calculate_relaxation_rates

    structure = struc.AtomArray(2)
    structure.res_id = np.array([1, 1])
    structure.res_name = np.array(["ALA", "ALA"])
    structure.atom_name = np.array(["N", "H"])

    # Mock spectral density to force r1_val to be exactly 0
    mocker.patch("synth_nmr.relaxation.spectral_density", return_value=0.0)

    rates = calculate_relaxation_rates(structure)
    assert 1 in rates
    assert np.isnan(rates[1]["NOE"])
    assert "R1 value for residue 1 is zero" in caplog.text


def test_calculate_relaxation_rates_exception(mocker):
    import biotite.structure as struc
    import numpy as np
    import pytest

    from synth_nmr.relaxation import calculate_relaxation_rates

    structure = struc.AtomArray(2)
    structure.res_id = np.array([1, 1])
    structure.res_name = np.array(["ALA", "ALA"])
    structure.atom_name = np.array(["N", "H"])

    mocker.patch(
        "synth_nmr.relaxation.predict_order_parameters", side_effect=Exception("Mock Exception")
    )

    with pytest.raises(Exception, match="Mock Exception"):
        calculate_relaxation_rates(structure)


def test_predict_order_parameters_exception(mocker):
    import biotite.structure as struc
    import numpy as np
    import pytest

    from synth_nmr.relaxation import predict_order_parameters

    structure = struc.AtomArray(1)
    structure.res_id = np.array([1])
    structure.res_name = np.array(["ALA"])
    structure.atom_name = np.array(["CA"])

    mocker.patch("biotite.structure.get_residue_starts", side_effect=Exception("Mock Exception"))

    with pytest.raises(Exception, match="Mock Exception"):
        predict_order_parameters(structure)


def test_predict_order_parameters_proline():
    import biotite.structure as struc
    import numpy as np

    from synth_nmr.relaxation import predict_order_parameters

    structure = struc.AtomArray(2)
    structure.res_id = np.array([1, 2])
    structure.res_name = np.array(["ALA", "PRO"])
    structure.atom_name = np.array(["CA", "N"])

    s2_map = predict_order_parameters(structure)

    # Check that residue 2 (PRO) still gets an S2 prediction for order parameters
    assert 2 in s2_map


def test_calculate_relaxation_rates_proline():
    import biotite.structure as struc
    import numpy as np

    from synth_nmr.relaxation import calculate_relaxation_rates

    structure = struc.AtomArray(3)
    structure.res_id = np.array([1, 1, 2])
    structure.res_name = np.array(["ALA", "ALA", "PRO"])
    structure.atom_name = np.array(["N", "H", "N"])

    # Proline lacks HN so relax rates shouldn't be calculated
    rates = calculate_relaxation_rates(structure)

    assert 1 in rates
    assert 2 not in rates


def test_sasa_fallback_nan(mocker):
    import biotite.structure as struc
    import numpy as np

    from synth_nmr.relaxation import predict_order_parameters

    structure = struc.AtomArray(1)
    structure.res_id = np.array([1])
    structure.res_name = np.array(["ALA"])
    structure.atom_name = np.array(["CA"])
    structure.coord = np.array([[0, 0, 0]])

    mocker.patch("biotite.structure.sasa", return_value=np.array([np.nan]))

    # Should fallback the NaN to 50.0 and process successfully
    s2_map = predict_order_parameters(structure)
    assert 1 in s2_map


def test_sasa_fallback_ion(mocker):
    import biotite.structure as struc
    import numpy as np

    from synth_nmr.relaxation import predict_order_parameters

    structure = struc.AtomArray(2)
    structure.res_id = np.array([1, 2])
    structure.res_name = np.array(["ALA", "ZN"])
    structure.atom_name = np.array(["CA", "ZN"])
    structure.coord = np.array([[0, 0, 0], [1, 1, 1]])

    # Should exclude ZN from SASA math but process the main chain correctly
    s2_map = predict_order_parameters(structure)
    assert 1 in s2_map
    assert 2 in s2_map


def test_sasa_fallback_ptm(mocker):
    import biotite.structure as struc
    import numpy as np

    from synth_nmr.relaxation import predict_order_parameters

    structure = struc.AtomArray(2)
    structure.res_id = np.array([1, 1])
    # TPO is a handled phosphothreonine
    structure.res_name = np.array(["TPO", "TPO"])
    structure.atom_name = np.array(["CA", "P"])
    structure.coord = np.array([[0, 0, 0], [1, 1, 1]])

    # Needs to strip P during SASA math
    s2_map = predict_order_parameters(structure)
    assert 1 in s2_map


def test_termini_effects():
    from synth_nmr.relaxation import _apply_termini_effects

    # Middle of sequence, shouldn't change
    assert _apply_termini_effects(10, 1, 20, 0.85) == 0.85
    # Very end of sequence
    assert _apply_termini_effects(1, 1, 20, 0.85) < 0.85


def test_termini_effects_out_of_bounds():
    from synth_nmr.relaxation import _apply_termini_effects

    # Ensure it works when pos somehow negative distance from end
    assert _apply_termini_effects(21, 1, 20, 0.85) < 0.85
    assert _apply_termini_effects(0, 1, 20, 0.85) < 0.85


def test_spectral_density_tau_f_internal():
    from synth_nmr.relaxation import spectral_density

    j = spectral_density(600e6, 10e-9, 0.8, tau_f=1.0)
    assert j > 0.0

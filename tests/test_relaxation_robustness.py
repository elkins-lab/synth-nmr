"""
Tests for the robustness and validation of the relaxation module.
"""

import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import os
from synth_nmr.relaxation import (
    spectral_density,
    predict_order_parameters,
    calculate_relaxation_rates,
)
import logging
from unittest.mock import patch

# Get the directory of the current test file
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def sample_structure():
    """Fixture to load a sample PDB file (Gly-Ala dipeptide)."""
    pdb_path = os.path.join(TEST_DATA_DIR, "test.pdb")
    return strucio.load_structure(pdb_path)


@pytest.fixture
def structure_no_n_atoms():
    """Fixture for a structure missing 'N' atoms."""
    atom1 = struc.Atom(
        coord=[0, 0, 0], atom_name="CA", element="C", res_id=1, res_name="GLY", chain_id="A"
    )
    atom2 = struc.Atom(
        coord=[0, 0, 1], atom_name="H", element="H", res_id=1, res_name="GLY", chain_id="A"
    )
    return struc.array([atom1, atom2])


@pytest.fixture
def structure_no_h_atoms():
    """Fixture for a structure missing 'H' atoms."""
    atom1 = struc.Atom(
        coord=[0, 0, 0], atom_name="N", element="N", res_id=1, res_name="GLY", chain_id="A"
    )
    atom2 = struc.Atom(
        coord=[0, 0, 1], atom_name="CA", element="C", res_id=1, res_name="GLY", chain_id="A"
    )
    return struc.array([atom1, atom2])


# --- Tests for spectral_density (validation logic moved to calling functions) ---


def test_spectral_density_invalid_s2_clip():
    """Test that s2 is clipped to 0-1 range within the function logic (implicitly)."""
    # This is an implicit test that the upstream validation in calculate_relaxation_rates
    # ensures S2 is within range. If S2 is outside, the spectral_density function itself
    # will still produce a numerical result.
    # The clip is happening in predict_order_parameters (0.01, 0.98), so this test
    # mainly ensures the jitted function doesn't crash with "invalid" S2 (e.g. 1.1)
    # if it somehow gets passed.
    j_high_s2 = spectral_density(omega=5e8, tau_m=10e-9, s2=1.1, tau_f=0.0)
    assert np.isfinite(j_high_s2)
    j_low_s2 = spectral_density(omega=5e8, tau_m=10e-9, s2=-0.1, tau_f=0.0)
    assert np.isfinite(j_low_s2)


# --- Tests for predict_order_parameters ---


def test_predict_s2_invalid_structure_input():
    """Test predict_order_parameters with invalid structure input types."""
    with pytest.raises(TypeError, match="Input 'structure' must be a biotite.structure.AtomArray."):
        predict_order_parameters("not_a_structure")
    with pytest.raises(TypeError, match="Input 'structure' must be a biotite.structure.AtomArray."):
        predict_order_parameters(None)


def test_predict_s2_empty_structure():
    """Test predict_order_parameters with an empty AtomArray."""
    empty_structure = struc.AtomArray(0)
    s2_map = predict_order_parameters(empty_structure)
    assert s2_map == {}


# --- Tests for calculate_relaxation_rates ---


def test_calc_rates_invalid_structure_input():
    """Test calculate_relaxation_rates with invalid structure input types."""
    with pytest.raises(TypeError, match="Input 'structure' must be a biotite.structure.AtomArray."):
        calculate_relaxation_rates("not_a_structure")
    with pytest.raises(TypeError, match="Input 'structure' must be a biotite.structure.AtomArray."):
        calculate_relaxation_rates(None)


def test_calc_rates_empty_structure():
    """Test calculate_relaxation_rates with an empty AtomArray."""
    empty_structure = struc.AtomArray(0)
    rates = calculate_relaxation_rates(empty_structure)
    assert rates == {}


def test_calc_rates_invalid_field_mhz():
    """Test calculate_relaxation_rates with invalid field_mhz values."""
    dummy_structure = struc.AtomArray(1)  # Need a non-empty structure for validation to be reached
    dummy_structure.add_annotation("atom_name", dtype="U4")
    dummy_structure.add_annotation("element", dtype="U1")
    dummy_structure.atom_name[:] = "N"
    dummy_structure.element[:] = "N"

    with pytest.raises(ValueError, match="Parameter 'field_mhz' must be a positive numeric value."):
        calculate_relaxation_rates(dummy_structure, field_mhz=0.0)
    with pytest.raises(ValueError, match="Parameter 'field_mhz' must be a positive numeric value."):
        calculate_relaxation_rates(dummy_structure, field_mhz=-100.0)
    with pytest.raises(ValueError, match="Parameter 'field_mhz' must be a positive numeric value."):
        calculate_relaxation_rates(dummy_structure, field_mhz="not_a_float")
    with pytest.raises(ValueError, match="Parameter 'field_mhz' must be a positive numeric value."):
        calculate_relaxation_rates(dummy_structure, field_mhz=None)


def test_calc_rates_invalid_tau_m_ns():
    """Test calculate_relaxation_rates with invalid tau_m_ns values."""
    dummy_structure = struc.AtomArray(1)
    dummy_structure.add_annotation("atom_name", dtype="U4")
    dummy_structure.add_annotation("element", dtype="U1")
    dummy_structure.atom_name[:] = "N"
    dummy_structure.element[:] = "N"

    with pytest.raises(ValueError, match="Parameter 'tau_m_ns' must be a positive numeric value."):
        calculate_relaxation_rates(dummy_structure, tau_m_ns=0.0)
    with pytest.raises(ValueError, match="Parameter 'tau_m_ns' must be a positive numeric value."):
        calculate_relaxation_rates(dummy_structure, tau_m_ns=-10.0)
    with pytest.raises(ValueError, match="Parameter 'tau_m_ns' must be a positive numeric value."):
        calculate_relaxation_rates(dummy_structure, tau_m_ns="not_a_float")
    with pytest.raises(ValueError, match="Parameter 'tau_m_ns' must be a positive numeric value."):
        calculate_relaxation_rates(dummy_structure, tau_m_ns=None)


def test_calc_rates_invalid_s2_map_input():
    """Test calculate_relaxation_rates with invalid s2_map input."""
    dummy_structure = struc.AtomArray(1)
    dummy_structure.add_annotation("atom_name", dtype="U4")
    dummy_structure.add_annotation("element", dtype="U1")
    dummy_structure.atom_name[:] = "N"
    dummy_structure.element[:] = "N"

    with pytest.raises(TypeError, match="Parameter 's2_map' must be a dictionary or None."):
        calculate_relaxation_rates(dummy_structure, s2_map="not_a_dict")
    with pytest.raises(TypeError, match="Parameter 's2_map' must be a dictionary or None."):
        calculate_relaxation_rates(dummy_structure, s2_map=[])


def test_calc_rates_no_nh_atoms(structure_no_n_atoms, structure_no_h_atoms):
    """Test calculate_relaxation_rates with structures missing N or H atoms."""
    rates_no_n = calculate_relaxation_rates(structure_no_n_atoms)
    assert rates_no_n == {}

    rates_no_h = calculate_relaxation_rates(structure_no_h_atoms)
    assert rates_no_h == {}


def test_calc_rates_r1_zero_noe_nan_logging(sample_structure, caplog):
    """
    Test the warning for R1=0 leading to NaN NOE calculation.
    This requires manipulating internal constants to force R1 to zero.
    """
    # This test is tricky because R1 is rarely exactly zero with valid parameters.
    # It tests the safety mechanism rather than a real-world scenario.

    # We will mock `spectral_density` to return values that result in R1 = 0
    # For R1 = d_sq * (j_diff + 3*j_wn + 6*j_sum) + c_sq * j_wn
    # If d_sq and c_sq are both non-zero, it's hard to make R1 exactly 0
    # unless all J terms are zero. Spectral density is always positive.
    # So, we will test the branch where `r1_val == 0` is true if it *could* happen.
    # For now, let's assume it's hard to force R1 to 0 directly.
    # We will instead test that if a `r1_val` *is* 0, NOE becomes NaN and a warning is logged.

    # Mock the internal physics constants to simplify.
    # Temporarily set some constants to make R1 zero for a single residue.
    # This requires reaching into the module's internals, which is generally bad practice,
    # but necessary for this very specific edge case without overcomplicating `spectral_density`.

    # A more realistic test might just check if np.nan values are handled.
    # Given the complexity, for now, we'll check that if r1_val somehow evaluates to 0,
    # the NOE handling and logging works.

    # This test currently won't hit the r1_val == 0 branch in normal calculation
    # due to the nature of spectral densities and constants.
    # The existing code already rounds R1 to 2 decimal places, so it's possible
    # a very small R1 could round to 0.0.

    # For a robust test here, we could mock the spectral_density call or the result of r1_val directly
    # within calculate_relaxation_rates.

    # Let's mock spectral_density to always return 0, which would make R1 = 0.
    with patch("synth_nmr.relaxation.spectral_density", return_value=0.0):
        caplog.set_level(logging.WARNING)
        rates = calculate_relaxation_rates(sample_structure)

        # We expect a warning for each residue where R1 is zero
        assert "R1 value for residue 1 is zero, NOE cannot be calculated." in caplog.text
        assert "R1 value for residue 2 is zero, NOE cannot be calculated." in caplog.text

        # All NOE values should be NaN
        for res_id, res_rates in rates.items():
            assert np.isnan(res_rates["NOE"])

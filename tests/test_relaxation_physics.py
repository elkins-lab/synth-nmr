import math

# These imports will fail initially because the helper functions do not exist yet.
# This follows strict TDD methodology.
from synth_nmr.relaxation import (
    _predict_s2_from_sasa,
    _apply_termini_effects,
    _calculate_dipolar_constant,
    _calculate_csa_constant,
)


def test_predict_s2_from_sasa():
    """
    Test the isolated modulation of S2 based on Relative Solvent Accessible Surface Area.

    Expected Rules:
    - Buried (rel_sasa=0.0): Bonus rigidity (+0.05) added to base.
    - Exposed (rel_sasa=1.0): Penalty flexibility (-0.15) subtracted from base.
    """
    base_s2_helix = 0.85

    # 1. Fully Buried (should gain +0.05)
    s2_buried = _predict_s2_from_sasa(rel_sasa=0.0, base_s2=base_s2_helix)
    assert math.isclose(s2_buried, 0.90, rel_tol=1e-5)

    # 2. Fully Exposed (should lose -0.15)
    s2_exposed = _predict_s2_from_sasa(rel_sasa=1.0, base_s2=base_s2_helix)
    assert math.isclose(s2_exposed, 0.70, rel_tol=1e-5)

    # 3. Half Exposed
    s2_half = _predict_s2_from_sasa(rel_sasa=0.5, base_s2=base_s2_helix)
    expected_half = 0.85 + 0.05 * (1.0 - 0.5) - 0.15 * (0.5)
    assert math.isclose(s2_half, expected_half, rel_tol=1e-5)


def test_apply_termini_effects():
    """
    Test the isolated logic that assigning lower order parameters to N/C termini due to fraying.

    Expected Rules:
    - Residues within 1 position of the start/end should be reduced to 0.50.
    """
    base_s2 = 0.85
    start_res = 1
    end_res = 100

    # N-Terminus (Residue 1)
    assert _apply_termini_effects(1, start_res, end_res, base_s2) == 0.50
    # N-Terminus Fray (Residue 2)
    assert _apply_termini_effects(2, start_res, end_res, base_s2) == 0.50
    # Core Residue (Residue 3) - Should be unchanged
    assert _apply_termini_effects(3, start_res, end_res, base_s2) == 0.85

    # C-Terminus Fray (Residue 99)
    assert _apply_termini_effects(99, start_res, end_res, base_s2) == 0.50
    # C-Terminus (Residue 100)
    assert _apply_termini_effects(100, start_res, end_res, base_s2) == 0.50


def test_calculate_dipolar_constant():
    """
    Test the isolated calculation of the squared Dipolar integration constant (d^2).
    """
    r_nh = 1.02e-10
    d_sq = _calculate_dipolar_constant(r_nh)

    # Expected magnitude check: ~ (1e-7 * 1e-34 * 1e8 * -1e7 / 1e-30)^2 ~ (1e9)^2 = 1e18
    # Exact value depends on precise constants, but should be > 0
    assert d_sq > 0
    assert isinstance(d_sq, float)


def test_calculate_spectral_density_tau_f():
    from synth_nmr.relaxation import spectral_density
    import numpy as np
    omega = 600e6 * 2 * np.pi
    tau_m = 10e-9
    s2 = 0.85
    tau_f = 2e-9
    j = spectral_density(omega, tau_m, s2, tau_f=tau_f)
    assert isinstance(j, float)


def test_calculate_csa_constant():
    """
    Test the isolated calculation of the squared CSA constant (c^2).
    """
    csa_n = -160e-6
    omega_n = -27.126e6 * 14.09  # Approx 600mhz wN
    c_sq = _calculate_csa_constant(csa_n, omega_n)

    # Expected magnitude check: ~ (1e-4 * 1e8)^2 ~ (1e4)^2 = 1e8
    assert c_sq > 0
    assert isinstance(c_sq, float)

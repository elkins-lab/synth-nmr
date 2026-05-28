"""
Tests for scientific validation metrics based on Montelione et al.
"""

import biotite.structure as struc
import numpy as np

from synth_nmr.structure_utils import calculate_c_beta_deviations
from synth_nmr.validation import calculate_cs_r_factor, calculate_dp_score, calculate_rpf_scores


def test_rpf_scores_and_dp_score():
    """
    Test RPF and DP scores with synthetic data.
    """
    # Create two lists of NOEs: predicted and experimental
    # In a perfect match, R=1, P=1, F=1, DP=1
    predicted = [{"seq_1": 1, "atom_1": "H", "seq_2": 10, "atom_2": "HA", "distance": 3.0}]
    experimental = [{"seq_1": 1, "atom_1": "H", "seq_2": 10, "atom_2": "HA", "dist": 5.0}]

    rpf = calculate_rpf_scores(predicted, experimental)
    assert rpf["recall"] == 1.0
    assert rpf["precision"] == 1.0
    assert rpf["f_measure"] == 1.0

    dp = calculate_dp_score(rpf)
    assert dp == 1.0

    # Test mismatch
    experimental_mismatch = [{"seq_1": 1, "atom_1": "H", "seq_2": 20, "atom_2": "HA", "dist": 5.0}]
    rpf_mismatch = calculate_rpf_scores(predicted, experimental_mismatch)
    assert rpf_mismatch["recall"] == 0.0
    assert rpf_mismatch["precision"] == 0.0
    assert rpf_mismatch["f_measure"] == 0.0


def test_cs_r_factor():
    """
    Test Chemical Shift R-factor with the corrected denominator.

    Correct formula: Rcs = sum(|calc - exp|) / sum(|exp - random_coil|)

    Without a res_name_map, the function uses the median CA random-coil
    value across all 20 standard amino acids (~56 ppm) as the fallback.
    """
    predicted = {"A": {1: {"CA": 55.0}, 2: {"CA": 60.0}}}
    reference = {"A": {1: {"CA": 55.5}, 2: {"CA": 60.5}}}

    # Numerator: |55.0 - 55.5| + |60.0 - 60.5| = 0.5 + 0.5 = 1.0
    # Denominator: |55.5 - median_rc_CA| + |60.5 - median_rc_CA|
    #   median_rc_CA ≈ 56 ppm (varies ±1 ppm depending on AA set).
    #   At median = 56: |55.5-56| + |60.5-56| = 0.5 + 4.5 = 5.0
    #   Rcs = 1.0 / 5.0 = 0.20
    # The exact value depends on the median across the 20 standard AAs.
    r_cs = calculate_cs_r_factor(predicted, reference, atom="CA")
    # Allow a wide range that is compatible with any reasonable RC median:
    assert 0.05 < r_cs < 0.50, f"Unexpected Rcs = {r_cs}"
    # And always verify it is less than 1.0 (the old formula gave ~0.0086 which
    # is clearly wrong — it was normalised by the raw 116 ppm CA value).
    assert r_cs < 1.0


def test_c_beta_deviations():
    """
    Test C-beta deviation calculation.
    """
    # Create a minimal residue (ALA)
    # Atoms: N, CA, C, CB
    # We'll create a "perfect" tetrahedral CB and then perturb it.

    atoms = []
    # Simplified coordinates
    atoms.append(
        struc.Atom(coord=[0, 0, 0], atom_name="CA", res_id=1, res_name="ALA", chain_id="A")
    )
    atoms.append(
        struc.Atom(coord=[-1.4, 0, 0], atom_name="N", res_id=1, res_name="ALA", chain_id="A")
    )
    atoms.append(
        struc.Atom(coord=[0.5, 1.4, 0], atom_name="C", res_id=1, res_name="ALA", chain_id="A")
    )

    # Calculate ideal CB direction
    v1 = np.array([-1.4, 0, 0])
    v2 = np.array([0.5, 1.4, 0])
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    cb_dir = -(v1 + v2)
    cb_dir /= np.linalg.norm(cb_dir)
    cb_coord = cb_dir * 1.53

    # Add CB with slight perturbation
    perturbed_cb = cb_coord + np.array([0.1, 0, 0])
    atoms.append(
        struc.Atom(coord=perturbed_cb, atom_name="CB", res_id=1, res_name="ALA", chain_id="A")
    )

    structure = struc.array(atoms)
    deviations = calculate_c_beta_deviations(structure)

    assert 1 in deviations
    assert deviations[1] > 0.05

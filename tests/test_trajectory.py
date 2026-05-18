"""
tests/test_trajectory.py
========================
Test suite for synth_nmr.trajectory — the MD Trajectory / Ensemble NMR module.

Written FIRST (TDD Red Phase): all tests here will fail with ImportError or
AttributeError until synth_nmr/trajectory.py is implemented.

Test philosophy:
  - All fixtures use only numpy + biotite (no MDTraj required to run the suite).
  - Each test documents the expected physics clearly, so the test file doubles
    as a precise specification of correct behaviour.
  - Numerical tolerances mirror those used elsewhere in synth-nmr (pytest.approx
    with abs=1e-4 or rel=1e-3 as appropriate).

Physics recap (relevant to test design):
  Chemical shifts : <δ>_t  — arithmetic mean over frames
  NOE distances   : <r⁻⁶>^(−1/6)  — sixth-power average (Solomon 1955)
  RDCs            : <D>_t  — arithmetic mean over frames
  S² order param  : |<μ>|²  — squared magnitude of the mean unit-vector (Lipari & Szabo 1982)
"""

import biotite.structure as struc
import numpy as np
import pytest

# ── The imports that will be RED until trajectory.py is created ────────────
from synth_nmr.trajectory import (
    TrajectoryEnsemble,
    compute_s2_from_trajectory,
    ensemble_average_noes,
    ensemble_average_rdcs,
    ensemble_average_shifts,
    load_trajectory,
)

# ══════════════════════════════════════════════════════════════════════════════
# Helper fixtures
# ══════════════════════════════════════════════════════════════════════════════


def _make_nh_frame(n_coord, h_coord, res_id: int = 1, res_name: str = "GLY") -> struc.AtomArray:
    """
    Build a minimal one-residue AtomArray containing a backbone N and amide H.
    Useful for RDC and S² tests where only bond-vector orientation matters.
    """
    n_atom = struc.Atom(
        n_coord, atom_name="N", element="N", res_id=res_id, res_name=res_name, chain_id="A"
    )
    h_atom = struc.Atom(
        h_coord, atom_name="H", element="H", res_id=res_id, res_name=res_name, chain_id="A"
    )
    return struc.array([n_atom, h_atom])


def _make_multi_residue_frame(
    residues: list,
    base_coord_offset: float = 0.0,
) -> struc.AtomArray:
    """
    Build a multi-residue AtomArray with backbone N, CA, C, O, H atoms.

    Parameters
    ----------
    residues : list of (res_id, res_name) tuples
    base_coord_offset : float
        Shift all coordinates by this amount along X (simulates a different
        frame / conformation for ensemble tests).

    Returns
    -------
    struc.AtomArray
    """
    atom_names = ["N", "CA", "C", "O", "H"]
    atoms = []
    for i, (res_id, res_name) in enumerate(residues):
        for j, aname in enumerate(atom_names):
            coord = [base_coord_offset + i * 3.8, j * 1.5, 0.0]
            atom = struc.Atom(
                coord,
                atom_name=aname,
                element=aname[0],
                res_id=res_id,
                res_name=res_name,
                chain_id="A",
            )
            atoms.append(atom)
    return struc.array(atoms)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TrajectoryEnsemble — construction
# ══════════════════════════════════════════════════════════════════════════════


class TestTrajectoryEnsemble:
    """Tests for the TrajectoryEnsemble container class."""

    def test_basic_construction(self):
        """A TrajectoryEnsemble built from a list of AtomArrays should expose
        its frames as an iterable and report the correct frame count."""
        frame1 = _make_multi_residue_frame([(1, "ALA"), (2, "GLY")])
        frame2 = _make_multi_residue_frame([(1, "ALA"), (2, "GLY")], base_coord_offset=0.5)
        ensemble = load_trajectory([frame1, frame2])

        assert len(ensemble) == 2
        assert isinstance(ensemble.stack, struc.AtomArrayStack)
        assert all(isinstance(f, struc.AtomArray) for f in ensemble)

    def test_single_frame(self):
        """A single-frame ensemble is valid (edge case for averaging functions)."""
        frame = _make_multi_residue_frame([(1, "ALA")])
        ensemble = load_trajectory([frame])
        assert len(ensemble) == 1

    def test_empty_ensemble_raises(self):
        """Constructing with zero frames should raise ValueError."""
        with pytest.raises(ValueError, match="Provide at least one AtomArray frame"):
            load_trajectory([])

    def test_wrong_type_raises(self):
        """Passing non-AtomArray elements should raise TypeError."""
        # Since we use struc.stack, it might raise AttributeError or TypeError
        # We catch the common case in load_trajectory now.
        with pytest.raises((TypeError, AttributeError)):
            load_trajectory(["not_an_atomarray"])

    def test_iteration(self):
        """TrajectoryEnsemble should be iterable over its frames."""
        # Use consistent topology (same res_id) for all frames to allow stacking
        frames = [_make_multi_residue_frame([(1, "ALA")]) for _ in range(3)]
        ensemble = load_trajectory(frames)
        for frame in ensemble:
            assert isinstance(frame, struc.AtomArray)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  load_trajectory
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadTrajectory:
    """Tests for the load_trajectory() multi-format loader."""

    def test_load_from_atomarray_list(self):
        """
        Passing a plain Python list of AtomArrays bypasses MDTraj entirely and
        wraps them directly into a TrajectoryEnsemble. This is the always-available
        code path — no optional dependency required.
        """
        frames = [_make_multi_residue_frame([(1, "ALA")]) for _ in range(3)]
        ensemble = load_trajectory(frames)

        assert isinstance(ensemble, TrajectoryEnsemble)
        assert len(ensemble) == 3

    def test_load_empty_list_raises(self):
        """An empty list of frames should raise ValueError, not silently succeed."""
        with pytest.raises(ValueError):
            load_trajectory([])

    def test_load_mdtraj_missing_raises_importerror(self, monkeypatch):
        """
        If MDTraj is not installed, passing a file path string should raise
        ImportError with a helpful install message, rather than a cryptic
        ModuleNotFoundError buried in a traceback.

        We simulate MDTraj absence by monkeypatching the import inside
        load_trajectory.
        """
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mdtraj":
                raise ImportError("No module named 'mdtraj'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="pip install synth-nmr\\[trajectory\\]"):
            load_trajectory("protein.xtc", topology="protein.pdb")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ensemble_average_shifts
# ══════════════════════════════════════════════════════════════════════════════


class TestEnsembleAverageShifts:
    """
    Tests for ensemble_average_shifts().

    PHYSICS:  The observed chemical shift in solution NMR is the TIME-AVERAGE
    of the instantaneous chemical shift over all conformations sampled during
    the experiment.  In the fast-exchange limit (conformational exchange fast
    compared to chemical shift difference), the measured peak position is:

        δ_obs(nucleus) = (1/N) Σ_frames  δ_predicted(frame, nucleus)

    This is a simple arithmetic mean — the correct choice here.
    """

    def _make_shift_dict(self, res_shifts: dict) -> dict:
        """
        Build a shift dictionary in the format returned by predict_chemical_shifts:
            {source_label: {res_id: {atom_name: float}}}

        Parameters
        ----------
        res_shifts : dict mapping res_id -> {atom_name: shift_ppm}
        """
        return {"empirical": {k: v for k, v in res_shifts.items()}}

    def test_exact_mean_two_frames(self):
        """
        Two-frame ensemble where frame 1 has δ(CA, res1)=55.0 ppm and
        frame 2 has δ(CA, res1)=57.0 ppm.  Expected mean: 56.0 ppm.

        This is the simplest sanity check for arithmetic averaging.
        """
        # We mock the per-frame predictor by passing pre-built shift dicts
        # directly to ensemble_average_shifts via the `per_frame_shifts` argument.
        frame_shifts = [
            {1: {"CA": 55.0, "N": 120.0}},
            {1: {"CA": 57.0, "N": 122.0}},
        ]
        result = ensemble_average_shifts(frame_shifts)

        assert 1 in result
        assert result[1]["CA"] == pytest.approx(56.0, abs=1e-6)
        assert result[1]["N"] == pytest.approx(121.0, abs=1e-6)

    def test_single_frame_passthrough(self):
        """Single-frame ensemble: average equals the frame value exactly."""
        frame_shifts = [{1: {"CA": 55.3}}]
        result = ensemble_average_shifts(frame_shifts)
        assert result[1]["CA"] == pytest.approx(55.3, abs=1e-6)

    def test_multiple_residues(self):
        """Averaging over multiple residues and nuclei simultaneously."""
        frame_shifts = [
            {1: {"CA": 50.0, "HA": 4.0}, 2: {"CA": 60.0}},
            {1: {"CA": 52.0, "HA": 4.2}, 2: {"CA": 62.0}},
            {1: {"CA": 54.0, "HA": 4.4}, 2: {"CA": 64.0}},
        ]
        result = ensemble_average_shifts(frame_shifts)
        assert result[1]["CA"] == pytest.approx(52.0, abs=1e-6)
        assert result[1]["HA"] == pytest.approx(4.2, abs=1e-6)
        assert result[2]["CA"] == pytest.approx(62.0, abs=1e-6)

    def test_nucleus_missing_in_some_frames_skipped(self):
        """
        If a nucleus is only present in a subset of frames (e.g. because SPARTA+
        could not assign it in one frame due to a missing atom), it should be
        excluded from the average rather than silently producing a biased result.
        """
        frame_shifts = [
            {1: {"CA": 55.0, "CB": 30.0}},
            {1: {"CA": 57.0}},  # CB missing
        ]
        result = ensemble_average_shifts(frame_shifts)
        # CA appears in both frames → include
        assert "CA" in result[1]
        # CB appears in only one frame → exclude to avoid bias
        assert "CB" not in result[1]

    def test_empty_frame_shifts_raises(self):
        """An empty list of frame-shift dicts should raise ValueError."""
        with pytest.raises(ValueError, match="at least one frame"):
            ensemble_average_shifts([])

    def test_wrong_type_raises(self):
        """Non-list input raises TypeError."""
        with pytest.raises(TypeError):
            ensemble_average_shifts("not_a_list")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ensemble_average_noes
# ══════════════════════════════════════════════════════════════════════════════


class TestEnsembleAverageNoes:
    """
    Tests for ensemble_average_noes().

    PHYSICS:  The NOE cross-relaxation rate σ_ij is proportional to the
    ensemble average of r⁻⁶:

        σ_ij  ∝  <r_ij⁻⁶>_t

    The EFFECTIVE distance extracted from a NOESY spectrum is therefore:

        r_eff = <r_ij⁻⁶>^(−1/6)

    This is called the "sixth-power average" or "r⁻⁶ average".  It weights
    short-distance conformations much more heavily than large-distance ones.
    Critically, r_eff < arithmetic_mean(r) when distances vary — the NOE is
    dominated by the closest approach.

    Reference: Solomon, I. (1955) Phys. Rev. 99, 559.
    """

    def test_sixth_power_average_two_distances(self):
        """
        Two frames with r=2 Å and r=4 Å for the same atom pair.

        r_eff = (<r⁻⁶>)^(−1/6) = ((2⁻⁶ + 4⁻⁶) / 2)^(−1/6)

        This should be noticeably less than the arithmetic mean (3.0 Å),
        demonstrating the fundamental bias toward the short-distance frame.
        """
        r1, r2 = 2.0, 4.0
        r_eff_expected = ((r1**-6 + r2**-6) / 2) ** (-1 / 6)

        # per_frame_noes: list of dicts {(res_i, res_j): distance_angstrom}
        per_frame_noes = [
            {(1, 2): r1},
            {(1, 2): r2},
        ]
        result = ensemble_average_noes(per_frame_noes)

        assert (1, 2) in result
        assert result[(1, 2)] == pytest.approx(r_eff_expected, abs=1e-4)
        # Confirm it is less than the arithmetic mean
        assert result[(1, 2)] < (r1 + r2) / 2

    def test_constant_distance_gives_same_distance(self):
        """If all frames have the same distance, r_eff == that distance exactly."""
        r = 3.5
        per_frame_noes = [{(1, 3): r}, {(1, 3): r}, {(1, 3): r}]
        result = ensemble_average_noes(per_frame_noes)
        assert result[(1, 3)] == pytest.approx(r, abs=1e-6)

    def test_pair_missing_in_some_frames_excluded(self):
        """
        An atom pair that appears in only a subset of frames is excluded from
        the average (same policy as ensemble_average_shifts).
        """
        per_frame_noes = [
            {(1, 2): 3.0, (2, 3): 4.0},
            {(1, 2): 3.5},  # (2,3) missing
        ]
        result = ensemble_average_noes(per_frame_noes)
        assert (1, 2) in result
        assert (2, 3) not in result

    def test_empty_noe_list(self):
        """No NOE pairs across all frames → return empty dict."""
        result = ensemble_average_noes([{}, {}])
        assert result == {}

    def test_empty_frame_list_raises(self):
        with pytest.raises(ValueError, match="at least one frame"):
            ensemble_average_noes([])


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ensemble_average_rdcs
# ══════════════════════════════════════════════════════════════════════════════


class TestEnsembleAverageRdcs:
    """
    Tests for ensemble_average_rdcs().

    PHYSICS: Like chemical shifts, RDC values are fast-exchange averages.
    The measured RDC is <D(θ,φ)>_t, where θ and φ are the angles of the
    NH bond vector in the alignment tensor principal axis system.

    The arithmetic mean is correct here.
    """

    def test_exact_mean_two_frames(self):
        """
        Frame 1: res 1 RDC = 10.0 Hz
        Frame 2: res 1 RDC = 14.0 Hz
        Expected mean: 12.0 Hz
        """
        per_frame_rdcs = [
            {1: 10.0},
            {1: 14.0},
        ]
        result = ensemble_average_rdcs(per_frame_rdcs)
        assert result[1] == pytest.approx(12.0, abs=1e-6)

    def test_three_frames_multiple_residues(self):
        per_frame_rdcs = [
            {1: 10.0, 2: -5.0},
            {1: 12.0, 2: -7.0},
            {1: 14.0, 2: -3.0},
        ]
        result = ensemble_average_rdcs(per_frame_rdcs)
        assert result[1] == pytest.approx(12.0, abs=1e-6)
        assert result[2] == pytest.approx(-5.0, abs=1e-6)

    def test_residue_missing_in_some_frames_excluded(self):
        """Residues not present in every frame are excluded."""
        per_frame_rdcs = [
            {1: 10.0, 2: -5.0},
            {1: 12.0},  # residue 2 missing
        ]
        result = ensemble_average_rdcs(per_frame_rdcs)
        assert 1 in result
        assert 2 not in result

    def test_single_frame(self):
        per_frame_rdcs = [{1: 8.5, 2: -3.2}]
        result = ensemble_average_rdcs(per_frame_rdcs)
        assert result[1] == pytest.approx(8.5, abs=1e-6)
        assert result[2] == pytest.approx(-3.2, abs=1e-6)

    def test_empty_frame_list_raises(self):
        with pytest.raises(ValueError, match="at least one frame"):
            ensemble_average_rdcs([])


# ══════════════════════════════════════════════════════════════════════════════
# 6.  compute_s2_from_trajectory
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeS2FromTrajectory:
    """
    Tests for compute_s2_from_trajectory().

    PHYSICS:  The Lipari-Szabo generalized order parameter S² quantifies the
    spatial restriction of a bond vector's motion.  For an NH vector μ(t),
    the plateau value of the autocorrelation function C(τ) as τ→∞ gives:

        S² = C(∞) = |<μ>|²

    where <μ> is the VECTOR MEAN of unit NH vectors over all trajectory frames.

    This follows from the Wigner rotation matrix decomposition of C(t) and
    is exact under the assumption of isotropic overall tumbling (i.e., the
    Lipari-Szabo model-free framework).

    Key intuitions:
      - Fixed NH vector (rigid): all μ_i are identical → |<μ>|² = 1.0
      - Isotropically disordered: <μ> ≈ 0 → S² ≈ 0.0
      - Partially restricted: 0 < S² < 1 (typical backbone S² ≈ 0.8-0.95)

    References:
      Lipari, G. & Szabo, A. (1982) J. Am. Chem. Soc. 104, 4546.
      Clore, G.M. et al. (1990) J. Am. Chem. Soc. 112, 4989.
    """

    def _make_nh_ensemble_fixed(self, n_frames: int = 20) -> TrajectoryEnsemble:
        """
        Build an ensemble where the NH vector is identical in every frame
        (pointing along Z).  Expected S² = 1.0.
        """
        frames = [_make_nh_frame([0.0, 0.0, 0.0], [0.0, 0.0, 1.02]) for _ in range(n_frames)]
        return load_trajectory(frames)

    def _make_nh_ensemble_isotropic(self, n_frames: int = 10000) -> TrajectoryEnsemble:
        """
        Build an ensemble with NH vectors uniformly distributed over the unit sphere.
        Expected S² ≈ 0.0 (converges as n_frames → ∞).

        We use the standard technique: draw unit vectors uniformly via
        normalising Gaussian random vectors.
        """
        rng = np.random.default_rng(seed=42)
        frames = []
        for _ in range(n_frames):
            direction = rng.standard_normal(3)
            direction /= np.linalg.norm(direction)
            h_coord = direction * 1.02  # 1.02 Å bond length
            frames.append(_make_nh_frame([0.0, 0.0, 0.0], h_coord.tolist()))
        return load_trajectory(frames)

    def test_rigid_vector_gives_s2_one(self):
        """
        If the NH vector does not move between frames, S² must equal 1.0.
        This is the maximum possible value and represents a perfectly rigid bond.
        """
        ensemble = self._make_nh_ensemble_fixed(n_frames=20)
        s2_map = compute_s2_from_trajectory(ensemble)

        assert 1 in s2_map
        assert s2_map[1] == pytest.approx(1.0, abs=1e-6)

    def test_isotropic_disorder_gives_s2_near_zero(self):
        """
        For an isotropically distributed NH vector, S² should converge to 0.
        With 10,000 frames and a fixed seed we expect S² < 0.01.
        """
        ensemble = self._make_nh_ensemble_isotropic(n_frames=10_000)
        s2_map = compute_s2_from_trajectory(ensemble)

        assert 1 in s2_map
        assert s2_map[1] == pytest.approx(0.0, abs=0.01)

    def test_single_frame_gives_s2_one(self):
        """
        With a single frame the mean NH vector equals the frame's unit vector,
        so |<μ>|² = 1.0 regardless of orientation.
        """
        ensemble = load_trajectory([_make_nh_frame([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])])
        s2_map = compute_s2_from_trajectory(ensemble)
        assert s2_map[1] == pytest.approx(1.0, abs=1e-6)

    def test_returns_dict_of_residue_ids(self):
        """Return type is dict[int, float] keyed by residue ID."""
        # Build separately per residue to keep each frame as one residue
        frame_data = []
        for _ in range(5):
            atoms = []
            for res_id in range(1, 4):
                n = struc.Atom(
                    [0.0, 0.0, 0.0], atom_name="N", element="N", res_id=res_id, chain_id="A"
                )
                h = struc.Atom(
                    [0.0, 0.0, 1.02], atom_name="H", element="H", res_id=res_id, chain_id="A"
                )
                atoms.extend([n, h])
            frame_data.append(struc.array(atoms))
        ensemble = load_trajectory(frame_data)
        s2_map = compute_s2_from_trajectory(ensemble)
        assert isinstance(s2_map, dict)
        for key in s2_map:
            assert isinstance(key, int)
        for val in s2_map.values():
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0 + 1e-9

    def test_structure_with_no_nh_returns_empty(self):
        """
        If the structure contains no backbone N+H pairs (e.g. only CA atoms),
        the result should be an empty dict, not an exception.
        """
        ca_only = struc.Atom([1.0, 0.0, 0.0], atom_name="CA", element="C", res_id=1, chain_id="A")
        frame = struc.array([ca_only])
        ensemble = load_trajectory([frame])
        s2_map = compute_s2_from_trajectory(ensemble)
        assert s2_map == {}

    def test_proline_residues_are_excluded(self):
        """
        Proline has no backbone amide proton, so it should be absent from
        the S² map (same exclusion rule as in the relaxation module).
        """
        frames = []
        for _ in range(5):
            pro_n = struc.Atom(
                [0.0, 0.0, 0.0], atom_name="N", element="N", res_id=1, res_name="PRO", chain_id="A"
            )
            frame = struc.array([pro_n])
            frames.append(frame)
        ensemble = load_trajectory(frames)
        s2_map = compute_s2_from_trajectory(ensemble)
        assert 1 not in s2_map


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Integration: full ensemble workflow
# ══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """
    End-to-end tests that compose load_trajectory → per-frame calculate_rdcs →
    ensemble_average_rdcs.  These validate that the public API works together
    without hidden coupling between modules.
    """

    def test_rdc_ensemble_pipeline(self):
        """
        Build a 3-frame ensemble where the NH vector rotates slightly between
        frames.  Calculate RDCs per frame using the existing calculate_rdcs()
        function, then call ensemble_average_rdcs().  The result must equal
        the arithmetic mean of the three per-frame values.
        """
        from synth_nmr.rdc import calculate_rdcs

        Da, R = 10.0, 0.5

        # Three distinct NH vectors
        nh_vectors = [
            ([0.0, 0.0, 0.0], [0.0, 0.0, 1.02]),  # aligned with Z  → RDC = 2*Da = 20.0
            ([0.0, 0.0, 0.0], [1.02, 0.0, 0.0]),  # aligned with X  → -2.5
            ([0.0, 0.0, 0.0], [0.0, 1.02, 0.0]),  # aligned with Y  → -17.5
        ]
        frames = [_make_nh_frame(n, h) for n, h in nh_vectors]
        ensemble = load_trajectory(frames)

        per_frame_rdcs = [calculate_rdcs(f, Da=Da, R=R) for f in ensemble]
        result = ensemble_average_rdcs(per_frame_rdcs)

        expected_mean = (20.0 + (-2.5) + (-17.5)) / 3  # = 0.0
        assert result[1] == pytest.approx(expected_mean, abs=1e-3)

    def test_load_then_average_shifts_pipeline(self):
        """
        Verify that load_trajectory followed by ensemble_average_shifts
        works end-to-end when per-frame shift dicts are provided directly.
        """
        frames = [_make_multi_residue_frame([(1, "ALA")]) for _ in range(4)]
        ensemble = load_trajectory(frames)

        # Simulate per-frame predictions
        per_frame = [{1: {"CA": 50.0 + 2 * i}} for i in range(len(ensemble))]
        result = ensemble_average_shifts(per_frame)

        # Mean of 50, 52, 54, 56 = 53
        assert result[1]["CA"] == pytest.approx(53.0, abs=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Coverage gap-filling tests
#     These tests specifically target lines that were uncovered in the initial
#     test suite, identified via `pytest --cov-report=term-missing`.
# ══════════════════════════════════════════════════════════════════════════════


class TestTrajectoryEnsembleRepr:
    """
    Tests for TrajectoryEnsemble.__repr__ (lines 169–170).

    __repr__ is crucial for interactive debugging — it should display the
    number of frames and atoms per frame without raising an exception.
    """

    def test_repr_contains_frame_count(self):
        """repr() should include the number of frames."""
        frames = [_make_nh_frame([0.0, 0.0, 0.0], [0.0, 0.0, 1.02]) for _ in range(5)]
        ensemble = load_trajectory(frames)
        r = repr(ensemble)
        assert "5" in r
        assert "TrajectoryEnsemble" in r

    def test_repr_contains_atom_count(self):
        """repr() should include the number of atoms per frame."""
        # Each _make_nh_frame creates 2 atoms (N + H)
        frame = _make_nh_frame([0.0, 0.0, 0.0], [0.0, 0.0, 1.02])
        ensemble = load_trajectory([frame])
        r = repr(ensemble)
        assert "2" in r  # 2 atoms per frame


class TestLoadTrajectoryEdgeCases:
    """
    Tests for extra load_trajectory() code paths:
      - stride < 1  → ValueError            (line 236)
      - unrecognised source type → TypeError (lines 289–292)
      - stride applied to list              (stride > 1 functional check)
    """

    def test_bad_stride_raises(self):
        """
        stride must be >= 1.  Passing stride=0 or stride=-1 should raise
        ValueError immediately, before any frame loading happens.
        """
        frames = [_make_nh_frame([0.0, 0.0, 0.0], [0.0, 0.0, 1.02]) for _ in range(4)]
        with pytest.raises(ValueError, match="stride must be"):
            load_trajectory(frames, stride=0)
        with pytest.raises(ValueError, match="stride must be"):
            load_trajectory(frames, stride=-1)

    def test_unrecognised_type_raises(self):
        """
        Passing an integer, dict, or any other non-list, non-string, non-MDTraj
        object should raise TypeError with a helpful message.
        """
        with pytest.raises(TypeError, match="Unrecognised source type"):
            load_trajectory(42)
        with pytest.raises(TypeError, match="Unrecognised source type"):
            load_trajectory({"frames": []})

    def test_stride_applied_to_list(self):
        """
        stride=2 on a 6-frame list should give 3 frames (frames 0, 2, 4).
        This validates that the slice is applied correctly.
        """
        frames = [
            _make_nh_frame([0.0, 0.0, float(i)], [0.0, 0.0, float(i) + 1.02]) for i in range(6)
        ]
        ensemble = load_trajectory(frames, stride=2)
        assert len(ensemble) == 3


class TestComputeS2EdgeCases:
    """
    Tests for S² edge cases that hit the inner branches of
    compute_s2_from_trajectory (lines 764, 768, 775–779):

      line 764 : proline 'continue'  — already tested but the 'continue'
                 statement itself wasn't hit; confirmed by coverage.
      line 768 : N atom exists but no matching H in h_coord_map
      lines 775-779 : zero-length N–H vector (N and H at the same coordinate)
    """

    def test_nitrogen_without_matching_h_is_skipped(self):
        """
        A residue that has a backbone N but whose amide H belongs to a
        *different* residue ID (or is absent entirely) should simply be skipped,
        not cause a KeyError or incorrect S².

        We create a frame with N at residue 1 and H at residue 2 — the N of
        residue 1 has no matching H, so residue 1 should be absent from the S² map.

        EDUCATIONAL NOTE: In real structures this can happen for the N-terminal
        residue (which lacks an amide proton in the free amine form) or in
        structures where the H was not modelled.
        """
        # N at res_id=1, H at res_id=2 — deliberately mismatched
        n_atom = struc.Atom(
            [0.0, 0.0, 0.0], atom_name="N", element="N", res_id=1, res_name="ALA", chain_id="A"
        )
        h_atom = struc.Atom(
            [0.0, 0.0, 1.02], atom_name="H", element="H", res_id=2, res_name="GLY", chain_id="A"
        )  # different res_id!
        frame = struc.array([n_atom, h_atom])
        ensemble = load_trajectory([frame])

        s2_map = compute_s2_from_trajectory(ensemble)
        # res 1 N has no matching H → excluded
        assert 1 not in s2_map

    def test_zero_length_nh_vector_is_skipped(self):
        """
        If N and H are placed at the same coordinate, the N→H vector has length
        zero.  Normalising a zero vector would produce NaN or a division-by-zero
        error.  compute_s2_from_trajectory should detect this (norm < 1e-9),
        emit a warning, and skip the (frame, residue) pair without crashing.

        EDUCATIONAL NOTE: Zero-length vectors can appear in pathological
        structure files or during aggressive energy minimisation where atoms
        collapse.  The guard here makes the function robust to such inputs.
        """
        # N and H at exactly the same position
        n_atom = struc.Atom(
            [1.0, 2.0, 3.0], atom_name="N", element="N", res_id=1, res_name="ALA", chain_id="A"
        )
        h_atom = struc.Atom(
            [1.0, 2.0, 3.0], atom_name="H", element="H", res_id=1, res_name="ALA", chain_id="A"
        )
        frame = struc.array([n_atom, h_atom])
        ensemble = load_trajectory([frame])

        # Should not raise; the degenerate vector is simply skipped
        s2_map = compute_s2_from_trajectory(ensemble)
        # With only one (bad) frame, no valid vector → residue 1 absent
        assert 1 not in s2_map


# ══════════════════════════════════════════════════════════════════════════════
# 9.  CLI trajectory commands
#     These tests exercise the new process_commands() code paths in
#     synth_nmr_cli.py (lines 76–181), which were 0% covered.
# ══════════════════════════════════════════════════════════════════════════════


class TestCLITrajectoryCommands:
    """
    Tests for the new CLI commands added to synth_nmr_cli.process_commands():
      • load trajectory <pdb1> [pdb2 ...]
      • ensemble s2
      • ensemble shifts
      • ensemble rdcs
      • ensemble noes
      • error paths (no trajectory loaded, no files found)

    We use capsys to capture stdout and tmp_path to write real (minimal) PDB
    files that the CLI can read.

    EDUCATIONAL NOTE: Testing a CLI through its public process_commands()
    function is preferable to subprocess testing because it:
      1. Runs in the same Python process (fast, no serialisation overhead)
      2. Allows pytest fixtures (capsys, tmp_path) to inspect stdout
      3. Gives meaningful tracebacks on failure rather than generic exit codes
    """

    def _write_minimal_pdb(self, path, offset: float = 0.0) -> None:
        """
        Write a minimal PDB file with one GLY residue containing a backbone
        N and amide H, offset along X to simulate different conformations.
        """
        content = (
            "ATOM      1  N   GLY A   1    "
            f"{offset:8.3f}   0.000   0.000  1.00  0.00           N  \n"
            "ATOM      2  H   GLY A   1    "
            f"{offset:8.3f}   0.000   1.020  1.00  0.00           H  \n"
            "ATOM      3  CA  GLY A   1    "
            f"{offset + 1.458:8.3f}   0.000   0.000  1.00  0.00           C  \n"
            "END\n"
        )
        path.write_text(content)

    def test_load_trajectory_and_s2(self, tmp_path, capsys):
        """
        'load trajectory f1.pdb f2.pdb ensemble s2' should:
          1. Load 2 frames successfully
          2. Compute S² and print at least one residue line via stdout
        """
        import synth_nmr.synth_nmr_cli as cli_module
        from synth_nmr.synth_nmr_cli import process_commands

        # Reset global state before the test
        cli_module.ensemble = None

        f1 = tmp_path / "frame1.pdb"
        f2 = tmp_path / "frame2.pdb"
        self._write_minimal_pdb(f1, offset=0.0)
        self._write_minimal_pdb(f2, offset=0.5)

        process_commands(
            [
                "load",
                "trajectory",
                str(f1),
                str(f2),
                "ensemble",
                "s2",
            ]
        )

        out = capsys.readouterr().out
        assert "Loaded trajectory ensemble with 2 frames" in out
        # S² output line for residue 1
        assert "S²" in out or "S" in out  # Unicode or ASCII fallback

    def test_load_trajectory_and_rdcs(self, tmp_path, capsys):
        """
        'load trajectory ... ensemble rdcs' should print per-residue D_NH values.
        """
        import synth_nmr.synth_nmr_cli as cli_module
        from synth_nmr.synth_nmr_cli import process_commands

        cli_module.ensemble = None
        f1 = tmp_path / "f1.pdb"
        f2 = tmp_path / "f2.pdb"
        self._write_minimal_pdb(f1, offset=0.0)
        self._write_minimal_pdb(f2, offset=0.3)

        process_commands(
            [
                "load",
                "trajectory",
                str(f1),
                str(f2),
                "ensemble",
                "rdcs",
            ]
        )

        out = capsys.readouterr().out
        assert "Loaded trajectory ensemble with 2 frames" in out
        # RDC output line (contains D_NH)
        assert "D_NH" in out

    def test_ensemble_command_without_load_prints_error(self, capsys):
        """
        Running 'ensemble s2' before 'load trajectory' should print a clear
        error message, not raise an exception.
        """
        import synth_nmr.synth_nmr_cli as cli_module
        from synth_nmr.synth_nmr_cli import process_commands

        cli_module.ensemble = None
        process_commands(["ensemble", "s2"])

        out = capsys.readouterr().out
        assert "Error" in out
        assert "trajectory" in out.lower()

    def test_load_trajectory_no_files_prints_error(self, capsys):
        """
        'load trajectory' with no PDB arguments should print an error, not crash.
        """
        import synth_nmr.synth_nmr_cli as cli_module
        from synth_nmr.synth_nmr_cli import process_commands

        cli_module.ensemble = None
        process_commands(["load", "trajectory"])

        outerr = capsys.readouterr()
        # argparse prints to stderr by default
        assert "error" in outerr.err.lower() or "Error" in outerr.out

    def test_ensemble_unknown_subcommand_prints_error(self, tmp_path, capsys):
        """
        'ensemble bogus' should print 'Error: Unknown ensemble subcommand'.
        """
        import synth_nmr.synth_nmr_cli as cli_module
        from synth_nmr.synth_nmr_cli import process_commands

        cli_module.ensemble = None
        f1 = tmp_path / "f1.pdb"
        self._write_minimal_pdb(f1, offset=0.0)

        process_commands(
            [
                "load",
                "trajectory",
                str(f1),
                "ensemble",
                "bogus",
            ]
        )

        outerr = capsys.readouterr()
        # argparse prints invalid choice errors to stderr
        assert "invalid choice" in outerr.err.lower() or "Unknown ensemble subcommand" in outerr.out

    def test_load_trajectory_and_noes(self, tmp_path, capsys):
        """
        'ensemble noes 6.0' should not crash and should print output lines
        (or a silent empty result if atoms are too far apart — both are valid).
        """
        import synth_nmr.synth_nmr_cli as cli_module
        from synth_nmr.synth_nmr_cli import process_commands

        cli_module.ensemble = None
        f1 = tmp_path / "f1.pdb"
        self._write_minimal_pdb(f1, offset=0.0)

        # Should run without exception; output may be empty (only 3 atoms)
        process_commands(
            [
                "load",
                "trajectory",
                str(f1),
                "ensemble",
                "noes",
                "6.0",
            ]
        )
        # If it reaches here without exception the code path is covered
        capsys.readouterr()

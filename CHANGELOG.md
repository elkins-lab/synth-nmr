# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.2] - 2026-03-04

### Fixed

- **`synth_nmr/relaxation.py`** — `calculate_relaxation_rates()`: Heteronuclear NOE
  values were identical for every residue (a flat horizontal line), regardless of
  backbone flexibility.  Root cause: when the fast internal motion timescale `tau_f`
  defaults to 0, the order parameter S² cancels exactly in the ratio of the
  cross-relaxation rate to R₁, making NOE a function of τ_m and field strength only.
  Fix: a per-residue `tau_f` is now derived from S² using the heuristic
  `tau_f = (1 − S²) × 500 ps + 50 ps` (range 50–550 ps), consistent with
  MD-derived backbone timescales.  Flexible residues (low S²) now correctly produce
  lower HetNOE values (approaching 0 and negative) while rigid secondary-structure
  elements (high S²) give HetNOE ≈ 0.5–0.8, matching experimental profiles.
  Reference: Lipari & Szabo (1982) *J Am Chem Soc* **104**:4546.

## [0.7.0] - 2026-03-03


### Added

- **MD Trajectory / Ensemble NMR** (`synth_nmr/trajectory.py`): New module for
  multi-frame NMR analysis.  Features:
    - `load_trajectory()` — load NMR ensembles or MD trajectories from a list of
      `AtomArray` frames (or MDTraj `Trajectory` objects).
    - `compute_s2_from_trajectory()` — backbone S² order parameters via
      Lipari-Szabo vector autocorrelation ($S^2 = |\langle\hat{\mu}\rangle|^2$).
    - `ensemble_average_shifts()` — arithmetic mean across conformers (fast exchange).
    - `ensemble_average_noes()` — $r^{-6}$-weighted effective distances
      ($r_\text{eff} = \langle r^{-6}\rangle^{-1/6}$).
    - `ensemble_average_rdcs()` — arithmetic mean RDC over ensemble.
    - Full test suite in `tests/test_trajectory.py` (45 tests, all passing).
- **SHIFTX2 integration** (`synth_nmr/chemical_shifts.py`): `predict_chemical_shifts()`
  now uses SHIFTX2 (Han et al. 2011) when available in PATH, achieving ~0.44 ppm Cα
  RMSD vs ~0.9–1.1 ppm for the built-in SPARTA+ model.  Automatic fallback to SPARTA+
  with a logged warning if SHIFTX2 is absent or crashes.  New `ShiftX2Predictor` class
  available for direct use.
- **New tutorial**: `docs/tutorials/ensemble_nmr_analysis.ipynb` — 8-cell Google Colab
  notebook demonstrating S², ensemble-averaged shifts, NOE r⁻⁶ vs arithmetic mean, and
  RDC ensemble averaging using the 1D3Z 10-conformer Ubiquitin NMR ensemble.
- **New science doc**: `docs/science/shiftx2.md` — comprehensive guide to SHIFTX2
  integration (accuracy comparison table, installation instructions, API reference,
  fallback logic diagram, and automated test coverage table).

### Changed

- **`synth_nmr_cli.py`**: CLI `ensemble` commands now correctly merge the nested
  `{method: {res_id: {atom: shift}}}` dict from `predict_chemical_shifts()` before
  passing to `ensemble_average_shifts()`.  Added `Dict` and `Tuple` to typing imports.
  Added `# type: ignore[attr-defined]` for `noe_dict.items()` where mypy cannot
  resolve the type.
- **`docs/tutorials/advanced_observables.ipynb`**: Upgraded with improved
  visualisations (J-coupling coloured by secondary structure, NOE dual-panel figure
  showing pair count vs. r⁻⁶ signal power, RDC bar chart with sign colouring) and a
  new Section 5 — ensemble NOE effects with r_eff vs r_arith scatter and per-residue
  error bar chart.
- **`docs/tutorials/relaxation_analysis.ipynb`**: Upgraded to a 4-panel relaxation
  figure (R₁, R₂, hetNOE, S²) and a new Section 5 comparing heuristic S² against
  trajectory-derived S² from the 10-conformer ensemble, with Pearson correlation plot.
- **`mkdocs.yml`**: Added Ensemble NMR Analysis tutorial and SHIFTX2 Integration
  doc page to site navigation.
- **`README.md`**: Added PyPI version, Python versions, Ruff, Mypy, and License
  badges; added fourth Colab badge for Ensemble NMR Analysis tutorial; added
  `pip install synth-nmr[trajectory]` install variant; added MD Trajectory /
  Ensemble NMR to Features list.

### Fixed

- **`synth_nmr/synth_nmr_cli.py`** (ruff E713): `not args[j].lower() in (...)` →
  `args[j].lower() not in (...)`.
- **`tests/test_trajectory.py`** (ruff F841): Removed unused `frames` list
  comprehension in `test_returns_dict_of_residue_ids`.
- **`synth_nmr/trajectory.py`** (mypy no-redef): Renamed local variable `frames` to
  `mdtraj_frames` in MDTraj object loading branch.
- **Tutorial notebooks**: Fixed `AttributeError` in ensemble NOE cells —
  `calculate_synthetic_noes()` returns a `list` of dicts, not a nested dict; replaced
  incorrect `raw.items()` iteration with a `for entry in raw:` loop.



## [0.6.1] - 2026-02-21

### Changed
- **Numpy Compatibility**: Relaxed numpy dependency pin to `<3.0` to resolve binary incompatibilities (`numpy.dtype size changed`) encountered when installed alongside `numpy 2.x`.

# SHIFTX2 Integration

SHIFTX2 is a state-of-the-art chemical shift prediction tool that combines a
hybrid machine-learning / empirical physics approach to predict backbone and
side-chain NMR chemical shifts from 3D protein coordinates.  When installed,
`synth-nmr` uses it automatically as the **primary** prediction engine, falling
back to its built-in SPARTA+ empirical model when SHIFTX2 is unavailable.

---

## Why SHIFTX2?  The Accuracy Hierarchy

Chemical shift prediction from structure is a hard problem.  Three broad approaches exist,
in increasing order of accuracy:

| Method | Basis | Typical Cα RMSD | Used in `synth-nmr` |
|---|---|---|---|
| **Random coil + secondary structure offsets** | Amino acid type + Ramachandran bin | ~2–3 ppm | Built-in fallback (SPARTA+ style) |
| **Database / fragment mining** | Nearest-neighbour search in known PDB+BMRB pairs | ~0.9–1.1 ppm | SPARTA+ (Han et al. 2011) |
| **SHIFTX2** | Hybrid ML + physics (ensemble of random forests, SVMs, ANNs) | **~0.44 ppm Cα** | **Primary (auto-detected)** |

The ~2× improvement in Cα RMSD between fragment-mining and SHIFTX2 is not merely
academic.  For structure validation, a 0.5 ppm artefact can mask or manufacture
secondary chemical shift signatures.  SHIFTX2's per-atom models also cover
side-chain nuclei (Hβ, Cγ, etc.) that simpler methods typically approximate
or omit entirely.

### What makes SHIFTX2 different?

SHIFTX2 (Han et al., 2011) trains separate machine-learning models for each heavy-atom
+ proton type (HA, CA, CB, C', N, HN …) using a curated reference database of
proteins with both known high-resolution structures and experimentally assigned
chemical shifts (the BMRB).  Each model is a committee of learners trained on
descriptor features that encode:

- Backbone torsion angles (φ, ψ, χ₁, χ₂)
- Hydrogen bond geometry (donor/acceptor distance and angle)
- Ring current effects (explicit geometry, not just intensity factors)
- Solvent accessibility (SASA)
- Neighbouring residue identity (i±1, i±2)

This is qualitatively richer than the SPARTA+ implementation in `synth-nmr`, which
applies per-secondary-structure mean offsets and a point-dipole ring current model.

---

## Automatic Detection and Fallback

`predict_chemical_shifts()` in `synth-nmr` follows this logic every time it is called:

```
┌─────────────────────────────────────────────────┐
│  predict_chemical_shifts(structure)              │
│                                                 │
│  1. Locate 'shiftx2.py' in:                     │
│     a. System PATH                              │
│     b. SHIFTX2_DIR environment variable         │
│     c. Typical locations (~/shiftx2/, /opt/...) │
│                                                 │
│  2. If found → run SHIFTX2 subprocess           │
│            └─ non-empty output? → return it ✓   │
│            └─ empty / crash?    → ⚠ log warning  │
│                                                 │
│  3. Fallback: predict_empirical_shifts()         │
│     (SPARTA+ offsets + ring current model)      │
└─────────────────────────────────────────────────┘
```

This design means:

- **Zero configuration required** — if SHIFTX2 is not installed, everything still
  works, just with ~2× higher shift RMSDs.
- **No silent failure** — a `logging.WARNING` is emitted whenever the fallback fires,
  so the user is never unaware of which engine is running.
- **Crash-safe** — if SHIFTX2 exits with a non-zero return code, the exception is
  caught and logged; the empirical model takes over.

---

## Installing SHIFTX2

SHIFTX2 is free for non-commercial use.  Two installation routes:

### Option 1 — Direct download (recommended)

1. Go to [http://www.shiftx2.ca/download.html](http://www.shiftx2.ca/download.html)
2. Download the version matching your OS (Linux/macOS).
3. Unpack the archive and make the script executable:

```bash
chmod +x shiftx2.py
```

4. Configure the location (choose one):

```bash
# A. Add to PATH in your shell profile
export PATH="/path/to/shiftx2_dir:$PATH"

# B. OR set the SHIFTX2_DIR environment variable
export SHIFTX2_DIR="/path/to/shiftx2_dir"

# C. OR move it to a typical location that synth-nmr searches automatically
mkdir -p ~/shiftx2
mv shiftx2.py ~/shiftx2/
```

5. Verify installation:

```bash
shiftx2.py --help
```

### Option 2 — SBGrid (academic institutions)

If your institution is an SBGrid subscriber:

```bash
sbgrid-cli install shiftx2
```

After installation, `shiftx2.py` will be available in the SBGrid PATH automatically.

---

## Verifying the Integration

```python
from synth_nmr.chemical_shifts import ShiftX2Predictor

predictor = ShiftX2Predictor()
print("SHIFTX2 available:", predictor.is_available())
print("Using executable at:", predictor.executable)
```

To verify end-to-end:

```python
import logging
import biotite.structure.io as strucio
from synth_nmr import predict_chemical_shifts

logging.basicConfig(level=logging.INFO)   # so you can see which engine is used

structure = strucio.load_structure("protein.pdb")
shifts = predict_chemical_shifts(structure)
```

The log output will say either:

```
INFO  synth_nmr.chemical_shifts: Successfully predicted chemical shifts using SHIFTX2.
```

or:

```
WARNING  synth_nmr.chemical_shifts: SHIFTX2 executable not found. Falling back to empirical SPARTA+ model.
         To use SHIFTX2, ensure it is in your PATH or set the SHIFTX2_DIR environment variable.
```

---

## Using SHIFTX2 Directly

You can also call the `ShiftX2Predictor` class directly if you want the raw SHIFTX2
output without the automatic fallback:

```python
from synth_nmr.chemical_shifts import ShiftX2Predictor
import biotite.structure.io as strucio

structure = strucio.load_structure("protein.pdb")

predictor = ShiftX2Predictor()           # default: looks for 'shiftx2.py' in PATH
# predictor = ShiftX2Predictor(executable="/opt/shiftx2/shiftx2.py")  # custom path

if predictor.is_available():
    shifts = predictor.predict(structure)
    # shifts: {"A": {1: {"CA": 52.3, "N": 122.1, ...}, 2: {...}, ...}}
else:
    print("SHIFTX2 not installed — see docs for setup instructions")
```

The returned dictionary format is identical to `predict_empirical_shifts()`, so all
downstream analysis functions (`calculate_csi()`, `ensemble_average_shifts()`, etc.)
work unchanged.

---

## Custom Executable Path

If `shiftx2.py` is not on your PATH or in a standard location, you can set the `SHIFTX2_DIR` environment variable to the directory containing `shiftx2.py`.

Alternatively, you can pass the path explicitly in code:

```python
predictor = ShiftX2Predictor(executable="/path/to/shiftx2.py")
```

---

## How `synth-nmr` Calls SHIFTX2

Under the hood, `ShiftX2Predictor.predict()` does exactly what you would do manually:

1. **Writes a temporary PDB** from the biotite `AtomArray` to a `tempfile.TemporaryDirectory`.
2. **Runs** `shiftx2.py -i input.pdb` as a subprocess.
3. **Parses** the `.cs` CSV output file that SHIFTX2 writes next to the input file.
4. **Returns** a `{chain: {res_id: {atom: shift}}}` dict and deletes the temp directory.

The subprocess output (stdout/stderr) is captured; a non-zero exit code raises
`RuntimeError` and triggers the fallback in `predict_chemical_shifts()`.

---

## Automated Test Coverage

The integration is tested without requiring SHIFTX2 to be installed, using `pytest-mock`
to simulate every branch:

| Test | What is verified |
|---|---|
| `test_shiftx2_is_available_mocked` | `is_available()` returns `True` when `shutil.which` finds the executable |
| `test_shiftx2_predict_not_available` | `predict()` raises `RuntimeError` when executable is missing |
| `test_shiftx2_predict_subprocess_error` | `predict()` raises `RuntimeError` on non-zero subprocess exit |
| `test_shiftx2_parse_output_missing_file` | `_parse_output()` raises `FileNotFoundError` for absent output |
| `test_shiftx2_parse_output_bad_format` | Malformed CSV lines are silently skipped |
| `test_predict_chemical_shifts_shiftx2_success` | `predict_chemical_shifts()` returns SHIFTX2 result when available |
| `test_predict_chemical_shifts_shiftx2_empty` | Falls back to SPARTA+ when SHIFTX2 returns empty dict |
| `test_predict_chemical_shifts_shiftx2_exception` | Falls back to SPARTA+ when `predict()` raises any exception |
| `test_shiftx2_predictor_predict_success` | Full subprocess + output parsing with realistic mocked CSV |

All tests pass without SHIFTX2 installed and run in CI on every pull request.

---

## Reference

> Han, B., Liu, Y., Ginzinger, S., & Wishart, D. (2011).
> **SHIFTX2: significantly improved protein chemical shift prediction.**
> *Journal of Biomolecular NMR*, 50(1), 43–57.
> [doi:10.1007/s10858-011-9478-4](https://doi.org/10.1007/s10858-011-9478-4)

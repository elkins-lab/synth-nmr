# Getting Started

Welcome to `synth-nmr`! This guide will get you up and running with calculating NMR observables from your structural models.

## Installation

`synth-nmr` requires Python 3.8 or greater and can be installed via `pip`:

```bash
pip install synth-nmr
```

### Performance Acceleration (Optional but Recommended)

For significant performance improvements on large structural ensembles (especially when calculating bulk distance restraints and relaxation rates), install the `performance` extra. This enables Numba Just-In-Time (JIT) compilation for the core physics kernels.

```bash
pip install "synth-nmr[performance]"
```

## Command-Line Interface (CLI)

`synth-nmr` provides a robust, interactive command-line interface perfect for quick data inspection.

You can launch the tool in **interactive mode**:

```bash
python -m synth_nmr.synth_nmr_cli
```

Once inside, you will be greeted by the interactive prompt:

```text
SynthNMR> read pdb 1UBQ.pdb
Loaded structure from 1UBQ.pdb (1231 atoms)
SynthNMR> calculate rdc 10.0 0.5
Predicted RDCs for 76 residues.
SynthNMR> predict shifts
Predicted chemical shifts for 76 residues.
SynthNMR> exit
```

### Non-Interactive (Scripting) Mode

For automated bash pipelines, you can pass commands directly as arguments:

```bash
# Calculate J-Couplings directly and write to standard output
python -m synth_nmr.synth_nmr_cli read pdb protein.pdb calculate j-coupling
```

## Quick Start (Python API)

If you're integrating `synth-nmr` directly into your ML training loop or analysis scripts, import the core functions:

```python
import biotite.structure.io as strucio
from synth_nmr import calculate_relaxation_rates, calculate_synthetic_noes

# 1. Load your protein structure (must contain Hydrogens)
structure = strucio.load_structure("model.pdb")

# 2. Extract distance restraints (NOEs)
noes = calculate_synthetic_noes(structure, cutoff=5.0)
print(f"Generated {len(noes)} distance restraints.")

# 3. Predict Lipari-Szabo Relaxation Rates
rates = calculate_relaxation_rates(
    structure,
    field_mhz=600.0,
    tau_m_ns=5.0
)
# Returns a dictionary, keyed by residue ID, containing R1, R2, S2, and NOE.
```

For detailed physical theory regarding what these functions calculate, proceed to the **[Scientific Background](science/index.md)**.

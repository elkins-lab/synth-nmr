# synth-nmr

[![Tests](https://github.com/elkins-lab/synth-nmr/actions/workflows/test.yml/badge.svg)](https://github.com/elkins-lab/synth-nmr/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/elkins-lab/synth-nmr/branch/master/graph/badge.svg)](https://codecov.io/gh/elkins-lab/synth-nmr)
[![PyPI version](https://img.shields.io/pypi/v/synth-nmr)](https://pypi.org/project/synth-nmr/)
[![Python versions](https://img.shields.io/pypi/pyversions/synth-nmr)](https://pypi.org/project/synth-nmr/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Mypy](https://img.shields.io/badge/type--checked-mypy-blue)](http://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-MkDocs-blue)](https://elkins-lab.github.io/synth-nmr/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19341451-blue)](https://doi.org/10.5281/zenodo.19341451)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/13129/badge)](https://www.bestpractices.dev/projects/13129)

<img src="https://raw.githubusercontent.com/elkins-lab/synth-nmr/master/images/NOE_Avenue.jpg" alt="NOE Avenue" width="50%">

**NMR spectroscopy calculations for protein structures**

A lightweight, standalone Python package for calculating NMR observables from protein structures. Originally extracted from the [synth-pdb](https://github.com/elkins-lab/synth-pdb) package to provide a focused toolkit that works with any protein structure source.

**[Read the full documentation here!](https://elkins-lab.github.io/synth-nmr/)**

## Features

- **NOE Calculations**: Synthetic NOE distance restraints
- **Relaxation Rates**: R1, R2, and heteronuclear NOE predictions
- **Chemical Shifts**: Integrates to SHIFTX2 predictions with SPARTA+ empirical fallback
- **J-Couplings**: Scalar coupling prediction using the Karplus equation with Vuister & Bax (1993) parameters.
- **RDC Calculations**: Prediction of residual dipolar couplings
- **MD Trajectory / Ensemble NMR**: S², ensemble-averaged shifts, NOEs (r⁻⁶), and RDCs from NMR ensembles or MD trajectories
- **NEF I/O**: Read and write NMR Exchange Format files
- **Secondary Structure**: Automatic classification for enhanced predictions

## Interactive Tutorials

Try out `synth-nmr` immediately in your browser using Google Colab!

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/synth-nmr/blob/master/examples/interactive_tutorials/basic_nmr_prediction.ipynb) **Basic NMR Prediction**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/synth-nmr/blob/master/examples/interactive_tutorials/advanced_observables.ipynb) **Advanced Observables: J-Couplings, NOEs, and RDCs**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/synth-nmr/blob/master/examples/interactive_tutorials/relaxation_analysis.ipynb) **Relaxation & Dynamics Analysis**
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins-lab/synth-nmr/blob/master/examples/interactive_tutorials/ensemble_nmr_analysis.ipynb) **Ensemble NMR Analysis: Proteins in Motion**


## Installation

```bash
pip install synth-nmr
```

For improved performance with JIT compilation:
```bash
pip install synth-nmr[performance]
```

For MD trajectory / ensemble NMR analysis:
```bash
pip install synth-nmr[trajectory]
```

## Command-Line Interface

`synth-nmr` provides a command-line interface for common tasks, allowing you to perform calculations directly from your terminal.

### Usage

You can run `synth-nmr` CLI commands directly or enter an interactive mode.

#### Non-Interactive Mode

Execute commands by passing them as arguments to the `synth_nmr.synth_nmr_cli` module:

```bash
python -m synth_nmr.synth_nmr_cli <command> [arguments]
```

**Examples:**

1.  **Read a PDB file and calculate RDCs:**
    ```bash
    python -m synth_nmr.synth_nmr_cli read pdb protein.pdb calculate rdc 10.0 0.5
    ```

2.  **Read a PDB file and predict chemical shifts:**
    ```bash
    python -m synth_nmr.synth_nmr_cli read pdb protein.pdb predict shifts
    ```

3.  **Read a PDB file and calculate J-couplings:**
    ```bash
    python -m synth_nmr.synth_nmr_cli read pdb protein.pdb calculate j-coupling
    ```

#### Interactive Mode

To enter interactive mode, run the CLI without any arguments:

```bash
python -m synth_nmr.synth_nmr_cli
```

Once in interactive mode, you will see a `SynthNMR>` prompt. Type `help` to see available commands:

```
SynthNMR> help
Commands:
  read pdb <filename>
  calculate rdc [Da] [R]
  predict shifts
  calculate j-coupling
  exit
SynthNMR> read pdb protein.pdb
SynthNMR> calculate rdc 10.0 0.5
SynthNMR> exit
```

### Available Commands

-   `read pdb <filename>`: Loads a protein structure from the specified PDB file. This command must be executed before any calculation commands.
-   `calculate rdc [Da] [R]`: Calculates Residual Dipolar Couplings.
    -   `Da`: (Optional) Axial component of the alignment tensor in Hz (default: 10.0).
    -   `R`: (Optional) Rhombicity of the alignment tensor (dimensionless) (default: 0.5).
-   `predict shifts`: Predicts chemical shifts using SPARTA+ with ring current corrections.
-   `calculate j-coupling`: Calculates ³J(HN-Hα) couplings using the Karplus equation.
-   `help`: (Interactive mode only) Displays a list of available commands.
-   `exit`: (Interactive mode only) Exits the CLI.

## Quick Start

```python
import biotite.structure.io as strucio
from synth_nmr import (
    calculate_synthetic_noes,
    calculate_relaxation_rates,
    predict_chemical_shifts,
    calculate_hn_ha_coupling,
    calculate_rdcs
)

# Load a protein structure
structure = strucio.load_structure("protein.pdb")

# Calculate NOEs
noes = calculate_synthetic_noes(structure, cutoff=5.0)

# Predict relaxation rates
relaxation = calculate_relaxation_rates(
    structure,
    field_strength=600.0,  # MHz
    temperature=298.0,      # K
    correlation_time=5.0    # ns
)

# Predict chemical shifts
shifts = predict_chemical_shifts(structure)

# Calculate J-couplings
j_couplings = calculate_hn_ha_coupling(structure)

# Predict RDCs
rdcs = calculate_rdcs(
    structure,
    Da=10.0, # Axial component of alignment tensor (Hz)
    R=0.5    # Rhombic component of alignment tensor
)
```

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- Biotite ≥ 0.35.0
- Numba ≥ 0.55.0 (optional, for performance)

## Documentation

### Core Functions

#### `calculate_synthetic_noes(structure, cutoff=5.0)`
Calculate synthetic NOE distance restraints.

**Parameters:**
- `structure`: biotite AtomArray
- `cutoff`: Distance cutoff in Ångströms (default: 5.0)

**Returns:** Dictionary of NOE restraints

#### `calculate_relaxation_rates(structure, field_strength, temperature, correlation_time)`
Predict NMR relaxation rates (R1, R2, heteronuclear NOE).

**Parameters:**
- `structure`: biotite AtomArray
- `field_strength`: Spectrometer frequency in MHz
- `temperature`: Temperature in Kelvin
- `correlation_time`: Molecular correlation time in nanoseconds

**Returns:** Dictionary of relaxation rates per residue

#### `predict_chemical_shifts(structure)`
Predict chemical shifts using SPARTA+ with ring current corrections.

**Parameters:**
- `structure`: biotite AtomArray

**Returns:** Dictionary of chemical shifts by residue and atom type

#### `calculate_hn_ha_coupling(structure)`
Calculate ³J(HN-Hα) couplings using the Karplus equation.

**Parameters:**
- `structure`: biotite AtomArray

**Returns:** Dictionary of J-coupling values per residue

#### `calculate_rdcs(structure, Da, R)`
Predict residual dipolar couplings (RDCs) for backbone N-H vectors.

**Parameters:**
- `structure`: biotite AtomArray
- `Da`: Axial component of the alignment tensor in Hz
- `R`: Rhombicity of the alignment tensor (dimensionless)

**Returns:** Dictionary of RDC values per residue

## Use Cases

- **Structure Validation**: Compare predicted vs experimental NMR data
- **MD Analysis**: Calculate NMR observables from molecular dynamics trajectories
- **Protein Design**: Predict NMR properties of designed structures
- **Data Integration**: Generate synthetic NMR data for machine learning

## Compatibility

Works with protein structures from any source:
- PDB files
- AlphaFold predictions
- Molecular dynamics simulations
- De novo structure generation (e.g., synth-pdb)

## Citation

If you use synth-nmr in your research, please cite:

```bibtex
@software{synth_nmr,
  author = {Elkins, George},
  title = {synth-nmr: NMR spectroscopy calculations for protein structures},
  year = {2026},
  doi = {10.5281/zenodo.18855129},
  url = {https://github.com/elkins-lab/synth-nmr}
}
```

## License

MIT License - see LICENSE file for details

## Related Projects

- [synth-pdb](https://github.com/elkins-lab/synth-pdb) — Core synthetic protein structure generator
- [synth-saxs](https://github.com/elkins-lab/synth-saxs) — SAXS profile simulator (use with synth-dynamics ensembles)
- [synth-cryo-em](https://github.com/elkins-lab/synth-cryo-em) — Cryo-EM map simulator
- [synth-dynamics](https://github.com/elkins-lab/synth-dynamics) — ANM/Langevin dynamics for NMR ensemble averaging
- [diff-biophys](https://github.com/elkins-lab/diff-biophys) — Differentiable JAX versions of all NMR kernels
- [diff-ensemble](https://github.com/elkins-lab/diff-ensemble) — IDP structural ensemble prediction with NMR restraints
- [Biotite](https://www.biotite-python.org/) — Computational biology toolkit

## References

This package relies on the following peer-reviewed research:

- **SPARTA+**: For chemical shift predictions.
  > Yang, Y., & Bax, A. (2011). *Journal of Biomolecular NMR*, 51(3), 259–274.

- **Karplus Equation**: For J-coupling calculations.
  > Karplus, M. (1959). *The Journal of Chemical Physics*, 30(1), 11–15.

- **NMR Relaxation**: The underlying theory for relaxation rate predictions.
  > Lipari, G., & Szabo, A. (1982). *Journal of the American Chemical Society*, 104(17), 4546–4559.

- **Residual Dipolar Couplings**: Seminal work on applying RDCs to proteins.
  > Bax, A., & Tjandra, N. (1997). *Journal of the American Chemical Society*, 119(49), 12041-12042.

- **Nuclear Overhauser Effect**: Foundational experimental observation.
  > Solomon, I. (1955). *Physical Review*, 99(2), 559.

- **2D NOESY**: Development of two-dimensional NOE spectroscopy for biomolecules.
  > Kumar, A., Ernst, R. R., & Wüthrich, K. (1980). *Biochemical and Biophysical Research Communications*, 95(1), 1-6.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please use the [GitHub issue tracker](https://github.com/elkins-lab/synth-nmr/issues).

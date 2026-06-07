# Architecture Overview

`synth-nmr` is designed to be highly modular, with clear separation between structure parsing, biophysical calculations, and the command-line interface.

## Core Dependencies

The foundation of `synth-nmr` relies on two primary external libraries:
* **`biotite`**: Used exclusively for parsing structural formats (PDB, mmCIF, trajectories) and constructing memory-efficient `AtomArray` objects.
* **`numpy`**: Provides the vectorized array operations required for fast coordinate math.

## Module Structure

The package is organized into targeted submodules:

* `synth_nmr_cli.py`: The entry point for the command-line interface. Handles argument parsing and interactive prompt looping.
* `chemical_shifts.py`: Implements SPARTA+ integration and ring-current shift calculations.
* `j_couplings.py`: Contains implementations of the Karplus equation with parameterizations for standard backbone dihedrals.
* `noes.py`: Calculates distance-based NOE restraints, applying the $r^{-6}$ distance dependence.
* `rdcs.py`: Implements alignment tensor math to calculate Residual Dipolar Couplings.
* `relaxation.py`: Calculates relaxation rates (R1, R2) and heteronuclear NOEs using Lipari-Szabo model-free formalism.
* `ensemble.py`: Averages NMR observables across an MD trajectory or structural ensemble.

## Data Flow

1. **Input**: A PDB file or trajectory is read into a `biotite.structure.AtomArray`.
2. **Processing**: The coordinates and topologies are passed to the specific biophysics module (e.g., `relaxation.py`).
3. **Output**: The module returns a standard Python dictionary mapping residue identifiers to the computed NMR observable.
4. **Formatting**: The CLI formats this dictionary into readable tables or NEF files for output.

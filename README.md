# synth-nmr

**NMR spectroscopy calculations for protein structures**

A lightweight, standalone Python package for calculating NMR observables from protein structures. Originally extracted from the [synth-pdb](https://github.com/gelkins/synth-pdb) package to provide a focused toolkit that works with any protein structure source.

## Features

- **NOE Calculations**: Synthetic NOE distance restraints
- **Relaxation Rates**: R1, R2, and heteronuclear NOE predictions
- **Chemical Shifts**: SPARTA-lite based predictions with ring current corrections
- **J-Couplings**: Karplus equation for scalar couplings
- **NEF I/O**: Read and write NMR Exchange Format files
- **Secondary Structure**: Automatic classification for enhanced predictions

## Installation

```bash
pip install synth-nmr
```

For improved performance with JIT compilation:
```bash
pip install synth-nmr[performance]
```

## Quick Start

```python
import biotite.structure.io as strucio
from synth_nmr import (
    calculate_synthetic_noes,
    calculate_relaxation_rates,
    predict_chemical_shifts,
    calculate_hn_ha_coupling
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
Predict chemical shifts using SPARTA-lite with ring current corrections.

**Parameters:**
- `structure`: biotite AtomArray

**Returns:** Dictionary of chemical shifts by residue and atom type

#### `calculate_hn_ha_coupling(structure)`
Calculate ³J(HN-Hα) couplings using the Karplus equation.

**Parameters:**
- `structure`: biotite AtomArray

**Returns:** Dictionary of J-coupling values per residue

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
  url = {https://github.com/gelkins/synth-nmr}
}
```

## License

MIT License - see LICENSE file for details

## Related Projects

- [synth-pdb](https://github.com/gelkins/synth-pdb) - Synthetic protein structure generation
- [Biotite](https://www.biotite-python.org/) - Computational biology toolkit

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please use the [GitHub issue tracker](https://github.com/gelkins/synth-nmr/issues).

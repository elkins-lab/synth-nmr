# 🧬 synth-nmr: NMR Spectroscopy Simulation

[![PyPI version](https://img.shields.io/pypi/v/synth-nmr.svg)](https://pypi.org/project/synth-nmr/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/synth-nmr.svg)](https://pypi.org/project/synth-nmr/)
[![Tests](https://github.com/elkins/synth-nmr/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/synth-nmr/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`synth-nmr` provides high-performance biophysical kernels for simulating NMR spectroscopy observables from 3D protein structures.

---

### 🧪 For Structural Biologists
*   **Full Observables Suite:** Calculate Chemical Shifts, RDCs (Residual Dipolar Couplings), NOEs, and J-Couplings from any PDB or MD trajectory.
*   **Scientific Accuracy:** Validated against experimental data and reference suites like SHIFTX2 and PALES.

### 🤖 For Machine Learning Geeks
*   **Fast Forward Kernels:** Highly optimized NumPy/Numba implementation for large-scale data processing.
*   **Differentiable Support:** Designed for seamless integration with `diff-biophys` for gradient-based structural refinement.

---

## 🚀 Supported Observables

*   **Chemical Shifts:** Random coil, secondary structure effects, and ring currents.
*   **RDCs:** Alignment tensor fitting and Q-factor calculation.
*   **Relaxation:** $R_1$, $R_2$, and NOE rates using Model-Free formalism.

## 📦 Installation

```bash
pip install synth-nmr
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

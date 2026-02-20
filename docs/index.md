# synth-nmr

**NMR spectroscopy calculations for protein structures**

A lightweight, standalone Python package for calculating NMR observables from protein structures. 

`synth-nmr` provides a focused toolkit that works with any protein structure source to predict experimental observables like NOEs, Relaxation rates, Chemical Shifts, and Residual Dipolar Couplings.

---

## Features

- **NOE Calculations**: Synthetic Nuclear Overhauser Effect distance restraints.
- **Relaxation Rates**: $R_{1}$, $R_{2}$, and heteronuclear NOE predictions based on Model-Free formalism.
- **Chemical Shifts**: SPARTA+ based predictions with ring current corrections.
- **J-Couplings**: Karplus equation applications for scalar couplings.
- **RDC Calculations**: Prediction of Residual Dipolar Couplings from alignment tensors.
- **NEF I/O**: Native support for the NMR Exchange Format.

## Why synth-nmr?

Modern structural biology increasingly relies on hybrid methods. Whether you're generating structures using AlphaFold, running Molecular Dynamics simulations, or building de novo generative models, `synth-nmr` enables you to quickly validate those theoretical structures against standard NMR experimental observables.

If you are an AI/ML researcher, bridging the gap between 3D Cartesian coordinates and the measurements that spectroscopists actually record is crucial for training multimodal models. `synth-nmr` acts as that rigorous biophysical bridge.

## Next Steps

- **[Installation & CLI Usage](getting_started.md)**: Learn how to install the package and use the command-line interface.
- **[Scientific Background](science/index.md)**: Dive deep into the physics and theory of NMR, including the seminal contributions of pioneers like Wüthrich and Bax.
- **[API Reference](api/index.md)**: Browse the Python API documentation for integrating `synth-nmr` into your own pipelines.

# Introduction to NMR Structural Biology

Nuclear Magnetic Resonance (NMR) spectroscopy is a unique and powerful technique for studying the three-dimensional structures, dynamics, and interactions of biomacromolecules *in solution*, at atomic resolution.

Unlike X-ray crystallography, which traps proteins in a rigid crystal lattice, NMR examines proteins under near-physiological conditions, tumbling freely in a magnetic field. This makes the technique exquisitely sensitive to molecular dynamics—from sub-nanosecond librations to millisecond conformational exchanges.

## The Pillars of Protein NMR

The development of modern protein NMR rests on the translation of 1D and 2D spectra into spatial coordinates. The pioneer of this methodology, **Kurt Wüthrich**, earned the 2002 Nobel Prize in Chemistry for establishing the foundational framework.

Wüthrich's lab at ETH Zürich developed the strategy of **Sequential Assignment**. Before you can determine a structure, you must map every peak in a spectrum to a specific atom in the protein chain. By combining:
1.  **Scalar Couplings (J-Coupling/COSY)**: Signals transferred through chemical bonds.
2.  **Nuclear Overhauser Effects (NOESY)**: Signals transferred through space.

Wüthrich demonstrated that researchers could systematically "walk" down the peptide backbone, connecting residue $i$ to residue $i+1$, entirely from experimental observables.

## The Synthetic Bridge: From Coordinates to Observables

In modern hybrid structural biology, computational researchers often find themselves operating in reverse. Rather than starting with experimental spectra and trying to solve a structure, they start with a predicted structure (e.g., from AlphaFold) and need to simulate what the corresponding NMR experiment *should* look like.

This is the primary directive of `synth-nmr`.

By calculating what the NOEs, Chemical Shifts, and Relaxation rates should be for a specific atomic model, researchers can quantitatively assess how closely their in silico predictions match physical reality.

---

### In This Section

We will explore the biophysics behind the observables that `synth-nmr` predicts, honoring the pioneers who formalized the theory:

*   **[Distance Restraints (NOEs)](noes.md)**: The phenomenon of cross-relaxation.
*   **[Molecular Dynamics (Relaxation)](relaxation.md)**: Probing flexibility through the Model-Free formalism.
*   **[Chemical Shifts](chemical_shifts.md)**: The relationship between local environment and Larmor frequency.
*   **[Dihedral Angles (J-Coupling)](j_couplings.md)**: The Karplus equation and parameterized structural dependencies.
*   **[Global Orientations (RDCs)](rdcs.md)**: Alignment tensors and structural elucidation.
*   **[Ensemble NMR & MD](trajectory.md)**: Time-averaging over dynamic conformational ensembles — accessible to NMR beginners.

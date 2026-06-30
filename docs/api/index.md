# API Reference

This section provides auto-generated documentation for the public API of `synth-nmr`.

The core modules are responsible for taking a `biotite.structure.AtomArray` structural object and converting its 3D coordinates into predicted Nuclear Magnetic Resonance observables.

Use the navigation sidebar to explore the specific modules:

*   **[Chemical Shifts](chemical_shifts.md)**: `predict_chemical_shifts()`
*   **[Relaxation](relaxation.md)**: `calculate_relaxation_rates()`, `predict_order_parameters()`
*   **[Nuclear Overhauser Effects (NOEs)](nmr.md)**: `calculate_synthetic_noes()`
*   **[J-Couplings](j_coupling.md)**: `calculate_hn_ha_coupling()`
*   **[Residual Dipolar Couplings (RDCs)](rdc.md)**: `calculate_rdcs()`
*   **[Data Pipeline](data_pipeline.md)**: Data loading and caching
*   **[NEF IO](nef_io.md)**: NMR Exchange Format parsing
*   **[Neural Shifts](neural_shifts.md)**: GNN-based shift prediction
*   **[Trajectory](trajectory.md)**: MD trajectory processing
*   **[Validation](validation.md)**: Observables validation against experiment
*   **[Structure Utils](structure_utils.md)**: Utilities for structural analysis

# Future Test Suite Improvements

This document outlines potential future enhancements for the `synth-nmr` test suite to further improve robustness, performance tracking, and integration confidence.

## 1. Property-Based Testing
**Tool:** `hypothesis`
**Description:** Introduce property-based testing for core mathematical and physics modules such as `j_coupling`, `relaxation`, and `chemical_shifts`. Instead of hardcoding static test cases, use `hypothesis` to automatically generate hundreds of edge-case inputs (e.g., extreme temperatures, unusual angles, or edge-case magnetic fields). This ensures the underlying math handles boundary conditions gracefully without crashing or returning `NaN`.

## 2. Performance Regression Testing
**Tool:** `pytest-benchmark`
**Description:** NMR predictions (especially SASA calculations and ML inference) can be computationally heavy. Implement benchmark tests wrapping core functions like `predict_order_parameters` or `predict_j_couplings`. Tracking these execution times across commits will help detect and prevent performance regressions.

## 3. End-to-End ML Integration Tests
**Description:** Currently, tests for GNN training (e.g., `train_gnn.py` or `test_gnn.py`) mostly use mocks or skip the PyTorch dependencies entirely. Add a lightweight, end-to-end integration test that executes a single training epoch on a tiny, synthetic dataset (e.g., 5 samples). This verifies that the actual PyTorch forward and backward pass logic remains functional end-to-end.

# Roadmap: synth-nmr

This document serves as a repository for future ideas, feature proposals, and potential architectural improvements for the `synth-nmr` project. It is organized into strategic development tracks.

## 1. Expanding the Physics & Observables

- [x] **Expanded J-Coupling Subtypes**:
    Currently, the project calculates the $^3J_{H_N, H_\alpha}$ coupling (using Vuister & Bax coefficients) as well as:
    - [x] $^3J_{H_\alpha, H_\beta}$ (derived from the $\chi_1$ side-chain angle).
    - [x] $^3J_{C', C_\gamma}$ couplings.
- [ ] **Paramagnetic Relaxation Enhancements (PRE) & Pseudocontact Shifts (PCS)**:
    Adding support for predicting PREs and PCSs. These spatial observables are critical in modern NMR for analyzing long-range distances and large multi-domain proteins or protein-ligand complexes.

## 2. Advancing the Machine Learning Pipeline

- [ ] **Structural Encoders & Pre-trained Embeddings**:
    The current GNN relies on Euclidean distances bounded by a `scipy.spatial.KDTree`. To give the GNN a deeper understanding of the chemical environment:
    - [ ] Integrate pre-trained protein language model embeddings (e.g., ESM-2).
    - [ ] Incorporate learned structural features (e.g., inverse-folding encodings like ProteinMPNN).
- [ ] **SE(3)-Equivariant Neural Architectures**:
    Upgrade the architecture from a standard Graph Attention Network (GAT) to an SE(3)-equivariant model (like EGNN). This naturally maintains the 3D rotational and translational invariance of protein atomic coordinates inherently better than standard message passing.

## 3. Usability & Computational Integrations

- [ ] **Molecular Dynamics (MD) Support**:
    Shift beyond parsing static `.pdb` / NEF files to evaluating ensembles.
    - [ ] Add direct parsing support for reading GROMACS (`.gro`/`.xtc`) or AMBER topology trajectories.
    - [ ] Implement logic to calculate time-averaged NMR observables across an entire simulation ensemble.
- [ ] **RDC Tensor Fitting (Inverse Problem)**:
    Currently, RDCs are predicted in a forward-direction based on assumed molecular alignment tensors ($D_a$ and $R$).
    - [ ] Write an inverse Singular Value Decomposition (SVD) algorithm that *fits* the alignment tensor given a coordinate structure and a user-provided set of *experimental* RDCs.
- [ ] **Web Interface / User Application**:
    - [ ] Wrap the `synth-nmr` backend in a FastAPI endpoint or a Streamlit UI.
    - [ ] Deploy as a web application allowing users to upload PDBs and instantly visualize predicted spectra directly in their browser without utilizing the CLI or writing Python code.
- [ ] **Interactive AI Documentation (NotebookLM)**:
    - [ ] Create a curated "source guide" (e.g., merging the README, ROADMAP, mathematical theory, and core module code) to upload to Google's NotebookLM.
    - [ ] Add a public NotebookLM link to the `README.md` allowing users to chat with the repository, learn the NMR physics, or ask how to use the CLI/API.

## 4. CI/CD & Documentation

- [x] **GitHub Actions Automation**:
    Leverage the 99% test coverage and strict `mypy` typing to harden the repository.
    - [x] Write a comprehensive `.github/workflows/ci.yml` pipeline. *(Successfully running: zero test failures across 324 tests!)*
    - [x] Automatically run `pytest`, `black`, `ruff`, and `mypy` on every push and pull request. *(Zero linting issues found)*
- [x] **MkDocs Automatic Deployment**:
    Formalize the documentation website referenced in the `README.md`.
    - [x] Configure `mkdocs-material` and deploy the living documentation automatically via GitHub Pages actions.

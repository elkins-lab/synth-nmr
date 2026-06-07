# Developer Guide

Welcome to the `synth-nmr` developer documentation! This section provides all the necessary information for setting up a local development environment, understanding the project architecture, testing the codebase, and releasing new versions.

## Overview

The `synth-nmr` package is a lightweight, standalone Python toolkit for calculating NMR observables from protein structures. It is designed to be easily extensible and highly robust.

If you are looking to contribute to the project, please start by reading our [Contributing Guidelines](../../CONTRIBUTING.md) and our [Code of Conduct](../../CODE_OF_CONDUCT.md).

## Navigating the Developer Docs

* **[Architecture](architecture.md)**: A high-level overview of how the code is structured and how data flows through the physics engines.
* **[Testing & CI](testing.md)**: Instructions for running the local test suite, linters, and type checkers.
* **[Release Process](release_process.md)**: The standardized pipeline for publishing new versions to PyPI.

## Local Environment Setup

To get started with development, you'll need to clone the repository and set up a virtual environment.

```bash
# Clone the repository
git clone https://github.com/elkins/synth-nmr.git
cd synth-nmr

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in editable mode with development dependencies
pip install -e ".[dev,test]"
```

Once installed, you can run the CLI directly from source:
```bash
python -m synth_nmr.synth_nmr_cli
```

"""
synth-nmr: NMR spectroscopy calculations for protein structures.

This package provides tools for calculating NMR observables from protein
structures, including:
- NOE distance restraints
- Relaxation rates (R1, R2, heteronuclear NOE)
- Chemical shifts (SPARTA+ with ring currents)
- J-couplings (Karplus equation)
- RDC calculations
- NEF format I/O for NMR data exchange
- MD trajectory loading and ensemble-averaged NMR observables

Originally extracted from the synth-pdb package to provide a lightweight,
standalone NMR toolkit that works with any protein structure source.
"""

__version__ = "0.11.6"

from .chemical_shifts import calculate_csi, predict_chemical_shifts
from .j_coupling import calculate_hn_ha_coupling
from .nmr import calculate_synthetic_noes
from .rdc import calculate_rdcs
from .relaxation import calculate_relaxation_rates, predict_order_parameters
from .trajectory import (
    TrajectoryEnsemble,
    compute_s2_from_trajectory,
    ensemble_average_j_couplings,
    ensemble_average_noes,
    ensemble_average_rdcs,
    ensemble_average_shifts,
    load_trajectory,
)

__all__ = [
    "calculate_synthetic_noes",
    "calculate_relaxation_rates",
    "predict_order_parameters",
    "predict_chemical_shifts",
    "calculate_csi",
    "calculate_hn_ha_coupling",
    "calculate_rdcs",
    # Trajectory / Ensemble NMR
    "TrajectoryEnsemble",
    "load_trajectory",
    "ensemble_average_shifts",
    "ensemble_average_noes",
    "ensemble_average_rdcs",
    "ensemble_average_j_couplings",
    "compute_s2_from_trajectory",
]

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

Originally extracted from the synth-pdb package to provide a lightweight,
standalone NMR toolkit that works with any protein structure source.
"""

__version__ = "0.3.0"

from .nmr import calculate_synthetic_noes
from .relaxation import calculate_relaxation_rates, predict_order_parameters
from .chemical_shifts import predict_chemical_shifts, calculate_csi
from .j_coupling import calculate_hn_ha_coupling
from .rdc import calculate_rdcs

__all__ = [
    "calculate_synthetic_noes",
    "calculate_relaxation_rates",
    "predict_order_parameters",
    "predict_chemical_shifts",
    "calculate_csi",
    "calculate_hn_ha_coupling",
    "calculate_rdcs",
]

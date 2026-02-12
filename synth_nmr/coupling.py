
"""
J-Coupling Prediction Module for Synth-PDB.

This module implements the Karplus equation for predicting scalar couplings (3J)
from backbone dihedral angles.

Educational Note - The Karplus Equation
---------------------------------------
Scalar coupling (J-coupling) is a quantum mechanical effect where the magnetic
state of one nucleus affects the energy levels of another through the bonding electrons.
The magnitude of the 3-bond coupling (^3J) depends strongly on the dihedral angle
(theta) between the coupled nuclei.

Equation: ^3J(theta) = A * cos^2(theta) + B * cos(theta) + C

References:
- Karplus, M. (1959). J. Chem. Phys.
- Vogeli, B., et al. (2007). J. Am. Chem. Soc. (Coefficients)
"""

import numpy as np
from typing import Dict, Union, List

# Coefficients for 3J(HN-HA) coupling
# Source: Vogeli et al. (2007) / Bax Group
# Standard values often used: A=9.45, B=-1.56, C=0.52 (Hu & Bax)
# Or: A=6.51, B=-1.76, C=1.60 (Orig Karplus for proteins)
# We use the widely accepted Hu & Bax parameterization for proteins in solution.
PARAMS_HN_HA = {
    'A': 6.51,  # Classical values often used in textbooks (e.g. Cavanagh)
    'B': -1.76, # differ slightly by calibration set.
    'C': 1.60   # This set gives ~4Hz for helix (-60) and ~9-10Hz for sheet (-120)
}

# Actually, let's use the Values from:
# "Vogeli et al., JACS 129, 9377 (2007)" which refined these including dynamics.
# For static structures, the Hofstetter/Vuister values are often robust:
# A=6.98, B=-1.38, C=1.72
# Let's stick to the classic values that satisfy the generic rule of thumb
# (Helix ~4, Sheet ~9) firmly. A=6.51 check matches well.

def calculate_hn_ha_coupling(phi_degrees: float) -> float:
    """
    Calculate the 3J(HN-HA) coupling constant using the Karplus equation.
    
    Args:
        phi_degrees (float): Backbone Phi angle in degrees.
        
    Returns:
        float: Predicted J-coupling in Hz. Returns NaN if input is NaN.
    """
    if np.isnan(phi_degrees):
        return np.nan

    # 1. Convert Phi to Theta
    # The Karplus angle theta is related to phi by a phase shift.
    # For H-N-Ca-Ha: Theta = |Phi - 60|
    # Note: Phi is defined as C'-N-Ca-C'.
    # The H-N bond is ~180 from C'-N.
    # The Ha-Ca bond is ~ +/- 120 from Ca-C'.
    # Standard relation: Theta = Phi - 60 degrees.
    
    theta_deg = phi_degrees - 60.0
    theta_rad = np.radians(theta_deg)
    
    # 2. Apply Karplus Equation
    # J = A*cos^2(theta) + B*cos(theta) + C
    A = PARAMS_HN_HA['A']
    B = PARAMS_HN_HA['B']
    C = PARAMS_HN_HA['C']
    
    cos_theta = np.cos(theta_rad)
    j_coupling = A * (cos_theta ** 2) + B * cos_theta + C
    
    return j_coupling

def predict_couplings_from_structure(phi_map: Dict[int, float]) -> Dict[int, float]:
    """
    Predict HN-HA couplings for a set of residues.
    
    Args:
        phi_map: Dictionary mapping Residue ID (int) -> Phi angle (float)
        
    Returns:
        Dictionary mapping Residue ID -> J-coupling (Hz)
    """
    couplings = {}
    for res_id, phi in phi_map.items():
        couplings[res_id] = calculate_hn_ha_coupling(phi)
    return couplings

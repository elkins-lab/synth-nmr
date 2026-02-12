"""
Scalar Coupling (J-coupling) calculations.

EDUCATIONAL NOTE - Karplus Equation:
====================================
Scalar couplings (J-couplings) are mediated through chemical bonds.
The 3-bond coupling (^3J) depends heavily on the torsion angle between the atoms.

For the backbone amide proton (HN) and alpha proton (HA), the coupling ^3J_HNHa
tells us about the Phi angle, and thus the secondary structure.

Formula:
  ^3J = A * cos^2(theta) + B * cos(theta) + C

Where theta = Phi - 60 degrees (phase shift).
Typical values:
- Alpha Helix (Phi ~ -60): theta ~ -120 -> J is small (~4 Hz)
- Beta Sheet (Phi ~ -120): theta ~ -180 -> J is large (~9 Hz)
- Random Coil: Averaged (~7 Hz)

This allows NMR spectroscopists to determine secondary structure just by measuring J-couplings!
"""

import numpy as np
import biotite.structure as struc
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Vuister & Bax parameters (J. Am. Chem. Soc. 1993, 115, 7772-7777)
KARPLUS_PARAMS = {
    'A': 6.51,
    'B': -1.76,
    'C': 1.60
}

def calculate_hn_ha_coupling(structure: struc.AtomArray) -> Dict[str, Dict[int, float]]:
    """
    Calculate 3J_HNHa coupling constants for the protein backbone.
    
    Args:
        structure: AtomArray containing the protein
        
    Returns:
        Dict keyed by Chain ID -> Residue ID -> J-coupling value (Hz)
    """
    logger.info("Calculating 3J_HNHa scalar couplings...")
    
    phi, psi, omega = struc.dihedral_backbone(structure)
    
    # biotite returns angles for each residue.
    # The first residue has no Phi (undefined).
    # The corresponding residues are structure residues that have backbone atoms.
    # We need to map these back to Res IDs.
    
    res_starts = struc.get_residue_starts(structure)
    # Filter to only amino acids? Usually safe.
    
    results = {}
    
    # Iterate over residues
    # Angles array matches number of residues
    if len(phi) != len(res_starts):
        logger.warning(f"Mismatch in backbone angles count ({len(phi)}) vs residue count ({len(res_starts)}).")
        return {}
        
    for i, start_idx in enumerate(res_starts):
        # Get residue info
        res_atoms = structure[start_idx : res_starts[i+1] if i+1 < len(res_starts) else None]
        chain_id = res_atoms.chain_id[0]
        res_id = res_atoms.res_id[0]
        res_name = res_atoms.res_name[0]
        
        if chain_id not in results:
            results[chain_id] = {}
            
        # Get Phi angle (in radians)
        phi_rad = phi[i]
        
        # Check for NaN (undefined, e.g. N-terminus)
        if np.isnan(phi_rad):
            # No coupling defined
            continue
            
        # Glycine has HA2/HA3, usually averaged or specific.
        # This equation assumes standard H-N-Ca-Ha geometry.
        # Standard Karplus: theta = phi - 60 deg (- pi/3)
        theta = phi_rad - (np.deg2rad(60.0))
        
        # Calculate J
        # J = A cos^2(theta) + B cos(theta) + C
        cos_theta = np.cos(theta)
        j_val = (KARPLUS_PARAMS['A'] * (cos_theta ** 2)) + \
                (KARPLUS_PARAMS['B'] * cos_theta) + \
                KARPLUS_PARAMS['C']
                
        # Add noise? Optional. Real measurements have error ~0.5 Hz.
        # For pure education, clean curves are better.
        
        results[chain_id][res_id] = round(j_val, 2)
        
    return results

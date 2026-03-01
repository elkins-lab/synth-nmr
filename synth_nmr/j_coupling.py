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
from typing import Dict

logger = logging.getLogger(__name__)

# Vuister & Bax parameters (J. Am. Chem. Soc. 1993, 115, 7772-7777)
KARPLUS_PARAMS = {"A": 6.51, "B": -1.76, "C": 1.60}


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

    results: Dict[str, Dict[int, float]] = {}

    # Iterate over residues
    # Angles array matches number of residues
    if len(phi) != len(res_starts):
        logger.warning(
            f"Mismatch in backbone angles count ({len(phi)}) vs residue count ({len(res_starts)})."
        )
        return {}

    for i, start_idx in enumerate(res_starts):
        # Get residue info
        res_atoms = structure[start_idx : res_starts[i + 1] if i + 1 < len(res_starts) else None]
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
        # For Glycine, which has two alpha protons (HA2/HA3), this calculation provides
        # a single value based on the averaged phi angle. It does not explicitly
        # distinguish between the potentially different couplings for HA2 and HA3.

        # EDUCATIONAL NOTE: The relationship theta = phi - 60 is a geometric consequence
        # of how the phi angle and the Karplus angle are defined.
        # - Phi (C'-N-Ca-C') measures the rotation around the N-Ca bond.
        # - Theta for 3J(HN-HA) is the dihedral H-N-Ca-HA.
        # In an ideal trans peptide plane, the HN and C' atoms are roughly 180 degrees
        # apart relative to the N-Ca bond. The HA proton is at a stereochemically
        # fixed position relative to the Ca-C' bond. This results in an approximate
        # 60-degree phase offset between the two dihedral angles.
        theta = phi_rad - (np.deg2rad(60.0))

        # Calculate J
        # J = A cos^2(theta) + B cos(theta) + C
        cos_theta = np.cos(theta)
        j_val = (
            (KARPLUS_PARAMS["A"] * (cos_theta**2))
            + (KARPLUS_PARAMS["B"] * cos_theta)
            + KARPLUS_PARAMS["C"]
        )

        results[chain_id][res_id] = round(j_val, 2)

    return results

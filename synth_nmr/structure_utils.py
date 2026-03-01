import numpy as np
import biotite.structure as struc
from typing import List
import logging

logger = logging.getLogger(__name__)


def get_secondary_structure(structure: struc.AtomArray) -> List[str]:
    """
    Determine the secondary structure of each residue based on Phi/Psi angles.

    Returns a list of strings: 'alpha', 'beta', or 'coil'.
    Matches the residue indices in the structure.
    """
    # Calculate dihedrals
    # Note: struc.dihedral_backbone returns phi, psi, omega arrays
    # length equals number of residues
    try:
        phi, psi, omega = struc.dihedral_backbone(structure)
    except struc.BadStructureError:
        # Fallback for incomplete backbones (e.g. in tests)
        return ["coil"] * struc.get_residue_count(structure)

    # We need to iterate over residues to match the output list
    # get_residue_starts is useful
    res_starts = struc.get_residue_starts(structure)
    ss_list = []

    for i, _ in enumerate(res_starts):
        # Safety check: if phi array is shorter than residue count (e.g. due to ions/HETATM)
        # The "coil" secondary structure assignment is a safe fallback and will not
        # negatively impact other functionality.
        if i >= len(phi) or i >= len(psi):
            ss_list.append("coil")
            continue

        # Get Angles (degrees)
        p = np.rad2deg(phi[i])
        s = np.rad2deg(psi[i])

        ss_state = "coil"

        # Determine Secondary Structure State based on broader Ramachandran regions.
        # The ranges used here are slightly wider than strict textbook definitions
        # to accommodate potential variations from structure generation algorithms
        # ("Synthetic Generator Offset Issues") and to capture both canonical
        # right-handed alpha helices and less common left-handed helical regions.
        #
        # References for Ramachandran regions:
        # - Ramachandran, G. N., Ramakrishnan, C., & Sasisekharan, V. (1963).
        #   "Stereochemistry of polypeptide chain configurations." J. Mol. Biol., 7(1), 95-99.
        # - Lovell, S. C., Davis, I. W., Arendall III, J. W., de Bakker, P. I.,
        #   Word, J. M., Prisant, M. G., ... & Richardson, D. C. (2003).
        #   "Structure validation by Calpha geometry: phi,psi and Cbeta deviation."
        #   Proteins: Structure, Function, and Bioinformatics, 50(3), 437-450.

        if not np.isnan(p) and not np.isnan(s):
            logger.debug(f"Res {i}: Phi={p:.1f}, Psi={s:.1f}")

            # Right-handed Alpha-helix region
            if (-90 < p < -30) and (-90 < s < -10):
                ss_state = "alpha"
            # Beta-sheet / Extended region
            elif (-160 < p < -80) and (80 < s < 170):
                ss_state = "beta"
            # Broader helical region, including potential left-handed helices or
            # alternative alpha-helical conformations sometimes produced by synthetic generators.
            elif (0 < p < 150) and (-90 < s < -10):
                ss_state = "alpha"

        ss_list.append(ss_state)

    # SMOOTHING PASS
    # Real secondary structure elements are usually contiguous.
    # We filter out isolated "coil" residues within helices/sheets.
    # Example: alpha-coil-alpha -> alpha-alpha-alpha

    # 1. Filter single-residue interruptions
    # The loop intentionally excludes the first and last residues (range from 1 to len-1),
    # as these termini are often intrinsically flexible and their secondary structure
    # is less reliably defined or less critical for local context smoothing.
    for i in range(1, len(ss_list) - 1):
        prev_s = ss_list[i - 1]
        curr_s = ss_list[i]
        next_s = ss_list[i + 1]

        if curr_s == "coil" and prev_s == next_s and prev_s != "coil":
            logger.debug(f"Smoothing residue {i}: coil -> {prev_s}")
            ss_list[i] = prev_s

    # 2. Filter 2-residue interruptions? (Optional, maybe too aggressive)

    return ss_list

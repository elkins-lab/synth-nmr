"""
NMR Spectroscopy utilities for synth-pdb.

This module is responsible for calculating synthetic NMR observables from
generated structures, such as Nuclear Overhauser Effects (NOEs) based on
inter-proton distances.
"""

import logging
import numpy as np
import biotite.structure as struc
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

def calculate_synthetic_noes(
    structure: struc.AtomArray,
    cutoff: float = 5.0,
    buffer: float = 0.5,
    exclude_intra_residue: bool = False
) -> List[Dict]:
    """
    Calculate synthetic NOE restraints from a structure.
    
    Finds all proton pairs (H-H) within the specified cutoff distance.
    Generates an upper bound restraint for each pair.

    # EDUCATIONAL NOTE - The Physics of NOEs
    # The Nuclear Overhauser Effect (NOE) allows us to measure distances between
    # protons in a molecule. 
    # NOE stands for Nuclear Overhauser Effect:
    # - "Nuclear": Involves atomic nuclei (protons).
    # - "Overhauser": Named after physicist Albert Overhauser who predicted it.
    # - "Effect": The phenomenon where spinning one nucleus affects the signal of its neighbor.
    #
    # The intensity of the NOE signal is proportional to the inverse 6th power of the distance (I ~ 1/r^6).
    #
    # This steep dependence means:
    # 1. Close protons give VERY strong signals.
    # 2. As distance increases, signal vanishes rapidly.
    # 3. The practical limit for detection is usually 5.0 - 6.0 Angstroms.
    #
    # In structure calculation, we treat these not as exact rulers, but as 
    # "Upper Distance Bounds". If we see an NOE, the atoms MUST be close.
    # If we don't see one, they might be far, or there might be motion/noise.
    
    Args:
        structure: The AtomArray containing the protein (must have Hydrogens).
        cutoff: Maximum distance (Angstroms) to consider an NOE. Must be > 0.
        buffer: Amount to add to actual distance for the Upper Bound. Must be >= 0.
        exclude_intra_residue: If True, ignore NOEs within same residue.
        
    Returns:
        List of restraint dictionaries, corrected to match documentation:
        {
            'index_1': int, 'res_name_1': str, 'atom_name_1': str, 'chain_1': str,
            'index_2': int, 'res_name_2': str, 'atom_name_2': str, 'chain_2': str,
            'distance': float,
            'upper_limit': float
        }
        
    Raises:
        TypeError: If the input structure is not a biotite.structure.AtomArray.
        ValueError: If cutoff or buffer have invalid values.
    """
    logger.info(
        f"Calculating synthetic NOEs with cutoff={cutoff} Å, buffer={buffer} Å, "
        f"exclude_intra_residue={exclude_intra_residue}."
    )

    # 1. Input Validation
    if not isinstance(structure, struc.AtomArray):
        raise TypeError("Input 'structure' must be a biotite.structure.AtomArray.")
    if cutoff <= 0:
        raise ValueError("'cutoff' distance must be positive.")
    if buffer < 0:
        raise ValueError("'buffer' must be non-negative.")
    if structure.array_length() == 0:
        logger.warning("Input 'structure' is empty. Returning no restraints.")
        return []

    try:
        # 1. Select only Protons (Element 'H')
        # Filter for element H
        h_mask = structure.element == "H"
        
        # Safety Check: If no hydrogens, we can't calculate NOEs
        if not np.any(h_mask):
            logger.warning(
                "No hydrogens found in structure. Cannot calculate NOEs. "
                "Consider adding hydrogens to the structure first."
            )
            return []
            
        protons = structure[h_mask]
        n_protons = protons.array_length()
        logger.debug(f"Found {n_protons} protons for NOE calculation.")
        
        # 2. Calculate Cell List for efficient neighbor search
        # We want pairs within cutoff.
        cell_list = struc.CellList(protons, cell_size=cutoff)
        
        # 3. Find neighbors
        # EDUCATIONAL NOTE: Neighbor Search Strategy
        # A naive search for all pairs of protons would be O(N^2), which is too slow for large proteins.
        # We use a CellList (also known as a grid search) to accelerate this to O(N).
        # 1. The space is divided into a grid of cells (size = cutoff).
        # 2. For each proton, we only search for neighbors in its own cell and adjacent cells.
        # This loop iterates through each proton, finds its neighbors using the efficient
        # `cell_list.get_atoms`, and then filters for unique pairs (j > i) to avoid duplicates.
        restraints = []
        
        # Let's iterate over all protons and find neighbors for each
        for i in range(n_protons):
            # Center atom
            center = protons[i]
            # Find indices of neighbors in 'protons' array
            indices = cell_list.get_atoms(center.coord, radius=cutoff)
            
            # Filter for unique pairs (j > i) to avoid duplicates and self-pairs
            indices = indices[indices > i]
            
            for j in indices:
                neighbor = protons[j]
                
                # Check exclusion logic for intra-residue pairs
                is_intra_residue = (center.res_id == neighbor.res_id) and (center.chain_id == neighbor.chain_id)

                if exclude_intra_residue and is_intra_residue:
                    continue
                
                dist = struc.distance(center, neighbor)

                # Explicitly exclude very close intra-residue geminal protons (e.g., HBx-HBx on same carbon)
                # These are usually trivial in NOE lists and have fixed distances (~1.77 Å).
                if is_intra_residue and dist < 2.0 and \
                   center.atom_name.startswith('HB') and neighbor.atom_name.startswith('HB'):
                    continue
                
                # Global exclusion for very short distances (e.g., direct bonds)
                if dist < 1.0: # Very short distances are likely errors or direct bonds
                    continue 
                    
                restraint = {
                    'index_1': center.res_id,
                    'res_name_1': center.res_name,
                    'atom_name_1': center.atom_name,
                    'chain_1': center.chain_id,
                    
                    'index_2': neighbor.res_id,
                    'res_name_2': neighbor.res_name,
                    'atom_name_2': neighbor.atom_name,
                    'chain_2': neighbor.chain_id,
                    
                    'distance': float(dist),
                    'upper_limit': float(dist + buffer)
                }
                restraints.append(restraint)
                
        logger.info(f"Generated {len(restraints)} synthetic NOE restraints.")
        return restraints

    except Exception as e:
        logger.error(f"An unexpected error occurred during NOE calculation: {e}", exc_info=True)
        # Re-raise the exception to not silently fail, allowing upstream handling
        raise


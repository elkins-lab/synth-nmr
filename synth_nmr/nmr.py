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
        cutoff: Maximum distance (Angstroms) to consider an NOE. Default 5.0.
        buffer: Amount to add to actual distance for the Upper Bound. Default 0.5.
        exclude_intra_residue: If True, ignore NOEs within same residue.
        
    Returns:
        List of restraint dictionaries:
        {
            'index_1': int, 'res_name_1': str, 'atom_name_1': str, 'chain_1': str,
            'index_2': int, 'res_name_2': str, 'atom_name_2': str, 'chain_2': str,
            'distance': float,
            'upper_limit': float
        }
    """
    logger.info(f"Calculating synthetic NOEs (cutoff={cutoff}A)...")
    
    # 1. Select only Protons (Element 'H')
    # Filter for element H
    h_mask = structure.element == "H"
    
    # Safety Check: If no hydrogens, we can't calculate NOEs
    if not np.any(h_mask):
        logger.warning("No hydrogens found in structure! Cannot calculate NOEs. Did you forget to add hydrogens/minimize?")
        return []
        
    protons = structure[h_mask]
    n_protons = len(protons)
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
        coord = center.coord
        
        # Find indices of neighbors in 'protons' array
        # radius search returns indices in 'protons'
        indices = cell_list.get_atoms(coord, radius=cutoff)
        
        # Filter indices to only keep j > i (unique pairs)
        indices = indices[indices > i]
        
        for j in indices:
            neighbor = protons[j]
            
            # Helper to get atom info
            idx1 = center.res_id
            name1 = center.res_name
            atom1 = center.atom_name
            chain1 = center.chain_id
            
            idx2 = neighbor.res_id
            name2 = neighbor.res_name
            atom2 = neighbor.atom_name
            chain2 = neighbor.chain_id
            
            # Check exclusion logic
            if exclude_intra_residue and (idx1 == idx2) and (chain1 == chain2):
                continue
                
            # Basic exclusion: don't restrain geminal protons (e.g. HB2-HB3) on same carbon?
            # Usually these are fixed distance. For "Synthetic" data, maybe we keep them or filter.
            # Standard NOE lists often exclude fixed distances (geminal/vicinal if fixed).
            # For simplicity in V1, we include everything not same atom (j > i handles same atom).
            # Let's exclude same residue for now if requested, but default keep it?
            # Actually, real NOE lists exclude intra-residue trivial ones often.
            # Let's check distance.
            
            dist = np.linalg.norm(center.coord - neighbor.coord)
            
            # Hard exclusion of very close atoms (bonded or geminal < ~1.8A)
            # Bonded H-H doesn't exist. Geminal H-H on CH2 is ~1.78A.
            # Real NOEs are usually > 1.8 or 2.0.
            if dist < 1.85:
                continue 
                
            restraint = {
                'residue_index_1': idx1,
                'res_name_1': name1,
                'atom_name_1': atom1,
                'chain_1': chain1,
                
                'residue_index_2': idx2,
                'res_name_2': name2,
                'atom_name_2': atom2,
                'chain_2': chain2,
                
                'distance': float(dist),
                'upper_limit': float(dist + buffer)
            }
            restraints.append(restraint)
            
    logger.info(f"Generated {len(restraints)} synthetic NOE restraints.")
    return restraints

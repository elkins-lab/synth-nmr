"""
Test fixtures and utilities for synth-nmr tests.

Provides helper functions for creating test structures without
requiring the full synth-pdb package.
"""

import numpy as np
import biotite.structure as struc


def create_test_structure(sequence_str, num_atoms_per_residue=5):
    """
    Create a minimal test structure for testing NMR functions.
    
    This is a simplified version that doesn't generate realistic coordinates,
    but provides the minimal structure needed for testing NMR calculations.
    
    Parameters
    ----------
    sequence_str : str
        Single-letter amino acid sequence
    num_atoms_per_residue : int
        Number of atoms per residue (default: 5 for N, CA, C, O, CB)
    
    Returns
    -------
    structure : biotite.structure.AtomArray
        Minimal test structure
    """
    n_residues = len(sequence_str)
    n_atoms = n_residues * num_atoms_per_residue
    
    structure = struc.AtomArray(n_atoms)
    
    # Map single-letter codes to three-letter codes
    aa_map = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
        'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
        'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
        'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
    }
    
    atom_names = ['N', 'CA', 'C', 'O', 'CB']
    
    for i, aa in enumerate(sequence_str):
        res_name = aa_map.get(aa.upper(), 'ALA')
        start_idx = i * num_atoms_per_residue
        end_idx = start_idx + num_atoms_per_residue
        
        structure.res_id[start_idx:end_idx] = i + 1
        structure.res_name[start_idx:end_idx] = res_name
        structure.chain_id[start_idx:end_idx] = 'A'
        structure.atom_name[start_idx:end_idx] = atom_names[:num_atoms_per_residue]
        
        # Create simple coordinates (not realistic, but sufficient for testing)
        for j in range(num_atoms_per_residue):
            structure.coord[start_idx + j] = [i * 3.8, j * 1.5, 0.0]
    
    return structure

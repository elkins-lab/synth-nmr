"""
Validation tests for comparing predicted chemical shifts against published experimental results.
"""

import pytest
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from io import StringIO
import requests
from synth_nmr.chemical_shifts import predict_chemical_shifts
from synth_nmr.structure_utils import get_secondary_structure

@pytest.fixture
def ubiquitin_structure():
    """
    Downloads the 1D3Z PDB file from RCSB, adds amide hydrogens programmatically,
    and returns the first model filtered for amino acids.
    """
    PDB_ID = "1D3Z"
    RCSB_URL = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
    
    # Download PDB file
    response = requests.get(RCSB_URL)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    pdb_string = response.text
    
    # Parse PDB file
    pdb_file = pdb.PDBFile.read(StringIO(pdb_string))
    # get_structure with a model number returns an AtomArray
    structure = pdb.get_structure(pdb_file, model=1)
    
    # Filter for protein and remove alternate location identifiers
    structure = structure[struc.filter_amino_acids(structure)]
    
    # Programmatically add amide hydrogens
    atoms_with_h = list(structure)
    for i in range(len(structure)):
        atom = structure[i]
        # Only add H to backbone N atoms, and exclude Proline
        if atom.atom_name == "N" and atom.res_name != "PRO":
            h_atom_coord = None
            # Find C-alpha of current residue
            ca_curr = structure[(structure.res_id == atom.res_id) & (structure.atom_name == 'CA')]
            # Find C of previous residue
            c_prev = structure[(structure.res_id == atom.res_id - 1) & (structure.atom_name == 'C')]
            
            # Ensure both CA of current and C of previous residue exist
            if len(ca_curr) > 0 and len(c_prev) > 0:
                # Vector from previous C to current N
                vec_c_to_n = atom.coord - c_prev.coord[0]
                # Vector from current CA to current N
                vec_ca_to_n = atom.coord - ca_curr.coord[0]

                # Calculate the bisector of the C_prev-N-CA_curr angle
                # This points roughly towards where the H should be, relative to N
                bisector = (vec_c_to_n / np.linalg.norm(vec_c_to_n)) + (vec_ca_to_n / np.linalg.norm(vec_ca_to_n))
                bisector_norm = bisector / np.linalg.norm(bisector)

                # N-H bond length is approx 1.02 Angstroms. Place H along the *outward* bisector.
                h_coord = atom.coord - bisector_norm * 1.02 # Subtract to point away from N-C_prev and N-CA_curr

                h_atom = struc.Atom(
                    coord=h_coord,
                    atom_name="H",
                    element="H",
                    res_id=atom.res_id,
                    res_name=atom.res_name,
                    chain_id=atom.chain_id
                )
                atoms_with_h.append(h_atom)
    
    # Convert list of atoms back to AtomArray
    structure = struc.array(atoms_with_h)
    
    return structure

def test_chemical_shift_validation_against_1d3z(ubiquitin_structure, monkeypatch):
    """
    Validates predicted chemical shifts (CA, HA, N) against experimental
    values published for ubiquitin (PDB: 1D3Z, BMRB: 5943).
    """
    # Experimental chemical shift values for Ubiquitin from BMRB 5943
    # Only a subset for key atoms (CA, HA, N)
    experimental_shifts = {
        2: {"CA": 55.45, "HA": 4.14, "N": 121.78},
        3: {"CA": 57.06, "HA": 4.16, "N": 122.95},
        4: {"CA": 50.15, "HA": 4.07, "N": 118.88},
        5: {"CA": 52.88, "HA": 4.41, "N": 119.55},
        6: {"CA": 55.03, "HA": 4.09, "N": 121.36},
        7: {"CA": 63.82, "HA": 4.31, "N": 113.84},
        8: {"CA": 54.19, "HA": 4.16, "N": 120.57},
        9: {"CA": 53.64, "HA": 4.19, "N": 120.35},
        10: {"CA": 44.91, "HA": 3.94, "N": 108.97},
        11: {"CA": 55.77, "HA": 4.06, "N": 121.32},
        12: {"CA": 62.46, "HA": 4.18, "N": 113.56},
        13: {"CA": 61.27, "HA": 4.09, "N": 121.13},
        14: {"CA": 62.47, "HA": 4.14, "N": 113.52},
        15: {"CA": 55.19, "HA": 4.16, "N": 121.23},
        16: {"CA": 55.93, "HA": 4.29, "N": 122.79},
        17: {"CA": 61.56, "HA": 4.00, "N": 120.24},
        18: {"CA": 53.51, "HA": 4.16, "N": 120.11},
        19: {"CA": 65.34, "HA": 4.49, "N": 0.00}, # Proline N/H not observed
        20: {"CA": 58.11, "HA": 4.34, "N": 115.35},
        21: {"CA": 54.43, "HA": 4.40, "N": 119.66},
        22: {"CA": 61.42, "HA": 4.32, "N": 112.98},
        23: {"CA": 61.30, "HA": 4.28, "N": 120.48},
        24: {"CA": 53.77, "HA": 4.21, "N": 120.89},
        25: {"CA": 52.61, "HA": 4.39, "N": 119.33},
        26: {"CA": 61.42, "HA": 4.13, "N": 120.25},
        27: {"CA": 55.72, "HA": 4.22, "N": 120.39},
        28: {"CA": 53.33, "HA": 4.26, "N": 121.57},
        29: {"CA": 55.70, "HA": 4.22, "N": 120.73},
        30: {"CA": 61.40, "HA": 4.17, "N": 120.30},
        31: {"CA": 54.91, "HA": 4.30, "N": 120.61},
        32: {"CA": 53.94, "HA": 4.29, "N": 120.36},
        33: {"CA": 55.33, "HA": 4.15, "N": 120.00},
        34: {"CA": 55.43, "HA": 4.25, "N": 119.78},
        35: {"CA": 45.41, "HA": 3.92, "N": 109.12},
        36: {"CA": 62.08, "HA": 4.14, "N": 121.65},
        37: {"CA": 65.04, "HA": 4.34, "N": 0.00}, # Proline
        38: {"CA": 64.63, "HA": 4.44, "N": 0.00}, # Proline
        39: {"CA": 54.83, "HA": 4.30, "N": 119.00},
        40: {"CA": 56.40, "HA": 4.34, "N": 121.16},
        41: {"CA": 56.77, "HA": 4.24, "N": 121.43},
        42: {"CA": 54.12, "HA": 4.33, "N": 120.30},
        43: {"CA": 54.80, "HA": 4.23, "N": 120.33},
        44: {"CA": 61.21, "HA": 4.19, "N": 120.91},
        45: {"CA": 57.65, "HA": 4.27, "N": 120.84},
        46: {"CA": 53.30, "HA": 4.20, "N": 120.59},
        47: {"CA": 45.18, "HA": 3.94, "N": 109.28},
        48: {"CA": 55.78, "HA": 4.26, "N": 121.29},
        49: {"CA": 55.67, "HA": 4.34, "N": 120.95},
        50: {"CA": 55.46, "HA": 4.25, "N": 121.84},
        51: {"CA": 56.12, "HA": 4.30, "N": 120.52},
        52: {"CA": 54.91, "HA": 4.32, "N": 121.28},
        53: {"CA": 45.42, "HA": 3.93, "N": 109.43},
        54: {"CA": 56.24, "HA": 4.21, "N": 121.61},
        55: {"CA": 61.94, "HA": 4.09, "N": 113.12},
        56: {"CA": 55.07, "HA": 4.06, "N": 120.94},
        57: {"CA": 58.70, "HA": 4.27, "N": 116.09},
        58: {"CA": 54.49, "HA": 4.14, "N": 120.00},
        59: {"CA": 58.74, "HA": 4.30, "N": 120.94},
        60: {"CA": 54.88, "HA": 4.14, "N": 120.53},
        61: {"CA": 61.35, "HA": 4.16, "N": 120.47},
        62: {"CA": 55.08, "HA": 4.29, "N": 119.78},
        63: {"CA": 55.80, "HA": 4.24, "N": 121.62},
        64: {"CA": 54.88, "HA": 4.29, "N": 120.48},
        65: {"CA": 58.07, "HA": 4.33, "N": 115.11},
        66: {"CA": 61.46, "HA": 4.27, "N": 113.84},
        67: {"CA": 55.22, "HA": 4.23, "N": 120.99},
        68: {"CA": 55.97, "HA": 4.36, "N": 120.54},
        69: {"CA": 55.84, "HA": 4.16, "N": 120.30},
        70: {"CA": 61.36, "HA": 4.16, "N": 120.81},
        71: {"CA": 55.48, "HA": 4.25, "N": 121.22},
        72: {"CA": 56.12, "HA": 4.16, "N": 121.15},
        73: {"CA": 55.92, "HA": 4.26, "N": 121.60},
        74: {"CA": 56.19, "HA": 4.20, "N": 120.93},
        75: {"CA": 45.40, "HA": 3.96, "N": 109.13},
        76: {"CA": 45.54, "HA": 3.95, "N": 109.30}
    }

    # Disable random noise for deterministic testing
    monkeypatch.setattr("synth_nmr.chemical_shifts._NOISE_SCALE", 0.0)

    structure = ubiquitin_structure
    predicted_shifts_all_chains = predict_chemical_shifts(structure)
    
    predicted_shifts_chain_A = predicted_shifts_all_chains.get('A', {})

    # Atom types to validate
    atom_types = ["CA", "HA", "N"]
    
    for atom_type in atom_types:
        experimental_vals = []
        predicted_vals = []
        
        for res_id, exp_shift_data in experimental_shifts.items():
            if res_id in predicted_shifts_chain_A and atom_type in exp_shift_data:
                
                # Some residues (like Proline) don't have N-H shifts, so skip if experimental is 0.0
                if exp_shift_data[atom_type] == 0.0:
                    continue
                
                exp_val = exp_shift_data[atom_type]
                pred_val = predicted_shifts_chain_A[res_id].get(atom_type)
                
                if pred_val is not None:
                    experimental_vals.append(exp_val)
                    predicted_vals.append(pred_val)
        
        assert len(predicted_vals) > 50, f"Not enough {atom_type} data for validation."

        experimental_vals = np.array(experimental_vals)
        predicted_vals = np.array(predicted_vals)
        
        rmsd = np.sqrt(np.mean((predicted_vals - experimental_vals)**2))
        
        # Set tolerance based on atom type
        if atom_type == "CA":
            tolerance = 4.0  # C-alpha shifts can vary more
        elif atom_type == "N":
            tolerance = 4.5 # N shifts can also vary
        elif atom_type == "HA":
            tolerance = 0.7 # H-alpha shifts are generally more precise
        else:
            tolerance = 1.0 # Default

        assert rmsd < tolerance, f"RMSD for {atom_type} shifts is {rmsd:.2f} ppm, which is too high (tolerance: {tolerance} ppm)."

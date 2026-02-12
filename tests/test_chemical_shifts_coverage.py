import pytest
import numpy as np
import biotite.structure as struc
from synth_pdb.chemical_shifts import (
    predict_chemical_shifts,
    calculate_csi,
    _get_aromatic_rings,
    _calculate_ring_current_shift
)
import synth_pdb.chemical_shifts as cs_mod

def test_ring_current_trp_coverage():
    """Test TRP ring identification and its effect on shifts."""
    # Create a structure with a TRP and a nearby Glycine to measure shifts
    # TRP at origin, GLY nearby
    residues = []
    
    # TRP Indole atoms
    trp_atoms = [
        struc.Atom(res_id=1, res_name="TRP", atom_name="N", coord=[-1,0,0]),
        struc.Atom(res_id=1, res_name="TRP", atom_name="CA", coord=[0,0,0]),
        struc.Atom(res_id=1, res_name="TRP", atom_name="C", coord=[0,1,0]),
        struc.Atom(res_id=1, res_name="TRP", atom_name="CB", coord=[1,0,0]),
        struc.Atom(res_id=1, res_name="TRP", atom_name="CG", coord=[2,0,0]),
        struc.Atom(res_id=1, res_name="TRP", atom_name="CD1", coord=[2,1,0]),
        struc.Atom(res_id=1, res_name="TRP", atom_name="CD2", coord=[3,0,0]),
        struc.Atom(res_id=1, res_name="TRP", atom_name="NE1", coord=[3,1,0]),
        struc.Atom(res_id=1, res_name="TRP", atom_name="CE2", coord=[4,0,0]),
        struc.Atom(res_id=1, res_name="TRP", atom_name="CE3", coord=[4,1,0]),
        struc.Atom(res_id=1, res_name="TRP", atom_name="CZ2", coord=[5,0,0]),
        struc.Atom(res_id=1, res_name="TRP", atom_name="CZ3", coord=[5,1,0]),
        struc.Atom(res_id=1, res_name="TRP", atom_name="CH2", coord=[6,0,0]),
    ]
    
    # Nearby GLY with HA
    gly_atoms = [
        struc.Atom(res_id=2, res_name="GLY", atom_name="N", coord=[4,0,3]), # 3A above ring plane
        struc.Atom(res_id=2, res_name="GLY", atom_name="CA", coord=[4,0,4]),
        struc.Atom(res_id=2, res_name="GLY", atom_name="HA", coord=[4,0,5]),
    ]
    
    peptide = struc.array(trp_atoms + gly_atoms)
    peptide.chain_id[:] = "A"
    
    # 1. Test _get_aromatic_rings for TRP
    rings = _get_aromatic_rings(peptide)
    assert rings.shape[0] > 0
    assert rings[0, 6] == 1.3 # TRP intensity
    
    # 2. Test predict_chemical_shifts (triggers _calculate_ring_current_shift)
    shifts = predict_chemical_shifts(peptide)
    assert "A" in shifts
    assert 2 in shifts["A"]
    # Check that HA of GLY has a shift
    assert "HA" in shifts["A"][2]

def test_ring_current_his_coverage():
    """Test HIS ring identification (HIS, HID, HIE, HIP)."""
    atoms = [
        struc.Atom(res_id=1, res_name="HIS", atom_name="CG", coord=[0,0,0]),
        struc.Atom(res_id=1, res_name="HIS", atom_name="ND1", coord=[1,0,0]),
        struc.Atom(res_id=1, res_name="HIS", atom_name="CD2", coord=[0,1,0]),
        struc.Atom(res_id=1, res_name="HIS", atom_name="CE1", coord=[1,1,0]),
        struc.Atom(res_id=1, res_name="HIS", atom_name="NE2", coord=[0.5,1.5,0]),
    ]
    peptide = struc.array(atoms)
    rings = _get_aromatic_rings(peptide)
    assert rings.shape[0] == 1
    assert rings[0, 6] == 0.5 # HIS intensity

def test_ring_current_calculation_singularity():
    """Test safe handling when proton is too close to ring center."""
    rings = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.2]]) # Center [0,0,0], Normal [0,0,1]
    proton_coord = np.array([0.1, 0.1, 0.1]) # Very close
    
    shift = _calculate_ring_current_shift(proton_coord, rings)
    assert shift == 0.0 # SINGULARITY check should trigger

def test_predict_shifts_no_random_coil_entry():
    """Test handling of unknown residue types."""
    atom = struc.Atom(res_id=1, res_name="XXX", atom_name="CA", coord=[0,0,0], chain_id="A")
    peptide = struc.array([atom])
    shifts = predict_chemical_shifts(peptide)
    assert "A" not in shifts # Should be skipped

def test_calculate_csi_missing_res_in_coil_shifts():
    """Test CSI calculation when residue is not in reference map."""
    shifts = {"A": {1: {"CA": 55.0}}}
    # Create structure where residue 1 is "XXX"
    atom = struc.Atom(res_id=1, res_name="XXX", atom_name="CA", coord=[0,0,0], chain_id="A")
    peptide = struc.array([atom])
    
    csi = calculate_csi(shifts, peptide)
    assert 1 not in csi["A"]



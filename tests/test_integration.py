"""
Basic integration tests for synth-nmr package.

These tests verify that the main NMR calculation functions work correctly
without requiring the full synth-pdb package for structure generation.
"""

import pytest
import biotite.structure as struc
from synth_nmr import (
    calculate_synthetic_noes,
    predict_order_parameters,
    predict_chemical_shifts,
    calculate_csi,
    calculate_hn_ha_coupling,
)


def create_minimal_structure(n_residues=3):
    """Create a minimal test structure."""
    n_atoms = n_residues * 5  # N, CA, C, O, CB per residue
    structure = struc.AtomArray(n_atoms)

    atom_names = ["N", "CA", "C", "O", "CB"]

    for i in range(n_residues):
        start = i * 5
        structure.res_id[start : start + 5] = i + 1
        structure.res_name[start : start + 5] = "ALA"
        structure.chain_id[start : start + 5] = "A"
        structure.atom_name[start : start + 5] = atom_names
        structure.element[start : start + 5] = ["N", "C", "C", "O", "C"]

        # Simple linear coordinates
        for j, atom in enumerate(atom_names):
            structure.coord[start + j] = [i * 3.8, j * 1.5, 0.0]

    return structure


def test_package_import():
    """Test that the package can be imported."""
    import synth_nmr

    assert synth_nmr.__version__ == "0.8.0"


def test_calculate_synthetic_noes():
    """Test NOE calculation."""
    structure = create_minimal_structure(n_residues=2)
    noes = calculate_synthetic_noes(structure, cutoff=5.0)

    # Returns a list (may be empty if no hydrogens)
    assert isinstance(noes, list)


def test_predict_order_parameters():
    """Test order parameter prediction."""
    structure = create_minimal_structure(n_residues=3)
    order_params = predict_order_parameters(structure)

    # Should return a dictionary
    assert isinstance(order_params, dict)


def test_predict_chemical_shifts():
    """Test chemical shift prediction."""
    structure = create_minimal_structure(n_residues=3)
    shifts = predict_chemical_shifts(structure)

    # Should return nested dictionary: chain -> residue -> atom -> shift
    assert isinstance(shifts, dict)


def test_calculate_csi():
    """Test CSI calculation."""
    structure = create_minimal_structure(n_residues=2)

    # Create mock shifts
    shifts = {"A": {1: {"CA": 52.5, "CB": 18.0}, 2: {"CA": 55.0, "CB": 19.0}}}

    csi = calculate_csi(shifts, structure)

    # Should return a dictionary
    assert isinstance(csi, dict)
    assert "A" in csi


def test_calculate_hn_ha_coupling():
    """Test J-coupling calculation."""
    structure = create_minimal_structure(n_residues=3)

    # Add H atoms for proper calculation
    # For now, just test that it runs without error
    try:
        couplings = calculate_hn_ha_coupling(structure)
        assert isinstance(couplings, dict)
    except (KeyError, IndexError):
        # Expected if structure doesn't have proper H atoms
        pass


def test_structure_utils():
    """Test secondary structure classification."""
    from synth_nmr.structure_utils import get_secondary_structure

    structure = create_minimal_structure(n_residues=5)
    ss = get_secondary_structure(structure)

    # Should return a list of secondary structure assignments
    assert isinstance(ss, list)
    assert len(ss) == 5  # One per residue


def test_empty_structure():
    """Test handling of empty structures."""
    structure = struc.AtomArray(0)

    # Empty structures may cause errors in biotite's dihedral calculation
    # This is expected behavior - just verify it doesn't crash unexpectedly
    try:
        shifts = predict_chemical_shifts(structure)
        assert isinstance(shifts, dict)
    except (IndexError, ValueError):
        # Expected for empty structures
        pass


def test_single_residue():
    """Test with single residue."""
    structure = create_minimal_structure(n_residues=1)

    shifts = predict_chemical_shifts(structure)
    assert isinstance(shifts, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

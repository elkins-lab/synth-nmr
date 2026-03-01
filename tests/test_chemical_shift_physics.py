import math

# These imports will fail initially because the helper functions do not exist yet.
from synth_nmr.chemical_shifts import (
    _get_random_coil_shifts,
    _apply_secondary_structure_offsets,
)


def test_get_random_coil_shifts():
    """
    Test the extraction of baseline Random Coil shifts for a given residue.

    Expected Rules:
    - Should return the dictionary of base shifts for valid residues (e.g. ALA).
    - Should return None or an empty dict for invalid/non-standard residues, allowing the main loop to skip safely.
    """
    # Valid residue
    ala_shifts = _get_random_coil_shifts("ALA")
    assert isinstance(ala_shifts, dict)
    assert "CA" in ala_shifts
    assert ala_shifts["CA"] == 52.5

    # Invalid residue
    invalid_shifts = _get_random_coil_shifts("XYZ")
    assert invalid_shifts == {}


def test_apply_secondary_structure_offsets():
    """
    Test the statistical SPARTA+ style offsets based on local geometry.

    Expected Rules:
    - Alpha Helix CA: +3.1
    - Beta Sheet CA: -1.5
    - Random Coil CA: 0.0
    """
    base_val = 50.0

    # Alpha Helix CA
    helix_val = _apply_secondary_structure_offsets("CA", "alpha", base_val)
    assert math.isclose(helix_val, 53.1, rel_tol=1e-5)

    # Beta Sheet CA
    sheet_val = _apply_secondary_structure_offsets("CA", "beta", base_val)
    assert math.isclose(sheet_val, 48.5, rel_tol=1e-5)

    # Random Coil / Unstructured
    coil_val = _apply_secondary_structure_offsets("CA", "coil", base_val)
    assert math.isclose(coil_val, 50.0, rel_tol=1e-5)

    # Atom type with no defined offset (e.g., an imaginary sidechain atom)
    unmapped_val = _apply_secondary_structure_offsets("CX", "alpha", base_val)
    assert math.isclose(unmapped_val, 50.0, rel_tol=1e-5)

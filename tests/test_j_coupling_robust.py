import pytest
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import os
from synth_nmr.j_coupling import (
    calculate_hn_ha_coupling,
    calculate_ha_hb_coupling,
    calculate_c_cg_coupling,
)


def test_j_coupling_on_real_pdb():
    """Test J-coupling calculations on a real PDB file (1UBQ)."""
    pdb_path = os.path.join("data", "1UBQ.pdb")
    if not os.path.exists(pdb_path):
        pytest.skip("1UBQ.pdb not found in data/")

    pdb_file = pdb.PDBFile.read(pdb_path)
    structure = pdb_file.get_structure()
    if isinstance(structure, struc.AtomArrayStack):
        structure = structure[0]

    # 1. HN-HA Couplings
    j_hnha = calculate_hn_ha_coupling(structure)
    assert len(j_hnha) > 0
    assert "A" in j_hnha
    # Ubiquitin has 76 residues. N-term (Res 1) has no Phi.
    # So we expect roughly 75 residues with J-couplings.
    assert len(j_hnha["A"]) >= 70

    # Check physical ranges
    for res_id, j_val in j_hnha["A"].items():
        assert 2.0 <= j_val <= 12.0

    # 2. Side-chain Ha-Hb
    j_hahb = calculate_ha_hb_coupling(structure)
    assert len(j_hahb) > 0
    # Not all residues have chi1 (Gly, Ala don't)
    assert len(j_hahb["A"]) < 70

    # 3. Side-chain C'-Cg
    j_ccg = calculate_c_cg_coupling(structure)
    assert len(j_ccg) > 0
    assert len(j_ccg["A"]) == len(j_hahb["A"])


def test_j_coupling_ensemble_averaging():
    """Test ensemble averaging logic for J-couplings."""
    from synth_nmr.trajectory import ensemble_average_j_couplings

    frame1 = {"A": {1: 4.0, 2: 9.0}}
    frame2 = {"A": {1: 6.0, 2: 7.0}}

    avg = ensemble_average_j_couplings([frame1, frame2])

    assert avg["A"][1] == 5.0
    assert avg["A"][2] == 8.0


def test_j_coupling_missing_residue_handling():
    """Test that missing residues in some frames are excluded from average."""
    from synth_nmr.trajectory import ensemble_average_j_couplings

    frame1 = {"A": {1: 4.0, 2: 9.0}}
    frame2 = {"A": {1: 6.0}}  # Res 2 missing

    avg = ensemble_average_j_couplings([frame1, frame2])

    assert 1 in avg["A"]
    assert 2 not in avg["A"]


def test_chi1_atom_selection_diversity():
    """Test that chi1 angles are correctly calculated for different amino acids."""
    # Create structure with VAL (CG1), SER (OG), CYS (SG), THR (OG1)
    atoms = []

    # 1. VAL - uses CG1 (or CG2)
    atoms.extend(
        [
            struc.Atom(
                atom_name="N", res_id=1, res_name="VAL", chain_id="A", element="N", coord=[0, 0, 0]
            ),
            struc.Atom(
                atom_name="CA", res_id=1, res_name="VAL", chain_id="A", element="C", coord=[1, 0, 0]
            ),
            struc.Atom(
                atom_name="CB", res_id=1, res_name="VAL", chain_id="A", element="C", coord=[1, 1, 0]
            ),
            struc.Atom(
                atom_name="CG1",
                res_id=1,
                res_name="VAL",
                chain_id="A",
                element="C",
                coord=[0, 1, 1],
            ),
        ]
    )

    # 2. SER - uses OG
    atoms.extend(
        [
            struc.Atom(
                atom_name="N", res_id=2, res_name="SER", chain_id="A", element="N", coord=[0, 0, 0]
            ),
            struc.Atom(
                atom_name="CA", res_id=2, res_name="SER", chain_id="A", element="C", coord=[1, 0, 0]
            ),
            struc.Atom(
                atom_name="CB", res_id=2, res_name="SER", chain_id="A", element="C", coord=[1, 1, 0]
            ),
            struc.Atom(
                atom_name="OG", res_id=2, res_name="SER", chain_id="A", element="O", coord=[0, 1, 1]
            ),
        ]
    )

    # 3. CYS - uses SG
    atoms.extend(
        [
            struc.Atom(
                atom_name="N", res_id=3, res_name="CYS", chain_id="A", element="N", coord=[0, 0, 0]
            ),
            struc.Atom(
                atom_name="CA", res_id=3, res_name="CYS", chain_id="A", element="C", coord=[1, 0, 0]
            ),
            struc.Atom(
                atom_name="CB", res_id=3, res_name="CYS", chain_id="A", element="C", coord=[1, 1, 0]
            ),
            struc.Atom(
                atom_name="SG", res_id=3, res_name="CYS", chain_id="A", element="S", coord=[0, 1, 1]
            ),
        ]
    )

    structure = struc.array(atoms)

    j_hahb = calculate_ha_hb_coupling(structure)
    j_ccg = calculate_c_cg_coupling(structure)

    assert 1 in j_hahb["A"]
    assert 2 in j_hahb["A"]
    assert 3 in j_hahb["A"]
    assert 1 in j_ccg["A"]
    assert 2 in j_ccg["A"]
    assert 3 in j_ccg["A"]

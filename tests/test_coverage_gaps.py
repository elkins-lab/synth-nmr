import logging
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

import biotite.structure as struc
import numpy as np
import pytest

from synth_nmr import synth_nmr_cli as cli
from synth_nmr.j_coupling import calculate_hn_ha_coupling, predict_couplings_from_phi_map
from synth_nmr.neural_shifts import NeuralShiftPredictor, build_graph_data
from synth_nmr.synth_nmr_cli import handle_command
from synth_nmr.trajectory import _mdtraj_to_stack, load_trajectory


def test_mdtraj_to_stack_mock() -> None:
    """Verify conversion from MDTraj to Biotite stack with scaling."""
    mock_traj = MagicMock()
    mock_atom = MagicMock()
    mock_atom.name = "CA"
    mock_atom.residue.name = "ALA"
    mock_atom.residue.resSeq = 1
    mock_atom.residue.chain.chain_id = "A"
    mock_atom.element.symbol = "C"

    mock_traj.topology.atoms = [mock_atom]
    mock_traj.topology.n_atoms = 1
    mock_traj.n_frames = 2
    # xyz in nm -> converted to Angstrom (* 10)
    mock_traj.xyz = np.array([[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]], dtype=np.float32)

    stack = _mdtraj_to_stack(mock_traj)

    assert stack.stack_depth() == 2
    assert stack.atom_name[0] == "CA"
    # 0.1 nm -> 1.0 A
    np.testing.assert_allclose(stack.coord[0, 0], [1.0, 2.0, 3.0])


def test_load_trajectory_mdtraj_mock() -> None:
    """Verify load_trajectory handles MDTraj objects via duck typing."""

    class MockMDTrajTraj:
        pass

    MockMDTrajTraj.__module__ = "mdtraj.core.trajectory"
    MockMDTrajTraj.__qualname__ = "Trajectory"

    mock_traj = MockMDTrajTraj()

    with patch("synth_nmr.trajectory._mdtraj_to_stack") as mock_conv:
        dummy_stack = struc.AtomArrayStack(2, 1)
        dummy_stack.res_id = np.array([1])
        dummy_stack.res_name = np.array(["ALA"])
        dummy_stack.atom_name = np.array(["CA"])
        dummy_stack.coord = np.zeros((2, 1, 3))

        mock_conv.return_value = dummy_stack

        ensemble = load_trajectory(mock_traj)
        assert ensemble.stack.stack_depth() == 2


def test_neural_shifts_gnn_branch() -> None:
    """Verify NeuralShiftPredictor with GNN model type."""
    pytest.importorskip("torch")
    # Initialise with random weights
    with patch("os.path.exists", return_value=False):
        predictor = NeuralShiftPredictor(model_type="gnn")
        assert predictor.model_type == "gnn"

        # Create a dummy structure
        structure = struc.AtomArray(3)
        structure.res_id = np.array([1, 1, 1])
        structure.res_name = np.array(["ALA"] * 3)
        structure.chain_id = np.array(["A"] * 3)
        structure.atom_name = np.array(["N", "CA", "C"])
        structure.element = np.array(["N", "C", "C"])
        structure.coord = np.random.rand(3, 3).astype(np.float32)

        # Mock empirical shifts return
        with patch(
            "synth_nmr.chemical_shifts.predict_empirical_shifts",
            return_value={"A": {1: {"N": 120.0, "CA": 55.0, "C": 175.0}}},
        ):
            shifts = predictor.predict(structure)
            assert "A" in shifts


def test_neural_shifts_empty_graph() -> None:
    """Verify build_graph_data returns None for empty/non-protein structures."""
    pytest.importorskip("torch")
    # Non-protein structure (HOH)
    structure = struc.AtomArray(1)
    structure.res_name = np.array(["HOH"])
    structure.atom_name = np.array(["O"])
    structure.element = np.array(["O"])

    data = build_graph_data(structure)
    assert data is None


def test_cli_malformed_rdc_file(tmp_path: Any) -> None:
    """Verify CLI handles malformed RDC validation files gracefully."""
    rdc_file = tmp_path / "bad_rdc.txt"
    rdc_file.write_text("ResID,Value\n1,invalid_float\n2,5.0")

    # Setup state
    cli.structure = struc.AtomArray(5)
    cli.structure.res_id = np.array([1, 2, 3, 4, 5])
    cli.structure.atom_name = np.array(["N", "CA", "C", "N", "CA"])  # minimal backbone
    cli.structure.res_name = np.array(["ALA"] * 5)
    cli.structure.chain_id = np.array(["A"] * 5)
    cli.structure.coord = np.zeros((5, 3))

    with patch("sys.stdout", new=StringIO()) as fake_out:
        handle_command(["validate", "rdc", str(rdc_file)])
        output = fake_out.getvalue()
        # Should finish without crashing and print header
        assert "RDC Validation" in output


def test_cli_export_no_structure() -> None:
    """Verify CLI error when exporting without a structure."""
    cli.structure = None
    with patch("sys.stdout", new=StringIO()) as fake_out:
        handle_command(["export", "nef", "out.nef"])
        output = fake_out.getvalue()
        assert "Error: No PDB file loaded" in output


def test_j_coupling_phi_map() -> None:
    """Verify predict_couplings_from_phi_map logic."""
    phi_map = {1: -60.0, 2: -120.0}
    couplings = predict_couplings_from_phi_map(phi_map)
    assert 1 in couplings
    assert 2 in couplings
    assert isinstance(couplings[1], float)


def test_j_coupling_length_mismatch(caplog: Any) -> None:
    """Verify warning on dihedral/residue count mismatch."""
    # Structure with 2 residues
    structure = struc.AtomArray(6)
    structure.res_id = np.array([1, 1, 1, 2, 2, 2])
    structure.atom_name = np.array(["N", "CA", "C", "N", "CA", "C"])
    structure.res_name = np.array(["ALA"] * 6)
    structure.chain_id = np.array(["A"] * 6)

    # Mock dihedral_backbone to return length 1 (mismatch with 2 residues)
    with caplog.at_level(logging.WARNING):
        with patch(
            "biotite.structure.dihedral_backbone",
            return_value=(np.array([-1.0]), np.array([-1.0]), np.array([-1.0])),
        ):
            res = calculate_hn_ha_coupling(structure)
            assert res == {}
            assert "Mismatch in backbone angles count" in caplog.text

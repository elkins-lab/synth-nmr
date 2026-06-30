import sys
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

import biotite.structure as struc
import numpy as np
import pytest

from synth_nmr.data_pipeline import load_matched_dataset, parse_bmrb_shifts
from synth_nmr.j_coupling import calculate_hn_ha_coupling
from synth_nmr.neural_shifts import NeuralShiftPredictor
from synth_nmr.relaxation import calculate_relaxation_rates
from synth_nmr.synth_nmr_cli import handle_command, main
from synth_nmr.trajectory import TrajectoryEnsemble, ensemble_average_j_couplings, load_trajectory
from synth_nmr.validation import (
    calculate_cs_r_factor,
    calculate_rdc_q_factor,
    validate_against_bmrb,
)


def test_cli_missing_command() -> None:
    args = MagicMock()
    args.command = None
    assert handle_command(args) is True


def test_cli_exceptions_during_ensemble(tmp_path: Any) -> None:
    # 239-240, 259-260
    import synth_nmr.synth_nmr_cli as cli

    stack = struc.AtomArrayStack(2, 1)
    stack.coord = np.zeros((2, 1, 3))
    stack.atom_name = np.array(["CA"])
    stack.res_name = np.array(["ALA"])
    stack.res_id = np.array([1])
    stack.chain_id = np.array(["A"])
    cli.ensemble = stack
    cli.structure = stack[0]

    args = MagicMock()
    args.command = "predict"
    args.subcommand = "shifts"
    args.ensemble = True

    with patch(
        "synth_nmr.synth_nmr_cli.predict_chemical_shifts", side_effect=Exception("mocked shift err")
    ):
        handle_command(args)

    args.subcommand = "noes"
    with patch(
        "synth_nmr.synth_nmr_cli.calculate_synthetic_noes", side_effect=Exception("mocked noe err")
    ):
        handle_command(args)


def test_cli_no_structure_checks() -> None:
    import synth_nmr.synth_nmr_cli as cli

    cli.structure = None

    for cmd, sub in [("calculate", "rdc"), ("predict", "shifts"), ("validate", "shifts")]:
        args = MagicMock()
        args.command = cmd
        args.subcommand = sub
        assert handle_command(args) is True


def test_cli_validate_structure() -> None:
    import synth_nmr.synth_nmr_cli as cli

    stack = struc.AtomArray(1)
    stack.coord = np.zeros((1, 3))
    stack.atom_name = np.array(["CA"])
    stack.res_name = np.array(["ALA"])
    stack.res_id = np.array([1])
    stack.chain_id = np.array(["A"])
    cli.structure = stack

    args = MagicMock()
    args.command = "validate"
    args.subcommand = "structure"

    with patch(
        "synth_nmr.synth_nmr_cli.calculate_c_beta_deviations", return_value={1: 0.3, 2: 0.1}
    ):
        with patch("sys.stdout", new=StringIO()):
            handle_command(args)


def test_cli_validate_rdc_no_file() -> None:
    import synth_nmr.synth_nmr_cli as cli

    cli.structure = struc.AtomArray(1)
    args = MagicMock()
    args.command = "validate"
    args.subcommand = "rdc"
    args.filename = "doesnotexist.txt"
    assert handle_command(args) is True


def test_cli_main_interactive() -> None:
    # 461, 464, 468-471
    with patch("sys.argv", ["synth_nmr_cli"]):
        with patch("sys.stdin.readline", side_effect=["\n", "invalid_cmd", EOFError()]):
            with patch("sys.stdout", new=StringIO()):
                main()

        with patch("sys.stdin.readline", side_effect=["raise_err", EOFError()]):
            with patch("synth_nmr.synth_nmr_cli.handle_command", side_effect=Exception("mock err")):
                with patch("sys.stdout", new=StringIO()):
                    main()


def test_trajectory_empty_stack() -> None:
    stack = struc.AtomArrayStack(0, 1)
    with pytest.raises(ValueError):
        TrajectoryEnsemble(stack=stack)


def test_trajectory_load_paths(tmp_path: Any) -> None:
    # 265-270
    stack = struc.AtomArrayStack(1, 1)
    stack.coord = np.zeros((1, 1, 3))
    stack.res_id = np.array([1])
    ens = load_trajectory(stack)
    assert ens.stack.stack_depth() == 1

    # 304-307
    mock_mdtraj = MagicMock()
    sys.modules["mdtraj"] = mock_mdtraj
    top = mock_mdtraj.Topology()
    ch = top.add_chain()
    res = top.add_residue("ALA", ch)
    top.add_atom("CA", mock_mdtraj.element.carbon, res)
    mock_mdtraj.Trajectory(xyz=np.zeros((1, 1, 3)), topology=top)

    with patch("synth_nmr.trajectory._mdtraj_to_stack", return_value=struc.AtomArrayStack(1, 1)):
        ens2 = load_trajectory("test.xtc", topology="test.pdb")
    assert ens2.stack.stack_depth() == 1


def test_trajectory_j_empty() -> None:
    with pytest.raises(ValueError):
        ensemble_average_j_couplings([])


def test_validation_rcs() -> None:
    # 246
    pred = {"A": {1: {"CA": 55.0}}}
    ref = {"B": {1: {"CA": 55.0}}}
    assert calculate_cs_r_factor(pred, ref) == 0.0

    # 262-263
    ref = {"A": {1: {"CA": 55.0}}}
    # Denominator zero -> 280
    assert calculate_cs_r_factor(pred, ref, res_name_map={1: "ALA"}) == 0.0
    assert calculate_cs_r_factor(pred, ref) == 0.0


def test_validation_q_factor() -> None:
    assert calculate_rdc_q_factor({1: 0.0}, {1: 0.0}) == 1.0


def test_validation_bmrb() -> None:
    # 353
    with patch("synth_nmr.data_pipeline.download_bmrb_file", return_value=None):
        with pytest.raises(RuntimeError):
            validate_against_bmrb(1, struc.AtomArray(1))

    # 359
    with patch("synth_nmr.data_pipeline.download_bmrb_file", return_value="dummy"):
        with patch("synth_nmr.data_pipeline.parse_bmrb_shifts", return_value={}):
            mock_predictor = MagicMock()
            mock_predictor.predict.return_value = {}
            validate_against_bmrb(1, struc.AtomArray(1), predictor=mock_predictor)
            assert mock_predictor.predict.called


def test_train_missing() -> None:
    pytest.importorskip("torch")
    from synth_nmr.scripts.train_gnn import train
    with patch("synth_nmr.scripts.train_gnn.load_matched_dataset", return_value=[]):
        train()


def test_data_pipeline_parse_bmrb_empty(tmp_path: Any) -> None:
    f = tmp_path / "bmrb.str"
    f.write_text("dummy")
    assert parse_bmrb_shifts(str(f)) == {}

    load_matched_dataset("nonexistent")


def test_neural_shifts_branches() -> None:
    pytest.importorskip("torch")
    # Coverage for empty neural shifts
    p = NeuralShiftPredictor()
    arr = struc.AtomArray(1)
    arr.res_id = np.array([1])
    arr.res_name = np.array(["HOH"])
    arr.atom_name = np.array(["O"])
    arr.element = np.array(["O"])
    p.predict(arr)


def test_relaxation_branches() -> None:
    arr = struc.AtomArray(1)
    arr.res_id = np.array([1])
    arr.res_name = np.array(["ALA"])
    arr.atom_name = np.array(["CA"])
    arr.coord = np.zeros((1, 3))
    # just hitting code paths
    calculate_relaxation_rates(arr, field_mhz=600.0)


def test_j_coupling_branches() -> None:
    arr = struc.AtomArray(1)
    arr.res_id = np.array([1])
    arr.res_name = np.array(["ALA"])
    arr.atom_name = np.array(["CA"])
    calculate_hn_ha_coupling(arr)

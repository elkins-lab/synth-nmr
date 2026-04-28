import runpy
import subprocess
import sys
from unittest import mock

import biotite.structure as struc
import numpy as np
import pytest


def _make_dummy_dataset():
    structure = struc.AtomArray(3)
    structure.res_id = np.array([1, 2, 3])
    structure.res_name = np.array(["ALA", "GLY", "VAL"])
    structure.chain_id = np.array(["A", "A", "A"])
    structure.atom_name = np.array(["CA", "CA", "CA"])
    structure.coord = np.zeros((3, 3))

    exp_shifts = {1: {"CA": 50.0}, 2: {"CA": 45.0}, 3: {"CA": 60.0}}

    return [(structure, exp_shifts)]


def test_train_gnn_script_execution(mocker, tmp_path):
    pytest.importorskip("torch")

    mocker.patch(
        "synth_nmr.scripts.train_gnn.load_matched_dataset", return_value=_make_dummy_dataset()
    )

    test_args = [
        "train_gnn.py",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--save-path",
        str(tmp_path / "model.pt"),
    ]
    with mock.patch.object(sys, "argv", test_args):
        # Clear module from sys.modules to avoid RuntimeWarning when running as script
        if "synth_nmr.scripts.train_gnn" in sys.modules:
            del sys.modules["synth_nmr.scripts.train_gnn"]
        runpy.run_module("synth_nmr.scripts.train_gnn", run_name="__main__")

    assert (tmp_path / "model.pt").exists()


def test_train_gnn_train_function(mocker, tmp_path):
    pytest.importorskip("torch")
    from synth_nmr.scripts import train_gnn

    mocker.patch(
        "synth_nmr.scripts.train_gnn.load_matched_dataset", return_value=_make_dummy_dataset()
    )

    train_gnn.train(epochs=1, save_path=str(tmp_path / "model2.pt"))
    assert (tmp_path / "model2.pt").exists()


def test_train_gnn_empty_dataset(mocker, caplog):
    pytest.importorskip("torch")
    from synth_nmr.scripts import train_gnn

    mocker.patch("synth_nmr.scripts.train_gnn.load_matched_dataset", return_value=[])

    train_gnn.train(epochs=1)
    assert "Empty dataset" in caplog.text


def test_train_gnn_import_error():
    # This runs the script in a fresh process without torch available
    script = "import sys\nsys.modules['torch'] = None\nimport synth_nmr.scripts.train_gnn\n"
    res = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert res.returncode == 1
    assert "PyTorch and PyTorch Geometric are required" in res.stdout

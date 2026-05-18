import logging
from unittest.mock import MagicMock, patch

import biotite.structure as struc
import numpy as np
import pytest

from synth_nmr.trajectory import (
    TrajectoryEnsemble,
    compute_s2_from_trajectory,
    ensemble_average_j_couplings,
    ensemble_average_noes,
    ensemble_average_rdcs,
    ensemble_average_shifts,
    load_trajectory,
)


def create_dummy_atom_array(n_atoms=10):
    structure = struc.AtomArray(n_atoms)
    structure.coord = np.random.rand(n_atoms, 3).astype(np.float32)
    structure.res_id = np.ones(n_atoms, dtype=int)
    structure.res_name = np.array(["ALA"] * n_atoms)
    structure.chain_id = np.array(["A"] * n_atoms)
    structure.atom_name = np.array(["CA"] * n_atoms)
    structure.element = np.array(["C"] * n_atoms)
    return structure


def test_trajectory_ensemble_basics():
    frames = [create_dummy_atom_array() for _ in range(5)]
    ensemble = load_trajectory(frames)

    assert len(ensemble) == 5
    # Accessing by index returns a frame view (AtomArray)
    assert np.allclose(ensemble[0].coord, frames[0].coord)
    assert len(ensemble[1:3]) == 2

    # Test iterator
    count = 0
    for _ in ensemble:
        count += 1
    assert count == 5


def test_trajectory_ensemble_errors():
    with pytest.raises(ValueError):
        load_trajectory([])


def test_load_trajectory_list():
    frames = [create_dummy_atom_array() for _ in range(3)]
    ensemble = load_trajectory(frames)
    assert isinstance(ensemble, TrajectoryEnsemble)
    assert len(ensemble) == 3


def test_ensemble_average_shifts():
    per_frame = [{1: {"CA": 50.0, "CB": 20.0}}, {1: {"CA": 51.0, "CB": 22.0}}]
    avg = ensemble_average_shifts(per_frame)
    assert avg[1]["CA"] == 50.5
    assert avg[1]["CB"] == 21.0

    with pytest.raises(ValueError):
        ensemble_average_shifts([])


def test_ensemble_average_noes_partial_coverage(caplog):
    # Test line 545: pair present in some but not all frames
    per_frame = [
        {(1, 2): 4.0, (1, 3): 5.0},
        {(1, 2): 4.2},  # (1, 3) missing in frame 2
    ]
    with caplog.at_level(logging.DEBUG):
        avg = ensemble_average_noes(per_frame)

    assert (1, 2) in avg
    assert (1, 3) not in avg
    assert "excluded from ensemble average" in caplog.text


def test_ensemble_average_rdcs():
    per_frame = [{1: 10.0, 2: 5.0}, {1: 12.0, 2: 7.0}]
    avg = ensemble_average_rdcs(per_frame)
    assert avg[1] == 11.0
    assert avg[2] == 6.0


def test_ensemble_average_j_couplings():
    per_frame = [{"A": {1: 7.5}}, {"A": {1: 8.5}}]
    avg = ensemble_average_j_couplings(per_frame)
    assert avg["A"][1] == 8.0


def test_compute_s2_from_trajectory():
    # Create an ensemble where a vector fluctuates
    frames = []
    for angle in [0.0, 0.1, -0.1]:
        struct = create_dummy_atom_array(2)
        # N-H vector along Z, slightly tilted
        struct.coord[0] = [0, 0, 0]  # N
        struct.coord[1] = [np.sin(angle), 0, np.cos(angle)]  # H
        struct.atom_name[0] = "N"
        struct.atom_name[1] = "H"
        frames.append(struct)

    ensemble = load_trajectory(frames)
    s2_map = compute_s2_from_trajectory(ensemble)
    assert 1 in s2_map
    assert 0 <= s2_map[1] <= 1.0


def test_load_trajectory_missing_topology_error():
    # Mock mdtraj so it appears to be installed
    mock_mdtraj = MagicMock()
    with patch.dict("sys.modules", {"mdtraj": mock_mdtraj}):
        with pytest.raises(ValueError) as excinfo:
            load_trajectory("some_file.xtc", topology=None)
        assert "A topology file path must be provided" in str(excinfo.value)


def test_load_trajectory_invalid_type():
    with pytest.raises(TypeError):
        load_trajectory(123)


def test_load_trajectory_no_mdtraj_error():
    with patch.dict("sys.modules", {"mdtraj": None}):
        with pytest.raises(ImportError) as excinfo:
            load_trajectory("some_file.xtc", topology="top.pdb")
        assert "requires MDTraj" in str(excinfo.value)

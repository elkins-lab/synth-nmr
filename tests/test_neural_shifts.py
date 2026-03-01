"""
TDD tests for synth_nmr.neural_shifts — NeuralShiftPredictor.

Tests are written BEFORE implementation (red → green → refactor).
Run with: pytest tests/test_neural_shifts.py -v
"""

import os
import sys
import tempfile
import unittest

import numpy as np


# ---------------------------------------------------------------------------
# Helpers — minimal synthetic biotite structure
# ---------------------------------------------------------------------------


def _make_test_structure(n_residues: int = 10, conformation: str = "alpha"):
    """Build a minimal biotite AtomArray for testing (backbone-only helix)."""
    import biotite.structure as struc

    AA_SEQUENCE = ["ALA", "GLY", "LEU", "VAL", "ILE", "SER", "THR", "ASP", "GLU", "LYS"]

    atoms = []
    atom_idx = 0

    # Backbone geometry for an ideal alpha helix
    # Using simplified geometry: 3.8 Å rise per residue, 100° rotation
    for i in range(n_residues):
        res_name = AA_SEQUENCE[i % len(AA_SEQUENCE)]
        res_id = i + 1
        angle = np.radians(i * 100.0)
        radius = 2.3  # Å from helix axis

        ca_x = radius * np.cos(angle)
        ca_y = radius * np.sin(angle)
        ca_z = i * 1.5  # ~1.5 Å rise per residue

        for atom_name, offset in [
            ("N", [-1.2, 0.0, -0.4]),
            ("CA", [0.0, 0.0, 0.0]),
            ("C", [1.2, 0.0, 0.4]),
            ("O", [1.5, 0.0, 1.5]),
        ]:
            a = struc.Atom(
                coord=[ca_x + offset[0], ca_y + offset[1], ca_z + offset[2]],
                chain_id="A",
                res_id=res_id,
                res_name=res_name,
                atom_name=atom_name,
                element=atom_name[0] if atom_name[0] in "NCOS" else "C",
                b_factor=10.0,
            )
            atoms.append(a)

    return struc.array(atoms)


# ---------------------------------------------------------------------------
# Feature extraction tests
# ---------------------------------------------------------------------------


class TestFeatureExtraction(unittest.TestCase):
    """Tests for build_residue_features() — the feature-engineering layer."""

    def setUp(self):
        self.structure = _make_test_structure(n_residues=10)
        # Import lazily so tests can run even without torch installed
        from synth_nmr.neural_shifts import build_residue_features

        self.build_residue_features = build_residue_features

    def test_feature_vector_length(self):
        """Each residue should be represented by a 74-dimensional vector."""
        X = self.build_residue_features(self.structure)
        self.assertEqual(X.shape[1], 74, f"Expected 74 features per residue, got {X.shape[1]}")

    def test_feature_row_count(self):
        """One feature row per residue."""
        X = self.build_residue_features(self.structure)
        import biotite.structure as struc

        n_res = struc.get_residue_count(self.structure)
        self.assertEqual(X.shape[0], n_res)

    def test_amino_acid_onehot_sums_to_one(self):
        """First 20 columns are one-hot → each row sums to 1."""
        X = self.build_residue_features(self.structure)
        aa_onehot = X[:, :20]
        row_sums = aa_onehot.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, err_msg="AA one-hot rows must sum to 1")

    def test_sin_cos_features_bounded(self):
        """sin/cos dihedral features must be in [−1, 1]."""
        X = self.build_residue_features(self.structure)
        # Columns 20–23: sin(φ), cos(φ), sin(ψ), cos(ψ)
        dihedral_feats = X[:, 20:24]
        self.assertTrue(np.all(dihedral_feats >= -1.0 - 1e-6))
        self.assertTrue(np.all(dihedral_feats <= 1.0 + 1e-6))

    def test_secondary_structure_onehot_sums_to_one(self):
        """Columns 24–26 are one-hot SS → each row sums to 1."""
        X = self.build_residue_features(self.structure)
        ss_onehot = X[:, 24:27]
        row_sums = ss_onehot.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, err_msg="SS one-hot rows must sum to 1")

    def test_sequence_position_bounded(self):
        """Sequence position feature must be in [0, 1]."""
        X = self.build_residue_features(self.structure)
        seq_pos = X[:, 27]
        self.assertTrue(np.all(seq_pos >= 0.0))
        self.assertTrue(np.all(seq_pos <= 1.0 + 1e-6))

    def test_neighbor_onehot_terminal_residues_zero(self):
        """
        The i−1 neighbor block (cols 34–53) of the first residue should be all zeros
        (no previous neighbour). Similarly, i+1 block (cols 54–73) of last residue should
        be all zeros.

        Feature layout:
          0-19  : current AA one-hot
          20-23 : sin/cos phi/psi
          24-26 : SS one-hot
          27    : seq position
          28-33 : random coil baseline
          34-53 : i-1 neighbour one-hot  ← tested here
          54-73 : i+1 neighbour one-hot  ← tested here
        """
        X = self.build_residue_features(self.structure)
        # i-1 block for first residue (index 0) — starts at col 34
        prev_first = X[0, 34:54]
        np.testing.assert_array_equal(
            prev_first, np.zeros(20), err_msg="N-terminal i-1 one-hot should be all zeros"
        )
        # i+1 block for last residue — starts at col 54
        next_last = X[-1, 54:74]
        np.testing.assert_array_equal(
            next_last, np.zeros(20), err_msg="C-terminal i+1 one-hot should be all zeros"
        )


# ---------------------------------------------------------------------------
# Model forward-pass tests
# ---------------------------------------------------------------------------


class TestNeuralShiftModel(unittest.TestCase):
    """Tests for the underlying nn.Module returned by NeuralShiftPredictor."""

    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from synth_nmr.neural_shifts import NeuralShiftPredictor

        self.predictor = NeuralShiftPredictor()  # random-weight model
        self.structure = _make_test_structure(n_residues=10)

    def test_forward_pass_output_shape(self):
        """Model output must be [N_residues, 6]."""
        import torch
        from synth_nmr.neural_shifts import build_residue_features

        X = build_residue_features(self.structure)
        x = torch.tensor(X, dtype=torch.float32)
        self.predictor.model.eval()
        with torch.no_grad():
            out = self.predictor.model(x)
        self.assertEqual(
            out.shape, (X.shape[0], 6), f"Expected [{X.shape[0]}, 6], got {tuple(out.shape)}"
        )

    def test_forward_pass_produces_finite_values(self):
        """Model output must not contain NaN or Inf."""
        import torch
        from synth_nmr.neural_shifts import build_residue_features

        X = build_residue_features(self.structure)
        x = torch.tensor(X, dtype=torch.float32)
        self.predictor.model.eval()
        with torch.no_grad():
            out = self.predictor.model(x)
        self.assertTrue(torch.isfinite(out).all(), "Model output contains NaN or Inf")


# ---------------------------------------------------------------------------
# Classifier API tests
# ---------------------------------------------------------------------------


class TestNeuralShiftPredictorAPI(unittest.TestCase):
    """Tests for NeuralShiftPredictor.predict() return contract."""

    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from synth_nmr.neural_shifts import NeuralShiftPredictor

        self.predictor = NeuralShiftPredictor()
        self.structure = _make_test_structure(n_residues=10)

    def test_predict_returns_dict(self):
        """predict() must return a nested dict {chain: {res_id: {atom: float}}}."""
        result = self.predictor.predict(self.structure)
        self.assertIsInstance(result, dict)

    def test_predict_has_chain_key(self):
        """Result must have at least one chain key."""
        result = self.predictor.predict(self.structure)
        self.assertGreater(len(result), 0)

    def test_predict_nucleus_keys(self):
        """Each residue dict must have a subset of [HA, CA, CB, C, N, H]."""
        result = self.predictor.predict(self.structure)
        valid_keys = {"HA", "CA", "CB", "C", "N", "H"}
        for chain, chain_data in result.items():
            for res_id, atom_shifts in chain_data.items():
                for key in atom_shifts:
                    self.assertIn(
                        key, valid_keys, f"Unexpected nucleus key '{key}' in residue {res_id}"
                    )

    def test_predict_values_are_floats_in_plausible_range(self):
        """Predicted shifts must be in a physically plausible range (0–220 ppm)."""
        result = self.predictor.predict(self.structure)
        for chain, chain_data in result.items():
            for res_id, atom_shifts in chain_data.items():
                for atom, val in atom_shifts.items():
                    self.assertIsInstance(val, float)
                    self.assertTrue(
                        0.0 <= val <= 220.0, f"{atom} shift {val:.2f} ppm out of expected range"
                    )

    def test_predict_same_format_as_empirical(self):
        """Neural and empirical predictors must return the same dict schema."""
        from synth_nmr.chemical_shifts import predict_chemical_shifts

        empirical = predict_chemical_shifts(self.structure)
        neural = self.predictor.predict(self.structure)

        self.assertEqual(
            set(empirical.keys()),
            set(neural.keys()),
            "Chain keys differ between empirical and neural predictors",
        )
        for chain in empirical:
            self.assertEqual(
                set(empirical[chain].keys()),
                set(neural[chain].keys()),
                f"Residue IDs differ in chain {chain}",
            )


# ---------------------------------------------------------------------------
# Checkpoint save/load test
# ---------------------------------------------------------------------------


class TestNeuralShiftCheckpoint(unittest.TestCase):

    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from synth_nmr.neural_shifts import NeuralShiftPredictor

        self.predictor = NeuralShiftPredictor()
        self.structure = _make_test_structure(n_residues=10)

    def test_save_and_load_roundtrip(self):
        """Neural ΔCS corrections must be identical before and after save/load.

        We compare the raw model output (not the full predict() output) to
        avoid non-determinism from the empirical predictor's Gaussian noise.
        """
        from synth_nmr.neural_shifts import NeuralShiftPredictor, build_residue_features
        import torch

        X = build_residue_features(self.structure)
        x = torch.tensor(X, dtype=torch.float32)

        self.predictor.model.eval()
        with torch.no_grad():
            delta_before = self.predictor.model(x).numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "test_model.pt")
            self.predictor.save(ckpt_path)

            loaded = NeuralShiftPredictor(model_path=ckpt_path)
            loaded.model.eval()
            with torch.no_grad():
                delta_after = loaded.model(x).numpy()

        np.testing.assert_allclose(
            delta_before,
            delta_after,
            rtol=0,
            atol=1e-5,
            err_msg="Neural ΔCS corrections differ after save/load roundtrip",
        )


# ---------------------------------------------------------------------------
# Import-safety test (no torch required at module level)
# ---------------------------------------------------------------------------


class TestImportSafety(unittest.TestCase):

    def test_module_imports_without_torch(self):
        """
        `synth_nmr.neural_shifts` must be importable even when torch is absent.
        We test this by reimporting the module with sys.modules manipulation.
        """
        saved = sys.modules.pop("synth_nmr.neural_shifts", None)
        saved_torch = sys.modules.pop("torch", None)

        # Patch torch to look absent
        sys.modules["torch"] = None

        try:
            # Should not raise ImportError here
            import synth_nmr.neural_shifts  # noqa: F401
        except ImportError:
            self.fail("Importing neural_shifts raised ImportError when torch is absent")
        except Exception:
            pass  # Any other exception (e.g., from reload) is fine
        finally:
            # Restore
            if saved_torch is not None:
                sys.modules["torch"] = saved_torch
            else:
                sys.modules.pop("torch", None)
            if saved is not None:
                sys.modules["synth_nmr.neural_shifts"] = saved
            else:
                sys.modules.pop("synth_nmr.neural_shifts", None)


if __name__ == "__main__":
    unittest.main()

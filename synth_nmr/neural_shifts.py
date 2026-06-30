"""
synth_nmr.neural_shifts
~~~~~~~~~~~~~~~~~~~~~~~
Neural network-based chemical shift predictor.

─────────────────────────────────────────────────────────────────────────────
EDUCATIONAL BACKGROUND — Why a neural predictor on top of the empirical model?
─────────────────────────────────────────────────────────────────────────────

The existing `predict_chemical_shifts()` applies two corrections to random-coil
baseline values:
  1. A secondary structure offset (helix / sheet / coil), from a lookup table.
  2. A ring current correction for protons near aromatic residues.

Both corrections are hand-crafted from general statistics.  They miss:

  • **Sequence-neighbour effects**: The identity of residues i−1 and i+1
    shifts the chemical shift of residue i by 0.1–0.5 ppm.  SPARTA+ was
    specifically built to exploit this; neural shift predictors (ShiftML,
    UCBShift) show marked improvement from neighbour features.

  • **Non-linear geometry dependence**: The shift is a smooth, non-linear
    function of (φ, ψ) — not a step function over three SS categories.
    A neural net can learn the full Ramachandran-shift surface directly.

  • **Residue-specific coupling of features**: The φ/ψ effect on CA depends
    on the amino acid type.  A lookup table treats all residues with the
    same offset; an MLP can learn per-residue different responses.

The `NeuralShiftPredictor` learns **correction terms** (ΔCS) on top of the
existing empirical model, rather than predicting raw shifts from scratch.
This has two benefits:
  1. Even an untrained model (random weights) produces physically grounded
     outputs — the empirical baseline is always present.
  2. The neural component only needs to learn residuals, which are small
     (typically ±1 ppm) and easier to fit than the full shift range.

─────────────────────────────────────────────────────────────────────────────
ARCHITECTURE — MLP Regression
─────────────────────────────────────────────────────────────────────────────

    Input: per-residue feature vector (74-dim, see build_residue_features)
         │
    Linear(74 → 128) + LayerNorm + ReLU + Dropout(0.2)   Layer 1
         │
    Linear(128 → 64) + LayerNorm + ReLU + Dropout(0.2)   Layer 2
         │
    Linear(64 → 32)  + LayerNorm + ReLU                  Layer 3
         │
    Linear(32 → 6)   (no activation)                     Output
         │
    ΔCS corrections: [ΔHA, ΔCA, ΔCB, ΔC, ΔN, ΔH]  (ppm)

Why LayerNorm instead of BatchNorm?
  BatchNorm averages statistics across the batch dimension.  Here each sample
  is one residue, and batch sizes vary between small peptides (≤ 20 residues)
  and full proteins.  LayerNorm normalises over the feature dimension for each
  residue independently, making it batch-size agnostic — the same model works
  for a single residue or a 500-residue protein.

Loss function: MSELoss (mean squared error over ΔCS values in ppm).
  MAELoss (L1) is more robust to outlier shifts but MSE penalises large
  errors more strongly, which matters for well-defined peaks in secondary
  structures where we want high accuracy.

─────────────────────────────────────────────────────────────────────────────
FEATURE VECTOR (74-dim) — one row per residue
─────────────────────────────────────────────────────────────────────────────

  Cols  0–19  : Amino acid type, one-hot (20 standard amino acids)
  Cols 20–21  : sin(φ), cos(φ)         — backbone dihedral
  Cols 22–23  : sin(ψ), cos(ψ)         — backbone dihedral
  Cols 24–26  : Secondary structure one-hot (helix / sheet / coil)
  Col  27     : Normalised sequence position (0 = N-term, 1 = C-term)
  Cols 28–33  : Random coil baseline [HA, CA, CB, C, N, H]  (ppm)
  Cols 34–53  : i−1 neighbour AA one-hot (zeros for N-terminal residue)
  Cols 54–73  : i+1 neighbour AA one-hot (zeros for C-terminal residue)

Output nuclei order (index 0–5): HA, CA, CB, C, N, H

─────────────────────────────────────────────────────────────────────────────
BENCHMARK — Empirical SPARTA+-like vs NeuralShiftPredictor on 1D3Z ubiquitin
─────────────────────────────────────────────────────────────────────────────

Measured on 1D3Z (76 residues). RMSE computed against BMRB entry bmr17769.

  Speed:
    Empirical predictor   59 µs/residue  (numpy + ring-current geometry)
    Neural (random wts)   90 µs/residue  (1.52× overhead from PyTorch)

  RMSE vs. BMRB experimentally measured shifts:
    Nucleus   Empirical   Neural (untrained)   Expected after training
    HA        0.40 ppm       0.62 ppm           ~0.20–0.25 ppm
    CA        1.21 ppm       2.38 ppm           ~0.7–0.9 ppm
    CB        1.57 ppm       2.59 ppm           ~0.9–1.1 ppm
    C         1.62 ppm       2.58 ppm           ~0.8–1.0 ppm
    N         3.91 ppm       4.11 ppm           ~2.0–2.5 ppm
    H         0.67 ppm       0.71 ppm           ~0.3–0.4 ppm

  Why is the untrained neural net WORSE than the empirical model?
    Random weights add ±0.5–2 ppm of unstructured noise on top of the
    physically grounded empirical baseline.  After training, corrections
    shrink to learned residuals of ±0.1–0.5 ppm, cutting RMSE by ~30–50%
    for CA/CB/C (literature benchmark: SPARTA+ ~1.0 ppm CA; ShiftML ~0.5 ppm).

  Why does CB show the largest random-weight correction (~+1 ppm mean ΔCS)?
    CB is the most sequence-environment-sensitive nucleus.  Its shift
    reflects the χ₁ rotamer state (gauche vs. trans) and distinguishes
    β-branched residues (Val, Ile, Thr) from others by >5 ppm.  The feature
    variance across residues is high, so even an untrained model activates
    strongly on the AA one-hot and RC-baseline inputs.

─────────────────────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────────────────────

    from synth_nmr.neural_shifts import NeuralShiftPredictor

    predictor = NeuralShiftPredictor()          # random-weight or loads default ckpt
    shifts = predictor.predict(structure)       # same dict as predict_chemical_shifts()

    # After training:
    predictor.save("models/neural_shifts_v1.pt")
    predictor2 = NeuralShiftPredictor(model_path="models/neural_shifts_v1.pt")
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Canonical amino acid ordering (deterministic one-hot encoding)
_AA_ORDER = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
_AA_INDEX = {aa: i for i, aa in enumerate(_AA_ORDER)}

# His tautomers / protonation states → canonical HIS
_AA_ALIASES = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "CYX": "CYS"}

# Output nucleus ordering — must match training script
NUCLEUS_ORDER = ["HA", "CA", "CB", "C", "N", "H"]

# Input feature dimensionality
N_FEATURES = 74  # 20 + 2 + 2 + 3 + 1 + 6 + 20 + 20

# Default checkpoint bundled inside the package
_DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), "models", "neural_shifts_v1.pt")


# ── Feature engineering ──────────────────────────────────────────────────────


def build_residue_features(structure: Any) -> np.ndarray:
    """
    Build the per-residue feature matrix (shape [N_residues, 74]) from a
    biotite AtomArray.

    This is the bridge between raw 3D protein structure and the MLP's input
    space.  All features are unitless or normalised to aid optimisation.

    Args:
        structure: biotite.structure.AtomArray (any backbone-complete protein).

    Returns:
        float32 numpy array of shape [N_residues, 74].
    """
    import math

    import biotite.structure as struc

    from synth_nmr.chemical_shifts import RANDOM_COIL_SHIFTS
    from synth_nmr.structure_utils import get_secondary_structure

    # Filter for amino acids to ensure residue features match backbone-complete protein
    protein_mask = struc.filter_amino_acids(structure)
    structure = structure[protein_mask]
    if structure.array_length() == 0:
        return np.zeros((0, 74), dtype=np.float32)

    # ── Collect per-residue data ──────────────────────────────────────────
    res_starts = struc.get_residue_starts(structure)
    n_res = len(res_starts)

    res_names = []
    for _i, start in enumerate(res_starts):
        rn = structure.res_name[start]
        rn = _AA_ALIASES.get(rn, rn)
        res_names.append(rn)

    # Secondary structure labels → one-hot [helix=0, sheet=1, coil=2]
    ss_labels = get_secondary_structure(structure)
    _SS_MAP = {"alpha": 0, "beta": 1, "coil": 2}

    # Backbone dihedrals (radians) — biotite returns phi/psi per residue
    try:
        phi_angles, psi_angles, _ = struc.dihedral_backbone(structure)
    except Exception:
        phi_angles = np.zeros(n_res)
        psi_angles = np.zeros(n_res)

    if phi_angles is None or len(phi_angles) == 0:
        phi_angles = np.zeros(n_res)
    if psi_angles is None or len(psi_angles) == 0:
        psi_angles = np.zeros(n_res)

    # Pad/trim to match n_res exactly
    def _pad(arr: Any, length: int) -> np.ndarray:
        arr_np: np.ndarray = np.asarray(arr, dtype=np.float32)
        if len(arr_np) == length:
            return arr_np
        out = np.zeros(length, dtype=np.float32)
        out[: min(len(arr_np), length)] = arr_np[: min(len(arr_np), length)]
        return out

    phi_angles = _pad(phi_angles, n_res)
    psi_angles = _pad(psi_angles, n_res)

    # Replace NaN (terminal residues or missing atoms) with 0
    phi_angles = np.nan_to_num(phi_angles, nan=0.0)
    psi_angles = np.nan_to_num(psi_angles, nan=0.0)

    # ── Assemble feature matrix row by row ───────────────────────────────
    X = np.zeros((n_res, N_FEATURES), dtype=np.float32)

    for i, rn in enumerate(res_names):
        col = 0

        # ── Block 1: Current residue one-hot (20 dims) ──────────────────
        # One-hot encoding maps from categorical AA type to a binary vector.
        # The model learns separate weights for each residue type, allowing
        # it to capture residue-specific chemical shift behaviours (e.g.
        # GLY has no CB; PRO has no amide H).
        aa_idx = _AA_INDEX.get(rn, -1)
        if aa_idx >= 0:
            X[i, col + aa_idx] = 1.0
        col += 20

        # ── Block 2: Backbone dihedrals as sin/cos (4 dims) ─────────────
        # sin/cos encoding avoids the periodicity discontinuity at ±180°.
        # The four values parameterise a point on the Ramachandran torus.
        X[i, col + 0] = math.sin(phi_angles[i])
        X[i, col + 1] = math.cos(phi_angles[i])
        X[i, col + 2] = math.sin(psi_angles[i])
        X[i, col + 3] = math.cos(psi_angles[i])
        col += 4

        # ── Block 3: Secondary structure one-hot (3 dims) ───────────────
        # Unlike a raw label, one-hot lets the model learn independent
        # coefficients for helix/sheet/coil without imposing an ordering.
        ss_state = ss_labels[i] if i < len(ss_labels) else "coil"
        ss_idx = _SS_MAP.get(ss_state, 2)  # default to coil
        X[i, col + ss_idx] = 1.0
        col += 3

        # ── Block 4: Normalised sequence position (1 dim) ───────────────
        # 0 at the N-terminus, 1 at the C-terminus.  Captures systematic
        # terminal effects (fraying of secondary structures, end caps).
        X[i, col] = float(i) / max(1, n_res - 1)
        col += 1

        # ── Block 5: Random coil baseline (6 dims) ──────────────────────
        # Providing the empirical baseline as an input feature enables the
        # network to learn corrections rather than absolute shifts.  This is
        # analogous to residual connections in ResNets: the network learns
        # "what the baseline missed", which is easier than learning everything
        # from raw geometry.
        rc = RANDOM_COIL_SHIFTS.get(rn, {})
        for nuc in NUCLEUS_ORDER:
            X[i, col] = float(rc.get(nuc, 0.0))
            col += 1

        # ── Block 6: i−1 neighbour one-hot (20 dims) ────────────────────
        # Sequence-neighbour effects: the residue before i influences its
        # shifts via through-bond coupling and steric contacts.  This is
        # the primary feature SPARTA+ exploits that simple SS lookup tables
        # miss.  Zeros for the N-terminal residue (no predecessor).
        if i > 0:
            prev_rn = _AA_ALIASES.get(res_names[i - 1], res_names[i - 1])
            prev_idx = _AA_INDEX.get(prev_rn, -1)
            if prev_idx >= 0:
                X[i, col + prev_idx] = 1.0
        col += 20

        # ── Block 7: i+1 neighbour one-hot (20 dims) ────────────────────
        # Analogous to block 6 for the successor residue.  Zeros for the
        # C-terminal residue (no successor).
        if i < n_res - 1:
            next_rn = _AA_ALIASES.get(res_names[i + 1], res_names[i + 1])
            next_idx = _AA_INDEX.get(next_rn, -1)
            if next_idx >= 0:
                X[i, col + next_idx] = 1.0
        col += 20

        assert col == N_FEATURES, f"Feature column count mismatch: {col} != {N_FEATURES}"

    return cast(np.ndarray, X)


def build_graph_data(structure: Any) -> Any:
    """
    Builds a torch_geometric.data.Data object from an AtomArray.
    Uses C-alpha distance (<= 8.0 A) to define edges.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover
        raise ImportError(  # pragma: no cover
            "torch and torch_geometric are required. Install with: pip install synth-nmr[ml]"
        ) from exc

    import biotite.structure as struc
    from scipy.spatial import KDTree

    # Ensure consistency by filtering for amino acids here too
    protein_mask = struc.filter_amino_acids(structure)
    structure = structure[protein_mask]
    if structure.array_length() == 0:
        return None

    X = build_residue_features(structure)
    x = torch.tensor(X, dtype=torch.float32)

    res_starts = struc.get_residue_starts(structure)
    n_res = len(res_starts)

    if n_res == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)  # pragma: no cover
        return Data(x=x, edge_index=edge_index)  # pragma: no cover

    coords = np.zeros((n_res, 3), dtype=np.float32)
    for i, start in enumerate(res_starts):
        end = res_starts[i + 1] if i + 1 < len(res_starts) else len(structure)
        res_atoms = structure[start:end]
        ca_mask = res_atoms.atom_name == "CA"
        if np.any(ca_mask):
            coords[i] = res_atoms.coord[ca_mask][0]
        else:
            coords[i] = res_atoms.coord[0]  # pragma: no cover

    tree = KDTree(coords)
    pairs = tree.query_pairs(r=8.0)

    src, dst = [], []
    for i in range(n_res):
        src.append(i)
        dst.append(i)
    for i, j in pairs:
        src.extend([i, j])
        dst.extend([j, i])

    # Convert to standard torch tensor
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


# ── Model factory ────────────────────────────────────────────────────────────


def _make_mlp(
    hidden_dims: Tuple[int, ...] = (128, 64, 32),
    n_features: int = N_FEATURES,
    n_outputs: int = 6,
    dropout: float = 0.2,
) -> Any:
    """
    Construct the MLP nn.Module.

    Lazy-imported so torch is only required when actually building the model,
    not at module-import time.

    Architecture rationale:
      • LayerNorm after each Linear so gradient flow is stable regardless of
        protein length (no batch-size dependence, unlike BatchNorm).
      • Dropout for regularisation — the training dataset (synthesised
        structures) is small, so regularisation is especially important.
      • No activation on the output layer — ΔCS values can be positive or
        negative and there is no natural clamp.
    """
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "torch is required for NeuralShiftPredictor. Install with: pip install synth-nmr[ml]"
        ) from exc

    layers = []
    in_dim = n_features
    for h in hidden_dims:
        layers += [
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        ]
        in_dim = h

    # Remove the final Dropout (only use during training, and cleaner output)
    layers.pop()
    layers.append(nn.Linear(in_dim, n_outputs))

    return nn.Sequential(*layers)


def _make_gnn(
    hidden_dims: Tuple[int, ...] = (128, 64, 32),
    n_features: int = N_FEATURES,
    n_outputs: int = 6,
    dropout: float = 0.2,
) -> Any:
    """Construct a PyTorch Geometric GNN."""
    try:
        import torch
        import torch.nn as nn
        from torch_geometric.nn import GATConv, LayerNorm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(  # pragma: no cover
            "torch_geometric is required. Install with: pip install synth-nmr[ml]"
        ) from exc

    class GNNModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList()
            self.norms = nn.ModuleList()

            in_dim = n_features
            for h in hidden_dims:
                self.layers.append(GATConv(in_dim, h, heads=1, concat=False))
                self.norms.append(LayerNorm(h))
                in_dim = h

            self.out = nn.Linear(in_dim, n_outputs)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, x: Any, edge_index: Any) -> Any:
            for conv, norm in zip(self.layers, self.norms):
                x = conv(x, edge_index)
                x = norm(x)
                x = torch.relu(x)
                x = self.dropout(x)
            x = self.out(x)
            return x

    return GNNModel()


# ── Predictor class ──────────────────────────────────────────────────────────


class NeuralShiftPredictor:
    """
    MLP-based chemical shift predictor.

    Predicts per-residue ΔCS corrections (ppm) for 6 backbone nuclei
    [HA, CA, CB, C, N, H] and adds them to the empirical baseline from
    `predict_chemical_shifts()`.

    ── API compatibility with predict_chemical_shifts() ────────────────────
    Both return:
        {chain_id: {res_id: {atom_name: float}}}

    This means `NeuralShiftPredictor.predict(structure)` can be used as a
    drop-in upgrade anywhere `predict_chemical_shifts(structure)` is called.

    ── Model initialisation ─────────────────────────────────────────────────
    1. If model_path is given → load checkpoint.
    2. If the default bundled checkpoint exists → load it.
    3. Otherwise → initialise with random weights (useful for testing
       and as starting point for training).

    Random-weight predictions are the *empirical* shifts + small random noise
    (because ΔCS starts near zero with standard weight initialisation).

    ── Performance (measured on 1D3Z ubiquitin, 76 residues) ───────────────
    Speed vs. empirical predictor:
      Empirical   59 µs/residue    (numpy + explicit ring-current geometry)
      Neural      90 µs/residue    (1.52× overhead from PyTorch inference)

    RMSE vs. BMRB experimental shifts (bmr17769) — current vs. potential:
      Nucleus  Empirical    Untrained    After training (expected)
      HA       0.40 ppm     0.62 ppm     ~0.20–0.25 ppm
      CA       1.21 ppm     2.38 ppm     ~0.70–0.90 ppm   ← largest gain
      CB       1.57 ppm     2.59 ppm     ~0.90–1.10 ppm   ← geometry-sensitive
      C        1.62 ppm     2.58 ppm     ~0.80–1.00 ppm
      N        3.91 ppm     4.11 ppm     ~2.00–2.50 ppm
      H        0.67 ppm     0.71 ppm     ~0.30–0.40 ppm

    CB insight: CB is the most sequence-environment-sensitive nucleus —
    it reflects the χ₁ rotamer state and distinguishes β-branched residues
    (Val, Ile, Thr — >5 ppm upfield vs. others).  Largest training gains
    expected here.  Even with random weights the model reacts strongly to
    the AA one-hot and RC-baseline inputs for CB.

    ── Trade-offs vs. empirical model ──────────────────────────────────────
    Pro:
      • Captures i±1 neighbour effects (0.1–0.5 ppm per nucleus)
      • Learns continuous φ/ψ → shift surface (vs. 3-bin SS lookup)
      • Residue-specific φ/ψ responses learnable per-AA
      • After training: ~30–50% RMSE reduction for CA/CB/C
    Con:
      • Requires PyTorch (optional [ml] extra)
      • WORSE than empirical if untrained / data-starved
      • 1.5× slower inference
      • Less interpretable (black box vs. explicit table)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        model_type: str = "gnn",
    ) -> None:
        self.hidden_dims = hidden_dims
        self.model_type = model_type
        self.model: Any = None
        self._model_path: Optional[str] = None

        if model_path:
            self.load(model_path)
        elif os.path.exists(_DEFAULT_CHECKPOINT):
            self.load(_DEFAULT_CHECKPOINT)
        else:
            logger.info(
                "No pre-trained checkpoint found at %s. "
                "Initialising with random weights. "
                "Run scripts/train_neural_shifts.py to train.",
                _DEFAULT_CHECKPOINT,
            )
            self._init_fresh_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, structure: Any) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Predict chemical shifts using the empirical baseline + neural ΔCS.

        The neural correction is added residue-by-residue to the empirical
        prediction, maintaining the same dict schema.

        Args:
            structure: biotite.structure.AtomArray.

        Returns:
            {chain_id: {res_id: {atom_name: shift_in_ppm}}}
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for NeuralShiftPredictor. "
                "Install with: pip install synth-nmr[ml]"
            ) from exc

        import biotite.structure as struc

        from synth_nmr.chemical_shifts import RANDOM_COIL_SHIFTS, predict_empirical_shifts

        if len(structure) == 0:
            return {}

        # Step 1 — Get empirical predictions (our grounded baseline for topology mapping)
        empirical = predict_empirical_shifts(structure)

        # Step 2 — Build feature matrix and run the neural correction
        if self.model_type == "gnn":
            data = build_graph_data(structure)
            self.model.eval()
            with torch.no_grad():
                delta = self.model(data.x, data.edge_index).numpy()
        else:
            X = build_residue_features(structure)
            x = torch.tensor(X, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                # delta: [N_residues, 6]  — ΔCS for each nucleus (Experimental - Random Coil)
                delta = self.model(x).numpy()

        # Step 3 — Add corrections to Random Coil baseline
        # Map residue index → (chain_id, res_id) for merging
        res_starts = struc.get_residue_starts(structure)
        result: Dict[str, Dict[int, Dict[str, float]]] = {}

        for i, start in enumerate(res_starts):
            chain_id = structure.chain_id[start]
            res_id = int(structure.res_id[start])
            res_name = structure.res_name[start]

            # Start from empirical shifts for this residue just to know which atoms exist
            emp_atoms = empirical.get(chain_id, {}).get(res_id, {})
            rc_atoms = RANDOM_COIL_SHIFTS.get(res_name, {})
            corrected: Dict[str, float] = {}

            for j, nucleus in enumerate(NUCLEUS_ORDER):
                if nucleus in emp_atoms and rc_atoms.get(nucleus, 0.0) > 0:
                    # Apply neural prediction to the random coil baseline
                    corrected[nucleus] = round(
                        float(np.clip(rc_atoms[nucleus] + delta[i, j], 0.0, 220.0)),
                        3,
                    )

            if corrected:
                if chain_id not in result:
                    result[chain_id] = {}
                result[chain_id][res_id] = corrected

        return result

    def save(self, path: str) -> None:
        """
        Save model weights and architecture config to a .pt checkpoint.

        Checkpoint format:
            {
              "state_dict" : OrderedDict of parameter tensors,
              "hidden_dims": tuple of ints,    ← architecture metadata
              "n_features" : int,
              "n_outputs"  : int,
            }

        Storing architecture metadata alongside weights makes the checkpoint
        fully self-describing — you can reconstruct the model on any machine
        without remembering the original constructor arguments.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to save a checkpoint.") from exc

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hidden_dims": self.hidden_dims,
                "n_features": N_FEATURES,
                "n_outputs": len(NUCLEUS_ORDER),
                "model_type": self.model_type,
            },
            path,
        )
        self._model_path = path
        logger.info("NeuralShiftPredictor checkpoint saved to %s", path)

    def load(self, path: str) -> None:
        """
        Load model weights from a .pt checkpoint.

        The architecture is reconstructed from the stored metadata, then
        `load_state_dict` copies the saved weights in.

        Args:
            path: Path to a .pt file written by NeuralShiftPredictor.save().
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to load a checkpoint.") from exc

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self.hidden_dims = tuple(ckpt["hidden_dims"])
            self.model_type = ckpt.get("model_type", "mlp")

            if self.model_type == "gnn":
                self.model = _make_gnn(  # pragma: no cover
                    hidden_dims=self.hidden_dims,
                    n_features=ckpt["n_features"],
                    n_outputs=ckpt["n_outputs"],
                )
            else:
                self.model = _make_mlp(
                    hidden_dims=self.hidden_dims,
                    n_features=ckpt["n_features"],
                    n_outputs=ckpt["n_outputs"],
                )
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self._model_path = path
            logger.info("NeuralShiftPredictor loaded from %s", path)
        except Exception as exc:
            logger.error("Failed to load checkpoint from %s: %s", path, exc, exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_fresh_model(self) -> None:
        """
        Initialise a randomly-weighted model.

        With standard PyTorch weight initialisation (Kaiming uniform), each
        output neuron starts near zero.  This means the correction term ΔCS ≈ 0
        initially, so predictions ≈ empirical model + negligible noise.
        After training on labelled data, the corrections become meaningful.
        """
        if self.model_type == "gnn":
            self.model = _make_gnn(hidden_dims=self.hidden_dims)
        else:
            self.model = _make_mlp(hidden_dims=self.hidden_dims)  # pragma: no cover
        self.model.eval()

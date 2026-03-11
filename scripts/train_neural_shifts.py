"""
Train the NeuralShiftPredictor on synthetic protein structures.

Data pipeline:
  1. Generate protein structures via synth-pdb (diverse conformations).
  2. Run the empirical predictor to get per-residue shift labels.
  3. Compute ΔCS = empirical shift − random coil baseline  (targets).
  4. Build 74-dim feature vectors via build_residue_features().
  5. Train MLP with MSELoss, AdamW, CosineAnnealingLR.
  6. Evaluate: report per-nucleus RMSE on held-out set.
  7. Save checkpoint.

Usage
-----
  # Quick dry-run (small dataset, few epochs):
  python scripts/train_neural_shifts.py --n-samples 60 --epochs 20

  # Full training:
  python scripts/train_neural_shifts.py --n-samples 500 --epochs 200

  # Use your own PDB directory instead of synthetic structures:
  python scripts/train_neural_shifts.py --pdb-dir /path/to/pdbs --epochs 200

Output
------
  synth_nmr/models/neural_shifts_v1.pt   (or the path given by --output)
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Nucleus output ordering — must match neural_shifts.py
NUCLEUS_ORDER = ["HA", "CA", "CB", "C", "N", "H"]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_synthetic_structures(n_samples: int, random_state: int = 42):
    """
    Generate protein structures using synth-pdb for training data.

    Returns a list of biotite AtomArray objects.
    """
    try:
        from synth_pdb.generator import generate_pdb_content
        import biotite.structure.io.pdb as pdb_io
        import io
    except ImportError:
        logger.warning(
            "synth-pdb not installed — cannot generate synthetic structures. "
            "Install with: pip install synth-pdb, or use --pdb-dir."
        )
        return []

    rng = np.random.default_rng(random_state)
    conformations = ["alpha", "beta", "random"]
    structures = []

    for i in range(n_samples):
        conf = conformations[i % len(conformations)]
        length = int(rng.integers(8, 25))    # short peptides for speed
        try:
            pdb_str = generate_pdb_content(length=length, conformation=conf,
                                           minimize_energy=False)
            struct = pdb_io.PDBFile.read(io.StringIO(pdb_str)).get_structure(model=1)
            structures.append(struct)
        except Exception as e:
            logger.warning("Skipping sample %d (%s, len=%d): %s", i, conf, length, e)

    logger.info("Generated %d / %d structures", len(structures), n_samples)
    return structures


def load_pdb_directory(pdb_dir: str):
    """Load all .pdb files from a directory as biotite AtomArrays."""
    import biotite.structure.io.pdb as pdb_io
    structures = []
    for fname in sorted(os.listdir(pdb_dir)):
        if not fname.endswith(".pdb"):
            continue
        path = os.path.join(pdb_dir, fname)
        try:
            struct = pdb_io.PDBFile.read(path).get_structure(model=1)
            structures.append(struct)
        except Exception as e:
            logger.warning("Skipping %s: %s", fname, e)
    logger.info("Loaded %d PDB files from %s", len(structures), pdb_dir)
    return structures


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_dataset(dataset_pairs: list):
    """
    Build (X, y) arrays from a list of (AtomArray, ExperimentalShifts) pairs.

    X : [N_residues_total, 74]  — feature vectors
    y : [N_residues_total,  6]  — ΔCS targets (experimental − random coil)

    Returns (X, y) as float32 numpy arrays, and the count of successfully
    processed residues.
    """
    from synth_nmr.neural_shifts import build_residue_features, NUCLEUS_ORDER
    from synth_nmr.chemical_shifts import RANDOM_COIL_SHIFTS
    import biotite.structure as struc

    X_list, y_list = [], []
    valid_residue_count = 0

    for idx, (struct, exp_shifts) in enumerate(dataset_pairs):
        try:
            X = build_residue_features(struct)
            res_starts = struc.get_residue_starts(struct)

            # We pre-allocate X and y, but might not use all rows if a residue lacks shifts
            X_valid = []
            y_valid = []

            for i, start in enumerate(res_starts):
                res_id = int(struct.res_id[start])
                res_name = struct.res_name[start]

                emp_atoms = exp_shifts.get(res_id, {})
                rc = RANDOM_COIL_SHIFTS.get(res_name, {})

                # Only include this residue if it has at least one experimental shift
                if not emp_atoms:
                    continue

                y_row = np.zeros(len(NUCLEUS_ORDER), dtype=np.float32)
                has_valid_nucleus = False

                for j, nuc in enumerate(NUCLEUS_ORDER):
                    if nuc in emp_atoms and rc.get(nuc, 0.0) > 0:
                        has_valid_nucleus = True
                        y_row[j] = emp_atoms[nuc] - rc[nuc]

                if has_valid_nucleus:
                    X_valid.append(X[i])
                    y_valid.append(y_row)
                    valid_residue_count += 1

            if len(X_valid) > 0:
                X_list.append(np.array(X_valid, dtype=np.float32))
                y_list.append(np.array(y_valid, dtype=np.float32))

        except Exception as e:
            logger.warning("Dataset build: skipping structure %d: %s", idx, e)

    if not X_list:
        raise RuntimeError("No structures could be processed for the dataset.")

    X_all = np.vstack(X_list)
    y_all = np.vstack(y_list)
    logger.info("Dataset: %d valid residues with assignments, X=%s, y=%s", valid_residue_count, X_all.shape, y_all.shape)
    return X_all, y_all


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(X_train, y_train, X_test, y_test, epochs: int = 100,
          hidden_dims=(128, 64, 32), batch_size: int = 64, lr: float = 1e-3):
    """
    Train the MLP and return the trained model.

    Loss:  MSELoss (mean squared error in ppm²).
    Opt:   AdamW with weight_decay=1e-4 for regularisation.
    Sched: CosineAnnealingLR — gradually reduces the learning rate from lr
           to lr/100, helping the model converge stably in the final epochs.
    """
    import torch
    import torch.nn as nn
    from synth_nmr.neural_shifts import _make_mlp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_te = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_te = torch.tensor(y_test,  dtype=torch.float32, device=device)

    model = _make_mlp(hidden_dims=hidden_dims).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d trainable parameters", n_params)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)

    n_train = len(X_tr)
    log_every = max(1, epochs // 10)

    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        # Mini-batch training — shuffle indices for each epoch
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start: start + batch_size]
            pred = model(X_tr[idx])
            loss = criterion(pred, y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % log_every == 0:
            model.eval()
            with torch.no_grad():
                test_mse = criterion(model(X_te), y_te).item()
            logger.info(
                "Epoch %4d/%d | train MSE=%.4f | test MSE=%.4f | lr=%.2e",
                epoch, epochs,
                epoch_loss / max(1, n_batches),
                test_mse,
                scheduler.get_last_lr()[0],
            )

    elapsed = time.perf_counter() - t0
    logger.info("Training complete in %.1f s", elapsed)
    model.eval()
    return model.cpu()


# ---------------------------------------------------------------------------
# Per-nucleus RMSE evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test):
    """Print per-nucleus RMSE on the test set."""
    import torch
    from synth_nmr.neural_shifts import NUCLEUS_ORDER

    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32))
    pred_np = pred.numpy()

    print("\n" + "=" * 50)
    print("  Per-nucleus RMSE on held-out test set")
    print("=" * 50)
    for j, nuc in enumerate(NUCLEUS_ORDER):
        rmse = float(np.sqrt(np.mean((pred_np[:, j] - y_test[:, j]) ** 2)))
        print(f"  {nuc:<4}  RMSE = {rmse:.3f} ppm")
    total_rmse = float(np.sqrt(np.mean((pred_np - y_test) ** 2)))
    print(f"\n  Overall RMSE = {total_rmse:.3f} ppm")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train NeuralShiftPredictor on Experimental Data")
    # Data sources (mutually exclusive)
    data_grp = parser.add_mutually_exclusive_group()
    data_grp.add_argument(
        "--experimental-data", action="store_true", default=True,
        help="Download and use experimental BMRB/PDB reference pairs (default).",
    )
    # Training hyperparameters
    parser.add_argument("--epochs",     type=int,   default=200,   help="Training epochs")
    parser.add_argument("--batch-size", type=int,   default=64,    help="Mini-batch size")
    parser.add_argument("--lr",         type=float, default=1e-3,  help="Initial learning rate")
    parser.add_argument("--test-size",  type=float, default=0.2,   help="Held-out fraction")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "synth_nmr", "models", "neural_shifts_v1.pt"),
        help="Output checkpoint path",
    )
    args = parser.parse_args()

    import importlib.util
    if importlib.util.find_spec("torch") is None:
        print("ERROR: torch is required. Install with: pip install synth-nmr[ml]")
        sys.exit(1)

    from sklearn.model_selection import train_test_split
    from synth_nmr.data_pipeline import load_matched_dataset

    # 1 — Load structures and experimental BMRB shifts
    logger.info("Initializing experimental data pipeline...")
    dataset_pairs = load_matched_dataset(data_dir="data")

    if not dataset_pairs:
        print("ERROR: No experimental structures available for training.")
        sys.exit(1)

    # 2 — Build feature/label dataset
    X, y = build_dataset(dataset_pairs)

    # 3 — Train/test split (stratify not needed for regression)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    logger.info("Split: %d train / %d test residues", len(X_train), len(X_test))

    # 4 — Train
    model = train(
        X_train, y_train, X_test, y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # 5 — Evaluate
    evaluate(model, X_test, y_test)

    # 6 — Save via NeuralShiftPredictor.save() so checkpoint is self-describing
    from synth_nmr.neural_shifts import NeuralShiftPredictor
    predictor = NeuralShiftPredictor.__new__(NeuralShiftPredictor)
    predictor.hidden_dims = (128, 64, 32)
    predictor.model = model
    predictor._model_path = None

    output_path = os.path.normpath(args.output)
    predictor.save(output_path)
    print(f"\nCheckpoint saved to: {output_path}")
    print("Load with: NeuralShiftPredictor(model_path='<path>')")


if __name__ == "__main__":
    main()

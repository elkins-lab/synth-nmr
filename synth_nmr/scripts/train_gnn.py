"""
Training script for the NeuralShiftPredictor (GNN architecture).
"""

import logging
import argparse
import sys
import numpy as np
import os

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch_geometric.loader import DataLoader
except ImportError:
    print("PyTorch and PyTorch Geometric are required to run training.")
    print("Please install them with: pip install synth-nmr[ml]")
    sys.exit(1)

import biotite.structure as struc

from synth_nmr.data_pipeline import load_matched_dataset
from synth_nmr.neural_shifts import NeuralShiftPredictor, NUCLEUS_ORDER, build_graph_data
from synth_nmr.chemical_shifts import RANDOM_COIL_SHIFTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_dataset(data_dir="data"):
    logger.info("Preparing dataset...")
    raw_data = load_matched_dataset(data_dir=data_dir)

    dataset = []
    for structure, exp_shifts in raw_data:
        # Build node features and edges
        data_obj = build_graph_data(structure)
        res_starts = struc.get_residue_starts(structure)
        n_res = len(res_starts)

        # Build y and mask (n_res, 6)
        y = torch.zeros((n_res, 6), dtype=torch.float32)
        mask = torch.zeros((n_res, 6), dtype=torch.bool)

        for i, start in enumerate(res_starts):
            # chain_id = structure.chain_id[start]
            res_id = int(structure.res_id[start])
            res_name = structure.res_name[start]

            # Find shifts for this residue
            atom_shifts = exp_shifts.get(res_id, {})
            rc_atoms = RANDOM_COIL_SHIFTS.get(res_name, {})

            for j, nucleus in enumerate(NUCLEUS_ORDER):
                if nucleus in atom_shifts and nucleus in rc_atoms:
                    # We train the model to predict ΔCS = exp - RC
                    delta = atom_shifts[nucleus] - rc_atoms[nucleus]
                    y[i, j] = float(delta)
                    mask[i, j] = True

        data_obj.y = y
        data_obj.mask = mask

        if mask.sum() > 0:
            dataset.append(data_obj)

    logger.info(f"Prepared {len(dataset)} valid training graphs.")
    return dataset


def train(
    epochs=10,
    batch_size=4,
    lr=1e-3,
    data_dir="data",
    save_path="synth_nmr/models/neural_shifts_v1.pt",
):
    dataset = prepare_dataset(data_dir)
    if not dataset:
        logger.error("Empty dataset. Aborting training.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Init predictor directly using GNN
    predictor = NeuralShiftPredictor(model_path=None, model_type="gnn")
    if predictor.model_type != "gnn":
        logger.info("Loaded an old MLP checkpoint by default. Re-initializing as a fresh GNN...")
        predictor.model_type = "gnn"
        predictor._init_fresh_model()

    model = predictor.model
    model.train()

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    logger.info("Starting training loop...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_atoms = 0

        for batch in loader:
            optimizer.zero_grad()

            # Forward pass
            out = model(batch.x, batch.edge_index)

            # Compute loss only on masked items
            masked_out = out[batch.mask]
            masked_y = batch.y[batch.mask]

            if len(masked_y) == 0:
                continue

            loss = loss_fn(masked_out, masked_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(masked_y)
            total_atoms += len(masked_y)

        avg_loss = total_loss / max(1, total_atoms)
        rmse = np.sqrt(avg_loss)
        logger.info(f"Epoch {epoch:03d} | MSE: {avg_loss:.4f} | RMSE: {rmse:.4f} ppm")

    # Verify save path directory exists
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    predictor.save(save_path)
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neural Shift Predictor GNN")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (number of graphs)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for BMRB/PDB data")
    parser.add_argument(
        "--save-path",
        type=str,
        default="synth_nmr/models/neural_shifts_v1.pt",
        help="Checkpoint save path",
    )

    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        data_dir=args.data_dir,
        save_path=args.save_path,
    )

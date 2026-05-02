"""
Phase 6: Behavioral cloning training script.

Trains a PolicyNetwork to imitate the MPC expert via supervised MSE regression
on clean (non-crashed) transitions from the expert dataset.

Training protocol (Amay's advice, 2026-05-01):
  Step 1 — Sanity check: train on all_normal only. If loss doesn't decrease,
            there's a bug. Fix before running ablations.
  Step 2 — Ablations: train one model per mixture, compare rollout performance.
  Step 3 — Combined: train on all four datasets merged for PPO warm-starting.

Usage:
    # Step 1 — sanity check (all_normal)
    python3 src/train_policy.py --dataset all_normal

    # Step 2 — ablations (one per mixture)
    python3 src/train_policy.py --dataset default_mix
    python3 src/train_policy.py --dataset cautious_heavy
    python3 src/train_policy.py --dataset aggressive_heavy

    # Step 3 — combined (all four merged)
    python3 src/train_policy.py --dataset all

    # Custom path
    python3 src/train_policy.py --dataset-path data/my_dataset.pkl --model-out models/my_policy.pt

Output:
    models/bc_policy_{dataset}.pt  — trained model weights
    Loss printed per epoch to stdout.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, str(Path(__file__).parent))
from policy_network import PolicyNetwork

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_PATHS = {
    "all_normal":        Path("data/expert_dataset_all_normal.pkl"),
    "default_mix":       Path("data/expert_dataset_default_mix.pkl"),
    "cautious_heavy":    Path("data/expert_dataset_cautious_heavy.pkl"),
    "aggressive_heavy":  Path("data/expert_dataset_aggressive_heavy.pkl"),
}

EPOCHS        = 100
BATCH_SIZE    = 256
LR            = 1e-3
VAL_FRACTION  = 0.1   # 10% held out for validation
SEED          = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_name: str, dataset_path: Path = None):
    """Load and return clean (obs, action) tensors for a named dataset or path."""
    if dataset_path is not None:
        paths = [dataset_path]
    elif dataset_name == "all":
        paths = list(DATASET_PATHS.values())
    else:
        paths = [DATASET_PATHS[dataset_name]]

    records = []
    for p in paths:
        with open(p, "rb") as f:
            data = pickle.load(f)
        clean = [r for r in data if not r["crashed"]]
        records.extend(clean)
        print(f"  Loaded {len(clean):>6} clean records from {p.name}")

    print(f"  Total clean records: {len(records)}")

    # Augment raw 25-dim obs with d_min (explicit gap) and step (episode progress)
    obs = torch.tensor(
        np.array([np.append(r["obs"].reshape(-1), [r["d_min"], float(r["step"])])
                  for r in records]), dtype=torch.float32
    )
    actions = torch.tensor(
        np.array([r["action"] for r in records]), dtype=torch.float32
    )
    return obs, actions


def compute_obs_stats(obs: torch.Tensor, train_indices):
    """Compute mean and std from training split only (no data leakage)."""
    train_obs = obs[train_indices]
    mean = train_obs.mean(dim=0)
    std  = train_obs.std(dim=0).clamp(min=1e-8)  # avoid division by zero
    return mean, std


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(dataset_name: str, dataset_path: Path, model_out: Path) -> None:
    torch.manual_seed(SEED)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading data...")
    obs, actions = load_dataset(dataset_name, dataset_path)
    print(f"  obs shape   : {obs.shape}")
    print(f"  action shape: {actions.shape}")
    print(f"  acc range   : [{actions[:,0].min():.3f}, {actions[:,0].max():.3f}]")

    # Train / val split
    n_val = int(len(obs) * VAL_FRACTION)
    n_train = len(obs) - n_val
    indices = torch.randperm(len(obs), generator=torch.Generator().manual_seed(SEED))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    # Obs normalization — fit on training split only (no data leakage)
    obs_mean, obs_std = compute_obs_stats(obs, train_idx)
    obs_norm = (obs - obs_mean) / obs_std
    print(f"  obs mean range: [{obs_mean.min():.3f}, {obs_mean.max():.3f}]")
    print(f"  obs std range : [{obs_std.min():.3f}, {obs_std.max():.3f}]")

    # Save normalization stats alongside the model
    stats_out = model_out.with_suffix(".npz")
    np.savez(stats_out, mean=obs_mean.numpy(), std=obs_std.numpy())
    print(f"  Norm stats saved to: {stats_out}")

    train_ds = TensorDataset(obs_norm[train_idx], actions[train_idx])
    val_ds   = TensorDataset(obs_norm[val_idx],   actions[val_idx])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"  train: {n_train}, val: {n_val}")

    # Model
    obs_dim    = obs.shape[1]   # 25
    action_dim = actions.shape[1]  # 2
    model     = PolicyNetwork(obs_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    loss_fn = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print(f"Training for {EPOCHS} epochs (batch={BATCH_SIZE}, lr={LR})...\n")
    print(f"{'Epoch':>6}  {'train_loss':>10}  {'val_loss':>10}  {'lr':>8}")
    print("-" * 42)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for obs_batch, act_batch in train_loader:
            pred = model(obs_batch)
            loss = loss_fn(pred, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_batch, act_batch in val_loader:
                pred = model(obs_batch)
                val_loss += loss_fn(pred, act_batch).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>10.5f}  {val_loss:>10.5f}  {current_lr:>8.2e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_out)

    print(f"\nBest val loss : {best_val_loss:.5f}")
    print(f"Model saved to: {model_out}")
    print(f"Norm stats at : {model_out.with_suffix('.npz')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_PATHS.keys()) + ["all"],
        default="all_normal",
        help="Named dataset to train on. Use 'all' to merge all four.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to a specific .pkl file (overrides --dataset).",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=None,
        help="Where to save the model weights. Defaults to models/bc_policy_{dataset}.pt",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch_size
    LR         = args.lr

    dataset_path = Path(args.dataset_path) if args.dataset_path else None
    model_out    = Path(args.model_out) if args.model_out else Path(f"models/bc_policy_{args.dataset}.pt")

    train(args.dataset, dataset_path, model_out)

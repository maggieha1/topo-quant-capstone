from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from windowing import DATA_PATH
from build_graph import TOPO_PATH
from dataset import MarketGraphDataset


def chronological_split_indices(
    n: int, gap: int, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> Tuple[List[int], List[int], List[int]]:
    """Oldest -> train, middle -> val, newest -> test, with `gap` windows dropped between each."""
    train_end = int(n * train_ratio)
    val_start = train_end + gap
    val_end = val_start + int(n * val_ratio)
    test_start = val_end + gap

    if test_start >= n:
        raise ValueError("Dataset too small for the requested split ratios and embargo gap.")

    return list(range(0, train_end)), list(range(val_start, val_end)), list(range(test_start, n))


def create_chronological_loaders(
    csv_path: str,
    topo_path: str,
    batch_size: int = 32,
    window_size: int = 30,
    horizon: int = 5,
    corr_threshold: float = 0.3,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    vol_quantile: float = 0.85,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Splits MarketGraphDataset chronologically with a strict purge/embargo gap
    between splits and returns PyG DataLoaders for train, val, and test.
    The volatility threshold is computed from TRAINING labels only.
    """
    gap = window_size + horizon  # 35 windows: no day appears in both a train label and a val/test input

    # Pass 1: forward volatilities don't depend on the threshold, so use them to set it
    probe = MarketGraphDataset(
        csv_path=csv_path, topo_path=topo_path, window_size=window_size,
        horizon=horizon, corr_threshold=corr_threshold,
    )
    train_idx, val_idx, test_idx = chronological_split_indices(len(probe), gap, train_ratio, val_ratio)
    vol_threshold = float(np.quantile(probe.forward_vols[train_idx], vol_quantile))

    # Pass 2: rebuild with the explicit, train-only threshold
    dataset = MarketGraphDataset(
        csv_path=csv_path, topo_path=topo_path, window_size=window_size,
        horizon=horizon, corr_threshold=corr_threshold, vol_threshold=vol_threshold,
    )

    # Shuffling inside train is safe once chronological boundaries and gaps are enforced.
    # A fixed seed makes the shuffle order reproducible.
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    WINDOW_SIZE, HORIZON, BATCH_SIZE = 30, 5, 32
    GAP = WINDOW_SIZE + HORIZON

    train_loader, val_loader, test_loader = create_chronological_loaders(
        csv_path=DATA_PATH, topo_path=TOPO_PATH, batch_size=BATCH_SIZE,
        window_size=WINDOW_SIZE, horizon=HORIZON,
    )

    ds = train_loader.dataset.dataset  # the underlying MarketGraphDataset
    splits = {
        "TRAIN": train_loader.dataset.indices,
        "VAL": val_loader.dataset.indices,
        "TEST": test_loader.dataset.indices,
    }
    tr, va, te = splits["TRAIN"], splits["VAL"], splits["TEST"]

    # --- Leakage checks ---
    assert max(tr) + GAP < min(va) and max(va) + GAP < min(te), "Embargo gap missing!"

    # Window i uses rows i ... i+29 as inputs; its label uses rows i+30 ... i+34.
    rows = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).index
    last_train_label_day = rows[WINDOW_SIZE - 1 + max(tr) + HORIZON]
    first_val_input_day = rows[min(va)]
    last_val_label_day = rows[WINDOW_SIZE - 1 + max(va) + HORIZON]
    first_test_input_day = rows[min(te)]
    assert last_train_label_day < first_val_input_day, "A training label shares a day with a validation input!"
    assert last_val_label_day < first_test_input_day, "A validation label shares a day with a test input!"

    # --- Batch shape checks ---
    batch = next(iter(train_loader))
    num_assets = ds.windows.shape[2]
    topo_dim = ds.topo_features.shape[1]
    assert batch.num_graphs == BATCH_SIZE, f"Expected {BATCH_SIZE} graphs, got {batch.num_graphs}"
    assert batch.x.shape == (BATCH_SIZE * num_assets, 4), f"Unexpected x shape: {tuple(batch.x.shape)}"
    assert batch.topo.shape == (BATCH_SIZE, topo_dim), f"Unexpected topo shape: {tuple(batch.topo.shape)}"
    assert batch.y.shape == (BATCH_SIZE,), f"Unexpected y shape: {tuple(batch.y.shape)}"
    assert batch.batch.shape == (BATCH_SIZE * num_assets,), f"Unexpected batch index shape: {tuple(batch.batch.shape)}"

    print("Success! Phase 2 DataLoaders verified:")
    print(f"  Volatility threshold (train only): {ds.vol_threshold:.5f}")
    for name, idx in splits.items():
        y = ds.labels[idx]
        print(f"  {name:5s}: {len(idx):4d} windows | {ds.dates[idx[0]]} to {ds.dates[idx[-1]]} | "
              f"stress {int(y.sum()):3d} ({y.mean().item():.1%})")
    print(f"  Embargo: {GAP} windows dropped before VAL and before TEST")
    print(f"  Last train label day {last_train_label_day.date()} < first val input day {first_val_input_day.date()}")
    print(f"  Last val label day   {last_val_label_day.date()} < first test input day {first_test_input_day.date()}")
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    print(f"  Sample batch:  {batch}")
    print(f"  Batch y positive count: {int(batch.y.sum().item())} / {len(batch.y)}")
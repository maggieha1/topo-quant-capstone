import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch, Data

from windowing import DATA_PATH, PROJECT_ROOT
from build_graph import TOPO_PATH
from dataset import MarketGraphDataset


def collate_graph_sequences(
    batch: List[Tuple[List[Data], torch.Tensor]]
) -> Tuple[List[Batch], torch.Tensor]:
    """
    Transposes B sequences of length L into L PyG Batch objects of size B,
    and stacks the aligned target labels into shape (B,).
    """
    seq_len = len(batch[0][0])
    batched_seq = [
        Batch.from_data_list([item[0][t] for item in batch])
        for t in range(seq_len)
    ]
    y = torch.stack([item[1] for item in batch], dim=0)
    return batched_seq, y


def embargo_gap(window_size: int, seq_len: int, horizon: int) -> int:
    """Windows to drop between splits so no day is in both a label and a later input."""
    return window_size + seq_len - 1 + horizon


def chronological_split_indices(
    n: int, gap: int, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> Tuple[List[int], List[int], List[int]]:
    """Oldest -> train, middle -> val, newest -> test, with `gap` samples dropped between each."""
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
    seq_len: int = 5,
    horizon: int = 5,
    corr_threshold: float = 0.3,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    vol_quantile: float = 0.85,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Chronological train/val/test sequence loaders with an embargo gap of
    window_size + seq_len - 1 + horizon, and a volatility threshold from TRAINING labels only.
    """
    gap = embargo_gap(window_size, seq_len, horizon)  # 39 for (30, 5, 5)

    # Pass 1: forward volatilities don't depend on the threshold, so use them to set it
    probe = MarketGraphDataset(
        csv_path=csv_path, topo_path=topo_path, window_size=window_size,
        seq_len=seq_len, horizon=horizon, corr_threshold=corr_threshold,
    )
    train_idx, val_idx, test_idx = chronological_split_indices(len(probe), gap, train_ratio, val_ratio)
    vol_threshold = float(np.quantile(probe.sample_forward_vols[train_idx], vol_quantile))

    # Pass 2: rebuild with the explicit, train-only threshold
    dataset = MarketGraphDataset(
        csv_path=csv_path, topo_path=topo_path, window_size=window_size,
        seq_len=seq_len, horizon=horizon, corr_threshold=corr_threshold,
        vol_threshold=vol_threshold,
    )

    # Each item is a whole sequence, so shuffling keeps every sequence intact and in order.
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,
        collate_fn=collate_graph_sequences, generator=generator,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, collate_fn=collate_graph_sequences,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, collate_fn=collate_graph_sequences,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))
    from tgnn import TopologicalGNN

    WINDOW_SIZE, SEQ_LEN, HORIZON, BATCH_SIZE = 30, 5, 5, 32
    GAP = embargo_gap(WINDOW_SIZE, SEQ_LEN, HORIZON)

    train_loader, val_loader, test_loader = create_chronological_loaders(
        csv_path=DATA_PATH, topo_path=TOPO_PATH, batch_size=BATCH_SIZE,
        window_size=WINDOW_SIZE, seq_len=SEQ_LEN, horizon=HORIZON,
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

    # Sample s uses input rows s ... (s + L - 1) + W - 1, and label rows up to (s + L - 1) + W - 1 + H.
    rows = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).index

    def last_label_row(s: int) -> int:
        return (s + SEQ_LEN - 1) + WINDOW_SIZE - 1 + HORIZON

    def first_input_row(s: int) -> int:
        return s

    last_train_label_day = rows[last_label_row(max(tr))]
    first_val_input_day = rows[first_input_row(min(va))]
    last_val_label_day = rows[last_label_row(max(va))]
    first_test_input_day = rows[first_input_row(min(te))]
    assert last_train_label_day < first_val_input_day, "A training label shares a day with a validation input!"
    assert last_val_label_day < first_test_input_day, "A validation label shares a day with a test input!"

    # --- Alignment check on an UNSHUFFLED loader: labels and last-step graphs match the dataset ---
    val_seq, val_y = next(iter(val_loader))
    first_val = va[: len(val_y)]
    assert torch.equal(val_y, ds.sample_labels[first_val]), "Validation labels are misaligned!"
    expected_topo = ds.topo_features[[ds.end_window(s) for s in first_val]]
    assert torch.equal(val_seq[-1].topo, expected_topo), "Last step's topo is not the sample's end window!"

    # --- End-to-end forward pass on a shuffled training batch ---
    torch.manual_seed(42)
    graph_seq, y = next(iter(train_loader))
    num_assets = ds.windows.shape[2]
    topo_dim = ds.topo_features.shape[1]

    model = TopologicalGNN()
    logits = model(graph_seq)
    loss = F.binary_cross_entropy_with_logits(logits, y)

    assert len(graph_seq) == SEQ_LEN, f"Expected sequence length {SEQ_LEN}, got {len(graph_seq)}"
    for g in graph_seq:
        assert g.num_graphs == BATCH_SIZE, "Every time step must hold the same batch of samples."
    assert graph_seq[-1].x.shape == (BATCH_SIZE * num_assets, 4), f"Unexpected x shape: {tuple(graph_seq[-1].x.shape)}"
    assert graph_seq[-1].topo.shape == (BATCH_SIZE, topo_dim), f"Unexpected topo shape: {tuple(graph_seq[-1].topo.shape)}"
    assert logits.shape == y.shape == (BATCH_SIZE,), f"Shape mismatch: logits {tuple(logits.shape)}, y {tuple(y.shape)}"
    assert torch.isfinite(loss), "Loss is not finite."

    print("Success! End-to-end sequence DataLoader -> TopologicalGNN verified.")
    print(f"  Volatility threshold (train only): {ds.vol_threshold:.5f}")
    for name, idx in splits.items():
        labels = ds.sample_labels[idx]
        print(f"  {name:5s}: {len(idx):4d} samples | {ds.sample_dates[idx[0]]} to {ds.sample_dates[idx[-1]]} | "
              f"stress {int(labels.sum()):3d} ({labels.mean().item():.1%})")
    print(f"  Embargo: {GAP} samples dropped before VAL and before TEST")
    print(f"  Last train label day {last_train_label_day.date()} < first val input day {first_val_input_day.date()}")
    print(f"  Last val label day   {last_val_label_day.date()} < first test input day {first_test_input_day.date()}")
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    print(f"  Sequence length: {len(graph_seq)} | Batch size: {len(y)} | Logits shape: {tuple(logits.shape)}")
    print(f"  Initial BCE loss: {loss.item():.4f}")
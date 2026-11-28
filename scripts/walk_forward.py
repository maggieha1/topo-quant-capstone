"""
Walk-forward evaluation folds.

For each test year Y:
    TRAIN = every sample before (VAL start - gap)
    VAL   = the `val_size` samples just before (TEST start - gap)     -> early stopping, decision threshold
    TEST  = every sample whose end date falls in year Y
Per fold, the stress threshold and the feature scaler come from that fold's TRAIN block only.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from dataloaders import collate_graph_sequences, embargo_gap
from dataset import MarketGraphDataset
from normalization import FeatureScaler, ScaledSequenceDataset


@dataclass
class Fold:
    name: str
    train_idx: List[int]
    val_idx: List[int]
    test_idx: List[int]


@dataclass
class FoldData:
    fold: Fold
    base: MarketGraphDataset
    scaler: FeatureScaler
    vol_threshold: float
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def walk_forward_folds(sample_dates: Sequence[str], test_years: Sequence[int], gap: int,
                       val_size: int = 126, min_train: int = 200) -> List[Fold]:
    years = pd.to_datetime(pd.Series(list(sample_dates))).dt.year.to_numpy()
    folds = []
    for year in test_years:
        test_idx = np.where(years == year)[0]
        if len(test_idx) == 0:
            raise ValueError(f"No samples end in {year}.")
        assert np.all(np.diff(test_idx) == 1), f"Test samples for {year} are not contiguous."

        test_start = int(test_idx[0])
        val_end = test_start - gap          # exclusive
        val_start = val_end - val_size
        train_end = val_start - gap         # exclusive
        if train_end < min_train:
            raise ValueError(f"Fold {year}: only {train_end} training samples (need {min_train}).")

        folds.append(Fold(str(year), list(range(0, train_end)),
                          list(range(val_start, val_end)), test_idx.tolist()))
    return folds


def check_fold_leakage(fold: Fold, rows: pd.Index, window_size: int, seq_len: int, horizon: int) -> None:
    """No trading day may appear in both an earlier block's label and a later block's input."""
    def last_label_row(s: int) -> int:
        return (s + seq_len - 1) + window_size - 1 + horizon

    assert max(fold.train_idx) < min(fold.val_idx) < min(fold.test_idx), f"Fold {fold.name}: blocks out of order."
    assert rows[last_label_row(max(fold.train_idx))] < rows[min(fold.val_idx)], \
        f"Fold {fold.name}: a training label shares a day with a validation input!"
    assert rows[last_label_row(max(fold.val_idx))] < rows[min(fold.test_idx)], \
        f"Fold {fold.name}: a validation label shares a day with a test input!"


def build_fold(csv_path: str, topo_path: str, fold: Fold, window_size: int = 30, seq_len: int = 5,
               horizon: int = 5, corr_threshold: float = 0.3, vol_quantile: float = 0.85,
               batch_size: int = 32, seed: int = 42, drop_topo: bool = False,
               probe: Optional[MarketGraphDataset] = None) -> FoldData:
    if probe is None:
        probe = MarketGraphDataset(csv_path=csv_path, topo_path=topo_path, window_size=window_size,
                                   seq_len=seq_len, horizon=horizon, corr_threshold=corr_threshold)

    # Stress threshold from this fold's TRAINING labels only
    vol_threshold = float(np.quantile(probe.sample_forward_vols[fold.train_idx], vol_quantile))
    base = MarketGraphDataset(csv_path=csv_path, topo_path=topo_path, window_size=window_size,
                              seq_len=seq_len, horizon=horizon, corr_threshold=corr_threshold,
                              vol_threshold=vol_threshold)

    # Feature scaler from this fold's TRAINING windows only
    scaler = FeatureScaler.fit(base, fold.train_idx)
    assert scaler.fit_windows[1] == max(fold.train_idx) + seq_len - 1, "Scaler used windows outside TRAIN."
    assert scaler.fit_windows[1] < min(fold.val_idx), "Scaler saw validation windows!"

    data = ScaledSequenceDataset(base, scaler, drop_topo=drop_topo)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(Subset(data, fold.train_idx), batch_size=batch_size, shuffle=True,
                              collate_fn=collate_graph_sequences, generator=generator)
    val_loader = DataLoader(Subset(data, fold.val_idx), batch_size=batch_size, shuffle=False,
                            collate_fn=collate_graph_sequences)
    test_loader = DataLoader(Subset(data, fold.test_idx), batch_size=batch_size, shuffle=False,
                             collate_fn=collate_graph_sequences)
    return FoldData(fold, base, scaler, vol_threshold, train_loader, val_loader, test_loader)

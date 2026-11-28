from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from windowing import DATA_PATH, create_sliding_windows
from build_graph import TOPO_PATH, build_market_graph


class MarketGraphDataset(Dataset):
    """
    Yields (graph_seq, y): graph_seq is a chronological list of `seq_len` PyG Data objects
    for windows [t-L+1, ..., t], and y is the forward regime-shift label of the LAST window (t).

    Indexing: sample idx uses windows idx ... idx + seq_len - 1.
      - Window-level arrays (windows, dates, forward_vols, labels) are indexed by window.
      - Sample-level arrays (sample_dates, sample_forward_vols, sample_labels) are indexed by sample.
    """

    def __init__(
        self,
        csv_path: str,
        topo_path: str,
        window_size: int = 30,
        seq_len: int = 5,
        horizon: int = 5,
        corr_threshold: float = 0.3,
        vol_threshold: Optional[float] = None,
        default_train_frac: float = 0.7,
    ):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1.")

        df_returns = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        windows, end_dates = create_sliding_windows(df_returns, window_size=window_size, step_size=1)

        topo_payload = torch.load(topo_path, weights_only=False)
        topo_features = topo_payload["features"]
        topo_dates = topo_payload["end_dates"]

        window_dates = [d.strftime("%Y-%m-%d") for d in end_dates]
        assert window_dates == topo_dates, "Date mismatch between windows and topo_features.pt!"
        assert topo_payload["tickers"] == list(df_returns.columns), "Ticker order mismatch with topo_features.pt!"

        # Equal-weighted market basket return for each day
        market_ret = df_returns.mean(axis=1).to_numpy(dtype=np.float64)

        # Forward H-day volatility for each WINDOW.
        # Window i closes at row (window_size - 1 + i); its label uses rows close+1 ... close+H.
        num_valid = len(windows) - horizon          # last H windows have no full future -> dropped
        num_samples = num_valid - seq_len + 1       # first L-1 windows have no full history -> no sample ends there
        if num_samples <= 0:
            raise ValueError("Not enough data for the chosen horizon and seq_len.")

        forward_vols = np.zeros(num_valid, dtype=np.float64)
        for i in range(num_valid):
            close_idx = window_size - 1 + i
            future_slice = market_ret[close_idx + 1 : close_idx + 1 + horizon]
            assert len(future_slice) == horizon, "Future slice is incomplete."
            forward_vols[i] = np.std(future_slice)

        # Each sample is labelled by its LAST window
        sample_forward_vols = forward_vols[seq_len - 1 :]

        # Temporary stand-in: 85th percentile of the first 70% of samples, stopping `horizon`
        # samples early. dataloaders.py always passes an explicit train-only threshold instead.
        if vol_threshold is None:
            train_cutoff = int(default_train_frac * num_samples)
            vol_threshold = float(np.quantile(sample_forward_vols[: train_cutoff - horizon], 0.85))

        # Window-level storage
        self.vol_threshold = vol_threshold
        self.seq_len = seq_len
        self.horizon = horizon
        self.corr_threshold = corr_threshold
        self.windows = windows[:num_valid]
        self.topo_features = topo_features[:num_valid]
        self.dates = window_dates[:num_valid]
        self.forward_vols = forward_vols
        self.labels = torch.tensor((forward_vols >= vol_threshold).astype(np.float32), dtype=torch.float32)

        # Sample-level views (sample idx -> its last window idx + seq_len - 1)
        self.sample_dates = self.dates[seq_len - 1 :]
        self.sample_forward_vols = sample_forward_vols
        self.sample_labels = self.labels[seq_len - 1 :]

    def __len__(self) -> int:
        return len(self.sample_labels)

    def end_window(self, idx: int) -> int:
        """Window index of the last (most recent) step of sample idx."""
        return idx + self.seq_len - 1

    def __getitem__(self, idx: int) -> Tuple[List[Data], torch.Tensor]:
        if not 0 <= idx < len(self):
            raise IndexError(f"Sample index {idx} out of range [0, {len(self)}).")

        graph_seq = [
            build_market_graph(
                self.windows[idx + step],
                self.topo_features[idx + step],
                threshold=self.corr_threshold,
            )
            for step in range(self.seq_len)
        ]
        # Target strictly aligned with the final graph in the sequence (time t)
        y = self.sample_labels[idx]
        return graph_seq, y


if __name__ == "__main__":
    SEQ_LEN = 5
    ds = MarketGraphDataset(
        csv_path=DATA_PATH,
        topo_path=TOPO_PATH,
        window_size=30,
        seq_len=SEQ_LEN,
        horizon=5,
        corr_threshold=0.3,
    )

    graph_seq, y = ds[0]
    assert len(graph_seq) == SEQ_LEN, f"Expected {SEQ_LEN} graphs, got {len(graph_seq)}"
    for g in graph_seq:
        g.validate(raise_on_error=True)
    assert y.dim() == 0, "y should be a single number per sample."

    pos_rate = ds.sample_labels.mean().item()
    assert 0.0 < pos_rate < 0.5, f"Unexpected class balance: {pos_rate:.2%}"

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    market = df.mean(axis=1)

    for i in [0, len(ds) // 2, len(ds) - 1]:
        seq, y_i = ds[i]
        e = ds.end_window(i)

        # 1) The steps are L consecutive windows, oldest first, ending on the sample's date
        step_dates = ds.dates[i : e + 1]
        assert step_dates[-1] == ds.sample_dates[i], f"Sample {i}: last step is not the sample's date."
        assert all(a < b for a, b in zip(step_dates, step_dates[1:])), f"Sample {i}: steps not in time order."

        # 2) The last graph really is the end window
        assert np.allclose(seq[-1].x[:, 3].numpy(), df.loc[ds.sample_dates[i]].values, atol=1e-6), \
            f"Sample {i}: last graph's last_ret does not match the end date."
        assert torch.equal(seq[-1].topo[0], ds.topo_features[e]), f"Sample {i}: last graph's topo misaligned."

        # 3) The label uses only the H trading days strictly AFTER the end date
        end = pd.Timestamp(ds.sample_dates[i])
        future = market[market.index > end].iloc[: ds.horizon]
        assert len(future) == ds.horizon and future.index.min() > end, f"Sample {i}: label uses a non-future day!"
        expected_vol = float(future.std(ddof=0))
        assert np.isclose(expected_vol, ds.sample_forward_vols[i], rtol=1e-6), f"Sample {i}: forward vol mismatch."
        assert y_i.item() == float(expected_vol >= ds.vol_threshold), f"Sample {i}: label mismatch."

    print(f"Success! Sequence dataset initialized with {len(ds)} samples (seq_len = {SEQ_LEN}).")
    print(f"  Sample dates (last step): {ds.sample_dates[0]} ... {ds.sample_dates[-1]}")
    print(f"  Sample 0 steps:           {ds.dates[0]} -> {ds.dates[ds.end_window(0)]}")
    print(f"  Volatility threshold (stand-in): {ds.vol_threshold:.5f}")
    print(f"  Positive rate overall:    {pos_rate:.2%}")
    print(f"  Sample[0]: {len(graph_seq)} graphs, last = {graph_seq[-1]}, y = {y.item():.0f}")
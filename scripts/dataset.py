from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from windowing import DATA_PATH, create_sliding_windows
from build_graph import TOPO_PATH, build_market_graph


class MarketGraphDataset(Dataset):
    """
    PyTorch Dataset yielding PyG Data objects for each 30-day market snapshot,
    paired with a forward-looking H-day volatility regime-shift label y.
    """

    def __init__(
        self,
        csv_path: str,
        topo_path: str,
        window_size: int = 30,
        horizon: int = 5,
        corr_threshold: float = 0.3,
        vol_threshold: Optional[float] = None,
        default_train_frac: float = 0.7,
    ):
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

        # Forward H-day volatility for each window.
        # Window i closes at row (window_size - 1 + i); its label uses rows close+1 ... close+H.
        num_valid = len(windows) - horizon  # the last H windows have no full future -> dropped
        if num_valid <= 0:
            raise ValueError("Not enough data for the chosen horizon.")

        forward_vols = np.zeros(num_valid, dtype=np.float64)
        for i in range(num_valid):
            close_idx = window_size - 1 + i
            future_slice = market_ret[close_idx + 1 : close_idx + 1 + horizon]
            assert len(future_slice) == horizon, "Future slice is incomplete."
            forward_vols[i] = np.std(future_slice)

        # Temporary stand-in until Day 15's proper split: 85th percentile of the first 70%,
        # stopping `horizon` windows early so no label used here looks past the cutoff.
        if vol_threshold is None:
            train_cutoff = int(default_train_frac * num_valid)
            vol_threshold = float(np.quantile(forward_vols[: train_cutoff - horizon], 0.85))

        self.vol_threshold = vol_threshold
        self.horizon = horizon
        self.corr_threshold = corr_threshold
        self.windows = windows[:num_valid]
        self.topo_features = topo_features[:num_valid]
        self.dates = window_dates[:num_valid]
        self.forward_vols = forward_vols

        # Binary labels: 1.0 = high-volatility regime shift, 0.0 = normal regime
        self.labels = torch.tensor((forward_vols >= vol_threshold).astype(np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        graph = build_market_graph(
            self.windows[idx],
            self.topo_features[idx],
            threshold=self.corr_threshold,
        )
        # Shape (1,) so PyG batches labels into shape (batch_size,)
        graph.y = self.labels[idx].unsqueeze(0)
        return graph


if __name__ == "__main__":
    ds = MarketGraphDataset(
        csv_path=DATA_PATH,
        topo_path=TOPO_PATH,
        window_size=30,
        horizon=5,
        corr_threshold=0.3,
    )

    sample = ds[0]
    sample.validate(raise_on_error=True)

    pos_rate = ds.labels.mean().item()
    assert sample.y.shape == (1,), f"Expected y shape (1,), got {tuple(sample.y.shape)}"
    assert 0.0 < pos_rate < 0.5, f"Unexpected class balance: {pos_rate:.2%}"

    # Independent look-ahead check using calendar DATES (not row arithmetic):
    # each label must use exactly the H trading days strictly AFTER the window's end date.
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    market = df.mean(axis=1)
    for i in [0, len(ds) // 2, len(ds) - 1]:
        end = pd.Timestamp(ds.dates[i])
        future = market[market.index > end].iloc[: ds.horizon]
        assert len(future) == ds.horizon, f"Window {i}: incomplete future."
        assert future.index.min() > end, f"Window {i}: label uses a non-future day!"
        expected_vol = float(future.std(ddof=0))
        assert np.isclose(expected_vol, ds.forward_vols[i], rtol=1e-6), f"Window {i}: forward vol mismatch."
        assert ds.labels[i].item() == float(expected_vol >= ds.vol_threshold), f"Window {i}: label mismatch."
    # The very last label must end exactly on the last day of data
    last_future = market[market.index > pd.Timestamp(ds.dates[-1])].iloc[: ds.horizon]
    assert last_future.index[-1] == market.index[-1], "Last label does not end on the last data day."

    cut = int(0.7 * len(ds))
    print(f"Success! Dataset initialized with {len(ds)} graph snapshots.")
    print(f"  Date range:            {ds.dates[0]} ... {ds.dates[-1]}")
    print(f"  Volatility threshold:  {ds.vol_threshold:.5f}")
    print(f"  Positive rate overall: {pos_rate:.2%}")
    print(f"  Positive rate first 70% (to {ds.dates[cut - 1]}): {ds.labels[:cut].mean().item():.2%}")
    print(f"  Positive rate last 30%:                       {ds.labels[cut:].mean().item():.2%}")
    print(f"  Sample[0]: {sample}  (date {ds.dates[0]}, y = {sample.y.item():.0f})")

    top = np.argsort(-ds.forward_vols)[:5]
    print("  Most turbulent upcoming weeks (window end date -> forward vol):")
    for i in top:
        print(f"    {ds.dates[i]}  ->  {ds.forward_vols[i]:.4f}")
"""
Compressed topological features: 8 summary numbers per window instead of the
300-number persistence landscape.

Each window's H0 and H1 persistence diagrams (the same ones topology.py computes)
are reduced to:
  h0_max_death     latest cluster merge (the "loneliest" day, e.g. 2021-01-27)
  h0_mean_death    typical merge distance (how spread out the days are)
  h0_std_death     how uneven the merges are
  h1_count         number of loops
  h1_max_life      longest-lived loop
  h1_total_life    total loop persistence (sum of lifetimes)
  h1_mean_birth    the scale at which loops form
  h1_entropy       persistent entropy: one dominant loop (low) vs. many similar ones (high)

Saved to data/topo_summary.pt with the same layout as topo_features.pt, so the
dataset and loaders can use it without changes.
"""
import os
import time

import numpy as np
import pandas as pd
import torch

from windowing import DATA_PATH, PROJECT_ROOT, create_sliding_windows
from point_cloud import window_to_point_cloud, compute_distance_matrix
from topology import extract_persistence_diagrams

SUMMARY_PATH = os.path.join(PROJECT_ROOT, "data", "topo_summary.pt")
LANDSCAPE_PATH = os.path.join(PROJECT_ROOT, "data", "topo_features.pt")
SUMMARY_FEATURE_NAMES = [
    "h0_max_death", "h0_mean_death", "h0_std_death",
    "h1_count", "h1_max_life", "h1_total_life", "h1_mean_birth", "h1_entropy",
]


def summarize_diagrams(diagrams) -> np.ndarray:
    """Reduces one window's H0/H1 diagrams to len(SUMMARY_FEATURE_NAMES) numbers."""
    h0_deaths = diagrams["H0"][:, 1]           # H0 births are all 0, so deaths = lifetimes
    h1 = diagrams["H1"]

    if len(h1) > 0:
        life = h1[:, 1] - h1[:, 0]
        p = life / life.sum()
        p = p[p > 0]
        entropy = float(-(p * np.log(p)).sum()) if len(p) > 1 else 0.0
        h1_stats = [len(h1), life.max(), life.sum(), h1[:, 0].mean(), entropy]
    else:
        h1_stats = [0, 0.0, 0.0, 0.0, 0.0]      # no loops in this window

    feats = np.array([h0_deaths.max(), h0_deaths.mean(), h0_deaths.std(), *h1_stats], dtype=np.float32)
    assert np.all(np.isfinite(feats)), "Non-finite summary feature."
    return feats


if __name__ == "__main__":
    df_returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    windows, end_dates = create_sliding_windows(df_returns, window_size=30, step_size=1)

    print(f"Summarizing {len(windows)} windows...")
    start = time.time()
    feats = np.stack([
        summarize_diagrams(extract_persistence_diagrams(compute_distance_matrix(window_to_point_cloud(w))))
        for w in windows
    ])
    print(f"  done in {time.time() - start:.1f}s")

    dates = [d.strftime("%Y-%m-%d") for d in end_dates]
    if os.path.exists(LANDSCAPE_PATH):
        landscape = torch.load(LANDSCAPE_PATH, weights_only=False)
        assert landscape["end_dates"] == dates, "Dates differ from topo_features.pt. Rebuild that file first."

    payload = {
        "features": torch.from_numpy(feats),
        "end_dates": dates,
        "tickers": list(df_returns.columns),
        "feature_names": SUMMARY_FEATURE_NAMES,
        "config": {"window_size": 30, "step_size": 1, "kind": "summary"},
    }
    torch.save(payload, SUMMARY_PATH)

    table = pd.DataFrame(feats, columns=SUMMARY_FEATURE_NAMES, index=dates)
    print(f"\nSuccess! Saved {SUMMARY_PATH} | Shape: {tuple(feats.shape)}")
    print("\nFirst window (compare with your Day 5 output: H0 max lifetime, H1 count, H1 max lifetime):")
    print(table.iloc[0].round(4).to_string())
    print("\nAcross all windows:")
    print(table.describe().loc[["mean", "std", "min", "max"]].round(3).to_string())

from typing import Tuple

import numpy as np
import pandas as pd
import torch

from windowing import DATA_PATH, create_sliding_windows


def build_correlation_edges(
    window: np.ndarray,
    threshold: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the Pearson correlation matrix for a (window_size, num_assets) window
    and returns PyG-formatted (edge_index, edge_attr) for |corr| >= threshold.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1].")

    # rowvar=False treats columns (assets) as variables -> shape (N, N)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr_matrix = np.corrcoef(window.astype(np.float64), rowvar=False)

    # Replace any NaNs (e.g., a flat/halted asset) with 0
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Keep |rho| >= threshold, and explicitly exclude the diagonal (self-loops)
    num_assets = corr_matrix.shape[0]
    keep = (np.abs(corr_matrix) >= threshold) & ~np.eye(num_assets, dtype=bool)
    src, dst = np.where(keep)

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_attr = torch.tensor(corr_matrix[src, dst], dtype=torch.float32).unsqueeze(-1)

    return edge_index, edge_attr


if __name__ == "__main__":
    df_returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    windows, end_dates = create_sliding_windows(df_returns, window_size=30, step_size=1)
    tickers = list(df_returns.columns)

    # Test on the first 30-day window
    edge_index, edge_attr = build_correlation_edges(windows[0], threshold=0.3)
    num_assets = windows.shape[2]

    # Sanity checks
    assert edge_index.shape[0] == 2, "edge_index must have shape (2, num_edges)."
    assert edge_index.shape[1] == edge_attr.shape[0], "Edge index and attribute count mismatch."
    assert edge_index.dtype == torch.long, "PyG requires edge_index to be torch.long (int64)."
    assert (edge_attr >= -1.0001).all() and (edge_attr <= 1.0001).all(), "Correlations out of bounds."
    assert not (edge_index[0] == edge_index[1]).any(), "Self-loops were not removed."

    # Symmetry check: every edge u->v must have a matching v->u with the same weight
    adj = torch.zeros(num_assets, num_assets)
    adj[edge_index[0], edge_index[1]] = edge_attr.squeeze(-1)
    assert torch.allclose(adj, adj.T), "Graph is not symmetric (undirected)."

    max_possible_edges = num_assets * (num_assets - 1)
    density = edge_index.shape[1] / max_possible_edges
    num_negative = int((edge_attr < 0).sum())

    print(f"Success! Window ending {end_dates[0].date()}:")
    print(f"  Edges retained: {edge_index.shape[1]} / {max_possible_edges} (Density: {density:.1%})")
    print(f"  Negative-correlation edges: {num_negative}")
    print(f"  edge_index shape: {tuple(edge_index.shape)} | edge_attr shape: {tuple(edge_attr.shape)}")

    # Show the 5 strongest connections (each pair once)
    pairs = [
        (edge_attr[k, 0].item(), tickers[u], tickers[v])
        for k, (u, v) in enumerate(edge_index.t().tolist())
        if u < v
    ]
    print("  Strongest connections:")
    for rho, a, b in sorted(pairs, key=lambda p: -abs(p[0]))[:5]:
        print(f"    {a:>4} - {b:<4}  rho = {rho:+.3f}")
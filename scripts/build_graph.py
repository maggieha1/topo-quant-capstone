import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from windowing import DATA_PATH, PROJECT_ROOT, create_sliding_windows
from graph_edges import build_correlation_edges

TOPO_PATH = os.path.join(PROJECT_ROOT, "data", "topo_features.pt")
NODE_FEATURE_NAMES = ["ret_5d", "volatility", "cum_ret_30d", "last_ret"]


def compute_node_features(window: np.ndarray) -> torch.Tensor:
    """
    Computes local asset-level features from a (window_size, num_assets) log-return array.
    Returns a tensor of shape (num_assets, 4): [ret_5d, volatility, cum_ret_30d, last_ret].
    """
    window = window.astype(np.float64)
    ret_5d = np.sum(window[-5:, :], axis=0)  # Short-term momentum (last 5 days)
    vol = np.std(window, axis=0)              # How jumpy the stock was
    cum_ret = np.sum(window, axis=0)          # Log-returns are time-additive
    last_ret = window[-1, :]                  # Most recent day's shock

    node_feats = np.stack([ret_5d, vol, cum_ret, last_ret], axis=1)
    return torch.tensor(node_feats, dtype=torch.float32)


def build_market_graph(
    window: np.ndarray,
    topo_feature: Optional[torch.Tensor] = None,
    threshold: float = 0.3,
) -> Data:
    """
    Constructs a PyG Data object with local node features, dynamic correlation edges,
    and (optionally) a (1, D) graph-level persistence landscape tensor.
    """
    x = compute_node_features(window)
    edge_index, edge_attr = build_correlation_edges(window, threshold=threshold)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if topo_feature is not None:
        # Enforce 2D shape (1, D) so PyG batches along dim 0 -> (batch_size, D)
        graph.topo = topo_feature.detach().clone().to(torch.float32).reshape(1, -1)

    return graph


if __name__ == "__main__":
    df_returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    windows, end_dates = create_sliding_windows(df_returns, window_size=30, step_size=1)
    tickers = list(df_returns.columns)
    num_assets = len(tickers)

    # Load precomputed persistence landscapes
    topo_payload = torch.load(TOPO_PATH, weights_only=False)
    topo_features = topo_payload["features"]
    topo_dates = topo_payload["end_dates"]
    topo_dim = topo_features.shape[1]

    # 1) Labels line up: same dates, same tickers, same window size
    window_dates = [d.strftime("%Y-%m-%d") for d in end_dates]
    assert window_dates == topo_dates, "Date mismatch between rolling windows and topo_features.pt!"
    assert topo_payload["tickers"] == tickers, "Ticker order mismatch with topo_features.pt!"
    assert topo_payload["config"]["window_size"] == 30, "topo_features.pt was built with a different window size."

    # 2) Content lines up: recompute window 0's landscape from scratch and compare to the saved row
    from point_cloud import window_to_point_cloud, compute_distance_matrix
    from topology import extract_persistence_diagrams
    from vectorize import vectorize_diagrams

    recomputed = vectorize_diagrams(
        extract_persistence_diagrams(compute_distance_matrix(window_to_point_cloud(windows[0])))
    )
    assert np.allclose(recomputed, topo_features[0].numpy(), atol=1e-5), \
        "Saved topo row 0 does not match window 0. Rebuild with build_topo_dataset.py."

    # Build two consecutive graph snapshots and test PyG batching
    g0 = build_market_graph(windows[0], topo_features[0], threshold=0.3)
    g1 = build_market_graph(windows[1], topo_features[1], threshold=0.3)
    g0.validate(raise_on_error=True)
    g1.validate(raise_on_error=True)

    batch = Batch.from_data_list([g0, g1])

    assert g0.topo.shape == (1, topo_dim), f"Expected (1, {topo_dim}), got {tuple(g0.topo.shape)}"
    assert batch.topo.shape == (2, topo_dim), f"Batching failed! Expected (2, {topo_dim}), got {tuple(batch.topo.shape)}"
    assert batch.x.shape == (2 * num_assets, 4), f"Expected ({2 * num_assets}, 4), got {tuple(batch.x.shape)}"
    assert batch.num_graphs == 2, "Batch should contain 2 graphs."
    assert torch.equal(batch.topo[1], g1.topo[0]), "Graph 1's topo row was scrambled during batching."

    print("Success! Topological features injected and batching verified:")
    print(f"  Dates aligned: {window_dates[0]} ... {window_dates[-1]} ({len(window_dates)} windows)")
    print(f"  Single graph:  {g0}")
    print(f"  Batched (2):   {batch}")
    print(f"  batch vector:  {batch.batch.tolist()}")
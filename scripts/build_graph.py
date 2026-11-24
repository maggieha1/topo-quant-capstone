import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from windowing import DATA_PATH, create_sliding_windows
from graph_edges import build_correlation_edges

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

    # Stack into shape (num_assets, 4)
    node_feats = np.stack([ret_5d, vol, cum_ret, last_ret], axis=1)
    return torch.tensor(node_feats, dtype=torch.float32)


def build_market_graph(window: np.ndarray, threshold: float = 0.3) -> Data:
    """
    Constructs a PyTorch Geometric Data object for a single 30-day market window.
    """
    x = compute_node_features(window)
    edge_index, edge_attr = build_correlation_edges(window, threshold=threshold)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


if __name__ == "__main__":
    df_returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    windows, end_dates = create_sliding_windows(df_returns, window_size=30, step_size=1)
    tickers = list(df_returns.columns)

    # Build the graph snapshot for the first 30-day window
    sample_graph = build_market_graph(windows[0], threshold=0.3)

    # PyG built-in integrity check (verifies index bounds, shapes, and dtypes)
    sample_graph.validate(raise_on_error=True)

    assert sample_graph.num_nodes == len(tickers), f"Expected {len(tickers)} nodes, got {sample_graph.num_nodes}"
    assert sample_graph.num_node_features == 4, f"Expected 4 features, got {sample_graph.num_node_features}"
    assert torch.isfinite(sample_graph.x).all(), "Non-finite values in node features."
    assert sample_graph.is_undirected(), "Graph should be undirected."

    # Last day's return must match the CSV on the window's end date (no look-ahead)
    expected_last = df_returns.loc[end_dates[0]].values
    assert np.allclose(sample_graph.x[:, 3].numpy(), expected_last, atol=1e-6), "last_ret misaligned."

    print(f"Success! Built valid PyG Data object for {end_dates[0].date()}:")
    print(f"  {sample_graph}")
    print(f"  Has isolated nodes: {sample_graph.has_isolated_nodes()}")
    print(f"  Is undirected:      {sample_graph.is_undirected()}")

    # Show each stock's node features
    feats = pd.DataFrame(sample_graph.x.numpy(), index=tickers, columns=NODE_FEATURE_NAMES)
    print("\nNode features (log returns):")
    print(feats.round(4).to_string())
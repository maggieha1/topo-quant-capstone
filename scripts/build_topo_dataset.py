import os
import time

import numpy as np
import pandas as pd
import torch

from windowing import DATA_PATH, PROJECT_ROOT, create_sliding_windows
from point_cloud import window_to_point_cloud, compute_distance_matrix
from topology import extract_persistence_diagrams
from vectorize import H0_RANGE, H1_RANGE, features_outside_grid, vectorize_diagrams

OUT_PATH = os.path.join(PROJECT_ROOT, "data", "topo_features.pt")


def build_and_save_dataset(
    csv_path: str,
    out_path: str,
    window_size: int = 30,
    step_size: int = 1,
    num_layers: int = 3,
    resolution: int = 50,
    h0_range: tuple = H0_RANGE,
    h1_range: tuple = H1_RANGE,
) -> None:
    df_returns = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    windows, end_dates = create_sliding_windows(df_returns, window_size=window_size, step_size=step_size)

    num_windows = windows.shape[0]
    feature_dim = 2 * num_layers * resolution
    features = np.zeros((num_windows, feature_dim), dtype=np.float32)

    # Coverage tracking: did any feature fall outside the fixed grids?
    outside_h0 = outside_h1 = 0
    h1_min_birth, h1_max_death = np.inf, -np.inf
    h1_counts = np.zeros(num_windows, dtype=np.int32)

    print(f"Processing {num_windows} rolling windows...")
    start_time = time.time()

    for i in range(num_windows):
        cloud = window_to_point_cloud(windows[i])
        dist_matrix = compute_distance_matrix(cloud)
        diagrams = extract_persistence_diagrams(dist_matrix, maxdim=1)
        features[i] = vectorize_diagrams(
            diagrams,
            num_layers=num_layers,
            resolution=resolution,
            h0_range=h0_range,
            h1_range=h1_range,
        )

        outside_h0 += features_outside_grid(diagrams["H0"], h0_range)
        outside_h1 += features_outside_grid(diagrams["H1"], h1_range)
        h1 = diagrams["H1"]
        h1_counts[i] = len(h1)
        if len(h1):
            h1_min_birth = min(h1_min_birth, h1[:, 0].min())
            h1_max_death = max(h1_max_death, h1[:, 1].max())

        if (i + 1) % 250 == 0 or (i + 1) == num_windows:
            elapsed = time.time() - start_time
            print(f"  [{i + 1}/{num_windows}] completed in {elapsed:.1f}s")

    # Convert to PyTorch tensor and run final integrity checks
    feature_tensor = torch.from_numpy(features)
    assert feature_tensor.shape == (num_windows, feature_dim), "Tensor shape mismatch."
    assert torch.isfinite(feature_tensor).all(), "Non-finite values detected in tensor."
    assert len(end_dates) == num_windows, "Date index and feature rows are misaligned."

    # Package tensor with date alignment and config for reproducibility
    payload = {
        "features": feature_tensor,
        "end_dates": [d.strftime("%Y-%m-%d") for d in end_dates],
        "tickers": list(df_returns.columns),
        "config": {
            "window_size": window_size,
            "step_size": step_size,
            "num_layers": num_layers,
            "resolution": resolution,
            "h0_range": list(h0_range),
            "h1_range": list(h1_range),
        },
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(payload, out_path)
    print(f"\nSuccess! Saved {out_path} | Shape: {tuple(feature_tensor.shape)}")

    # Coverage report across ALL windows
    print("\nGrid coverage report:")
    print(f"  Features outside grid -> H0: {outside_h0}, H1: {outside_h1}")
    print(f"  H1 loops per window   -> min {h1_counts.min()}, mean {h1_counts.mean():.1f}, max {h1_counts.max()}")
    if np.isfinite(h1_min_birth):
        print(f"  H1 values observed    -> earliest birth {h1_min_birth:.3f}, latest death {h1_max_death:.3f} "
              f"(grid is {h1_range[0]} to {h1_range[1]})")


if __name__ == "__main__":
    build_and_save_dataset(csv_path=DATA_PATH, out_path=OUT_PATH)
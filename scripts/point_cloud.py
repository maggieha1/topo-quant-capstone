import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from windowing import DATA_PATH, create_sliding_windows


def window_to_point_cloud(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Standardizes a (window_size, num_assets) array along the time axis
    so each asset has zero mean and unit variance within the window.
    """
    window = window.astype(np.float64)  # extra precision for the distance math
    mean = np.mean(window, axis=0, keepdims=True)
    std = np.std(window, axis=0, keepdims=True)

    # eps prevents division by zero for flat/halted assets
    point_cloud = (window - mean) / (std + eps)
    return point_cloud


def compute_distance_matrix(point_cloud: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Computes the symmetric (window_size, window_size) pairwise distance matrix.
    """
    return squareform(pdist(point_cloud, metric=metric))


if __name__ == "__main__":
    df_returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    windows, end_dates = create_sliding_windows(df_returns, window_size=30, step_size=1)

    # Isolate the very first 30-day window
    sample_window = windows[0]
    cloud = window_to_point_cloud(sample_window)
    dist_matrix = compute_distance_matrix(cloud)

    # Sanity checks: correct shapes
    assert cloud.shape == (30, df_returns.shape[1]), f"Unexpected cloud shape: {cloud.shape}"
    assert dist_matrix.shape == (30, 30), f"Unexpected distance matrix shape: {dist_matrix.shape}"

    # Sanity checks: Z-scoring worked (each asset has mean 0, std 1 inside the window)
    assert np.allclose(cloud.mean(axis=0), 0.0, atol=1e-6), "Assets are not centered."
    assert np.allclose(cloud.std(axis=0), 1.0, atol=1e-4), "Assets are not unit-variance."

    # Sanity checks: valid metric space properties
    assert np.allclose(dist_matrix, dist_matrix.T), "Distance matrix is not symmetric."
    assert np.allclose(np.diag(dist_matrix), 0.0), "Diagonal elements must be zero."
    assert np.all(dist_matrix >= 0.0), "Distances cannot be negative."

    print(f"Success! Point cloud built for window ending {end_dates[0].date()}.")
    print(f"Max pairwise distance (diameter of point cloud): {dist_matrix.max():.4f}")
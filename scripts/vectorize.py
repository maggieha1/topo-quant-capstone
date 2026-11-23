from typing import Dict, Tuple

import numpy as np
import pandas as pd

from windowing import DATA_PATH, create_sliding_windows
from point_cloud import window_to_point_cloud, compute_distance_matrix
from topology import extract_persistence_diagrams

# Fixed filtration grids, one per homology dimension.
# They must be the SAME for every window so the vectors are comparable.
H0_RANGE: Tuple[float, float] = (0.0, 10.0)  # H0 tents span [0, death]; deaths are ~2-5
H1_RANGE: Tuple[float, float] = (1.5, 6.5)   # H1 loops are short-lived and sit in a narrow band


def persistence_landscape(
    dgm: np.ndarray,
    num_layers: int = 3,
    resolution: int = 50,
    t_min: float = 0.0,
    t_max: float = 10.0,
) -> np.ndarray:
    """
    Converts a (M, 2) persistence diagram into a fixed-size (num_layers, resolution)
    persistence landscape matrix.
    """
    landscape = np.zeros((num_layers, resolution), dtype=np.float32)
    if len(dgm) == 0:
        return landscape

    grid = np.linspace(t_min, t_max, resolution)

    # Broadcast tent functions: shape (M, resolution)
    births = dgm[:, 0:1]
    deaths = dgm[:, 1:2]
    tents = np.maximum(0.0, np.minimum(grid - births, deaths - grid))

    # Sort descending along the feature axis (axis=0) to extract top-K layers
    sorted_tents = np.sort(tents, axis=0)[::-1]

    k_available = min(num_layers, sorted_tents.shape[0])
    landscape[:k_available, :] = sorted_tents[:k_available, :]

    return landscape


def vectorize_diagrams(
    diagrams: Dict[str, np.ndarray],
    num_layers: int = 3,
    resolution: int = 50,
    h0_range: Tuple[float, float] = H0_RANGE,
    h1_range: Tuple[float, float] = H1_RANGE,
) -> np.ndarray:
    """
    Computes and concatenates flattened H0 and H1 persistence landscapes
    into a single 1D feature vector of length (2 * num_layers * resolution).
    """
    h0_land = persistence_landscape(diagrams["H0"], num_layers, resolution, *h0_range)
    h1_land = persistence_landscape(diagrams["H1"], num_layers, resolution, *h1_range)
    return np.concatenate([h0_land.ravel(), h1_land.ravel()])


def features_outside_grid(dgm: np.ndarray, t_range: Tuple[float, float]) -> int:
    """Counts features whose (birth, death) interval is not fully inside the grid."""
    if len(dgm) == 0:
        return 0
    return int(np.sum((dgm[:, 0] < t_range[0]) | (dgm[:, 1] > t_range[1])))


if __name__ == "__main__":
    df_returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    windows, end_dates = create_sliding_windows(df_returns, window_size=30, step_size=1)

    cloud = window_to_point_cloud(windows[0])
    dist_matrix = compute_distance_matrix(cloud)
    diagrams = extract_persistence_diagrams(dist_matrix, maxdim=1)

    num_layers, resolution = 3, 50
    feature_vec = vectorize_diagrams(diagrams, num_layers=num_layers, resolution=resolution)

    # Sanity checks
    expected_dim = 2 * num_layers * resolution
    assert feature_vec.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {feature_vec.shape}"
    assert np.all(np.isfinite(feature_vec)), "Vector contains NaN or Inf."
    assert np.all(feature_vec >= 0.0), "Persistence landscape values cannot be negative."
    assert np.linalg.norm(feature_vec) > 0.0, "Feature vector is all zeros."

    print(f"Success! Vectorized window ending {end_dates[0].date()} to shape {feature_vec.shape}.")
    print(f"L2 norm of topological feature vector: {np.linalg.norm(feature_vec):.4f}")

    # Coverage check: is the grid fine and wide enough to see the loops?
    h1 = diagrams["H1"]
    h1_part = feature_vec[num_layers * resolution:]
    true_peak = (h1[:, 1] - h1[:, 0]).max() / 2 if len(h1) else 0.0
    print(f"H1 tallest tent: true {true_peak:.4f} | captured on grid {h1_part.max():.4f}")
    print(f"Features outside grid -> H0: {features_outside_grid(diagrams['H0'], H0_RANGE)}, "
          f"H1: {features_outside_grid(h1, H1_RANGE)}")
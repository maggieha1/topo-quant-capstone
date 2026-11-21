from typing import Dict

import numpy as np
import pandas as pd
from ripser import ripser

from windowing import DATA_PATH, create_sliding_windows
from point_cloud import window_to_point_cloud, compute_distance_matrix


def extract_persistence_diagrams(dist_matrix: np.ndarray, maxdim: int = 1) -> Dict[str, np.ndarray]:
    """
    Computes H0 and H1 persistence diagrams from a precomputed distance matrix
    and strips infinite death times from H0.
    """
    result = ripser(dist_matrix, distance_matrix=True, maxdim=maxdim)
    dgms = result["dgms"]

    # Filter out the single global component with death == inf in H0
    h0 = dgms[0]
    h0_finite = h0[np.isfinite(h0[:, 1])]

    # H1 features are always finite for a finite metric space
    h1 = dgms[1] if maxdim >= 1 else np.empty((0, 2))

    return {"H0": h0_finite, "H1": h1}


def summarize_diagram(dgm: np.ndarray, name: str) -> None:
    """Prints feature count and lifetime (death - birth) statistics."""
    if len(dgm) == 0:
        print(f"{name}: 0 features detected.")
        return

    lifetimes = dgm[:, 1] - dgm[:, 0]
    print(f"{name}: {len(dgm)} finite features | "
          f"Max lifetime: {lifetimes.max():.4f} | "
          f"Mean lifetime: {lifetimes.mean():.4f}")


if __name__ == "__main__":
    df_returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    windows, end_dates = create_sliding_windows(df_returns, window_size=30, step_size=1)

    # Build point cloud and distance matrix for the first window
    cloud = window_to_point_cloud(windows[0])
    dist_matrix = compute_distance_matrix(cloud)

    # Extract and inspect H0 and H1 diagrams
    diagrams = extract_persistence_diagrams(dist_matrix, maxdim=1)
    n_points = dist_matrix.shape[0]

    # Sanity checks
    assert diagrams["H0"].shape == (n_points - 1, 2), \
        f"Expected {n_points - 1} finite H0 features, got {diagrams['H0'].shape}"
    assert np.all(np.isfinite(diagrams["H0"])), "inf/NaN left in H0."
    assert np.all(np.isfinite(diagrams["H1"])), "inf/NaN found in H1."
    assert np.all(diagrams["H0"][:, 0] == 0.0), "Every H0 feature should be born at 0."
    assert np.all(diagrams["H0"][:, 1] >= diagrams["H0"][:, 0]), "H0 death < birth detected."
    if len(diagrams["H1"]) > 0:
        assert np.all(diagrams["H1"][:, 1] >= diagrams["H1"][:, 0]), "H1 death < birth detected."
    assert diagrams["H0"][:, 1].max() <= dist_matrix.max() + 1e-6, \
        "H0 deaths cannot exceed the point cloud diameter."

    print(f"Persistence summary for window ending {end_dates[0].date()}:")
    summarize_diagram(diagrams["H0"], "H0")
    summarize_diagram(diagrams["H1"], "H1")
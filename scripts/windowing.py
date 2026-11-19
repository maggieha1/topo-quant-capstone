import os
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# Always point at the project's data/ folder, no matter where the script is run from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sp500_log_returns.csv")


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int = 30,
    step_size: int = 1,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Chunks a (T, N) DataFrame of log-returns into overlapping sliding windows.

    Returns:
        windows: np.ndarray of shape (num_windows, window_size, num_assets)
        end_dates: pd.DatetimeIndex of the closing date for each window
    """
    if len(df) < window_size:
        raise ValueError(f"Data length ({len(df)}) must be >= window_size ({window_size}).")
    if step_size < 1:
        raise ValueError("step_size must be >= 1.")

    data = df.to_numpy(dtype=np.float32)

    # sliding_window_view along axis=0 appends the window dim to the end:
    # shape becomes (T - window_size + 1, num_assets, window_size)
    raw_windows = sliding_window_view(data, window_shape=window_size, axis=0)

    # Transpose to standard sequence format: (num_windows, window_size, num_assets)
    windows = raw_windows.transpose(0, 2, 1)[::step_size]

    # Index strictly by the END date of each window to prevent look-ahead bias
    end_dates = df.index[window_size - 1 :][::step_size]

    return np.ascontiguousarray(windows), end_dates


if __name__ == "__main__":
    df_returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    windows, end_dates = create_sliding_windows(df_returns, window_size=30, step_size=1)

    # Quick sanity checks (mini unit tests)
    assert windows.shape[0] == len(end_dates), "Window count and date index mismatch."
    assert windows.shape[1] == 30, "Window size mismatch."
    assert windows.shape[2] == df_returns.shape[1], "Asset count mismatch."
    assert np.allclose(windows[0, -1, :], df_returns.iloc[29].values), "Alignment error on first window."
    assert np.allclose(windows[-1, -1, :], df_returns.iloc[-1].values), "Alignment error on last window."
    assert end_dates[-1] == df_returns.index[-1], "Last window must close on the last available date."

    print(f"Success! Created {windows.shape[0]} windows of shape {windows.shape[1:]}.")
    print(f"First window closes on: {end_dates[0].date()}")
    print(f"Last window closes on:  {end_dates[-1].date()}")
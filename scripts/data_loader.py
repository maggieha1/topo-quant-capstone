import os

import numpy as np
import pandas as pd
import yfinance as yf

# Always point at the project's data/ folder, no matter where the script is run from.
# __file__ = .../topo-quant-capstone/scripts/data_loader.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def fetch_log_returns(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads split- and dividend-adjusted close prices and computes daily log returns.
    """
    print(f"Downloading data for {len(tickers)} tickers...")

    # auto_adjust=True makes 'Close' the adjusted price (yfinance no longer returns 'Adj Close')
    prices = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )["Close"]

    if prices.empty:
        raise RuntimeError("No data downloaded. Check your internet connection and try again.")

    # Keep the columns in the same order as the ticker list
    prices = prices[tickers]

    # Vectorized log return computation: ln(P_t / P_{t-1})
    log_returns = np.log(prices / prices.shift(1)).dropna()

    return log_returns


if __name__ == "__main__":
    # Cross-sector subset for clear correlation structures
    target_tickers = ["AAPL", "MSFT", "JPM", "GS", "XOM", "CVX", "UNH", "JNJ", "PG", "WMT"]

    # 5-year window capturing different market regimes
    df_returns = fetch_log_returns(target_tickers, "2021-01-01", "2026-01-01")

    # Ensure data directory exists and save
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "sp500_log_returns.csv")
    df_returns.to_csv(out_path)

    print(f"Success. Log returns saved to {out_path} with shape {df_returns.shape}")
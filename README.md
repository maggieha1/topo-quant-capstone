# Topological Quant Capstone

Portfolio optimization that combines **topological data analysis (TDA)**,
**graph neural networks**, and **convex optimization** on real market data.

The core idea: describe the *shape* of recent market behaviour with persistent
homology, then use that shape as a feature for learning and allocation.

## Status
- ✅ **Phase 1: Topological feature pipeline** (complete)
- ⏳ Phase 2: Graph dataset & GNN (PyTorch Geometric)
- ⏳ Phase 3: Portfolio optimization & backtest (cvxpy)

## Phase 1 Pipeline

```
prices → log returns → 30-day windows → Z-scored point clouds → distance matrices
       → persistent homology (H0, H1) → persistence landscapes → (1225, 300) tensor
```

| Script | What it does |
|---|---|
| `scripts/data_loader.py` | Downloads adjusted prices for 10 S&P 500 stocks (2021–2025) via `yfinance` and computes daily log returns |
| `scripts/windowing.py` | Builds 1,225 overlapping 30-day windows with `sliding_window_view`, each labelled by its **end date** |
| `scripts/point_cloud.py` | Z-scores each asset within a window and computes the 30 × 30 Euclidean distance matrix |
| `scripts/topology.py` | Runs Vietoris–Rips persistent homology with `ripser` to extract H0 (clusters) and H1 (loops) |
| `scripts/vectorize.py` | Converts variable-length persistence diagrams into fixed-size persistence landscapes (3 layers × 50 points per dimension) |
| `scripts/build_topo_dataset.py` | Runs the pipeline over every window and saves features, dates, tickers and config to `data/topo_features.pt` |

**Universe:** AAPL, MSFT (tech) · JPM, GS (financials) · XOM, CVX (energy) ·
UNH, JNJ (healthcare) · PG, WMT (consumer staples)

## Key Design Decisions
- **No look-ahead bias:** each window is labelled by its closing date, so a feature
  dated *t* uses only data available at the close of *t*.
- **Within-window Z-scoring:** stops high-volatility stocks from dominating the
  geometry, so distances reflect co-movement patterns rather than raw volatility.
- **Infinite-death filtering:** the single persistent H0 component (death = ∞) is
  removed before vectorization to keep tensors finite.
- **Per-dimension landscape grids:** H0 uses [0, 10]; H1 uses a finer [0.5, 6.0] grid
  because H1 loops are short-lived. A coverage check across all 1,225 windows
  confirmed that **no feature falls outside either grid**.
- **Offline feature engineering:** topology is computed once and saved, so model
  training never re-runs `ripser`.
- **Built-in sanity checks:** every module asserts shapes, alignment, metric
  properties and finiteness before passing data on.

## Early Observations
- In the first window (ending 2021-02-17), the most topologically isolated day was
  **2021-01-27**, the peak of the GameStop short squeeze. Its isolation distance
  (4.485) matches the window's largest H0 death exactly.
- The second most isolated day was **2021-01-06**, a sector-rotation day
  (financials and energy up, tech lagging) after the Georgia Senate runoffs, which
  shows the method picks up shifts in cross-asset *relationships*, not just large moves.
- Across all windows, H1 loop counts range from 0 to 16 (mean 6.7), which suggests
  the market's topological structure varies meaningfully over time.

## Project Structure
```
topo-quant-capstone/
├── data/        # generated data (not tracked; rebuilt by scripts)
├── models/      # model definitions and saved weights
├── notebooks/   # exploratory analysis
├── scripts/     # pipeline scripts (see table above)
└── requirements.txt
```

## Setup
```bash
git clone https://github.com/maggieha1/topo-quant-capstone.git
cd topo-quant-capstone
python -m venv venv
source venv/bin/activate        # Windows Git Bash: source venv/Scripts/activate
pip install -r requirements.txt
```

## Reproduce Phase 1
Run from the project root:
```bash
python scripts/data_loader.py          # download prices → data/sp500_log_returns.csv
python scripts/build_topo_dataset.py   # build features → data/topo_features.pt
```

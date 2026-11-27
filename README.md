# Topological Quant Capstone

Forecasting market stress and building portfolios by combining **topological data
analysis (TDA)**, **graph neural networks**, and **convex optimization** on real
market data.

The core idea: describe the *shape* of recent market behaviour with persistent
homology, describe *who moves with whom* as a dynamic correlation graph, and use
both to forecast short-term volatility regimes.

## Status
- ✅ **Phase 1: Topological feature pipeline** (complete)
- ✅ **Phase 2: Graph dataset & data loaders** (complete)
- 🔄 **Phase 3: T-GNN model, evaluation, portfolio optimization & backtest** (starting)

## Pipeline Overview

```
prices → log returns → 30-day windows ─┬─→ Z-scored point clouds → persistent homology
                                       │       → persistence landscapes → topo (1, 300)
                                       ├─→ correlation graph → edges (2, E), weights (E, 1)
                                       ├─→ node features → x (10, 4)
                                       └─→ forward 5-day volatility → label y
                                                     ↓
                    one PyG graph per trading day (1,220) → chronological train / val / test loaders
```

## Scripts

### Phase 1: Topology
| Script | What it does |
|---|---|
| `scripts/data_loader.py` | Downloads adjusted prices for 10 S&P 500 stocks (2021–2025) via `yfinance` and computes daily log returns |
| `scripts/windowing.py` | Builds 1,225 overlapping 30-day windows with `sliding_window_view`, each labelled by its **end date** |
| `scripts/point_cloud.py` | Z-scores each asset within a window and computes the 30 × 30 Euclidean distance matrix |
| `scripts/topology.py` | Runs Vietoris–Rips persistent homology with `ripser` to extract H0 (clusters) and H1 (loops) |
| `scripts/vectorize.py` | Converts variable-length persistence diagrams into fixed-size persistence landscapes (3 layers × 50 points per dimension) |
| `scripts/build_topo_dataset.py` | Runs the pipeline over every window and saves features, dates, tickers and config to `data/topo_features.pt` |

### Phase 2: Graphs & Data Loading
| Script | What it does |
|---|---|
| `scripts/graph_edges.py` | Computes each window's 10 × 10 Pearson correlation matrix and keeps edges with \|ρ\| ≥ 0.3 in PyG COO format |
| `scripts/build_graph.py` | Builds a PyG `Data` object per window: node features `x`, correlation edges, and the window's persistence landscape as a graph-level `topo` attribute |
| `scripts/dataset.py` | `MarketGraphDataset`: pairs each graph with a forward-looking volatility label `y`, with built-in look-ahead checks |
| `scripts/dataloaders.py` | Chronological train / validation / test split with a 35-window embargo, a train-only volatility threshold, and PyG `DataLoader`s |

**Universe:** AAPL, MSFT (tech) · JPM, GS (financials) · XOM, CVX (energy) ·
UNH, JNJ (healthcare) · PG, WMT (consumer staples)

## Graph Design
Each trading day becomes one graph:

| Component | Shape | Content |
|---|---|---|
| Nodes | 10 | One per stock |
| Node features `x` | (10, 4) | 5-day return, 30-day volatility, 30-day cumulative return, last-day return |
| Edges `edge_index` | (2, E) | Stock pairs with \|ρ\| ≥ 0.3, stored in both directions |
| Edge weights `edge_attr` | (E, 1) | The pairwise correlation |
| Topology `topo` | (1, 300) | H0 + H1 persistence landscapes for the whole market |
| Label `y` | (1,) | 1 = high-volatility week ahead, 0 = calm |

A batch of 32 graphs becomes `x` (320, 4), `topo` (32, 300) and `y` (32,), with PyG tracking which
nodes belong to which graph.

## Prediction Target
For a window closing on day *t*, the label looks **only** at days *t+1 … t+5*:
- Compute the standard deviation of the equal-weighted basket's daily returns over those 5 days.
- Label **y = 1** if it is at or above the **85th percentile of the training split** (threshold = **0.01077**),
  otherwise **y = 0**.
- The final 5 windows have no complete future and are dropped, which leaves **1,220 labelled graphs**
  (2021-02-17 to 2025-12-23).

## Data Splits
Chronological, never shuffled across time. **35 windows** (window size + horizon) are dropped between
blocks, so no trading day appears in both a training label and a validation or test input.

| Split | Windows | Dates (window end) | Stress windows |
|---|---|---|---|
| Train | 854 | 2021-02-17 to 2024-07-10 | 128 (15.0%) |
| Validation | 183 | 2024-08-29 to 2025-05-22 | 30 (16.4%) |
| Test | 113 | 2025-07-16 to 2025-12-23 | **0 (0.0%)** |

Leakage is verified with calendar dates: the last training label day (2024-07-17) comes before the first
validation input day (2024-07-19), and the last validation label day (2025-05-30) before the first test
input day (2025-06-03). Training batches are shuffled with a fixed seed for reproducibility.

**Finding: the chronological test block contains no stress windows.** Under the training threshold,
July–December 2025 was entirely calm; the April 2025 tariff shock falls in validation. A single
train/test split can therefore only measure false alarms in a calm market, not stress detection, so
final evaluation will use **walk-forward folds** (below).

## Key Design Decisions

**Avoiding look-ahead bias**
- Every window is labelled by its **closing date**, so features dated *t* use only data available at the close of *t*.
- Labels use only days strictly **after** *t*. This is checked independently using calendar dates, not just row arithmetic.
- The volatility threshold is computed **explicitly from the training indices** and passed into the dataset,
  so it stays leak-free even if the split ratios change.
- A 35-window embargo separates the splits, which is stricter than the 5-window minimum needed to stop labels overlapping.

**Topology**
- **Within-window Z-scoring** stops high-volatility stocks from dominating the geometry.
- **Infinite-death filtering** removes the single persistent H0 component before vectorization.
- **Per-dimension landscape grids:** H0 on [0, 10]; H1 on a finer [0.5, 6.0] grid because H1 loops are short-lived.
  A coverage check across all 1,225 windows confirmed **no feature falls outside either grid**.
- **Offline feature engineering:** topology is computed once and saved, so training never reruns `ripser`.

**Graphs**
- **Graph-level topology:** the 300 landscape values describe the whole market, so they are stored once per graph
  as `topo` (1, 300) instead of being copied onto every node, which would bury each stock's own features.
- **Non-redundant node features:** a 30-day mean return was replaced by a 5-day return, because the mean is
  exactly the cumulative return ÷ 30 and would add no information.
- **Undirected, self-loop-free edges** are verified on every graph.

**Correlation threshold (\|ρ\| ≥ 0.3)**, tested across all 1,225 windows:

| Threshold | Avg. density | Windows with an isolated stock |
|---|---|---|
| 0.20 | 59.1% | 56 |
| **0.30** | **42.1%** | **402** |
| 0.36 | 34.0% | 574 |
| 0.40 | 29.2% | 691 |
| 0.50 | 19.2% | 962 |

0.3 keeps density in a healthy range while limiting isolated stocks. With 30 observations, |ρ| ≈ 0.36 is needed
for 5% significance, so 0.3 deliberately keeps some weaker edges and relies on graph attention to down-weight them.

## Observations So Far
- **Topology finds real events.** In the first window (ending 2021-02-17), the most topologically isolated day was
  **2021-01-27**, the peak of the GameStop short squeeze. Its isolation distance (4.485) matches the window's
  largest H0 death exactly. The second was **2021-01-06**, a sector-rotation day after the Georgia Senate runoffs
  (banks up, tech down), which shows the method picks up changes in cross-asset *relationships*, not just big moves.
- **Graphs reflect the economy.** In the same window, the strongest edges were JPM–GS (0.90) and XOM–CVX (0.85),
  followed by a banks–energy cluster (≈0.57), consistent with the early-2021 "reflation trade". Banks and energy
  also led 30-day returns (XOM +29.5%, GS +18.0%), while defensive stocks lagged (UNH −6.5%, PG −6.2%).
- **Graphs change daily.** Moving the window forward one trading day changed the edge count from 42 to 34.
- **Stocks without a sector twin get isolated.** UNH (164 windows), WMT (135) and JNJ (80) were most often left
  with no edges, while JPM, GS, XOM and CVX, each paired with a highly correlated peer, never were.
- **Labels capture real stress.** The five most turbulent upcoming weeks all followed the **April 2025 tariff
  announcements**, led by the window ending 2025-04-02, whose next five days held both the crash and the rebound.
- **Stress is unevenly spread over time.** Training holds 15.0% stress windows (including the 2022 bear market),
  validation 16.4% (including April 2025), and the most recent half-year none.
- H1 loop counts range from 0 to 16 per window (mean 6.7).

## Evaluation Plan
- **Walk-forward folds:** train on everything before each test year (with the 35-window embargo), then test on
  that year, so every stress episode (2022 rate hikes, March 2023 banking stress, August 2024 sell-off,
  April 2025 tariff shock) is tested exactly once. Scores are reported per fold and averaged.
- **Metrics:** precision, recall and PR-AUC rather than accuracy. With rare stress, always predicting "calm"
  scores high accuracy while detecting nothing.
- **Baselines:** the GNN must beat a simple rule based on current volatility, because volatility tends to persist
  (volatility clustering). Ablations will compare models with and without the topological features.
- **Scaling:** node and topological features will be standardized using training-split statistics only.

## Known Limitations
- 10 stocks is a small universe. Stocks without a sector peer (e.g. UNH) are often isolated in the graph.
- Overlapping windows mean far fewer *independent* stress episodes than the raw example count suggests.
- The most recent half-year (Jul–Dec 2025) contains no stress windows, so a single chronological test split
  cannot measure stress detection.
- Some stress is driven by news that prices cannot anticipate (e.g. policy announcements made after the close).
- Hyperparameters (window size, correlation threshold, landscape grids) are not yet tuned on a validation split.

## Project Structure
```
topo-quant-capstone/
├── data/        # generated data (not tracked; rebuilt by scripts)
├── models/      # model definitions and saved weights
├── notebooks/   # exploratory analysis
├── scripts/     # pipeline scripts (see tables above)
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

## Reproduce
Run from the project root:
```bash
python scripts/data_loader.py          # download prices → data/sp500_log_returns.csv
python scripts/build_topo_dataset.py   # topological features → data/topo_features.pt
python scripts/dataset.py              # build and verify the labelled graph dataset
python scripts/dataloaders.py          # build and verify the chronological splits and loaders
```

# Topological Quant Capstone

Can the *shape* of market behaviour help forecast market stress?

This project builds a full pipeline that turns daily stock returns into **topological
features** (persistent homology) and **dynamic correlation graphs**, feeds them to a
**spatial-temporal graph neural network** (GAT + GRU), and tests whether that beats
simple volatility baselines at forecasting high-volatility weeks.

**Main finding:** under strict walk-forward evaluation (2023–2025, multiple seeds,
block-bootstrap intervals), **topological features did not detectably improve stress
forecasts beyond current volatility**, and no learned model reliably beat a simple
5-day volatility rule. The sections below show how this was tested and why it is
likely the case.

## Status
- ✅ **Phase 1: Topological feature pipeline**
- ✅ **Phase 2: Graph dataset & data loaders**
- ✅ **Phase 3a: Model, walk-forward evaluation, baselines & topology tests**
- ⏳ Phase 3b: Portfolio optimization & backtest (cvxpy)

## Pipeline Overview

```
prices → log returns → 30-day windows ─┬─→ Z-scored point clouds → persistent homology (H0, H1)
                                       │       ├─→ persistence landscapes      → topo (1, 300)
                                       │       └─→ 8 summary statistics        → topo (1, 8)
                                       ├─→ correlation graph → edges (2, E), weights (E, 1)
                                       ├─→ node features → x (10, 4)
                                       └─→ forward 5-day volatility → label y
                                                     ↓
           sequences of 5 consecutive daily graphs (1,216 samples) → GAT + GRU → stress probability
```

## Scripts

### Phase 1: Topology
| Script | What it does |
|---|---|
| `scripts/data_loader.py` | Downloads adjusted prices for 10 S&P 500 stocks (2021–2025) via `yfinance` and computes daily log returns |
| `scripts/windowing.py` | Builds 1,225 overlapping 30-day windows with `sliding_window_view`, each labelled by its **end date** |
| `scripts/point_cloud.py` | Z-scores each asset within a window and computes the 30 × 30 Euclidean distance matrix |
| `scripts/topology.py` | Runs Vietoris–Rips persistent homology with `ripser` to extract H0 (clusters) and H1 (loops) |
| `scripts/vectorize.py` | Converts persistence diagrams into fixed-size persistence landscapes (3 layers × 50 points per dimension) |
| `scripts/build_topo_dataset.py` | Saves the 300-number landscape for every window to `data/topo_features.pt` |
| `scripts/build_topo_summary.py` | Saves 8 compressed topology statistics per window to `data/topo_summary.pt` |

### Phase 2: Graphs & Data Loading
| Script | What it does |
|---|---|
| `scripts/graph_edges.py` | Computes each window's 10 × 10 Pearson correlation matrix and keeps edges with \|ρ\| ≥ 0.3 in PyG COO format |
| `scripts/build_graph.py` | Builds a PyG `Data` object per window: node features `x`, correlation edges, and graph-level `topo` |
| `scripts/dataset.py` | `MarketGraphDataset`: sequences of 5 consecutive graphs, labelled by the last day's forward volatility, with look-ahead checks |
| `scripts/dataloaders.py` | Chronological train / validation / test split with a 39-window embargo and a train-only threshold |

### Phase 3a: Model & Evaluation
| Script | What it does |
|---|---|
| `models/tgnn.py` | `TopologicalGNN`: two-layer multi-head GAT (with correlation edge attention) + topology MLP, fused per day, then a GRU over 5 days |
| `scripts/normalization.py` | Standardizes node features and topology using **training-window statistics only** |
| `scripts/walk_forward.py` | Walk-forward folds (one per test year) with embargo gaps, per-fold thresholds and scalers |
| `scripts/evaluation.py` | Volatility baselines and metrics (PR-AUC, ROC-AUC, precision, recall) |
| `scripts/train.py` | Training loop with AdamW weight decay, early stopping, and train-vs-validation tracking |
| `scripts/run_walk_forward.py` | Runs the full GNN comparison across folds and seeds, with verdicts and an overfitting report |
| `scripts/topo_value_test.py` | Logistic regression test of topology's value beyond volatility, with block-bootstrap intervals |

**Universe:** AAPL, MSFT (tech) · JPM, GS (financials) · XOM, CVX (energy) ·
UNH, JNJ (healthcare) · PG, WMT (consumer staples)

## Prediction Target
For a sample whose last day is *t*, the label looks **only** at days *t+1 … t+5*:
- Compute the standard deviation of the equal-weighted basket's daily returns over those 5 days.
- Label **y = 1 ("stress")** if it is at or above the **85th percentile of the training period**, otherwise **y = 0**.
- The threshold is recomputed from each fold's training data only.

## Graph & Model Design
Each trading day becomes one graph; each sample is a sequence of **5 consecutive daily graphs**.

| Component | Shape | Content |
|---|---|---|
| Nodes | 10 | One per stock |
| Node features `x` | (10, 4) | 5-day return, 30-day volatility, 30-day cumulative return, last-day return |
| Edges | (2, E), (E, 1) | Stock pairs with \|ρ\| ≥ 0.3 in both directions, weighted by correlation |
| Topology `topo` | (1, 300) or (1, 8) | Persistence landscape, or 8 summary statistics |
| Label `y` | scalar | Stress in the 5 trading days after the last graph |

| Model part | Parameters (landscape) | Parameters (summary) |
|---|---|---|
| Graph encoder: 2 × GATConv, 4 heads, `edge_dim=1` | 5,056 | 5,056 |
| Topology encoder (MLP) | 21,536 | 2,848 |
| Temporal encoder: GRU over 5 days | 9,472 | 9,472 |
| Classification head | 545 | 545 |
| **Total** | **36,609** | **17,921** |

The 8 topology summary statistics are: `h0_max_death`, `h0_mean_death`, `h0_std_death`, `h1_count`,
`h1_max_life`, `h1_total_life`, `h1_mean_birth` and `h1_entropy` (persistent entropy of H1 lifetimes).

## Evaluation Design

### Why not a single train/test split
A chronological 70/15/15 split left the test block (Jul–Dec 2025) with **zero stress windows**
under the training threshold, so it could not measure stress detection at all. Evaluation therefore
uses **walk-forward folds**.

### Walk-forward folds
For each test year, the model trains on everything before it. About 6 months of validation
(for early stopping and choosing the alarm threshold) sit between training and test, with a
**39-window embargo** (30-day window + 4 extra days of sequence history + 5-day horizon) on each side.
No trading day appears in both an earlier block's label and a later block's input; this is
checked with calendar dates for every fold.

| Fold | Train | Validation | Test | Test stress |
|---|---|---|---|---|
| 2023 | 265 samples (2021-02 → 2022-03), 15.1% stress | 2022-05 → 2022-11, **61.9%** stress | 250 samples | 20 (8.0%) |
| 2024 | 515 samples (2021-02 → 2023-03), 15.1% | 2023-05 → 2023-11, 0.8% | 252 samples | **4 (1.6%)** |
| 2025 | 767 samples (2021-02 → 2024-03), 15.0% | 2024-05 → 2024-11, 4.8% | 245 samples | 23 (9.4%) |

The 2024 fold has only 4 stress samples, so its scores are very noisy.

### Leakage controls
- Every window is labelled by its **closing date**; labels use only days strictly **after** it.
- Stress thresholds, feature standardization and baseline fits all use **training data only**.
- Splits are chronological; neighbouring samples (which share 29 of 30 days) never straddle a boundary.

### Baselines
- **vol5_rule:** the current 5-day basket volatility, used directly as the stress score (no fitting).
- **vol_logistic:** logistic regression on log 5-day and 30-day basket volatility, fit on training data.

Volatility tends to persist ("volatility clustering"), so these are strong, honest baselines.

### Metrics
**PR-AUC** (average precision) is the primary metric, since stress is rare. Always predicting "calm"
scores high accuracy but detects nothing, so accuracy is not used. ROC-AUC, precision and recall are
also reported, with alarm thresholds chosen on **validation**, never on test.

## Results

### 1. GNN with persistence landscapes (3 seeds)
Mean test PR-AUC over the three folds:

| Model | Seed 42 | Seed 1 | Seed 2 | Mean |
|---|---|---|---|---|
| GAT + GRU, no topology | 0.213 | 0.230 | 0.213 | **0.219** |
| vol5_rule | 0.211 | 0.211 | 0.211 | **0.211** |
| vol_logistic | 0.138 | 0.138 | 0.138 | 0.138 |
| GAT + GRU + 300-number landscape | 0.175 | 0.106 | 0.096 | **0.126** |

Removing the landscape **improved** results in every seed (+0.038, +0.125, +0.117). With it, the model
memorized its training data within 5–10 epochs (training PR-AUC 0.94–0.999) while validation worsened.

### 2. GNN with 8 compressed topology features (3 seeds)

| Model | Mean test PR-AUC | Seed range |
|---|---|---|
| vol5_rule | **0.211** | fixed |
| GAT + GRU, no topology | 0.197 | 0.137 – 0.242 |
| GAT + GRU + 8 summary features | 0.160 | 0.138 – 0.179 |
| vol_logistic | 0.138 | fixed |

Compression **removed the memorization** (average training PR-AUC 0.63, down from 0.94–0.999) but did
**not** improve test performance: topology changed mean PR-AUC by −0.037 and helped in only 1 of 3 seeds.

(The no-topology model scores slightly differently between the two experiments because the second
run also reseeds the training shuffle for each seed. That difference is itself a reminder of how
sensitive small-data models are.)

### 3. Logistic regression test: does topology add anything beyond volatility?
No neural network and no random seeds: the same folds and thresholds, with plain logistic regressions.

| Model | Mean PR-AUC | Mean ROC-AUC |
|---|---|---|
| Volatility only | **0.138** | 0.591 |
| Topology only (8 features) | 0.090 | 0.632 |
| Volatility + topology | 0.117 | **0.663** |
| Volatility + topology + 5-day changes | 0.096 | 0.573 |

A **block bootstrap** over test days (2,000 resamples, 10-day blocks, since labels overlap) estimates
the uncertainty of the difference:

| Comparison | Δ mean PR-AUC | 90% interval | Topology helps in |
|---|---|---|---|
| Volatility + topology vs. volatility only | −0.021 | [−0.093, +0.047] | 31% of resamples |
| + 5-day changes vs. volatility only | −0.043 | [−0.115, +0.023] | 13% of resamples |

Both intervals include zero: **no detectable improvement** from topology.

Some topology patterns were **stable across all three training periods** (same coefficient sign every fold):
- `h1_max_life` **positive**: one long-lived loop → higher stress probability
- `h1_total_life` **negative**: many small loops → lower stress probability
- `h0_std_death` **negative**: uneven cluster merges → lower stress probability

These patterns are interpretable but did not translate into better test forecasts.

## Conclusions
1. **No learned model reliably beat a simple 5-day volatility rule.** The best GNN (graph only, no topology)
   matched it within seed-to-seed noise.
2. **Topological features did not detectably improve stress forecasts** beyond volatility. This holds in
   three independent tests: GNN with landscapes, GNN with summary statistics, and bootstrapped
   logistic regression.
3. **The 300-number landscape caused severe overfitting** with only 265–767 training samples per fold.
   Compressing it to 8 statistics fixed the overfitting but not the lack of signal.
4. **Topology may carry weak general information** about market state (higher ROC-AUC), but it does not
   help pick out the stress weeks themselves (PR-AUC).
5. **Results for individual folds vary strongly with the random seed** (e.g. 0.14 to 0.41 for the same model
   and fold), so single-run results should not be trusted.

### Why topology likely did not help
Each window is Z-scored per stock (Phase 1, Day 4) so that volatile stocks do not dominate the geometry.
That makes the point cloud **scale-free**: it captures the *shape* of co-movement but discards the
**volatility level**, which is the strongest predictor of forward volatility. Other plausible factors:
30 points in 10 dimensions is a small, noisy point cloud, and the data contain only a handful of
independent stress episodes. These are hypotheses for future testing.

## Observations Along the Way
- **Topology finds real events.** In the first window (ending 2021-02-17), the most topologically isolated
  day was **2021-01-27** (the GameStop short squeeze); its isolation distance (4.485) matches the window's
  largest H0 death exactly. Second was **2021-01-06**, a sector-rotation day after the Georgia Senate runoffs
  (banks up, tech down).
- **Graphs reflect the economy.** The strongest edges were JPM–GS (0.90) and XOM–CVX (0.85), followed by
  a banks–energy cluster (≈0.57), consistent with the early-2021 "reflation trade".
- **Graphs change daily.** Moving the window forward one trading day changed the edge count from 42 to 34.
- **Stocks without a sector twin get isolated.** UNH (164 windows), WMT (135) and JNJ (80) were most often
  left with no edges; JPM, GS, XOM and CVX never were.
- **Labels capture real stress.** The most turbulent upcoming weeks all followed the **April 2025 tariff
  announcements**.

## Key Design Decisions
- **Per-dimension landscape grids** (H0 on [0, 10], H1 on [0.5, 6.0]), with a coverage check confirming no
  feature falls outside either grid across all windows.
- **Graph-level topology:** stored once per graph instead of being copied onto every node.
- **Non-redundant node features:** a 5-day return replaced a 30-day mean return (which equals the cumulative return ÷ 30).
- **Correlation threshold |ρ| ≥ 0.3**, chosen after testing 0.2–0.6 across all windows (42% average density;
  isolated stocks in 33% of windows). With 30 observations, |ρ| ≈ 0.36 is needed for 5% significance.
- **Offline feature engineering:** topology is computed once and saved, so training never reruns `ripser`.
- **Seed-aware verdicts:** a model only "beats" a baseline if it does so in every seed.

## Known Limitations
- Only 10 stocks; stocks without a sector peer (e.g. UNH) are often isolated in the graph.
- Few independent stress episodes; overlapping windows make the effective sample size much smaller than the sample count.
- The 2024 test year contains only 4 stress samples.
- Validation periods can differ sharply from training (e.g. 61.9% stress in the 2022 validation block),
  which distorts early stopping and alarm thresholds.
- Hyperparameters (window size, correlation threshold, model size) were not tuned.

## Possible Next Steps
- **Topology on non-Z-scored windows**, to test whether keeping the volatility level changes the result.
- **A target topology should suit better**, such as a forward jump in average correlation (stocks suddenly moving together).
- **A larger universe** (e.g. 50+ stocks) for richer graphs and point clouds.
- **Phase 3b:** use the model and features in portfolio optimization and backtesting.

## Project Structure
```
topo-quant-capstone/
├── data/        # generated data (not tracked; rebuilt by scripts)
├── models/      # model definition (tgnn.py)
├── notebooks/   # exploratory analysis
├── results/     # experiment outputs (CSV)
├── scripts/     # pipeline, evaluation and experiment scripts
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
python scripts/data_loader.py            # prices → data/sp500_log_returns.csv
python scripts/build_topo_dataset.py     # 300-number landscapes → data/topo_features.pt
python scripts/build_topo_summary.py     # 8 topology statistics → data/topo_summary.pt

# GNN comparison (several minutes per training run)
python scripts/run_walk_forward.py --topo landscape summary --ablation --seeds 42 1 2

# Topology value test (under a minute)
python scripts/topo_value_test.py
```

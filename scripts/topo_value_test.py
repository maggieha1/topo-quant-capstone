"""
Does topology add information beyond volatility? A simple, low-noise test.

Same walk-forward folds, embargo gaps and train-only stress thresholds as
run_walk_forward.py, but with plain logistic regressions instead of a neural network
(no random seeds, runs in seconds):

  vol_only      log 5-day and 30-day basket volatility                 (= vol_logistic)
  topo_only     the 8 topology summary features
  vol_topo      volatility + the 8 topology summary features
  vol_topo_chg  volatility + topology + the 5-day CHANGE in each topology feature

All features use only data up to each sample's date. The key comparison is
vol_topo vs. vol_only; a block bootstrap over test days estimates whether the
difference is larger than chance.

Usage (from the project folder):
  python scripts/build_topo_summary.py      # once, if data/topo_summary.pt does not exist
  python scripts/topo_value_test.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from windowing import DATA_PATH, PROJECT_ROOT
from build_topo_summary import SUMMARY_FEATURE_NAMES, SUMMARY_PATH
from dataloaders import embargo_gap
from dataset import MarketGraphDataset
from evaluation import trailing_vol_features
from walk_forward import check_fold_leakage, walk_forward_folds

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
WINDOW_SIZE, SEQ_LEN, HORIZON = 30, 5, 5
TEST_YEARS = [2023, 2024, 2025]
N_BOOT, BLOCK = 2000, 10  # bootstrap replicates; block length in trading days (labels overlap ~5 days)


def feature_sets(base, idx):
    """Builds every feature set for the given sample indices (past data only)."""
    vol = np.log(trailing_vol_features(base, idx) + 1e-12)                  # (n, 2)
    end = np.array([base.end_window(s) for s in idx])
    start = end - (SEQ_LEN - 1)                                             # first window of the sequence
    topo = base.topo_features[end].numpy().astype(np.float64)               # (n, 8)
    topo_chg = topo - base.topo_features[start].numpy().astype(np.float64)  # change over the last 5 days
    return {
        "vol_only": vol,
        "topo_only": topo,
        "vol_topo": np.hstack([vol, topo]),
        "vol_topo_chg": np.hstack([vol, topo, topo_chg]),
    }


def fit_predict(X_tr, y_tr, X_te):
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=1.0, max_iter=5000))
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1], model


def block_bootstrap_indices(n, rng):
    """Circular block bootstrap: keeps runs of neighbouring days together (labels overlap)."""
    starts = rng.integers(0, n, size=int(np.ceil(n / BLOCK)))
    idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel() % n
    return idx[:n]


def main():
    if not os.path.exists(SUMMARY_PATH):
        sys.exit("Missing data/topo_summary.pt. Run: python scripts/build_topo_summary.py")
    warnings.filterwarnings("ignore", category=UserWarning)

    gap = embargo_gap(WINDOW_SIZE, SEQ_LEN, HORIZON)
    probe = MarketGraphDataset(csv_path=DATA_PATH, topo_path=SUMMARY_PATH, window_size=WINDOW_SIZE,
                               seq_len=SEQ_LEN, horizon=HORIZON)
    rows = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).index
    folds = walk_forward_folds(probe.sample_dates, TEST_YEARS, gap)

    results, coefs, test_preds = [], [], []
    for fold in folds:
        check_fold_leakage(fold, rows, WINDOW_SIZE, SEQ_LEN, HORIZON)
        # Same train-only threshold as run_walk_forward.py
        thr = float(np.quantile(probe.sample_forward_vols[fold.train_idx], 0.85))
        base = MarketGraphDataset(csv_path=DATA_PATH, topo_path=SUMMARY_PATH, window_size=WINDOW_SIZE,
                                  seq_len=SEQ_LEN, horizon=HORIZON, vol_threshold=thr)
        y_tr = base.sample_labels[fold.train_idx].numpy().astype(int)
        y_te = base.sample_labels[fold.test_idx].numpy().astype(int)
        F_tr, F_te = feature_sets(base, fold.train_idx), feature_sets(base, fold.test_idx)

        preds = {"fold": fold.name, "y": y_te}
        for name in F_tr:
            p, model = fit_predict(F_tr[name], y_tr, F_te[name])
            preds[name] = p
            both = 0 < y_te.sum() < len(y_te)
            results.append({
                "fold": fold.name, "model": name, "n_stress": int(y_te.sum()), "base_rate": y_te.mean(),
                "pr_auc": average_precision_score(y_te, p) if both else np.nan,
                "roc_auc": roc_auc_score(y_te, p) if both else np.nan,
            })
            if name == "vol_topo":
                names = ["log_vol5", "log_vol30"] + SUMMARY_FEATURE_NAMES
                coefs.append(pd.Series(model[-1].coef_[0], index=names, name=fold.name))
        test_preds.append(preds)

    res = pd.DataFrame(results)
    pd.set_option("display.width", 160)
    print("=== Test PR-AUC by fold (logistic regression, train-only thresholds) ===")
    table = res.pivot(index="fold", columns="model", values="pr_auc")[["vol_only", "topo_only", "vol_topo", "vol_topo_chg"]]
    table.insert(0, "base_rate", res.groupby("fold")["base_rate"].first().round(3))
    table.insert(1, "n_stress", res.groupby("fold")["n_stress"].first())
    print(table.round(3).to_string())

    print("\n=== Mean over folds ===")
    print(res.groupby("model")[["pr_auc", "roc_auc"]].mean()
          .loc[["vol_only", "topo_only", "vol_topo", "vol_topo_chg"]].round(3).to_string())

    # --- Block bootstrap: is (vol_topo - vol_only) bigger than chance? ---
    rng = np.random.default_rng(0)
    for challenger in ["vol_topo", "vol_topo_chg"]:
        diffs = []
        for _ in range(N_BOOT):
            per_fold = []
            for tp in test_preds:
                b = block_bootstrap_indices(len(tp["y"]), rng)
                yb = tp["y"][b]
                if yb.sum() == 0 or yb.sum() == len(yb):
                    continue
                per_fold.append(average_precision_score(yb, tp[challenger][b]) -
                                average_precision_score(yb, tp["vol_only"][b]))
            if per_fold:
                diffs.append(np.mean(per_fold))
        diffs = np.array(diffs)
        observed = res[res.model == challenger]["pr_auc"].mean() - res[res.model == "vol_only"]["pr_auc"].mean()
        lo, hi = np.percentile(diffs, [5, 95])
        print(f"\n{challenger} minus vol_only (mean test PR-AUC): {observed:+.3f}")
        print(f"  90% bootstrap interval: [{lo:+.3f}, {hi:+.3f}] | share of resamples where topology helps: "
              f"{(diffs > 0).mean():.0%}")
        if lo > 0:
            print("  -> topology adds information beyond volatility (interval above 0).")
        elif hi < 0:
            print("  -> adding topology makes things worse (interval below 0).")
        else:
            print("  -> no detectable difference: the interval includes 0.")

    print("\n=== vol_topo coefficients (standardized; larger |value| = more weight; + raises stress probability) ===")
    coef_df = pd.concat(coefs, axis=1)
    coef_df["mean"] = coef_df.mean(axis=1)
    print(coef_df.reindex(coef_df["mean"].abs().sort_values(ascending=False).index).round(3).to_string())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "topo_value_test.csv"), index=False)
    coef_df.to_csv(os.path.join(RESULTS_DIR, "topo_value_coefficients.csv"))
    print("\nSaved: results/topo_value_test.csv and results/topo_value_coefficients.csv")


if __name__ == "__main__":
    main()

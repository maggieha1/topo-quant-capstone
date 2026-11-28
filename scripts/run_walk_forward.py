"""
Walk-forward experiment. For each test year, compares:
  vol5_rule        current 5-day basket volatility, no fitting (the "volatility persists" rule)
  vol_logistic     logistic regression on current 5- and 30-day basket volatility, fit on TRAIN
  tgnn_landscape   GAT + GRU + the 300-number persistence landscape
  tgnn_summary     GAT + GRU + 8 compressed topology summary features
  tgnn_no_topo     GAT + GRU with topology zeroed out (only with --ablation)
Each TGNN variant is trained once per seed; results are averaged over seeds.

Usage (from the project folder):
  python scripts/run_walk_forward.py --topo landscape summary --ablation --seeds 42 1 2
  python scripts/run_walk_forward.py --topo summary --ablation --seeds 42 1 2
  python scripts/run_walk_forward.py --quick            # 3 epochs, one seed: checks everything runs
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from windowing import DATA_PATH, PROJECT_ROOT
from build_graph import TOPO_PATH
from build_topo_summary import SUMMARY_PATH
from dataloaders import embargo_gap
from dataset import MarketGraphDataset
from evaluation import classification_report, pick_threshold, volatility_baselines
from train import overfitting_summary, predict, train_model
from walk_forward import build_fold, check_fold_leakage, walk_forward_folds

sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))
from tgnn import TopologicalGNN, count_params  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
WINDOW_SIZE, SEQ_LEN, HORIZON = 30, 5, 5
TOPO_FILES = {"landscape": TOPO_PATH, "summary": SUMMARY_PATH}
BASELINES = ("vol5_rule", "vol_logistic")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--topo", nargs="+", default=["landscape"], choices=list(TOPO_FILES),
                        help="which topology features to train with (one model per choice)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--ablation", action="store_true", help="also train the model without topology")
    parser.add_argument("--val-size", type=int, default=126, help="validation samples per fold (~6 months)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=32, help="model width; must be divisible by 4 (try 16)")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--verbose", action="store_true", help="print the per-epoch training log")
    parser.add_argument("--quick", action="store_true", help="3 epochs per model, first seed only (smoke test)")
    args = parser.parse_args()
    if args.quick:
        args.epochs, args.patience, args.seeds = 3, 3, args.seeds[:1]

    for kind in args.topo:
        if not os.path.exists(TOPO_FILES[kind]):
            sys.exit(f"Missing {TOPO_FILES[kind]}. Run: python scripts/"
                     f"{'build_topo_dataset' if kind == 'landscape' else 'build_topo_summary'}.py")

    gap = embargo_gap(WINDOW_SIZE, SEQ_LEN, HORIZON)
    probe = MarketGraphDataset(csv_path=DATA_PATH, topo_path=TOPO_PATH, window_size=WINDOW_SIZE,
                               seq_len=SEQ_LEN, horizon=HORIZON)
    rows = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).index
    folds = walk_forward_folds(probe.sample_dates, args.years, gap, val_size=args.val_size)
    n_trainings = len(folds) * len(args.seeds) * (len(args.topo) + int(args.ablation))
    print(f"Walk-forward: {len(folds)} folds x {len(args.seeds)} seeds x "
          f"{len(args.topo) + int(args.ablation)} TGNN variants = {n_trainings} trainings")

    results, overfit = [], []
    start = time.time()

    for fold in folds:
        check_fold_leakage(fold, rows, WINDOW_SIZE, SEQ_LEN, HORIZON)
        common = dict(window_size=WINDOW_SIZE, seq_len=SEQ_LEN, horizon=HORIZON, probe=probe)

        # One FoldData per topology kind (same samples, labels and threshold; different topo features)
        fds = {kind: build_fold(DATA_PATH, TOPO_FILES[kind], fold, **common) for kind in args.topo}
        if args.ablation:
            fds["no_topo"] = build_fold(DATA_PATH, TOPO_FILES[args.topo[0]], fold, drop_topo=True, **common)
        ref = next(iter(fds.values()))
        assert all(np.isclose(fd.vol_threshold, ref.vol_threshold) for fd in fds.values()), "Thresholds differ."

        d = ref.base.sample_dates
        y = {name: ref.base.sample_labels[idx].numpy() for name, idx in
             [("train", fold.train_idx), ("val", fold.val_idx), ("test", fold.test_idx)]}
        train_rate = float(y["train"].mean())

        print(f"\n=== Fold {fold.name} | stress threshold (train only) {ref.vol_threshold:.5f} ===")
        for name, idx in [("TRAIN", fold.train_idx), ("VAL", fold.val_idx), ("TEST", fold.test_idx)]:
            yy = ref.base.sample_labels[idx]
            print(f"  {name:5s} {len(idx):4d} samples | {d[idx[0]]} to {d[idx[-1]]} | "
                  f"stress {int(yy.sum()):3d} ({yy.mean().item():.1%})")

        # --- Baselines (deterministic: computed once, recorded with seed -1) ---
        for name, s in volatility_baselines(ref.base, fold.train_idx, fold.val_idx, fold.test_idx).items():
            thr = pick_threshold(y["val"], s["val"], s["train"], train_rate)
            results.append({"fold": fold.name, "model": name, "seed": -1,
                            **classification_report(y["test"], s["test"], thr)})

        # --- TGNN variants x seeds ---
        for kind, fd in fds.items():
            name = f"tgnn_{kind}"
            topo_dim = fd.base.topo_features.shape[1]
            for seed in args.seeds:
                torch.manual_seed(seed)
                fd.train_loader.generator.manual_seed(seed)  # shuffle order also depends on the seed
                model = TopologicalGNN(topo_in_dim=topo_dim, hidden_dim=args.hidden_dim,
                                       topo_embed_dim=args.hidden_dim, dropout=args.dropout)
                history, best_epoch = train_model(
                    model, fd.train_loader, fd.val_loader, pos_weight=(1 - train_rate) / train_rate,
                    max_epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                    patience=args.patience, verbose=args.verbose,
                )
                p_tr, _ = predict(model, fd.train_loader)
                p_va, y_va = predict(model, fd.val_loader)
                p_te, y_te = predict(model, fd.test_loader)
                thr = pick_threshold(y_va, p_va, p_tr, train_rate)
                rep = classification_report(y_te, p_te, thr)
                results.append({"fold": fold.name, "model": name, "seed": seed, **rep})
                of = overfitting_summary(history, best_epoch)
                overfit.append({"fold": fold.name, "model": name, "seed": seed, **of})
                print(f"  {name:15s} seed {seed:3d} | {count_params(model):6,} params | best epoch {best_epoch:2d} | "
                      f"train PR-AUC {of['train_pr_auc']:.3f} | val PR-AUC {of['val_pr_auc']:.3f} | "
                      f"TEST PR-AUC {rep['pr_auc']:.3f}")

    res = pd.DataFrame(results)
    ovf = pd.DataFrame(overfit)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(RESULTS_DIR, "walk_forward_results.csv"), index=False)
    ovf.to_csv(os.path.join(RESULTS_DIR, "overfitting_report.csv"), index=False)
    pd.set_option("display.width", 160)

    # Per fold: mean (and spread) of test PR-AUC over seeds
    print(f"\n=== Test PR-AUC by fold: mean over seeds (std in brackets) | {time.time() - start:.0f}s ===")
    by_fold = res.groupby(["fold", "model"])["pr_auc"].agg(["mean", "std"])
    cell = by_fold.apply(lambda r: f"{r['mean']:.3f}" + ("" if np.isnan(r["std"]) else f" ({r['std']:.3f})"), axis=1)
    base_rate = res.groupby("fold")["base_rate"].first()
    table = cell.unstack("model")
    table.insert(0, "base_rate", base_rate.round(3))
    table.insert(1, "n_stress", res.groupby("fold")["n_stress"].first())
    print(table.to_string())

    # Overall: mean over folds for each seed, then mean / std / min / max over seeds
    per_seed = res.groupby(["model", "seed"])[["pr_auc", "pr_auc_lift", "roc_auc", "precision", "recall"]].mean()
    summary = per_seed.groupby("model").agg(
        pr_auc=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"), pr_auc_min=("pr_auc", "min"),
        pr_auc_max=("pr_auc", "max"), lift=("pr_auc_lift", "mean"), roc_auc=("roc_auc", "mean"),
        precision=("precision", "mean"), recall=("recall", "mean"),
    ).sort_values("pr_auc", ascending=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_by_model.csv"))
    print("\n=== Mean over folds, then over seeds (std/min/max show seed-to-seed spread) ===")
    print(summary.round(3).to_string())

    print("\n=== Overfitting check (averaged over folds and seeds) ===")
    ovf["pr_auc_gap"] = ovf["train_pr_auc"] - ovf["val_pr_auc"]
    ovf_summary = ovf.groupby("model")[["best_epoch", "epochs_run", "train_pr_auc", "val_pr_auc", "pr_auc_gap"]].mean()
    print(ovf_summary.round(3).to_string())
    for m, r in ovf_summary.iterrows():
        if r["pr_auc_gap"] > 0.25:
            print(f"  WARNING: {m}: train PR-AUC exceeds val PR-AUC by {r['pr_auc_gap']:.2f} on average (memorizing).")

    few = res[(res["model"] == "vol5_rule") & (res["n_stress"] < 10)]
    if len(few):
        print("  CAUTION: fewer than 10 stress samples in TEST for fold(s):",
              ", ".join(f"{r.fold} ({r.n_stress})" for r in few.itertuples()), "-> PR-AUC there is very noisy.")

    # Verdicts, judged seed by seed so a lucky seed cannot carry the result
    seed_means = res.groupby(["model", "seed"])["pr_auc"].mean()
    best_base = max(BASELINES, key=lambda m: seed_means[m].mean())
    best_base_score = seed_means[best_base].mean()
    print()
    for m in [m for m in summary.index if m.startswith("tgnn")]:
        scores = seed_means[m]
        wins = int((scores > best_base_score).sum())
        diff = scores.mean() - best_base_score
        verdict = "BEATS" if diff > 0 and wins == len(scores) else ("ties / mixed vs." if wins > 0 else "does NOT beat")
        print(f"Verdict: {m} {verdict} the best baseline ({best_base}, {best_base_score:.3f}): "
              f"mean {scores.mean():.3f} ({diff:+.3f}); above it in {wins} of {len(scores)} seeds.")
    if "tgnn_no_topo" in seed_means:
        nt = seed_means["tgnn_no_topo"]
        for kind in args.topo:
            m = f"tgnn_{kind}"
            delta = seed_means[m] - nt.reindex(seed_means[m].index)
            print(f"Topology effect ({kind} vs. no topology): {delta.mean():+.3f} mean PR-AUC; "
                  f"helps in {int((delta > 0).sum())} of {len(delta)} seeds.")
    if args.quick:
        print("\nNOTE: this was a --quick run (3 epochs, 1 seed). Use it only to check the code runs.")
    print("\nSaved: results/walk_forward_results.csv, results/summary_by_model.csv, results/overfitting_report.csv")


if __name__ == "__main__":
    main()

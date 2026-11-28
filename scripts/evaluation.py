"""
Metrics and simple volatility baselines.

Metrics focus on the rare "stress" class: PR-AUC (average precision), ROC-AUC, and
precision / recall at a decision threshold chosen on VALIDATION, never on test.
"""
from typing import Dict, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NAN = float("nan")


def pick_threshold(y_val: np.ndarray, s_val: np.ndarray, s_train: np.ndarray, train_base_rate: float) -> float:
    """
    Decision threshold that maximizes F1 on validation.
    If validation has no stress samples, fall back to the score level that would flag the same
    share of TRAINING samples as the training stress rate.
    """
    y_val = np.asarray(y_val).astype(int)
    if y_val.sum() == 0:
        return float(np.quantile(s_train, 1.0 - train_base_rate))
    candidates = np.unique(s_val)
    f1s = [f1_score(y_val, (s_val >= c).astype(int), zero_division=0) for c in candidates]
    return float(candidates[int(np.argmax(f1s))])


def classification_report(y_true: Sequence[float], scores: Sequence[float], threshold: float) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    pred = (s >= threshold).astype(int)
    n_pos = int(y.sum())
    both_classes = 0 < n_pos < len(y)

    pr_auc = average_precision_score(y, s) if both_classes else NAN
    base_rate = float(y.mean())
    return {
        "n": len(y),
        "n_stress": n_pos,
        "base_rate": base_rate,
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / base_rate if both_classes else NAN,   # 1.0 = no better than random
        "roc_auc": roc_auc_score(y, s) if both_classes else NAN,
        "precision": precision_score(y, pred, zero_division=0) if pred.sum() > 0 else NAN,
        "recall": recall_score(y, pred, zero_division=0) if n_pos > 0 else NAN,
        "alarm_rate": float(pred.mean()),
    }


def trailing_vol_features(base, sample_indices: Sequence[int]) -> np.ndarray:
    """
    For each sample, the equal-weighted basket's volatility over the last 5 and last 30 days
    of its END window. Uses only data up to the sample's date, so there is no look-ahead.
    """
    feats = []
    for s in sample_indices:
        w = base.windows[base.end_window(s)].astype(np.float64)
        basket = w.mean(axis=1)
        feats.append([np.std(basket[-5:]), np.std(basket)])
    return np.asarray(feats)


def volatility_baselines(base, train_idx, val_idx, test_idx) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns scores for two baselines on each split:
      - "vol5_rule":   current 5-day basket volatility (no fitting at all)
      - "vol_logistic": logistic regression on log 5-day and log 30-day basket volatility, fit on TRAIN
    """
    F = {name: trailing_vol_features(base, idx) for name, idx in
         [("train", train_idx), ("val", val_idx), ("test", test_idx)]}
    y_train = base.sample_labels[train_idx].numpy().astype(int)

    logit = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    logit.fit(np.log(F["train"] + 1e-12), y_train)

    return {
        "vol5_rule": {split: F[split][:, 0] for split in F},
        "vol_logistic": {split: logit.predict_proba(np.log(F[split] + 1e-12))[:, 1] for split in F},
    }

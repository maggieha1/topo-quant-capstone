"""
Training loop with overfitting safeguards:
  - AdamW with weight decay (penalizes large weights)
  - early stopping on VALIDATION loss, restoring the best epoch's weights
  - per-epoch train vs. validation loss and PR-AUC, so the generalization gap is visible
  - gradient clipping for stability
"""
import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score


@torch.no_grad()
def predict(model: nn.Module, loader) -> Tuple[np.ndarray, np.ndarray]:
    """Stress probabilities and true labels for every sample in the loader (eval mode, no dropout)."""
    model.eval()
    probs, ys = [], []
    for graph_seq, y in loader:
        probs.append(torch.sigmoid(model(graph_seq)))
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


@torch.no_grad()
def mean_loss(model: nn.Module, loader, loss_fn) -> float:
    model.eval()
    total, n = 0.0, 0
    for graph_seq, y in loader:
        total += loss_fn(model(graph_seq), y).item() * len(y)
        n += len(y)
    return total / n


def safe_pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def train_model(model: nn.Module, train_loader, val_loader, pos_weight: float,
                max_epochs: int = 60, lr: float = 1e-3, weight_decay: float = 1e-2,
                patience: int = 10, grad_clip: float = 1.0, log_every: int = 5,
                verbose: bool = True) -> Tuple[List[Dict[str, float]], int]:
    """
    Trains with class-weighted BCE (pos_weight = negatives / positives in TRAIN) and early stopping.
    Returns the per-epoch history and the best epoch; the model ends with the best epoch's weights.
    """
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[Dict[str, float]] = []
    best_val, best_epoch, best_state, bad_epochs = float("inf"), 0, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for graph_seq, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(graph_seq), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Evaluate train and val the same way (eval mode) so the gap is a fair comparison
        train_loss = mean_loss(model, train_loader, loss_fn)
        val_loss = mean_loss(model, val_loader, loss_fn)
        p_tr, y_tr = predict(model, train_loader)
        p_va, y_va = predict(model, val_loader)
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                  "train_pr_auc": safe_pr_auc(y_tr, p_tr), "val_pr_auc": safe_pr_auc(y_va, p_va)}
        history.append(record)

        improved = val_loss < best_val - 1e-4
        if improved:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1

        if verbose and (epoch == 1 or epoch % log_every == 0 or improved and epoch < 3):
            print(f"    epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
                  f"train PR-AUC {record['train_pr_auc']:.3f} | val PR-AUC {record['val_pr_auc']:.3f}")

        if bad_epochs >= patience:
            if verbose:
                print(f"    early stop at epoch {epoch} (no val improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    return history, best_epoch


def overfitting_summary(history: List[Dict[str, float]], best_epoch: int) -> Dict[str, float]:
    """Generalization gap at the chosen epoch, and how much val loss worsened after it."""
    best = history[best_epoch - 1]
    last = history[-1]
    return {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_loss": best["train_loss"],
        "val_loss": best["val_loss"],
        "loss_gap": best["val_loss"] - best["train_loss"],
        "train_pr_auc": best["train_pr_auc"],
        "val_pr_auc": best["val_pr_auc"],
        "val_loss_rise_after_best": last["val_loss"] - best["val_loss"],
    }

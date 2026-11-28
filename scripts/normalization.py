"""
Train-only feature standardization for the sequence dataset.

FeatureScaler learns means and standard deviations from TRAINING windows only, then
applies them unchanged to validation and test. ScaledSequenceDataset wraps a
MarketGraphDataset so every graph it returns is standardized (and cached, since
graphs do not change between epochs).
"""
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from build_graph import build_market_graph, compute_node_features


class FeatureScaler:
    """Z-scores node features (per feature, pooled over stocks) and topo (per dimension)."""

    def __init__(self, x_mean: torch.Tensor, x_std: torch.Tensor,
                 topo_mean: torch.Tensor, topo_std: torch.Tensor):
        self.x_mean, self.x_std = x_mean, x_std
        self.topo_mean, self.topo_std = topo_mean, topo_std
        self.fit_windows: Optional[tuple] = None  # (first, last) window index used for fitting

    @staticmethod
    def windows_used(dataset, sample_indices: Sequence[int]) -> List[int]:
        """Every window that appears in any step of the given samples."""
        used = set()
        for s in sample_indices:
            used.update(range(s, s + dataset.seq_len))
        return sorted(used)

    @classmethod
    def fit(cls, dataset, train_indices: Sequence[int], min_std: float = 1e-6) -> "FeatureScaler":
        wins = cls.windows_used(dataset, train_indices)

        x = torch.stack([compute_node_features(dataset.windows[w]) for w in wins])  # (W, N, 4)
        x = x.reshape(-1, x.shape[-1])                                                # (W*N, 4)
        topo = dataset.topo_features[wins].to(torch.float32)                          # (W, 300)

        x_std = x.std(dim=0, unbiased=False)
        topo_std = topo.std(dim=0, unbiased=False)
        # Constant features (e.g. landscape grid points that are always 0) get std 1 so they stay 0
        x_std = torch.where(x_std < min_std, torch.ones_like(x_std), x_std)
        topo_std = torch.where(topo_std < min_std, torch.ones_like(topo_std), topo_std)

        scaler = cls(x.mean(dim=0), x_std, topo.mean(dim=0), topo_std)
        scaler.fit_windows = (wins[0], wins[-1])
        return scaler

    def transform(self, graph):
        g = graph.clone()
        g.x = (g.x - self.x_mean) / self.x_std
        g.topo = (g.topo - self.topo_mean) / self.topo_std
        return g

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"x_mean": self.x_mean, "x_std": self.x_std,
                "topo_mean": self.topo_mean, "topo_std": self.topo_std}


class ScaledSequenceDataset(Dataset):
    """
    Same samples as the wrapped MarketGraphDataset, but every graph is standardized with a
    train-fitted scaler. drop_topo=True replaces topo with zeros (for the "no topology" ablation).
    """

    def __init__(self, base, scaler: Optional[FeatureScaler], drop_topo: bool = False):
        self.base = base
        self.scaler = scaler
        self.drop_topo = drop_topo
        self._cache: Dict[int, object] = {}

    def __len__(self) -> int:
        return len(self.base)

    def _graph(self, w: int):
        if w not in self._cache:
            g = build_market_graph(self.base.windows[w], self.base.topo_features[w],
                                   threshold=self.base.corr_threshold)
            if self.scaler is not None:
                g = self.scaler.transform(g)
            if self.drop_topo:
                g.topo = torch.zeros_like(g.topo)
            self._cache[w] = g
        return self._cache[w]

    def __getitem__(self, idx: int):
        if not 0 <= idx < len(self):
            raise IndexError(f"Sample index {idx} out of range [0, {len(self)}).")
        graph_seq = [self._graph(idx + step) for step in range(self.base.seq_len)]
        return graph_seq, self.base.sample_labels[idx]

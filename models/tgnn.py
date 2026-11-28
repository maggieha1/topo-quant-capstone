import os
import sys
from typing import List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")


class TopologicalGNN(nn.Module):
    """
    Spatial-Temporal Topological Graph Neural Network for financial regime-shift forecasting.
    Processes sequences of L consecutive market graph snapshots + persistence landscapes.
    """

    def __init__(
        self,
        node_in_dim: int = 4,
        topo_in_dim: int = 300,
        hidden_dim: int = 32,
        topo_embed_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads."

        self.node_in_dim = node_in_dim
        self.topo_in_dim = topo_in_dim
        self.hidden_dim = hidden_dim
        self.topo_embed_dim = topo_embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Branch 1: Multi-head graph attention encoder (uses edge_attr via edge_dim=1)
        self.gat1 = GATConv(
            in_channels=node_in_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=1,
            dropout=dropout,
        )
        self.gat_norm1 = nn.LayerNorm(hidden_dim)

        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=False,
            edge_dim=1,
            dropout=dropout,
        )
        self.gat_norm2 = nn.LayerNorm(hidden_dim)

        # Branch 2: Persistence landscape projection MLP (300 -> 64 -> 32)
        self.topo_encoder = nn.Sequential(
            nn.Linear(topo_in_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, topo_embed_dim),
            nn.LayerNorm(topo_embed_dim),
            nn.GELU(),
        )

        # Temporal aggregation: 1-layer GRU over L consecutive fused market states
        fusion_dim = hidden_dim + topo_embed_dim
        self.temporal_encoder = nn.GRU(
            input_size=fusion_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Classification head (final GRU hidden state -> raw logit)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init for MLP layers; orthogonal init for GRU recurrent weights."""
        for block in (self.topo_encoder, self.head):
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        for name, param in self.temporal_encoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def encode_graph(self, data) -> torch.Tensor:
        """Runs 2-layer GAT message passing and pools nodes to (B, hidden_dim)."""
        h = self.gat1(data.x, data.edge_index, edge_attr=data.edge_attr)
        h = F.gelu(self.gat_norm1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.gat2(h, data.edge_index, edge_attr=data.edge_attr)
        h = F.gelu(self.gat_norm2(h))

        return global_mean_pool(h, data.batch)

    def encode_step(self, data) -> torch.Tensor:
        """Fuses graph embedding and persistence landscape for one time step -> (B, fusion_dim)."""
        graph_emb = self.encode_graph(data)      # (B, hidden_dim)
        topo_emb = self.topo_encoder(data.topo)  # (B, topo_embed_dim)
        return torch.cat([graph_emb, topo_emb], dim=-1)

    def forward(self, graph_seq: Union[Sequence, object]) -> torch.Tensor:
        """
        Args:
            graph_seq: List of L PyG Batch objects ordered chronologically [t-L+1, ..., t].
                       A single Batch is treated as a sequence of length 1.
        Returns:
            logits: Tensor of shape (B,) for the regime shift following time t.
        """
        if not isinstance(graph_seq, (list, tuple)):
            graph_seq = [graph_seq]

        batch_size = graph_seq[0].num_graphs
        assert all(g.num_graphs == batch_size for g in graph_seq), "All time steps must have the same batch size."

        step_embs = [self.encode_step(g) for g in graph_seq]
        seq_tensor = torch.stack(step_embs, dim=1)    # (B, L, fusion_dim)

        _, h_n = self.temporal_encoder(seq_tensor)    # h_n: (1, B, hidden_dim)
        final_state = self.temporal_norm(h_n[-1])     # (B, hidden_dim)

        logits = self.head(final_state).squeeze(-1)   # (B,)
        return logits


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    sys.path.insert(0, SCRIPTS_DIR)
    from windowing import DATA_PATH
    from build_graph import TOPO_PATH
    from dataloaders import create_chronological_loaders

    torch.manual_seed(42)  # reproducible weight initialization

    train_loader, _, _ = create_chronological_loaders(
        csv_path=DATA_PATH,
        topo_path=TOPO_PATH,
        batch_size=32,
        seq_len=5,
    )
    graph_seq, targets = next(iter(train_loader))  # real 5-step sequences; targets = last-step labels
    seq_len = len(graph_seq)

    model = TopologicalGNN(
        node_in_dim=4,
        topo_in_dim=300,
        hidden_dim=32,
        topo_embed_dim=32,
        num_heads=4,
        dropout=0.2,
    )

    # Forward + backward with the real loss
    model.train()
    logits = model(graph_seq)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()

    assert logits.shape == targets.shape, f"Unexpected logits shape: {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "Non-finite logits detected."

    # Gradients must reach every part of the model
    for name, part in [("gat1", model.gat1), ("gat2", model.gat2),
                       ("topo_encoder", model.topo_encoder), ("temporal_encoder", model.temporal_encoder)]:
        for pname, p in part.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"No/invalid gradient in {name}.{pname}"
    assert model.temporal_encoder.weight_hh_l0.grad.abs().sum() > 0, "No gradient flowing into GRU!"
    assert model.gat1.att_edge.grad.abs().sum() > 0, "No gradient flowing into GAT edge weights!"

    model.eval()
    with torch.no_grad():
        out_fwd = model(graph_seq)
        out_rev = model(graph_seq[::-1])       # same days, reversed order
        out_single = model(graph_seq[-1])      # a single batch = sequence of length 1
    assert not torch.allclose(out_fwd, out_rev), "Reversing time order changed nothing: GRU ignores order!"
    assert out_single.shape == out_fwd.shape, "Single-batch input should also work."

    print("Success! Spatial-Temporal TopologicalGNN (GAT + GRU) verified.")
    print(f"  Total trainable parameters: {count_params(model):,}")
    print(f"    graph encoder (GAT):  {sum(count_params(m) for m in (model.gat1, model.gat_norm1, model.gat2, model.gat_norm2)):,}")
    print(f"    topology encoder:     {count_params(model.topo_encoder):,}")
    print(f"    temporal GRU (+norm): {count_params(model.temporal_encoder) + count_params(model.temporal_norm):,}")
    print(f"    head:                 {count_params(model.head):,}")
    print(f"  Sequence: {seq_len} steps x {len(targets)} samples (from the training DataLoader)")
    print(f"  GRU recurrent grad norm:    {model.temporal_encoder.weight_hh_l0.grad.norm().item():.6f}")
    print(f"  Order matters: mean |forward - reversed| logit difference = {(out_fwd - out_rev).abs().mean().item():.4f}")
    print(f"  Initial BCE loss: {loss.item():.4f}")
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")


class TopologicalGNN(nn.Module):
    """
    Dual-branch Topological Graph Neural Network for financial regime-shift forecasting.
    Fuses asset graph embeddings (GAT) with persistent homology landscapes.
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
            concat=True,          # 4 heads x 8 features -> 32
            edge_dim=1,
            dropout=dropout,
        )
        self.gat_norm1 = nn.LayerNorm(hidden_dim)

        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=False,         # average the 4 heads -> 32
            edge_dim=1,
            dropout=dropout,
        )
        self.gat_norm2 = nn.LayerNorm(hidden_dim)

        # Temporal encoder placeholder (to be added on Day 18)
        self.temporal_encoder = None

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

        # Fusion & classification head (outputs a raw logit)
        fusion_dim = hidden_dim + topo_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_linear_weights()

    def _init_linear_weights(self) -> None:
        """Applies Xavier uniform initialization to MLP Linear layers (leaving GATConv defaults intact)."""
        for block in (self.topo_encoder, self.head):
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def encode_graph(self, data) -> torch.Tensor:
        """Runs 2-layer GAT message passing and pools nodes to a graph-level vector."""
        h = self.gat1(data.x, data.edge_index, edge_attr=data.edge_attr)
        h = F.gelu(self.gat_norm1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.gat2(h, data.edge_index, edge_attr=data.edge_attr)
        h = F.gelu(self.gat_norm2(h))

        # Pool (B * 10, hidden_dim) -> (B, hidden_dim)
        return global_mean_pool(h, data.batch)

    def forward(self, data) -> torch.Tensor:
        graph_emb = self.encode_graph(data)      # (B, hidden_dim)
        topo_emb = self.topo_encoder(data.topo)  # (B, topo_embed_dim)

        fused = torch.cat([graph_emb, topo_emb], dim=-1)
        logits = self.head(fused).squeeze(-1)    # (B,)
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
    )
    batch = next(iter(train_loader))

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
    logits = model(batch)
    loss = F.binary_cross_entropy_with_logits(logits, batch.y)
    loss.backward()

    assert logits.shape == (batch.y.shape[0],), f"Expected {(batch.y.shape[0],)}, got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "Non-finite logits detected."

    # Gradients must reach every part of the model
    for name, part in [("gat1", model.gat1), ("gat2", model.gat2), ("topo_encoder", model.topo_encoder)]:
        for pname, p in part.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"No/invalid gradient in {name}.{pname}"
    assert model.gat1.att_edge.grad.abs().sum() > 0, "No gradient flowing into GAT edge_attr weights!"

    # In eval mode (no dropout) the model must be deterministic
    model.eval()
    with torch.no_grad():
        out_a, out_b = model(batch), model(batch)
        graph_emb = model.encode_graph(batch)
    assert torch.allclose(out_a, out_b), "Eval-mode outputs are not deterministic."
    assert graph_emb.shape == (batch.num_graphs, model.hidden_dim), f"Unexpected graph_emb shape {tuple(graph_emb.shape)}"
    assert graph_emb.abs().sum() > 0, "Graph branch is still all zeros."

    graph_params = sum(count_params(m) for m in (model.gat1, model.gat_norm1, model.gat2, model.gat_norm2))
    print("Success! Multi-head GATConv spatial encoder verified.")
    print(f"  Total trainable parameters: {count_params(model):,}")
    print(f"    graph encoder (GAT):  {graph_params:,}")
    print(f"    topology encoder:     {count_params(model.topo_encoder):,}")
    print(f"    head:                 {count_params(model.head):,}")
    print(f"  Graph embedding shape:  {tuple(graph_emb.shape)}  (no longer zeros)")
    print(f"  GAT1 edge-attention grad norm: {model.gat1.att_edge.grad.norm().item():.6f}")
    print(f"  Initial BCE loss:       {loss.item():.4f}")
import os
import sys

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")


class TopologicalGNN(nn.Module):
    """
    Dual-branch Topological Graph Neural Network for financial regime-shift forecasting.
    Fuses asset graph embeddings with persistent homology landscapes.
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
        self.node_in_dim = node_in_dim
        self.topo_in_dim = topo_in_dim
        self.hidden_dim = hidden_dim
        self.topo_embed_dim = topo_embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # Branch 1: Graph encoder (to be implemented on Days 17-18)
        self.spatial_encoder = None
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

        # Fusion & classification head (outputs a raw logit for regime-shift probability)
        fusion_dim = hidden_dim + topo_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Applies Xavier uniform initialization to all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data) -> torch.Tensor:
        """
        Placeholder forward pass to verify the topological branch and head shapes.
        Returns raw logits of shape (B,). Never apply sigmoid here.
        """
        topo_emb = self.topo_encoder(data.topo)  # (B, topo_embed_dim)

        # Temporary zero-tensor stand-in for the graph branch until Day 17
        batch_size = topo_emb.size(0)
        graph_emb = torch.zeros(batch_size, self.hidden_dim, device=topo_emb.device)

        fused = torch.cat([graph_emb, topo_emb], dim=-1)  # (B, hidden_dim + topo_embed_dim)
        logits = self.head(fused).squeeze(-1)             # (B,)
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

    logits = model(batch)

    assert logits.shape == (batch.y.shape[0],), f"Expected {(batch.y.shape[0],)}, got {tuple(logits.shape)}"
    assert torch.isfinite(logits).all(), "Non-finite logits detected."

    # Gradient check: one backward pass must reach the very first topology layer
    loss = nn.functional.binary_cross_entropy_with_logits(logits, batch.y)
    loss.backward()
    first_grad = model.topo_encoder[0].weight.grad
    assert first_grad is not None and first_grad.abs().sum() > 0, "No gradient reached the topology encoder."

    probs = torch.sigmoid(logits.detach())
    print("Success! TopologicalGNN scaffold verified.")
    print(f"  Trainable parameters (current): {count_params(model):,}")
    print(f"    topology encoder:   {count_params(model.topo_encoder):,}")
    print(f"      first layer only: {count_params(model.topo_encoder[0]):,}")
    print(f"    head:               {count_params(model.head):,}")
    print(f"  Output logits shape:  {tuple(logits.shape)}")
    print(f"  Logit range:          {logits.min().item():+.3f} to {logits.max().item():+.3f}")
    print(f"  Predicted prob range: {probs.min().item():.3f} to {probs.max().item():.3f}")
    print(f"  Initial BCE loss:     {loss.item():.4f}  (an untrained model is usually around 0.6-0.8)")
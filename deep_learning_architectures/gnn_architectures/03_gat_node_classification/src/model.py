import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNodeClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = GATConv(
            input_dim,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
        )
        self.conv2 = GATConv(
            hidden_dim * num_heads,
            num_classes,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


if __name__ == "__main__":
    num_nodes = 6
    input_dim = 8
    hidden_dim = 4
    num_heads = 2
    num_classes = 3
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
        ],
        dtype=torch.long,
    )

    model = GATNodeClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_heads=num_heads,
        dropout=0.6,
    )
    logits = model(x, edge_index)
    print(f"Input x shape: {tuple(x.shape)}")
    print(f"edge_index shape: {tuple(edge_index.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")

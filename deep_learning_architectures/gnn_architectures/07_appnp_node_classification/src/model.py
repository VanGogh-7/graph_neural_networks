import torch
from torch import nn
from torch_geometric.nn import APPNP


class APPNPNodeClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        propagation_steps: int,
        teleport_probability: float,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.propagation = APPNP(
            K=propagation_steps,
            alpha=teleport_probability,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.propagation(x, edge_index)
        return x


if __name__ == "__main__":
    num_nodes = 6
    input_dim = 8
    hidden_dim = 16
    num_classes = 3
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
        ],
        dtype=torch.long,
    )

    model = APPNPNodeClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=0.5,
        propagation_steps=10,
        teleport_probability=0.1,
    )
    logits = model(x, edge_index)
    print(f"Input x shape: {tuple(x.shape)}")
    print(f"edge_index shape: {tuple(edge_index.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")

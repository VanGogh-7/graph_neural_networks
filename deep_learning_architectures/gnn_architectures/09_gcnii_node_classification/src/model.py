import math

import torch
from torch import nn
from torch_geometric.nn import GCNConv

try:
    from torch_geometric.nn import GCN2Conv
except ImportError:  # pragma: no cover - used only with older PyG versions.
    GCN2Conv = None


class EducationalGCN2Conv(nn.Module):
    """Small fallback that illustrates the two core GCNII ingredients."""

    def __init__(
        self,
        channels: int,
        alpha: float,
        theta: float,
        layer: int,
        shared_weights: bool,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = math.log(theta / layer + 1.0) if theta > 0 else 0.0
        self.shared_weights = shared_weights
        self.propagation = GCNConv(channels, channels)
        self.linear = nn.Linear(channels, channels, bias=False)
        self.initial_linear = None
        if not shared_weights:
            self.initial_linear = nn.Linear(channels, channels, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        x_0: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        propagated = self.propagation(x, edge_index)
        residual_mix = (1 - self.alpha) * propagated + self.alpha * x_0
        transformed = self.linear(residual_mix)
        if self.initial_linear is not None:
            transformed = transformed + self.initial_linear(x_0)
        return (1 - self.beta) * residual_mix + self.beta * transformed


def make_gcnii_layer(
    channels: int,
    alpha: float,
    theta: float,
    layer: int,
    shared_weights: bool,
) -> nn.Module:
    if GCN2Conv is not None:
        return GCN2Conv(
            channels=channels,
            alpha=alpha,
            theta=theta,
            layer=layer,
            shared_weights=shared_weights,
        )
    return EducationalGCN2Conv(
        channels=channels,
        alpha=alpha,
        theta=theta,
        layer=layer,
        shared_weights=shared_weights,
    )


class GCNIINodeClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        alpha: float,
        theta: float,
        shared_weights: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList(
            [
                make_gcnii_layer(
                    channels=hidden_dim,
                    alpha=alpha,
                    theta=theta,
                    layer=layer,
                    shared_weights=shared_weights,
                )
                for layer in range(1, num_layers + 1)
            ]
        )
        self.output_linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_linear(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x_0 = x

        for conv in self.convs:
            x = conv(x, x_0, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)

        logits = self.output_linear(x)
        return logits


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

    model = GCNIINodeClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=4,
        alpha=0.1,
        theta=0.5,
        shared_weights=True,
        dropout=0.5,
    )
    logits = model(x, edge_index)
    print(f"Input x shape: {tuple(x.shape)}")
    print(f"edge_index shape: {tuple(edge_index.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")

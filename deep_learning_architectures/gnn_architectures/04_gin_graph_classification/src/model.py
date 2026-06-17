import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_add_pool


def make_gin_mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class GINGraphClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        current_dim = input_dim
        for _ in range(num_layers):
            self.convs.append(GINConv(make_gin_mlp(current_dim, hidden_dim)))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        for conv, batch_norm in zip(self.convs, self.batch_norms, strict=True):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = torch.relu(x)

        graph_embeddings = global_add_pool(x, batch)
        return self.classifier(graph_embeddings)


if __name__ == "__main__":
    graph_1 = Data(
        x=torch.randn(4, 6),
        edge_index=torch.tensor(
            [
                [0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2],
            ],
            dtype=torch.long,
        ),
        y=torch.tensor([0]),
    )
    graph_2 = Data(
        x=torch.randn(3, 6),
        edge_index=torch.tensor(
            [
                [0, 1, 1, 2],
                [1, 0, 2, 1],
            ],
            dtype=torch.long,
        ),
        y=torch.tensor([1]),
    )
    batch = Batch.from_data_list([graph_1, graph_2])

    model = GINGraphClassifier(
        input_dim=6,
        hidden_dim=16,
        num_classes=2,
        num_layers=3,
        dropout=0.5,
    )
    logits = model(batch.x, batch.edge_index, batch.batch)
    print(f"x shape: {tuple(batch.x.shape)}")
    print(f"edge_index shape: {tuple(batch.edge_index.shape)}")
    print(f"batch shape: {tuple(batch.batch.shape)}")
    print(f"y shape: {tuple(batch.y.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")

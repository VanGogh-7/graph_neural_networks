import torch
from torch import nn
from torch_geometric.nn import GCNConv, JumpingKnowledge


class JKNetNodeClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        jk_mode: str,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if jk_mode not in {"cat", "max", "lstm"}:
            raise ValueError("jk_mode must be one of: cat, max, lstm.")

        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.jumping_knowledge = JumpingKnowledge(
            mode=jk_mode,
            channels=hidden_dim,
            num_layers=num_layers,
        )
        classifier_input_dim = hidden_dim * num_layers if jk_mode == "cat" else hidden_dim
        self.classifier = nn.Linear(classifier_input_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden_states = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
            hidden_states.append(x)

        x = self.jumping_knowledge(hidden_states)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    num_nodes = 6
    input_dim = 8
    hidden_dim = 16
    num_classes = 3
    num_layers = 4
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
        ],
        dtype=torch.long,
    )

    model = JKNetNodeClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        jk_mode="cat",
        dropout=0.5,
    )
    logits = model(x, edge_index)
    print(f"Input x shape: {tuple(x.shape)}")
    print(f"edge_index shape: {tuple(edge_index.shape)}")
    print(f"Logits shape: {tuple(logits.shape)}")

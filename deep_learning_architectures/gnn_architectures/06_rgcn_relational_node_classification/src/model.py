import torch
from torch import nn
from torch_geometric.nn import RGCNConv


class RGCNNodeClassifier(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_relations: int,
        dropout: float,
        use_node_embeddings: bool,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.use_node_embeddings = use_node_embeddings
        if use_node_embeddings:
            self.node_embeddings = nn.Embedding(num_nodes, input_dim)
        else:
            self.node_embeddings = None

        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations)
        self.conv2 = RGCNConv(hidden_dim, num_classes, num_relations)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor | None,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_node_embeddings:
            node_ids = torch.arange(self.num_nodes, device=edge_index.device)
            x = self.node_embeddings(node_ids)
        elif x is None:
            raise ValueError("x must be provided when use_node_embeddings is False.")

        x = self.conv1(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type)
        return x


if __name__ == "__main__":
    num_nodes = 6
    input_dim = 8
    hidden_dim = 16
    num_classes = 3
    num_relations = 4
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 4, 5],
            [1, 0, 2, 1, 3, 2, 5, 4],
        ],
        dtype=torch.long,
    )
    edge_type = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)

    model = RGCNNodeClassifier(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_relations=num_relations,
        dropout=0.2,
        use_node_embeddings=True,
    )
    logits = model(None, edge_index, edge_type)
    print(f"edge_index shape: {tuple(edge_index.shape)}")
    print(f"edge_type shape: {tuple(edge_type.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")

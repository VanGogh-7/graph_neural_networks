import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GAEEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.conv2(x, edge_index)


class VGAEEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logstd = GCNConv(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index).clamp(max=10)
        return mu, logstd


class GraphAutoEncoderLinkPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
        model_type: str = "GAE",
    ) -> None:
        super().__init__()
        self.model_type = model_type.upper()
        if self.model_type == "GAE":
            self.encoder = GAEEncoder(input_dim, hidden_dim, latent_dim, dropout)
        elif self.model_type == "VGAE":
            self.encoder = VGAEEncoder(input_dim, hidden_dim, latent_dim, dropout)
        else:
            raise ValueError("model_type must be either 'GAE' or 'VGAE'.")

    def reparameterize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(logstd)
            return mu + noise * torch.exp(logstd)
        return mu

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.model_type == "VGAE":
            mu, logstd = self.encoder(x, edge_index)
            z = self.reparameterize(mu, logstd)
            return z, mu, logstd
        return self.encoder(x, edge_index)

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        source = z[edge_label_index[0]]
        target = z[edge_label_index[1]]
        return (source * target).sum(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encode(x, edge_index)
        z = encoded[0] if self.model_type == "VGAE" else encoded
        return self.decode(z, edge_label_index)


if __name__ == "__main__":
    num_nodes = 6
    input_dim = 8
    hidden_dim = 16
    latent_dim = 4
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3],
        ],
        dtype=torch.long,
    )
    edge_label_index = torch.tensor(
        [
            [0, 1, 2, 4],
            [2, 3, 4, 5],
        ],
        dtype=torch.long,
    )

    model = GraphAutoEncoderLinkPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=0.0,
        model_type="GAE",
    )
    z = model.encode(x, edge_index)
    edge_logits = model(x, edge_index, edge_label_index)
    print(f"x shape: {tuple(x.shape)}")
    print(f"edge_index shape: {tuple(edge_index.shape)}")
    print(f"z shape: {tuple(z.shape)}")
    print(f"edge_label_index shape: {tuple(edge_label_index.shape)}")
    print(f"edge_logits shape: {tuple(edge_logits.shape)}")

    variational_model = GraphAutoEncoderLinkPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=0.0,
        model_type="VGAE",
    )
    variational_model.eval()
    vgae_z, mu, logstd = variational_model.encode(x, edge_index)
    print(f"VGAE z shape: {tuple(vgae_z.shape)}")
    print(f"VGAE mu shape: {tuple(mu.shape)}")
    print(f"VGAE logstd shape: {tuple(logstd.shape)}")

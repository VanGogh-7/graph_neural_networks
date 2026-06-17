from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "cora"
    seed: int = 42
    hidden_dim: int = 64
    dropout: float = 0.5
    propagation_steps: int = 10
    teleport_probability: float = 0.1
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    n_epochs: int = 200
    model_path: str = "appnp_cora.pt"
    history_path: str = "training_history.json"
    curve_path: str = "training_curves.png"
    output_dir: str = "outputs"
    num_predictions: int = 10

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")
        if self.propagation_steps <= 0:
            raise ValueError("propagation_steps must be positive.")
        if not 0 <= self.teleport_probability <= 1:
            raise ValueError("teleport_probability must be in the range [0, 1].")
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be positive.")

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "cora"
    seed: int = 42
    model_type: str = "GAE"
    hidden_dim: int = 64
    latent_dim: int = 32
    dropout: float = 0.0
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    kl_weight: float = 0.001
    n_epochs: int = 200
    validation_ratio: float = 0.05
    test_ratio: float = 0.1
    model_path: str = "gae_cora.pt"
    history_path: str = "training_history.json"
    curve_path: str = "training_curves.png"
    output_dir: str = "outputs"
    num_predictions: int = 10

    def __post_init__(self) -> None:
        self.model_type = self.model_type.upper()
        if self.model_type not in {"GAE", "VGAE"}:
            raise ValueError("model_type must be either 'GAE' or 'VGAE'.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")
        if self.kl_weight < 0:
            raise ValueError("kl_weight must be non-negative.")
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be positive.")
        if not 0 < self.validation_ratio < 1:
            raise ValueError("validation_ratio must be between 0 and 1.")
        if not 0 < self.test_ratio < 1:
            raise ValueError("test_ratio must be between 0 and 1.")
        if self.validation_ratio + self.test_ratio >= 1:
            raise ValueError("validation_ratio + test_ratio must be less than 1.")

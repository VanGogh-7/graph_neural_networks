from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "mutag"
    dataset_name: str = "MUTAG"
    seed: int = 42
    batch_size: int = 32
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.5
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    n_epochs: int = 100
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    model_path: str = "gin_mutag.pt"
    history_path: str = "training_history.json"
    curve_path: str = "training_curves.png"
    output_dir: str = "outputs"
    num_predictions: int = 10

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be positive.")
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1.")
        if not 0 < self.valid_ratio < 1:
            raise ValueError("valid_ratio must be between 0 and 1.")
        if self.train_ratio + self.valid_ratio >= 1:
            raise ValueError("train_ratio + valid_ratio must be less than 1.")

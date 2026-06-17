from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class TrainConfig:
    # All datasets are stored in the repository-level datasets/ directory.
    data_root: Path = REPO_ROOT / "datasets" / "aifb"
    dataset_name: str = "AIFB"
    seed: int = 42
    valid_ratio: float = 0.2
    embedding_dim: int = 16
    hidden_dim: int = 16
    dropout: float = 0.2
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    n_epochs: int = 100
    model_path: str = "rgcn_aifb.pt"
    history_path: str = "training_history.json"
    curve_path: str = "training_curves.png"
    output_dir: str = "outputs"
    num_predictions: int = 10

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty.")
        if not 0 < self.valid_ratio < 1:
            raise ValueError("valid_ratio must be between 0 and 1.")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be positive.")

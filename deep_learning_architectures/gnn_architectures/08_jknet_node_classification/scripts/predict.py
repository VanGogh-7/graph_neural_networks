import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import load_cora_dataset, print_dataset_statistics
from src.model import JKNetNodeClassifier
from src.utils import get_device, load_checkpoint


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    dataset, data = load_cora_dataset(config.data_root)
    print_dataset_statistics(dataset, data)
    data = data.to(device)

    model = JKNetNodeClassifier(
        input_dim=dataset.num_node_features,
        hidden_dim=config.hidden_dim,
        num_classes=dataset.num_classes,
        num_layers=config.num_layers,
        jk_mode=config.jk_mode,
        dropout=config.dropout,
    ).to(device)
    checkpoint_path = PROJECT_ROOT / config.model_path
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        predictions = logits.argmax(dim=1)

    test_node_ids = data.test_mask.nonzero(as_tuple=False).view(-1)
    for node_id in test_node_ids[: config.num_predictions].tolist():
        predicted_class = int(predictions[node_id].item())
        true_class = int(data.y[node_id].item())
        correct = predicted_class == true_class
        print(
            f"Node id: {node_id}, "
            f"predicted class: {predicted_class}, "
            f"true class: {true_class}, "
            f"correct: {correct}"
        )


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_link_prediction_splits, print_dataset_statistics
from src.model import GraphAutoEncoderLinkPredictor
from src.utils import get_device, load_checkpoint


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    dataset, original_data, train_data, validation_data, test_data = (
        build_link_prediction_splits(
            data_root=config.data_root,
            validation_ratio=config.validation_ratio,
            test_ratio=config.test_ratio,
            seed=config.seed,
        )
    )
    print_dataset_statistics(dataset, original_data, train_data, validation_data, test_data)
    test_data = test_data.to(device)

    model = GraphAutoEncoderLinkPredictor(
        input_dim=dataset.num_node_features,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
        model_type=config.model_type,
    ).to(device)
    checkpoint_path = PROJECT_ROOT / config.model_path
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    with torch.no_grad():
        encoded = model.encode(test_data.x, test_data.edge_index)
        z = encoded[0] if config.model_type == "VGAE" else encoded

        positive_edges = test_data.pos_edge_label_index[:, : config.num_predictions]
        positive_logits = model.decode(z, positive_edges)
        positive_probabilities = torch.sigmoid(positive_logits)

    for edge_index in range(positive_edges.size(1)):
        source = int(positive_edges[0, edge_index].item())
        target = int(positive_edges[1, edge_index].item())
        probability = float(positive_probabilities[edge_index].item())
        predicted_label = int(probability >= 0.5)
        print(
            f"Source node: {source}, "
            f"target node: {target}, "
            f"predicted link probability: {probability:.4f}, "
            f"predicted label: {predicted_label}"
        )


if __name__ == "__main__":
    main()

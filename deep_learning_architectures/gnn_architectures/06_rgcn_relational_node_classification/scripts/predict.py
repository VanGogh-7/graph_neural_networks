import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_relational_node_data, print_dataset_statistics
from src.model import RGCNNodeClassifier
from src.utils import get_device, load_checkpoint


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    dataset, data, y, train_idx, validation_idx, test_idx = build_relational_node_data(
        data_root=config.data_root,
        dataset_name=config.dataset_name,
        valid_ratio=config.valid_ratio,
        seed=config.seed,
    )
    print_dataset_statistics(
        dataset_name=config.dataset_name,
        dataset=dataset,
        data=data,
        train_idx=train_idx,
        validation_idx=validation_idx,
        test_idx=test_idx,
    )

    x = data.x if hasattr(data, "x") and data.x is not None else None
    use_node_embeddings = x is None
    input_dim = config.embedding_dim if use_node_embeddings else dataset.num_node_features

    edge_index = data.edge_index.to(device)
    edge_type = data.edge_type.to(device)
    y = y.to(device)
    test_idx = test_idx.to(device)
    if x is not None:
        x = x.to(device)

    model = RGCNNodeClassifier(
        num_nodes=data.num_nodes,
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_classes=dataset.num_classes,
        num_relations=dataset.num_relations,
        dropout=config.dropout,
        use_node_embeddings=use_node_embeddings,
    ).to(device)
    checkpoint_path = PROJECT_ROOT / config.model_path
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    with torch.no_grad():
        logits = model(x, edge_index, edge_type)
        predictions = logits.argmax(dim=1)

    for node_id in test_idx[: config.num_predictions].tolist():
        predicted_class = int(predictions[node_id].item())
        true_class = int(y[node_id].item())
        correct = predicted_class == true_class
        print(
            f"Node id: {node_id}, "
            f"predicted class: {predicted_class}, "
            f"true class: {true_class}, "
            f"correct: {correct}"
        )


if __name__ == "__main__":
    main()

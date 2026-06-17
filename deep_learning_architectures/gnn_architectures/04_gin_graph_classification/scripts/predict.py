import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_dataloaders, print_dataset_statistics
from src.model import GINGraphClassifier
from src.utils import get_device, load_checkpoint


def main() -> None:
    config = TrainConfig()
    device = get_device()
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, dataset = build_dataloaders(
        data_root=config.data_root,
        dataset_name=config.dataset_name,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        valid_ratio=config.valid_ratio,
        seed=config.seed,
    )
    print_dataset_statistics(
        dataset_name=config.dataset_name,
        dataset=dataset,
        train_graphs=len(train_loader.dataset),
        valid_graphs=len(valid_loader.dataset),
        test_graphs=len(test_loader.dataset),
    )

    model = GINGraphClassifier(
        input_dim=dataset.num_node_features,
        hidden_dim=config.hidden_dim,
        num_classes=dataset.num_classes,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    checkpoint_path = PROJECT_ROOT / config.model_path
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    printed = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            predictions = logits.argmax(dim=1)
            graph_ids = batch.graph_id.view(-1)

            for item_index in range(batch.num_graphs):
                graph_index = int(graph_ids[item_index].item())
                predicted_class = int(predictions[item_index].item())
                true_class = int(batch.y[item_index].item())
                correct = predicted_class == true_class
                print(
                    f"Graph index: {graph_index}, "
                    f"predicted class: {predicted_class}, "
                    f"true class: {true_class}, "
                    f"correct: {correct}"
                )
                printed += 1
                if printed >= config.num_predictions:
                    return


if __name__ == "__main__":
    main()

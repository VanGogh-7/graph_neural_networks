import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import load_cora_dataset, print_dataset_statistics
from src.engine import evaluate, train
from src.model import APPNPNodeClassifier
from src.utils import clear_memory, get_device, load_checkpoint, save_json, set_seed
from src.visualize import plot_training_curves


def main() -> None:
    config = TrainConfig()
    clear_memory()
    set_seed(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    dataset, data = load_cora_dataset(config.data_root)
    print_dataset_statistics(dataset, data)
    data = data.to(device)

    model = APPNPNodeClassifier(
        input_dim=dataset.num_node_features,
        hidden_dim=config.hidden_dim,
        num_classes=dataset.num_classes,
        dropout=config.dropout,
        propagation_steps=config.propagation_steps,
        teleport_probability=config.teleport_probability,
    ).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    print(f"x shape: {tuple(data.x.shape)}")
    print(f"edge_index shape: {tuple(data.edge_index.shape)}")
    print(f"y shape: {tuple(data.y.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_path = PROJECT_ROOT / config.model_path
    output_dir = PROJECT_ROOT / config.output_dir
    history_path = output_dir / config.history_path
    curve_path = output_dir / config.curve_path

    history = train(
        model=model,
        data=data,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=config.n_epochs,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    load_checkpoint(model, checkpoint_path, device)
    test_accuracy = evaluate(model, data, data.test_mask)
    history["test_accuracies"].append(test_accuracy)
    save_json(history, history_path)
    plot_training_curves(history, curve_path)

    print(f"Best model saved to {checkpoint_path}")
    print(f"Training history saved to {history_path}")
    print(f"Training curves saved to {curve_path}")
    print(f"Final test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()

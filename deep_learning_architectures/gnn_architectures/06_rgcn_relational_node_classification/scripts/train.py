import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_relational_node_data, print_dataset_statistics
from src.engine import evaluate, train
from src.model import RGCNNodeClassifier
from src.utils import clear_memory, get_device, load_checkpoint, save_json, set_seed
from src.visualize import plot_training_curves


def main() -> None:
    config = TrainConfig()
    clear_memory()
    set_seed(config.seed)
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
    train_idx = train_idx.to(device)
    validation_idx = validation_idx.to(device)
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
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_type)
    print(f"Uses node embeddings: {use_node_embeddings}")
    if x is not None:
        print(f"x shape: {tuple(x.shape)}")
    print(f"edge_index shape: {tuple(edge_index.shape)}")
    print(f"edge_type shape: {tuple(edge_type.shape)}")
    print(f"y shape: {tuple(y.shape)}")
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
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=y,
        train_idx=train_idx,
        validation_idx=validation_idx,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=config.n_epochs,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    load_checkpoint(model, checkpoint_path, device)
    test_accuracy = evaluate(model, x, edge_index, edge_type, y, test_idx)
    history["test_accuracies"].append(test_accuracy)
    save_json(history, history_path)
    plot_training_curves(history, curve_path)

    print(f"Best model saved to {checkpoint_path}")
    print(f"Training history saved to {history_path}")
    print(f"Training curves saved to {curve_path}")
    print(f"Final test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TrainConfig
from src.data import build_link_prediction_splits, print_dataset_statistics
from src.engine import evaluate, train
from src.model import GraphAutoEncoderLinkPredictor
from src.utils import clear_memory, get_device, load_checkpoint, save_json, set_seed
from src.visualize import plot_training_curves


def main() -> None:
    config = TrainConfig()
    clear_memory()
    set_seed(config.seed)
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
    train_data = train_data.to(device)
    validation_data = validation_data.to(device)
    test_data = test_data.to(device)

    model = GraphAutoEncoderLinkPredictor(
        input_dim=dataset.num_node_features,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
        model_type=config.model_type,
    ).to(device)
    sample_edge_label_index = train_data.pos_edge_label_index[:, :8]
    model.eval()
    with torch.no_grad():
        encoded = model.encode(train_data.x, train_data.edge_index)
        z = encoded[0] if config.model_type == "VGAE" else encoded
        edge_logits = model.decode(z, sample_edge_label_index)
    print(f"x shape: {tuple(train_data.x.shape)}")
    print(f"edge_index shape: {tuple(train_data.edge_index.shape)}")
    print(f"z shape: {tuple(z.shape)}")
    print(f"edge_label_index shape: {tuple(sample_edge_label_index.shape)}")
    print(f"edge_logits shape: {tuple(edge_logits.shape)}")

    loss_fn = nn.BCEWithLogitsLoss()
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
        train_data=train_data,
        validation_data=validation_data,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=config.n_epochs,
        kl_weight=config.kl_weight,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    load_checkpoint(model, checkpoint_path, device)
    test_loss, test_auc, test_ap = evaluate(model, test_data, loss_fn)
    history["test_aucs"].append(test_auc)
    history["test_average_precisions"].append(test_ap)
    save_json(history, history_path)
    plot_training_curves(history, curve_path)

    print(f"Best model saved to {checkpoint_path}")
    print(f"Training history saved to {history_path}")
    print(f"Training curves saved to {curve_path}")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test AUC: {test_auc:.4f}")
    print(f"Final test AP: {test_ap:.4f}")


if __name__ == "__main__":
    main()

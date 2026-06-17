from pathlib import Path

import torch
from torch import nn
from tqdm import trange

from src.utils import save_checkpoint, save_json


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    x: torch.Tensor | None,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad()

    logits = model(x, edge_index, edge_type)
    loss = loss_fn(logits[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()

    train_accuracy = accuracy(logits[train_idx], y[train_idx])
    return loss.item(), train_accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x: torch.Tensor | None,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    y: torch.Tensor,
    node_idx: torch.Tensor,
) -> float:
    model.eval()
    logits = model(x, edge_index, edge_type)
    return accuracy(logits[node_idx], y[node_idx])


def train(
    model: nn.Module,
    x: torch.Tensor | None,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    validation_idx: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    n_epochs: int,
    checkpoint_path: Path,
    history_path: Path,
) -> dict[str, list[float]]:
    history = {
        "train_losses": [],
        "train_accuracies": [],
        "validation_accuracies": [],
        "test_accuracies": [],
    }
    best_validation_accuracy = 0.0

    progress_bar = trange(n_epochs, desc="Training", leave=True)
    for epoch in progress_bar:
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            y=y,
            train_idx=train_idx,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )
        validation_accuracy = evaluate(
            model=model,
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            y=y,
            node_idx=validation_idx,
        )

        history["train_losses"].append(train_loss)
        history["train_accuracies"].append(train_accuracy)
        history["validation_accuracies"].append(validation_accuracy)
        save_json(history, history_path)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            save_checkpoint(model, checkpoint_path)
            checkpoint_status = "saved best checkpoint"
        else:
            checkpoint_status = "checkpoint unchanged"

        progress_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_accuracy:.4f}",
            val_acc=f"{validation_accuracy:.4f}",
        )
        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {train_loss:.4f}, "
            f"train accuracy: {train_accuracy:.4f}, "
            f"validation accuracy: {validation_accuracy:.4f}, "
            f"{checkpoint_status}"
        )

    return history

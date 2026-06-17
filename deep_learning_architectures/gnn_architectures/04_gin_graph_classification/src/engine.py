from pathlib import Path

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import trange

from src.utils import save_checkpoint, save_json


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> int:
    predictions = logits.argmax(dim=1)
    return int((predictions == labels).sum().item())


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_graphs = 0

    for batch in data_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(logits, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_correct += accuracy(logits, batch.y)
        total_graphs += batch.num_graphs

    return total_loss / total_graphs, total_correct / total_graphs


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_correct = 0
    total_graphs = 0

    for batch in data_loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        total_correct += accuracy(logits, batch.y)
        total_graphs += batch.num_graphs

    return total_correct / total_graphs


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
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
            data_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        validation_accuracy = evaluate(model, valid_loader, device)

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

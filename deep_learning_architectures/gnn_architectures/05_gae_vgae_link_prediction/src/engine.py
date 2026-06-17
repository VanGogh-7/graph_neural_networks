from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from tqdm import trange

from src.utils import save_checkpoint, save_json


def binary_auc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    scores = logits.detach().cpu().numpy()
    targets = labels.detach().cpu().numpy()
    positive_count = int((targets == 1).sum())
    negative_count = int((targets == 0).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    positive_rank_sum = ranks[targets == 1].sum()
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def average_precision(logits: torch.Tensor, labels: torch.Tensor) -> float:
    scores = logits.detach().cpu().numpy()
    targets = labels.detach().cpu().numpy()
    positive_count = int((targets == 1).sum())
    if positive_count == 0:
        return float("nan")

    order = np.argsort(-scores)
    sorted_targets = targets[order]
    true_positives = np.cumsum(sorted_targets == 1)
    ranks = np.arange(1, len(sorted_targets) + 1)
    precision_at_k = true_positives / ranks
    return float(precision_at_k[sorted_targets == 1].sum() / positive_count)


def make_edge_labels(
    positive_edge_index: torch.Tensor,
    negative_edge_index: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_label_index = torch.cat([positive_edge_index, negative_edge_index], dim=1)
    edge_labels = torch.cat(
        [
            torch.ones(positive_edge_index.size(1), device=device),
            torch.zeros(negative_edge_index.size(1), device=device),
        ]
    )
    return edge_label_index, edge_labels


def reconstruction_loss(
    model: nn.Module,
    z: torch.Tensor,
    positive_edge_index: torch.Tensor,
    negative_edge_index: torch.Tensor,
    loss_fn: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_label_index, edge_labels = make_edge_labels(
        positive_edge_index=positive_edge_index,
        negative_edge_index=negative_edge_index,
        device=z.device,
    )
    logits = model.decode(z, edge_label_index)
    loss = loss_fn(logits, edge_labels)
    return loss, logits, edge_labels


def vgae_kl_loss(mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(
        torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd), dim=1)
    )


def train_one_epoch(
    model: nn.Module,
    train_data: Data,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    kl_weight: float,
) -> float:
    model.train()
    optimizer.zero_grad()

    negative_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.pos_edge_label_index.size(1),
        method="sparse",
    )

    if model.model_type == "VGAE":
        z, mu, logstd = model.encode(train_data.x, train_data.edge_index)
        loss, _, _ = reconstruction_loss(
            model=model,
            z=z,
            positive_edge_index=train_data.pos_edge_label_index,
            negative_edge_index=negative_edge_index,
            loss_fn=loss_fn,
        )
        loss = loss + kl_weight * vgae_kl_loss(mu, logstd)
    else:
        z = model.encode(train_data.x, train_data.edge_index)
        loss, _, _ = reconstruction_loss(
            model=model,
            z=z,
            positive_edge_index=train_data.pos_edge_label_index,
            negative_edge_index=negative_edge_index,
            loss_fn=loss_fn,
        )

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: Data,
    loss_fn: nn.Module,
) -> tuple[float, float, float]:
    model.eval()
    if model.model_type == "VGAE":
        z, _, _ = model.encode(data.x, data.edge_index)
    else:
        z = model.encode(data.x, data.edge_index)

    loss, logits, labels = reconstruction_loss(
        model=model,
        z=z,
        positive_edge_index=data.pos_edge_label_index,
        negative_edge_index=data.neg_edge_label_index,
        loss_fn=loss_fn,
    )
    return loss.item(), binary_auc(logits, labels), average_precision(logits, labels)


def train(
    model: nn.Module,
    train_data: Data,
    validation_data: Data,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    n_epochs: int,
    kl_weight: float,
    checkpoint_path: Path,
    history_path: Path,
) -> dict[str, list[float]]:
    history = {
        "train_losses": [],
        "validation_losses": [],
        "validation_aucs": [],
        "validation_average_precisions": [],
        "test_aucs": [],
        "test_average_precisions": [],
    }
    best_validation_auc = 0.0

    progress_bar = trange(n_epochs, desc="Training", leave=True)
    for epoch in progress_bar:
        train_loss = train_one_epoch(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            loss_fn=loss_fn,
            kl_weight=kl_weight,
        )
        validation_loss, validation_auc, validation_ap = evaluate(
            model=model,
            data=validation_data,
            loss_fn=loss_fn,
        )

        history["train_losses"].append(train_loss)
        history["validation_losses"].append(validation_loss)
        history["validation_aucs"].append(validation_auc)
        history["validation_average_precisions"].append(validation_ap)
        save_json(history, history_path)

        if validation_auc > best_validation_auc:
            best_validation_auc = validation_auc
            save_checkpoint(model, checkpoint_path)
            checkpoint_status = "saved best checkpoint"
        else:
            checkpoint_status = "checkpoint unchanged"

        progress_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{validation_loss:.4f}",
            val_auc=f"{validation_auc:.4f}",
            val_ap=f"{validation_ap:.4f}",
        )
        print(
            f"Epoch {epoch + 1}/{n_epochs}, "
            f"train loss: {train_loss:.4f}, "
            f"validation loss: {validation_loss:.4f}, "
            f"validation AUC: {validation_auc:.4f}, "
            f"validation AP: {validation_ap:.4f}, "
            f"{checkpoint_status}"
        )

    return history

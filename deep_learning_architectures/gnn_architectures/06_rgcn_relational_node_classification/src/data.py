from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Entities


def load_entities_dataset(data_root: Path, dataset_name: str = "AIFB") -> tuple[Entities, Data]:
    dataset = Entities(root=str(data_root), name=dataset_name)
    data = dataset[0]
    return dataset, data


def build_label_vector(data: Data) -> torch.Tensor:
    y = torch.full((data.num_nodes,), -1, dtype=torch.long)
    y[data.train_idx] = data.train_y
    y[data.test_idx] = data.test_y
    return y


def split_train_validation(
    train_idx: torch.Tensor,
    valid_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(train_idx.numel(), generator=generator)
    valid_size = max(1, int(train_idx.numel() * valid_ratio))
    valid_idx = train_idx[permutation[:valid_size]]
    new_train_idx = train_idx[permutation[valid_size:]]
    return new_train_idx, valid_idx


def build_relational_node_data(
    data_root: Path,
    dataset_name: str,
    valid_ratio: float,
    seed: int,
) -> tuple[Entities, Data, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset, data = load_entities_dataset(data_root, dataset_name)
    y = build_label_vector(data)
    train_idx, validation_idx = split_train_validation(
        train_idx=data.train_idx,
        valid_ratio=valid_ratio,
        seed=seed,
    )
    return dataset, data, y, train_idx, validation_idx, data.test_idx


def dataset_statistics(
    dataset: Entities,
    data: Data,
    train_idx: torch.Tensor,
    validation_idx: torch.Tensor,
    test_idx: torch.Tensor,
) -> dict[str, int]:
    return {
        "num_nodes": data.num_nodes,
        "num_edges": data.edge_index.size(1),
        "num_relations": dataset.num_relations,
        "num_classes": dataset.num_classes,
        "train_nodes": int(train_idx.numel()),
        "validation_nodes": int(validation_idx.numel()),
        "test_nodes": int(test_idx.numel()),
    }


def print_dataset_statistics(
    dataset_name: str,
    dataset: Entities,
    data: Data,
    train_idx: torch.Tensor,
    validation_idx: torch.Tensor,
    test_idx: torch.Tensor,
) -> None:
    stats = dataset_statistics(dataset, data, train_idx, validation_idx, test_idx)
    print(f"Dataset: {dataset_name}")
    print(f"Number of nodes: {stats['num_nodes']}")
    print(f"Number of edges: {stats['num_edges']}")
    print(f"Number of relations: {stats['num_relations']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Train nodes: {stats['train_nodes']}")
    print(f"Validation nodes: {stats['validation_nodes']}")
    print(f"Test nodes: {stats['test_nodes']}")

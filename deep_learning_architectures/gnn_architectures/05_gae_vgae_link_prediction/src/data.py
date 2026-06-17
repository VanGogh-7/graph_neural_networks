from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit


def load_cora_dataset(data_root: Path) -> tuple[Planetoid, Data]:
    dataset = Planetoid(root=str(data_root), name="Cora")
    data = dataset[0]
    return dataset, data


def build_link_prediction_splits(
    data_root: Path,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[Planetoid, Data, Data, Data, Data]:
    dataset, data = load_cora_dataset(data_root)
    torch.manual_seed(seed)
    transform = RandomLinkSplit(
        num_val=validation_ratio,
        num_test=test_ratio,
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=False,
    )
    train_data, validation_data, test_data = transform(data)
    return dataset, data, train_data, validation_data, test_data


def positive_edge_count(data: Data) -> int:
    return int(data.pos_edge_label_index.size(1))


def dataset_statistics(
    dataset: Planetoid,
    original_data: Data,
    train_data: Data,
    validation_data: Data,
    test_data: Data,
) -> dict[str, int]:
    return {
        "num_nodes": original_data.num_nodes,
        "original_edges": original_data.num_edges,
        "num_node_features": dataset.num_node_features,
        "train_positive_edges": positive_edge_count(train_data),
        "validation_positive_edges": positive_edge_count(validation_data),
        "test_positive_edges": positive_edge_count(test_data),
    }


def print_dataset_statistics(
    dataset: Planetoid,
    original_data: Data,
    train_data: Data,
    validation_data: Data,
    test_data: Data,
) -> None:
    stats = dataset_statistics(dataset, original_data, train_data, validation_data, test_data)
    print("Dataset: Cora")
    print(f"Number of nodes: {stats['num_nodes']}")
    print(f"Original edges: {stats['original_edges']}")
    print(f"Number of node features: {stats['num_node_features']}")
    print(f"Train positive edges: {stats['train_positive_edges']}")
    print(f"Validation positive edges: {stats['validation_positive_edges']}")
    print(f"Test positive edges: {stats['test_positive_edges']}")

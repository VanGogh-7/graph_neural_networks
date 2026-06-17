from pathlib import Path

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


def load_cora_dataset(data_root: Path) -> tuple[Planetoid, Data]:
    dataset = Planetoid(root=str(data_root), name="Cora")
    data = dataset[0]
    return dataset, data


def dataset_statistics(dataset: Planetoid, data: Data) -> dict[str, int]:
    return {
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_node_features": dataset.num_node_features,
        "num_classes": dataset.num_classes,
        "train_nodes": int(data.train_mask.sum().item()),
        "validation_nodes": int(data.val_mask.sum().item()),
        "test_nodes": int(data.test_mask.sum().item()),
    }


def print_dataset_statistics(dataset: Planetoid, data: Data) -> None:
    stats = dataset_statistics(dataset, data)
    print("Dataset: Cora")
    print(f"Number of nodes: {stats['num_nodes']}")
    print(f"Number of edges: {stats['num_edges']}")
    print(f"Number of node features: {stats['num_node_features']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Train nodes: {stats['train_nodes']}")
    print(f"Validation nodes: {stats['validation_nodes']}")
    print(f"Test nodes: {stats['test_nodes']}")

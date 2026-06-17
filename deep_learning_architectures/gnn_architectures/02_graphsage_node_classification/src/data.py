from pathlib import Path

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Reddit


PLANETOID_DATASETS = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}


def dataset_root(data_root: Path, dataset_name: str) -> Path:
    return data_root / dataset_name.lower()


def load_node_dataset(data_root: Path, dataset_name: str) -> tuple[object, Data]:
    normalized_name = dataset_name.lower()
    root = dataset_root(data_root, dataset_name)

    if normalized_name in PLANETOID_DATASETS:
        dataset = Planetoid(root=str(root), name=PLANETOID_DATASETS[normalized_name])
    elif normalized_name == "reddit":
        dataset = Reddit(root=str(root))
    else:
        supported_names = ", ".join([*PLANETOID_DATASETS.values(), "Reddit"])
        raise ValueError(f"Unsupported dataset_name '{dataset_name}'. Use one of: {supported_names}.")

    data = dataset[0]
    return dataset, data


def dataset_statistics(dataset: object, data: Data) -> dict[str, int]:
    return {
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_node_features": dataset.num_node_features,
        "num_classes": dataset.num_classes,
        "train_nodes": int(data.train_mask.sum().item()),
        "validation_nodes": int(data.val_mask.sum().item()),
        "test_nodes": int(data.test_mask.sum().item()),
    }


def print_dataset_statistics(dataset_name: str, dataset: object, data: Data) -> None:
    stats = dataset_statistics(dataset, data)
    print(f"Dataset: {dataset_name}")
    print(f"Number of nodes: {stats['num_nodes']}")
    print(f"Number of edges: {stats['num_edges']}")
    print(f"Number of node features: {stats['num_node_features']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Train nodes: {stats['train_nodes']}")
    print(f"Validation nodes: {stats['validation_nodes']}")
    print(f"Test nodes: {stats['test_nodes']}")

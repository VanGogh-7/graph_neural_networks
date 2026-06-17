from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


def load_mutag_dataset(data_root: Path, dataset_name: str = "MUTAG") -> TUDataset:
    dataset = TUDataset(root=str(data_root), name=dataset_name)
    if dataset.num_node_features == 0 or dataset[0].x is None:
        raise ValueError(
            f"{dataset_name} does not provide node features in this configuration. "
            "This educational GIN example expects graph data objects with x."
        )
    return dataset


def split_dataset(
    dataset: TUDataset,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> tuple[list[Data], list[Data], list[Data]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()

    train_size = int(len(dataset) * train_ratio)
    valid_size = int(len(dataset) * valid_ratio)
    train_indices = indices[:train_size]
    valid_indices = indices[train_size : train_size + valid_size]
    test_indices = indices[train_size + valid_size :]

    def collect_graphs(split_indices: list[int]) -> list[Data]:
        graphs = []
        for graph_index in split_indices:
            graph = dataset[graph_index].clone()
            graph.graph_id = torch.tensor([graph_index], dtype=torch.long)
            graphs.append(graph)
        return graphs

    return (
        collect_graphs(train_indices),
        collect_graphs(valid_indices),
        collect_graphs(test_indices),
    )


def build_dataloaders(
    data_root: Path,
    dataset_name: str,
    batch_size: int,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader, TUDataset]:
    dataset = load_mutag_dataset(data_root, dataset_name)
    train_dataset, valid_dataset, test_dataset = split_dataset(
        dataset=dataset,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader, dataset


def dataset_statistics(
    dataset: TUDataset,
    train_graphs: int,
    valid_graphs: int,
    test_graphs: int,
) -> dict[str, float | int]:
    total_nodes = sum(graph.num_nodes for graph in dataset)
    total_edges = sum(graph.num_edges for graph in dataset)
    return {
        "num_graphs": len(dataset),
        "num_classes": dataset.num_classes,
        "num_node_features": dataset.num_node_features,
        "average_nodes_per_graph": total_nodes / len(dataset),
        "average_edges_per_graph": total_edges / len(dataset),
        "train_graphs": train_graphs,
        "validation_graphs": valid_graphs,
        "test_graphs": test_graphs,
    }


def print_dataset_statistics(
    dataset_name: str,
    dataset: TUDataset,
    train_graphs: int,
    valid_graphs: int,
    test_graphs: int,
) -> None:
    stats = dataset_statistics(dataset, train_graphs, valid_graphs, test_graphs)
    print(f"Dataset: {dataset_name}")
    print(f"Number of graphs: {stats['num_graphs']}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Number of node features: {stats['num_node_features']}")
    print(f"Average nodes per graph: {stats['average_nodes_per_graph']:.2f}")
    print(f"Average edges per graph: {stats['average_edges_per_graph']:.2f}")
    print(f"Train graphs: {stats['train_graphs']}")
    print(f"Validation graphs: {stats['validation_graphs']}")
    print(f"Test graphs: {stats['test_graphs']}")

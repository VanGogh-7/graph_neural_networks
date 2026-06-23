from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.nn import Node2Vec


# Resolve paths from this file so the script also works when run from ch2/.
CHAPTER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CHAPTER_DIR.parent
DATA_PATH = PROJECT_ROOT / "datasets" / "polbooks" / "polbooks.gml"
OUTPUT_DIR = CHAPTER_DIR / "outputs"
EMBEDDINGS_PATH = OUTPUT_DIR / "polbooks_node2vec_embeddings.pt"
GRAPH_PLOT_PATH = OUTPUT_DIR / "polbooks_graph.png"
EMBEDDING_PLOT_PATH = OUTPUT_DIR / "polbooks_node2vec_projection.png"


def load_polbooks_graph() -> nx.Graph:
    """Read the Political Books graph from the project dataset folder."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    graph = nx.read_gml(DATA_PATH)
    print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    # The Political Books dataset stores the political leaning in the node
    # attribute named "value": l = liberal, c = conservative, n = neutral.
    value_counts = Counter(attributes["value"] for _, attributes in graph.nodes(data=True))
    print(f"Node label counts: {dict(value_counts)}")
    return graph


def graph_to_edge_index(graph: nx.Graph) -> tuple[torch.Tensor, dict[str, int], list[str]]:
    """Convert a NetworkX graph with arbitrary node labels into PyG edge_index."""
    nodes = list(graph.nodes())

    # PyG Node2Vec expects integer node IDs from 0 to num_nodes - 1.
    node_to_id = {node: index for index, node in enumerate(nodes)}
    id_to_node = nodes

    edges = []
    for source, target in graph.edges():
        source_id = node_to_id[source]
        target_id = node_to_id[target]
        edges.append((source_id, target_id))

        # The Political Books graph is undirected, so add the reverse direction
        # to make the random walks symmetric in PyG.
        if not graph.is_directed() and source_id != target_id:
            edges.append((target_id, source_id))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index, node_to_id, id_to_node


def train_node2vec(edge_index: torch.Tensor, num_nodes: int) -> Node2Vec:
    """Train a small PyG Node2Vec model for educational use."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = edge_index.to(device)

    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=64,
        walk_length=30,
        context_size=10,
        walks_per_node=20,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        num_nodes=num_nodes,
        sparse=False,
    ).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 101):
        model.train()
        total_loss = 0.0

        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, loss: {total_loss / len(loader):.4f}")

    return model


def save_embeddings(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    node_to_id: dict[str, int],
    id_to_node: list[str],
) -> None:
    """Save embeddings with the mapping back to the original book labels."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "edge_index": edge_index,
            "node_to_id": node_to_id,
            "id_to_node": id_to_node,
        },
        EMBEDDINGS_PATH,
    )
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")


def node_colors(graph: nx.Graph, id_to_node: list[str]) -> list[str]:
    """Return one plot color per node, ordered by PyG integer node ID."""
    color_map = {"l": "blue", "c": "red", "n": "gray"}
    return [color_map.get(graph.nodes[node].get("value"), "gray") for node in id_to_node]


def plot_graph(graph: nx.Graph) -> None:
    """Plot the original graph colored by political leaning."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(15, 15))
    pos = nx.kamada_kawai_layout(graph)
    colors = node_colors(graph, list(graph.nodes()))

    nx.draw_networkx_nodes(graph, pos, node_color=colors, alpha=0.7)
    nx.draw_networkx_edges(graph, pos, alpha=0.1)
    nx.draw_networkx_labels(graph, pos, font_size=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(GRAPH_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved graph plot to {GRAPH_PLOT_PATH}")


def plot_embedding_projection(graph: nx.Graph, embeddings: torch.Tensor, id_to_node: list[str]) -> None:
    """Reduce Node2Vec embeddings to two dimensions and plot them."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    try:
        from sklearn.manifold import TSNE

        tsne_model = TSNE(
            n_components=2,
            learning_rate="auto",
            init="random",
            perplexity=30,
            random_state=42,
        )
        features_2d = tsne_model.fit_transform(embeddings.numpy())
        title = "t-SNE Visualization of PyG Node2Vec Embeddings"
    except ImportError:
        # Keep the script runnable without scikit-learn by using a small PCA
        # projection written with torch operations.
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        features_2d = (centered @ vh[:2].T).numpy()
        title = "PCA Visualization of PyG Node2Vec Embeddings"

    plt.figure(figsize=(12, 8))
    plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        color=node_colors(graph, id_to_node),
        alpha=0.7,
    )
    plt.xlabel("Embedding Feature 0")
    plt.ylabel("Embedding Feature 1")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(EMBEDDING_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved embedding plot to {EMBEDDING_PLOT_PATH}")


def main() -> None:
    graph = load_polbooks_graph()
    plot_graph(graph)

    edge_index, node_to_id, id_to_node = graph_to_edge_index(graph)
    print(f"Converted graph to edge_index with shape {tuple(edge_index.shape)}")

    model = train_node2vec(edge_index, num_nodes=len(id_to_node))

    # Each row is the learned embedding for the node with that integer ID.
    embeddings = model.embedding.weight.detach().cpu()
    save_embeddings(embeddings, edge_index, node_to_id, id_to_node)
    plot_embedding_projection(graph, embeddings, id_to_node)


if __name__ == "__main__":
    main()

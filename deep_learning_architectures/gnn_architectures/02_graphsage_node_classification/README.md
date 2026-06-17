# GraphSAGE Node Classification

This project implements a two-layer GraphSAGE model for node classification
using PyTorch and PyTorch Geometric.

The default dataset is Cora so the first version stays fast, full-batch, and
easy to run on a typical machine. The code includes a `dataset_name` option in
`src/config.py` and can be switched to `Reddit` later, but Reddit is much
larger and is usually trained with neighbor sampling.

## GraphSAGE

GraphSAGE is a graph neural network architecture that learns how to aggregate
features from a node's local neighborhood. It computes a new representation for
each node by combining the node's own feature vector with a learned summary of
neighbor features.

GraphSAGE is historically important because it helped popularize inductive
learning on graphs. Instead of learning one embedding table tied to a fixed
graph, GraphSAGE learns aggregation functions that can be applied to nodes that
were not seen during training, as long as their features and neighborhoods are
available.

## GCN vs GraphSAGE

GCN uses a normalized graph convolution over the graph structure. Its classic
semi-supervised setup is transductive: the full graph is available during
training, and the model learns using labeled nodes while propagating information
through all nodes.

GraphSAGE also aggregates neighborhoods, but it explicitly frames the operation
as a learnable sampling-and-aggregation process. This makes it especially
natural for inductive settings and for large graphs where training on the full
graph every step is too expensive.

## Neighbor Aggregation

The core idea is neighbor aggregation. For each node, the model gathers
representations from neighboring nodes, summarizes them, and combines that
summary with the node's own representation. Stacking two GraphSAGE layers lets a
node use information from two-hop neighborhoods.

## Mean Aggregator

This project uses the mean aggregator by default. The mean aggregator averages
neighbor features before applying learnable transformations. In PyTorch
Geometric, this is implemented with:

```python
SAGEConv(input_dim, hidden_dim, aggr="mean")
```

The model in this project is:

```text
SAGEConv(input_dim, hidden_dim)
ReLU
Dropout
SAGEConv(hidden_dim, num_classes)
```

The model returns raw logits. Softmax is not applied inside `forward` because
`CrossEntropyLoss` expects unnormalized class scores.

## Neighbor Sampling

Large-scale GraphSAGE is often trained with neighbor sampling. Instead of
loading the entire graph into every training step, the training loop samples a
fixed number of neighbors for each batch of target nodes. This makes datasets
like Reddit more practical.

This educational first version intentionally starts with a simple full-batch
implementation. Full-batch training is appropriate for the default Cora setup
and keeps the code easier to read before moving on to sampled mini-batches.

## Node Classification Task

The default Cora task is a citation-network node classification problem. Nodes
represent scientific papers, edges represent citation links, node features are
bag-of-words vectors, and labels are paper topic classes. The model trains on
the provided training mask, selects the best checkpoint using validation
accuracy, and evaluates the test mask only after training finishes.

Datasets are stored under the repository-level directory:

```text
datasets/
```

To switch datasets, edit:

```python
dataset_name = "Cora"
```

in `src/config.py`. Supported names are `Cora`, `CiteSeer`, `PubMed`, and
`Reddit`.

## Tensor Shapes

- `x`: `[num_nodes, num_node_features]`
- `edge_index`: `[2, num_edges]`
- `y`: `[num_nodes]`
- `logits`: `[num_nodes, num_classes]`

For the standard Cora dataset, the expected statistics are:

- Nodes: `2708`
- Edges: `10556`
- Node features: `1433`
- Classes: `7`
- Train nodes: `140`
- Validation nodes: `500`
- Test nodes: `1000`

## Project Structure

```text
02_graphsage_node_classification/
├── scripts/
│   ├── train.py
│   └── predict.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── engine.py
│   ├── model.py
│   ├── utils.py
│   └── visualize.py
├── README.md
└── requirements.txt
```

## Train

From this project directory:

```bash
python scripts/train.py
```

## Predict

After training:

```bash
python scripts/predict.py
```

The prediction script loads the saved checkpoint and prints several test node
predictions with node id, predicted class, true class, and correctness.

## Key Paper

Inductive Representation Learning on Large Graphs  
https://arxiv.org/abs/1706.02216

This project is an educational PyTorch Geometric implementation, not a full
production graph learning system.

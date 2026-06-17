# GCN Node Classification

This project implements a two-layer Graph Convolutional Network (GCN) for
semi-supervised node classification on the real public Cora citation network
using PyTorch and PyTorch Geometric.

## Graph Convolutional Networks

A Graph Convolutional Network is a neural network for graph-structured data.
Instead of treating each example as independent, a GCN updates each node
representation by combining the node's own features with information from its
neighbors.

GCN is historically important because it made graph neural networks simple,
scalable, and practical for semi-supervised learning on citation networks and
other relational datasets. The Kipf and Welling GCN showed that a compact
first-order graph convolution could produce strong results while being easy to
train end to end.

## Neighborhood Aggregation

The core idea is neighborhood aggregation. For each node, the model gathers
features from connected nodes, mixes those features with learnable weights, and
passes the result through nonlinear layers. After one GCN layer, a node has
information from its immediate neighbors. After two layers, it can use
information from two-hop neighborhoods.

## Normalized Graph Convolution

Raw neighbor summation can make high-degree nodes dominate the computation.
GCN uses a normalized graph convolution so messages are scaled by node degree.
PyTorch Geometric's `GCNConv` implements this normalized propagation, including
self-loops by default, so every node keeps part of its own feature signal while
also receiving neighbor information.

The model in this project is:

```text
GCNConv(input_dim, hidden_dim)
ReLU
Dropout
GCNConv(hidden_dim, num_classes)
```

The model returns raw logits. Softmax is not applied inside `forward` because
`CrossEntropyLoss` expects unnormalized class scores.

## Cora Node Classification

Cora is a citation network where nodes are scientific papers and edges are
citation links. Each node has a bag-of-words feature vector, and the task is to
predict each paper's topic class. The Planetoid split provides training,
validation, and test masks, so the model trains only on labeled training nodes
while still passing messages over the full graph.

This project uses:

```python
Planetoid(root=..., name="Cora")
```

The dataset is stored under the repository-level directory:

```text
datasets/cora/
```

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
01_gcn_node_classification/
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

The best checkpoint is selected by validation accuracy. The test set is
evaluated only after training finishes.

## Predict

After training:

```bash
python scripts/predict.py
```

The prediction script loads the saved checkpoint and prints several test node
predictions with node id, predicted class, true class, and correctness.

## Key Paper

Semi-Supervised Classification with Graph Convolutional Networks  
https://arxiv.org/abs/1609.02907

This project is an educational PyTorch Geometric implementation, not a full
production graph learning system.

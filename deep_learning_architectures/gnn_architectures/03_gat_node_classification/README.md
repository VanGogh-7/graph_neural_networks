# GAT Node Classification

This project implements a two-layer Graph Attention Network (GAT) for
semi-supervised node classification on the real public Cora citation network
using PyTorch and PyTorch Geometric.

## Graph Attention Networks

A Graph Attention Network is a graph neural network that learns how strongly a
node should attend to each of its neighbors. Instead of treating all neighbors
with a fixed normalization rule, GAT computes attention weights over local
neighborhoods and uses those weights to combine neighbor information.

GAT is historically important because it brought attention mechanisms into
graph neural networks in a clean, practical architecture. It showed that graph
models could learn different neighbor importances directly from node features
and graph structure without requiring expensive spectral graph operations.

## GCN vs GAT

GCN aggregates neighbors with a normalized graph convolution. The normalization
is determined by graph structure, usually based on node degrees, and every edge
contributes according to that fixed rule.

GAT learns attention scores for edges. This lets the model decide that some
neighbors are more useful than others for a particular node classification
decision. The graph still defines which nodes can exchange information, but the
model learns the relative importance of those messages.

## Attention for Neighbor Aggregation

Attention is useful for neighbor aggregation because graph neighborhoods can be
noisy. A cited paper, social contact, or connected item may not always be
equally informative. GAT learns attention coefficients so the representation of
each node can focus more on the most relevant neighbors.

## Multi-Head Attention

GAT often uses multi-head attention. Multiple attention heads learn different
ways to weight the same neighborhood. In the first layer, this project
concatenates the outputs of several heads to build a richer hidden
representation. In the final layer, it uses one head with `concat=False` to
return class logits.

The model in this project is:

```text
GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
ELU
Dropout
GATConv(hidden_dim * num_heads, num_classes, heads=1, concat=False, dropout=dropout)
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
03_gat_node_classification/
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

Graph Attention Networks  
https://arxiv.org/abs/1710.10903

This project is an educational PyTorch Geometric implementation, not a full
production graph learning system.

# R-GCN Relational Node Classification

This project implements a Relational Graph Convolutional Network (R-GCN) for
relational node classification on the real public AIFB dataset using PyTorch
and PyTorch Geometric.

## Relational Graph Convolutional Networks

A Relational Graph Convolutional Network is a graph neural network for graphs
with typed edges. Instead of treating every edge as the same kind of connection,
R-GCN uses the relation type of each edge to choose relation-specific message
passing parameters.

R-GCN is historically important because it brought graph convolutional learning
to knowledge graphs and relational data. It showed how neural message passing
can handle many relation types while still training end to end for tasks such
as entity classification and link prediction.

## Homogeneous vs Relational Graphs

Ordinary homogeneous graph models such as GCN usually assume one edge type.
Every edge means the same kind of connection, so the same transformation can be
used for every neighbor message.

Relational graphs contain different edge types. In a knowledge graph, one edge
may mean "works at", another may mean "published by", and another may mean
"located in". These relations should not all share the same transformation.

## Edge Types

`edge_type` is a vector with one relation id per edge:

```text
edge_index: [2, num_edges]
edge_type: [num_edges]
```

For each edge in `edge_index`, the matching value in `edge_type` tells R-GCN
which relation-specific parameters to use.

## Why Relation-Specific Parameters Matter

Different relations carry different meanings. A message from a node connected
by one relation type may need a different transformation than a message through
another relation type. R-GCN handles this by learning separate transformations
for relation types inside `RGCNConv`.

## AIFB Task

AIFB is a relational entity classification dataset. PyTorch Geometric loads it
with:

```python
Entities(root=..., name="AIFB")
```

The dataset has no input node feature matrix, so this project safely uses
learnable node embeddings as the initial node representations. The provided
training nodes are split deterministically into train and validation nodes
using `config.seed` and `config.valid_ratio`. The original test split is used
only after training finishes.

Datasets are stored under the repository-level directory:

```text
datasets/aifb/
```

## Tensor Shapes

- `edge_index`: `[2, num_edges]`
- `edge_type`: `[num_edges]`
- `y`: `[num_nodes]`
- `logits`: `[num_nodes, num_classes]`

For the standard AIFB dataset, the expected statistics are:

- Nodes: `8285`
- Edges: `58086`
- Node features: `0`
- Classes: `4`

## Project Structure

```text
06_rgcn_relational_node_classification/
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

The best checkpoint is selected by validation accuracy. The test split is
evaluated only after training finishes.

## Predict

After training:

```bash
python scripts/predict.py
```

The prediction script loads the saved checkpoint and prints several test node
predictions with node id, predicted class, true class, and correctness.

## Key Paper

Modeling Relational Data with Graph Convolutional Networks  
https://arxiv.org/abs/1703.06103

This project is an educational PyTorch Geometric implementation, not a full
production knowledge graph system.

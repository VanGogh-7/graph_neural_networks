# GIN Graph Classification

This project implements a Graph Isomorphism Network (GIN) for graph
classification on the real public MUTAG molecular graph dataset using PyTorch
and PyTorch Geometric.

## Graph Isomorphism Networks

A Graph Isomorphism Network is a graph neural network designed to produce
expressive graph representations. GIN updates node representations with a
learned transformation of the node's own features and an aggregated summary of
neighbor features. After several layers, node embeddings are pooled into a
graph-level embedding for classification.

GIN is historically important because it connected graph neural network
expressiveness to the Weisfeiler-Lehman graph isomorphism test. The paper
showed that common GNN aggregation choices can lose structural information and
that sum aggregation is especially powerful for distinguishing graph
structures.

## Node Classification vs Graph Classification

Node classification predicts one label per node inside a single graph, such as
predicting a paper topic in Cora. Graph classification predicts one label for
each whole graph. In MUTAG, every molecular graph has one class label, so the
model must summarize all nodes and edges into a graph-level representation.

## Graph-Level Representation

GIN first computes node embeddings with stacked GIN layers. A global pooling
operation then converts all node embeddings in each graph into one graph
embedding. This project uses `global_add_pool`, which sums node embeddings per
graph according to the PyG `batch` vector.

## Sum Aggregation

GIN uses sum aggregation because sums can preserve more information about
multisets of neighbor features than mean or max aggregation. This matters when
different neighborhoods have similar averages but different counts or
structures.

## Weisfeiler-Lehman Connection

The Weisfeiler-Lehman graph isomorphism test repeatedly updates node labels
based on each node's current label and the multiset of neighbor labels. GIN
mirrors this idea with neural message passing: it aggregates the multiset of
neighbor embeddings and applies an MLP to produce updated node representations.

## MUTAG Graph Classification

MUTAG is a molecular graph classification dataset. Each graph represents a
molecule, nodes represent atoms, edges represent chemical bonds, and the label
is a graph-level class. This project loads MUTAG with:

```python
TUDataset(root=..., name="MUTAG")
```

Datasets are stored under the repository-level directory:

```text
datasets/mutag/
```

The dataset is split deterministically into train, validation, and test sets
using `config.seed`.

## Tensor Shapes

- `x`: `[num_nodes_in_batch, num_node_features]`
- `edge_index`: `[2, num_edges_in_batch]`
- `batch`: `[num_nodes_in_batch]`
- `y`: `[num_graphs_in_batch]`
- `logits`: `[num_graphs_in_batch, num_classes]`

For the standard MUTAG dataset, the expected statistics are approximately:

- Graphs: `188`
- Classes: `2`
- Node features: `7`
- Average nodes per graph: about `17.93`
- Average edges per graph: about `39.59`

## Project Structure

```text
04_gin_graph_classification/
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

The prediction script loads the saved checkpoint and prints several test graph
predictions with graph index, predicted class, true class, and correctness.

## Key Paper

How Powerful are Graph Neural Networks?  
https://arxiv.org/abs/1810.00826

This project is an educational PyTorch Geometric implementation, not a full
production graph learning system.

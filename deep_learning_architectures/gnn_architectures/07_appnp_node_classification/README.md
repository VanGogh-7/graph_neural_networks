# APPNP Node Classification

This project implements APPNP for semi-supervised node classification on the
real public Cora citation network using PyTorch and PyTorch Geometric.

## APPNP

APPNP stands for Approximate Personalized Propagation of Neural Predictions.
It first uses a small neural network to predict class logits from node
features, then propagates those predictions over the graph with an approximate
Personalized PageRank-style diffusion.

APPNP is historically important because it clearly separated neural prediction
from graph propagation. This made it easier to use expressive feature
predictors while controlling how information spreads across the graph.

## GCN vs APPNP

GCN mixes feature transformation and graph propagation inside each graph
convolution layer. APPNP decouples these two steps. The MLP first learns local
node predictions from features, and the APPNP propagation step then smooths
those predictions through the graph.

## Predict Then Propagate

The core idea is "predict then propagate." The model computes initial logits:

```text
Linear(input_dim, hidden_dim)
ReLU
Dropout
Linear(hidden_dim, num_classes)
```

Then it applies:

```text
APPNP(K=propagation_steps, alpha=teleport_probability)
```

The model returns raw logits. Softmax is not applied inside `forward` because
`CrossEntropyLoss` expects unnormalized class scores.

## Personalized PageRank Propagation

APPNP propagation repeatedly mixes neighbor information with the original
predictions. This is inspired by Personalized PageRank, where a random walk can
return to its starting distribution instead of drifting indefinitely.

## K: Propagation Steps

`K`, exposed as `propagation_steps`, controls how many propagation iterations
are applied. Larger values let information travel farther across the graph, but
can also make node predictions more similar.

## Alpha: Teleport Probability

`alpha`, exposed as `teleport_probability`, controls how much of the original
prediction is retained at each propagation step. A higher alpha keeps the model
closer to the MLP prediction; a lower alpha allows stronger graph smoothing.

## Over-Smoothing

Deep message passing networks can suffer from over-smoothing, where node
representations become too similar after many propagation steps. APPNP helps by
teleporting back to the original neural predictions during propagation, which
preserves node-specific information.

## Cora Node Classification

Cora is a citation network where nodes are scientific papers and edges are
citation links. Each node has a bag-of-words feature vector, and the task is to
predict each paper's topic class. The Planetoid split provides training,
validation, and test masks.

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
07_appnp_node_classification/
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

Predict then Propagate: Graph Neural Networks meet Personalized PageRank  
https://arxiv.org/abs/1810.05997

This project is an educational PyTorch Geometric implementation, not a full
production graph learning system.

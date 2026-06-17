# GCNII Node Classification on Cora

This project implements GCNII for semi-supervised node classification on the real Cora citation graph.

## What GCNII Is

GCNII is a deep graph convolutional network designed to make many-layer GNNs easier to train. It keeps the useful graph smoothing behavior of GCNs, but adds two important ideas:

- an initial residual connection, which repeatedly mixes each layer with the first hidden representation `x_0`,
- an identity mapping, which keeps each deep layer close to an identity transformation when useful.

This project uses `torch_geometric.nn.GCN2Conv` when it is available. A small fallback layer is included in `src/model.py` to show the educational structure of the update if an older PyTorch Geometric version does not provide `GCN2Conv`.

## Why GCNII Is Historically Important

Ordinary GCNs often become difficult to train when stacked deeply. As more graph convolution layers are applied, node embeddings can become too similar. This is called over-smoothing. Deep GCNs can also lose the original node information after repeated neighborhood averaging.

GCNII is historically important because it showed that simple residual and identity mechanisms can make much deeper graph convolutional networks practical.

## Initial Residual Connection

GCNII preserves the first hidden representation `x_0`. Each GCNII layer mixes propagated features with `x_0`. This gives every layer direct access to the transformed input features, even in a deep network.

The `alpha` parameter controls how much of `x_0` is mixed into each layer. A larger `alpha` keeps more of the initial representation.

## Identity Mapping

GCNII also uses an identity mapping term. This helps deep layers avoid drifting too far from useful earlier representations. The `theta` parameter controls the strength of this identity mapping schedule across layers.

In PyTorch Geometric, `GCN2Conv` receives `theta` and the layer index. The effective transformation strength changes by depth.

## GCN, JK-Net, APPNP, and GCNII

- GCN stacks graph convolution layers and usually uses only the final layer representation.
- JK-Net stacks GNN layers and combines representations from several depths.
- APPNP separates neural prediction from graph propagation using Personalized PageRank-style propagation.
- GCNII keeps a deep graph convolution stack, but stabilizes it with initial residual connections and identity mapping.

These models all address how information should move across graph neighborhoods, but they do it with different architectural choices.

## Cora Node Classification Task

Cora is a citation network. Nodes are papers, edges are citation links, node features are bag-of-words paper features, and labels are paper topic classes. The task is to predict the class of each paper using a small labeled training set and the graph structure.

The dataset is loaded with:

```python
Planetoid(root=..., name="Cora")
```

## Input and Output Shapes

Input tensors:

- `x`: `[num_nodes, num_node_features]`
- `edge_index`: `[2, num_edges]`
- `y`: `[num_nodes]`

Output tensor:

- `logits`: `[num_nodes, num_classes]`

The model returns raw logits. It does not apply softmax inside `forward`, because `CrossEntropyLoss` expects raw logits.

## Training

Run:

```bash
python scripts/train.py
```

The training script:

- loads Cora from the repository-level `datasets/` directory,
- prints dataset statistics,
- trains on `train_mask`,
- tracks training loss, training accuracy, validation accuracy, and final test accuracy,
- saves the best checkpoint based on validation accuracy.

## Prediction

Run:

```bash
python scripts/predict.py
```

The prediction script loads the saved checkpoint and prints several test-node predictions:

- node id,
- predicted class,
- true class,
- whether the prediction is correct.

## Key Paper

Simple and Deep Graph Convolutional Networks  
https://arxiv.org/abs/2007.02133

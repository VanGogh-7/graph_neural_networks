# JK-Net Node Classification on Cora

This project implements a Jumping Knowledge Network, usually called JK-Net, for semi-supervised node classification on the real Cora citation graph.

## What JK-Net Is

JK-Net is a graph neural network architecture that combines node representations from multiple GNN layers instead of using only the final layer. Each GCN layer aggregates information from a wider graph neighborhood. A shallow layer mostly sees nearby nodes, while a deeper layer sees nodes farther away.

The main idea is simple:

1. Run several graph convolution layers.
2. Save the hidden representation after each layer.
3. Combine those layer-wise representations with a Jumping Knowledge module.
4. Feed the combined representation into a classifier.

This project uses `GCNConv` as the base convolution and `JumpingKnowledge` from PyTorch Geometric for layer aggregation.

## Why JK-Net Is Historically Important

Early GNNs commonly used a fixed number of message-passing layers. That means every node used the same neighborhood range. JK-Net showed that this is too rigid: different nodes may need different receptive field sizes depending on graph structure, label distribution, and local connectivity.

JK-Net is historically important because it made deep graph neural networks more flexible. It also helped clarify the over-smoothing problem, where repeated graph propagation can make node embeddings too similar. By preserving and combining earlier-layer representations, JK-Net can keep useful local information while still using deeper neighborhood information.

## Fixed Neighborhood Range in Deep GNNs

In a normal stacked GCN, the final layer decides the representation used by the classifier. With four layers, every node representation has mixed information from roughly four-hop neighborhoods. That can help some nodes, but it can hurt others if their useful signal is mostly local.

JK-Net avoids forcing one fixed receptive field size on every node. The model can use representations from shallow and deep layers together.

## Aggregation Modes

This project supports these Jumping Knowledge modes:

- `cat`: Concatenates all layer outputs. This keeps all layer information and increases the classifier input size.
- `max`: Takes an element-wise maximum over layer outputs. This keeps the strongest activation at each feature dimension.
- `lstm`: Uses an LSTM-based attention mechanism to adaptively combine layer outputs.

The default mode is `cat` because it is easy to inspect and works well for an educational first version.

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

Representation Learning on Graphs with Jumping Knowledge Networks  
https://arxiv.org/abs/1806.03536

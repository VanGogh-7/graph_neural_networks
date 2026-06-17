# GAE and VGAE Link Prediction

This project implements a Graph Autoencoder (GAE) for link prediction on the
real public Cora citation network using PyTorch and PyTorch Geometric. It also
includes a Variational Graph Autoencoder (VGAE) option in the same educational
code path.

## Graph Autoencoder

A Graph Autoencoder learns node embeddings from graph structure and node
features, then reconstructs whether edges should exist between pairs of nodes.
The encoder maps each node to a latent vector `z`, and the decoder scores node
pairs as likely or unlikely links.

## Variational Graph Autoencoder

A Variational Graph Autoencoder extends GAE with a probabilistic latent space.
Instead of producing one embedding directly, the encoder predicts `mu` and
`logstd` for each node. A latent vector is sampled with the reparameterization
trick, and training adds a KL loss so the latent distribution stays regular.

This project trains `GAE` by default. To try VGAE, edit `model_type = "VGAE"`
in `src/config.py`.

## Historical Importance

Graph autoencoders are historically important because they showed how graph
neural networks can be trained without node labels by reconstructing graph
structure. VGAE is a key early example of combining graph convolutional
encoders with variational latent-variable modeling.

## Node Classification vs Link Prediction

Node classification predicts a class for each node. Link prediction predicts
whether an edge should exist between two nodes. For Cora link prediction, the
model sees a training graph with some edges held out, then learns to score
held-out positive edges higher than sampled non-existing edges.

## Encoder-Decoder Structure

The encoder uses `GCNConv` layers to produce node embeddings from `x` and
`edge_index`. The decoder receives pairs of nodes in `edge_label_index` and
returns one raw logit per pair.

## Inner-Product Decoder

The decoder uses an inner product:

```text
score(i, j) = z_i dot z_j
```

Large positive logits mean the model believes an edge is likely. The training
loss uses `BCEWithLogitsLoss`, so sigmoid is not applied inside the model.

## Positive and Negative Edges

Positive edges are real citation links. Negative edges are sampled node pairs
that are not present as edges. Training samples fresh negative edges each epoch.
Validation and test splits use fixed positive and negative edges created by
PyTorch Geometric's `RandomLinkSplit`.

## Metrics

Link prediction is evaluated with:

- **AUC**: how often positive edges receive higher scores than negative edges.
- **Average Precision (AP)**: precision averaged over the ranked list of edge
  predictions, emphasizing how well positives are ranked near the top.

## Cora Link Prediction

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
- `edge_label_index`: `[2, num_edges_to_predict]`
- `edge_logits`: `[num_edges_to_predict]`

For the standard Cora dataset, the expected graph statistics are:

- Nodes: `2708`
- Original edges: `10556`
- Node features: `1433`

## Project Structure

```text
05_gae_vgae_link_prediction/
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

The best checkpoint is selected by validation AUC. The test set is evaluated
only after training finishes.

## Predict

After training:

```bash
python scripts/predict.py
```

The prediction script loads the saved checkpoint and prints several held-out
edge predictions with source node, target node, predicted link probability, and
predicted label.

## Key Paper

Variational Graph Auto-Encoders  
https://arxiv.org/abs/1611.07308

This project is an educational PyTorch Geometric implementation, not a full
production graph learning system.

# gnn_models.py

This module provides simple graph neural network layers. Graph neural networks operate on data structured as nodes and
edges, capturing relationships that traditional architectures cannot.

## Layers

- **`normalize_adjacency`** – computes a symmetrically normalized adjacency matrix. This step stabilizes graph
  convolutions by scaling by node degrees.
- **`GCNLayer`** – implements a basic graph convolution. Node features are propagated by multiplying with the normalized
  adjacency matrix and passing through a linear transformation.
- **`GATLayer`** – introduces attention on edges. Each node attends to its neighbours via learnable attention weights,
  allowing the model to focus on the most relevant connections.
- **`GraphSAGELayer`** – aggregates neighbourhood information using mean, sum, or max operations and concatenates it with
  the node's own features.

## Python Techniques

All layers subclass `nn.Module`. Matrix operations use PyTorch tensor algebra, including `@` for matrix multiplication
and `torch.einsum` in the attention layer. Parameters are registered via `nn.Parameter` so PyTorch tracks them during
optimization.

## Theoretical Motivation

GCN layers approximate spectral convolutions on graphs, blending each node with its neighbours. GAT layers apply
self-attention to learn importance weights. GraphSAGE samples and aggregates neighbourhoods, enabling scalable training
on large graphs. These mechanisms empower models to learn from relational data such as social networks or molecular
structures.

The field of geometric deep learning frames these layers as extensions of
convolution to non-Euclidean domains. By respecting the graph's connectivity,
models capture complex relational patterns that regular CNNs cannot. Attention
mechanisms like those in GAT further refine message passing by letting each node
weight information from its neighbours according to their relevance for the
prediction task.

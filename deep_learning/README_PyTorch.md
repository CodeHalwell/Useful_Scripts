# PyTorch.py

This script presents a minimal example of constructing a feed-forward network for tabular data using the PyTorch
framework. The code defines a `TabularModel` class derived from `nn.Module` and demonstrates how to initialize and use
it.

## Model Architecture

- The constructor builds a sequence of `nn.Linear` layers separated by `nn.ReLU` activations. Layer sizes are provided
  via the `hidden_layers` list.
- The `forward` method defines how inputs flow through these layers.

## Python Features

PyTorch modules subclass `nn.Module` and must implement a `forward` method. The use of `nn.Sequential` collects layers
into a single callable block. Comprehensions are used to dynamically create intermediate layers.

## Theory

Neural networks learn nonlinear mappings from inputs to outputs by composing linear transformations and non-linear
activation functions. Here we demonstrate a simple fully connected architecture suitable for tabular regression or
classification tasks. During training, PyTorch uses automatic differentiation to compute gradients for backpropagation.

At its core, PyTorch represents models as dynamic computation graphs. This means
the graph is defined on-the-fly as tensors flow through operations, making it
easy to debug and allowing for flexible model architectures. Each parameter is a
tensor with attached gradients, and optimizers adjust these parameters by
descending along the gradient of a loss function. This mechanism enables the
iterative refinement of the network's weights so that predictions gradually
improve on the training data.

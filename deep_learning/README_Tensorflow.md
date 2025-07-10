# Tensorflow.py

This file collects two TensorFlow examples: a dense network for tabular data and a simple convolutional neural network.

## Dense Model

`create_model` builds a sequential stack of `Dense` layers. Activation functions are referenced by name from
`tensorflow.keras.activations`. The script shows how to compile the model for regression (MSE loss) or classification
(using e.g., `binary_crossentropy`).

Training is performed with `model.fit`, where the resulting `history` object records metrics over epochs.

## CNN Example

`create_cnn_model` constructs a tiny CNN with convolutional and pooling layers followed by dense layers. After
compiling, the model is trained on image data and evaluated on a test set. The placeholders in the code indicate where
you would load your own dataset.

## Theory

Dense layers perform affine transformations followed by nonlinear activations, suitable for structured data. Convolution
layers slide learnable filters over images to detect features, while pooling layers downsample to reduce spatial
resolution. Dropout randomly zeros activations to mitigate overfitting. These building blocks underpin many modern
computer vision models.

TensorFlow builds static computation graphs that are optimized before execution.
This allows for efficient deployment across different hardware backends but
requires model structures to be defined up front. The Keras API offers a higher
level abstraction where layers are composed to form a model object that can be
compiled and trained with minimal code. Underneath, gradients are calculated via
automatic differentiation, enabling the training of deep networks across a wide
range of domains.

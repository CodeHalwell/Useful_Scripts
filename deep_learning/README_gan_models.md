# gan_models.py

This file defines a minimal Generative Adversarial Network (GAN) using PyTorch. GANs pit two neural networks—the
Generator and the Discriminator—against each other in a game-theoretic setup.

## Components

- **`Generator`** – transforms random noise vectors into images via a series of linear layers with ReLU activations and
a final `tanh` to scale outputs to [-1, 1].
- **`Discriminator`** – takes images and attempts to classify them as real or fake using a small feed-forward network
with LeakyReLU activations.

## Python Aspects

Both classes inherit from `nn.Module` and define a `forward` method. Sequential containers (`nn.Sequential`) make layer
construction concise. The architecture is intentionally simple for instructional purposes.

## Theory

GANs train the generator to produce realistic samples that fool the discriminator, while simultaneously training the
discriminator to correctly distinguish real from generated data. This adversarial process often leads to high-quality
synthetic images once the two networks reach equilibrium.

From a probabilistic perspective, the generator implicitly defines a model of the
data distribution. The discriminator provides feedback by estimating how far the
generated samples deviate from real examples. Training proceeds as a minimax
game: the generator minimizes the discriminator's ability to tell real from
fake, while the discriminator maximizes it. When this game converges, the
generator approximates the true data distribution, producing convincing samples
despite never seeing explicit probability densities.

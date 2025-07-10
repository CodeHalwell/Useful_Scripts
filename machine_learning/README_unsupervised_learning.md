# unsupervised_learning.py

The unsupervised learning helpers cover clustering and dimensionality reduction techniques implemented in scikit-learn.

## Functions

- **`run_kmeans`** – performs k-means clustering, returning the fitted model and the labels assigned to each point.
- **`run_pca`** – reduces the dimensionality of data with Principal Component Analysis and returns the transformed
  components.
- **`run_tsne`** – computes a nonlinear embedding using t-SNE, useful for visualising high-dimensional data.

## Python Notes

The functions rely on scikit-learn estimators. Type hints specify the expected argument types and default parameter
values. Returning both the estimator object and the transformed data allows for later inspection or re-use.

## Theory

K-means partitions observations into clusters by minimizing within-cluster variance. PCA projects data onto orthogonal
axes that capture maximum variance, enabling compression and noise reduction. t-SNE maps data into a lower-dimensional
space while preserving local structure, commonly used for visualising embeddings.

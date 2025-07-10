from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def run_kmeans(X, n_clusters: int = 8, random_state: int = 42):
    """Cluster ``X`` using k-means and return the fitted model and labels."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return model, labels


def run_pca(X, n_components: int = 2):
    """Reduce dimensionality of ``X`` using PCA."""
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X)
    return pca, components


def run_tsne(X, n_components: int = 2, random_state: int = 42):
    """Run t-SNE on ``X`` and return the embedding."""
    tsne = TSNE(n_components=n_components, random_state=random_state)
    embedding = tsne.fit_transform(X)
    return tsne, embedding

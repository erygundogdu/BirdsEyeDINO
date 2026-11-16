"""Make the README figures from synthetic embeddings, without using local datasets."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def main():
    rng = np.random.default_rng(42)
    groups = np.repeat(np.arange(4), 40)
    centers = rng.normal(size=(4, 64))
    embeddings = centers[groups] + rng.normal(scale=0.65, size=(len(groups), 64))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(embeddings)
    projections = {
        'pca': PCA(n_components=2, random_state=42).fit_transform(embeddings),
        'tsne': TSNE(n_components=2, random_state=42, perplexity=30,
                     init='random', learning_rate='auto').fit_transform(embeddings),
    }
    destination = Path(__file__).resolve().parent / 'images'
    destination.mkdir(parents=True, exist_ok=True)
    for name, coordinates in projections.items():
        title = 'PCA' if name == 'pca' else 't-SNE'
        with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11}):
            fig, ax = plt.subplots(figsize=(8, 5.2), facecolor='#f8fafc')
            ax.set_facecolor('#f8fafc')
            for cluster in range(4):
                mask = labels == cluster
                ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=30,
                           color=plt.get_cmap('tab20')(cluster), alpha=0.9,
                           edgecolors='white', linewidths=0.45,
                           label=f'Cluster {cluster} (n={mask.sum()})')
            ax.set_title(f'{title}: a view of the embedding space', loc='left', pad=18, fontweight='bold')
            ax.set_xlabel(f'{title} dimension 1')
            ax.set_ylabel(f'{title} dimension 2')
            ax.spines[['top', 'right']].set_visible(False)
            ax.legend(frameon=False, loc='best', fontsize=9)
            fig.text(0.5, 0.025, 'Synthetic example only; no aircraft data or real run results.',
                     ha='center', fontsize=9, color='#475569')
            fig.tight_layout(rect=[0, 0.055, 1, 1])
            path = destination / f'clustering_{name}.png'
            fig.savefig(path, dpi=160)
            plt.close(fig)
            print(path)


if __name__ == '__main__':
    main()

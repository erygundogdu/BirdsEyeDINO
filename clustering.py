import warnings

import joblib
import numpy as np
from sklearn.cluster import KMeans

from runtime import write_csv


def cluster(embeddings, instance_ids, config, directory):
    x = np.asarray(embeddings, dtype=np.float32)
    n, k = len(x), config['k']
    directory.mkdir(parents=True, exist_ok=True)
    if n < k:
        return None, f'Insufficient candidates: N={n} < K={k}. K was not changed.'
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite embeddings cannot be clustered.')
    model = KMeans(n_clusters=k, random_state=config['random_state'], n_init=config['n_init'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        labels = model.fit_predict(x)
    distances = np.linalg.norm(x - model.cluster_centers_[labels], axis=1)
    rows = [dict(instance_id=i, cluster_id=int(label), centroid_distance=float(distance))
            for i, label, distance in zip(instance_ids, labels, distances)]
    write_csv(directory / 'assignments.csv', rows, ['instance_id', 'cluster_id', 'centroid_distance'])
    summary = [dict(cluster_id=c, count=int(np.sum(labels == c))) for c in range(k)]
    write_csv(directory / 'cluster_summary.csv', summary, ['cluster_id', 'count'])
    np.save(directory / 'cluster_centers.npy', model.cluster_centers_)
    joblib.dump(model, directory / 'kmeans.joblib')
    return dict(labels=labels, distances=distances, counts={str(s['cluster_id']): s['count'] for s in summary},
                warnings=[str(w.message) for w in caught]), None

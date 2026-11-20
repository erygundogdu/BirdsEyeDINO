"""Read MAR20 ground-truth boxes and evaluate unsupervised clusters."""

from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from runtime import write_csv


class MAR20Dataset:
    def __init__(self, root):
        self.root = root
        self.image_root = root / 'JPEGImages'
        annotation_root = root / 'Annotations' / 'Horizontal Bounding Boxes'
        self.annotations = {}
        names = set()
        for path in sorted(annotation_root.glob('*.xml')):
            objects = []
            for obj in ET.parse(path).getroot().findall('object'):
                name = obj.findtext('name')
                if not name or not name.strip():
                    raise ValueError(f'Missing object class in {path}')
                name = name.strip()
                box = obj.find('bndbox')
                # VOC-style boxes use one-based, inclusive coordinates.
                coords = [float(box.findtext(key)) for key in ('xmin', 'ymin', 'xmax', 'ymax')]
                coords[0] -= 1
                coords[1] -= 1
                if not np.isfinite(coords).all() or coords[2] <= coords[0] or coords[3] <= coords[1]:
                    raise ValueError(f'Invalid bounding box in {path}')
                objects.append((name, coords))
                names.add(name)
            self.annotations[path.stem] = objects
        if not self.annotations or not names:
            raise ValueError(f'No labeled MAR20 horizontal XML annotations found in {annotation_root}')
        self.class_names = sorted(names)
        self.class_ids = {name: i for i, name in enumerate(self.class_names)}
        self.paths = sorted(p for p in self.image_root.iterdir()
                            if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
        stems = [p.stem for p in self.paths]
        if len(set(stems)) != len(stems):
            raise ValueError('MAR20 contains multiple images with the same stem.')
        missing_images = set(self.annotations) - set(stems)
        missing_labels = set(stems) - set(self.annotations)
        if missing_images or missing_labels:
            raise ValueError(f'MAR20 image/annotation mismatch: {len(missing_images)} missing images, '
                             f'{len(missing_labels)} missing annotations.')

    def detections(self, path):
        return np.asarray([coords + [1.0, self.class_ids[name]]
                           for name, coords in self.annotations[path.stem]], dtype=np.float32).reshape(-1, 6)


def evaluate_clusters(records, instance_ids, labels, class_names, directory):
    by_id = {r['instance_id']: r for r in records}
    truth = np.array([by_id[i]['class_id'] for i in instance_ids], dtype=int)
    labels = np.asarray(labels, dtype=int)
    if not len(truth) or len(labels) != len(truth):
        raise ValueError('Evaluation requires matching nonempty instance IDs and cluster labels.')
    cluster_ids, columns = np.unique(labels, return_inverse=True)
    counts = np.zeros((len(class_names), len(cluster_ids)), dtype=int)
    np.add.at(counts, (truth, columns), 1)
    rows, cols = linear_sum_assignment(-counts)
    mapping = {int(cluster_ids[c]): int(r) for r, c in zip(rows, cols) if counts[r, c] > 0}
    matches = sum(counts[r, c] for r, c in zip(rows, cols))
    write_csv(directory / 'class_cluster_counts.csv',
              [dict(class_id=r, class_name=name, cluster_id=int(cid), count=int(counts[r, c]))
               for r, name in enumerate(class_names) for c, cid in enumerate(cluster_ids)],
              ['class_id', 'class_name', 'cluster_id', 'count'])
    write_csv(directory / 'evaluated_assignments.csv',
              [dict(instance_id=i, class_id=int(t), class_name=class_names[t], cluster_id=int(c),
                    matched_class_name=class_names[mapping[c]] if c in mapping else '')
               for i, t, c in zip(instance_ids, truth, labels)],
              ['instance_id', 'class_id', 'class_name', 'cluster_id', 'matched_class_name'])
    return dict(evaluated_instances=len(truth), class_counts=dict(Counter(class_names[t] for t in truth)),
                adjusted_rand_index=float(adjusted_rand_score(truth, labels)),
                normalized_mutual_information=float(normalized_mutual_info_score(truth, labels)),
                matched_accuracy=float(matches / len(truth)),
                cluster_to_class={str(c): class_names[t] for c, t in mapping.items()})

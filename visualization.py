import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from candidate_extractor import materialize_crop
from runtime import read_image, write_csv, write_image


def color(cluster_id):
    return plt.get_cmap('tab20')(int(cluster_id) % 20)[:3]


def bgr(cluster_id):
    return tuple(int(v * 255) for v in color(cluster_id)[::-1])


def box(image, coords, text, shade):
    x1, y1, x2, y2 = (int(round(v)) for v in coords)
    cv2.rectangle(image, (x1, y1), (x2, y2), shade, 2)
    cv2.putText(image, text, (max(0, x1), max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, shade, 1, cv2.LINE_AA)


def annotate(paths, records, assignments, root, directory):
    grouped = {}
    for record in records:
        if record.get('valid', True) and record['instance_id'] in assignments:
            grouped.setdefault(record['image_path'], []).append(record)
    for path in paths:
        if str(path) not in grouped:
            continue
        image = read_image(path)
        for record in grouped.get(str(path), []):
            cid = assignments.get(record['instance_id'])
            if cid is not None:
                box(image, record['bbox'], f"{record['instance_id']} cluster {cid} {record['confidence']:.2f}", bgr(cid))
        write_image(directory / 'annotated_images' / path.relative_to(root), image)


def rejected_debug(records, directory):
    for record in records:
        if record['valid']:
            continue
        image = read_image(record['image_path'])
        if record['crop_coordinates']:
            box(image, record['crop_coordinates'], f"{record.get('crop_mode', 'contextual')} crop", (255, 255, 0))
        if np.isfinite(record['bbox']).all():
            box(image, record['bbox'], record['instance_id'], (0, 0, 255))
        for other in record['interfering_detections']:
            box(image, other['bbox'], f"interfering d{other['index']} class {other['class_id']}", (0, 165, 255))
        cv2.putText(image, record['rejection_reason'], (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)
        write_image(directory / 'rejected_candidates' / f"{record['instance_id']}.png", image)


def galleries(records, labels, k, config, directory, padding_value):
    for cid in range(k):
        folder = directory / f'cluster_{cid}'
        folder.mkdir(parents=True, exist_ok=True)
        members = [r for r, label in zip(records, labels) if label == cid]
        page_size, columns = config['gallery_page_size'], config['gallery_columns']
        for start in range(0, len(members), page_size):
            page = members[start:start + page_size]
            sheet = np.full((math.ceil(len(page) / columns) * 190, columns * 240, 3), 245, np.uint8)
            for index, record in enumerate(page):
                crop = read_image(record['crop_path']) if record['crop_path'] else materialize_crop(
                    read_image(record['image_path']), record, padding_value)
                write_image(folder / f"{record['instance_id']}.png", crop)
                scale = min(230 / crop.shape[1], 140 / crop.shape[0])
                thumb = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)),
                                          max(1, round(crop.shape[0] * scale))))
                x, y = (index % columns) * 240, (index // columns) * 190
                sheet[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
                cv2.putText(sheet, record['image_filename'][:32], (x + 2, y + 155),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (30, 30, 30), 1)
                cv2.putText(sheet, record['instance_id'], (x + 2, y + 175),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr(cid), 1)
            write_image(folder / f'gallery_{start // page_size:03d}.png', sheet)


def projections(x, ids, labels, config, seed, directory, model_name='DINOv2'):
    directory.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    notes = []
    for name, enabled in [('pca', config['save_pca']), ('tsne', config['save_tsne'])]:
        if not enabled:
            continue
        if n < 2:
            notes.append(f'{name} skipped: at least two samples required.')
            continue
        if np.allclose(x, x[0], rtol=0, atol=1e-8):
            coords = np.zeros((n, 2))
            notes.append(f'{name}: identical embeddings represented at the origin.')
        elif name == 'pca':
            coords = PCA(n_components=min(2, n, x.shape[1]), random_state=seed).fit_transform(x)
            if coords.shape[1] == 1:
                coords = np.column_stack((coords, np.zeros(n)))
        else:
            perplexity = min(float(config['tsne_perplexity']), float(n - 1))
            coords = TSNE(n_components=2, random_state=seed, perplexity=perplexity,
                          init='random', learning_rate='auto').fit_transform(x)
            notes.append(f'tsne effective perplexity: {perplexity}')
        rows = [dict(instance_id=i, cluster_id=int(c), x=float(v[0]), y=float(v[1]))
                for i, c, v in zip(ids, labels, coords)]
        write_csv(directory / f'clustering_{name}.csv', rows, ['instance_id', 'cluster_id', 'x', 'y'])
        fig, ax = plt.subplots(figsize=(9, 7))
        for cid in sorted(set(labels)):
            mask = labels == cid
            ax.scatter(coords[mask, 0], coords[mask, 1], s=25, color=color(cid),
                       label=f'Cluster {cid} (n={int(mask.sum())})')
        title = 'PCA' if name == 'pca' else 't-SNE'
        ax.set(title=f'{title} of normalized {model_name} aircraft embeddings',
               xlabel=f'{title} dimension 1', ylabel=f'{title} dimension 2')
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f'clustering_{name}.png', dpi=config['dpi'])
        plt.close(fig)
    return notes

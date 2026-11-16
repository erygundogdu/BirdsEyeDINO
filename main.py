import gc
import logging
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import torch

from candidate_extractor import MANIFEST_FIELDS, extract_candidates
from clustering import cluster
from detector import Detector
from feature_extractor import FeatureExtractor
from runtime import backbone_settings, load_config, read_image, write_csv, write_json
from visualization import annotate, galleries, projections, rejected_debug


def run(config_path=Path(__file__).with_name('config.yaml')):
    cfg = load_config(config_path)
    output = cfg['output']['directory']
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f'Result directory is not empty: {output}. Select a new output.directory in config.yaml.')
    timings, records, errors, all_detections = {}, [], [], []
    summary = dict(status='running', configuration=cfg, total_images_processed=0,
                   total_detections=0, class_4_detections=0, accepted_candidates=0,
                   rejected_candidates=0, number_of_clusters=0, requested_clusters=cfg['clustering']['k'],
                   instances_per_cluster={}, embedding_dimension=0, processing_seconds=timings,
                   warnings=[], image_errors=errors)
    try:
        backbone = backbone_settings(cfg['dino'])
        summary['backbone'] = {k: v for k, v in backbone.items() if k != 'settings'}
        FeatureExtractor.prepare(cfg['dino'])
        start = time.perf_counter()
        detector = Detector(cfg['detector'])
        paths = detector.image_paths()
        summary['total_images_discovered'] = len(paths)
        summary['detector_attention_modules'] = detector.attention_modules
        summary['detector_compatibility_repairs'] = detector.compatibility_repairs
        summary['detector_class_names'] = detector.model.names
        processed = []
        for index, path in enumerate(paths):
            try:
                image = read_image(path)
            except ValueError as error:
                errors.append(dict(image_path=str(path), error=str(error)))
                continue
            detections = detector.predict(image)
            processed.append(path)
            summary['total_detections'] += len(detections)
            relative = path.relative_to(cfg['detector']['image_dir'])
            all_detections.append(dict(image_path=str(path), image_shape=list(image.shape),
                                       detections=detections.tolist()))
            records.extend(extract_candidates(image, path, relative, index, detections,
                                               cfg['candidates'], output, cfg['output']['save_crops']))
            if (index + 1) % 25 == 0 or index == len(paths) - 1:
                print(f'Detection: {index + 1}/{len(paths)} images, {sum(r["valid"] for r in records)} accepted', flush=True)
        timings['detection_and_candidates'] = time.perf_counter() - start
        summary['total_images_processed'] = len(processed)
        accepted = [r for r in records if r['valid']]
        rejected = [r for r in records if not r['valid']]
        summary.update(class_4_detections=len(records), accepted_candidates=len(accepted), rejected_candidates=len(rejected))
        write_csv(output / 'manifest.csv', records, MANIFEST_FIELDS)
        write_csv(output / 'rejected_candidates.csv', rejected, MANIFEST_FIELDS)
        write_json(output / 'detections.json', all_detections)
        del detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        start = time.perf_counter()
        dimension = backbone['dimension'] * (2 if cfg['dino']['embedding_type'] == 'cls_mean' else 1)
        embeddings, ids = torch.empty((0, dimension)), []
        if accepted:
            extractor = FeatureExtractor(cfg['dino'])
            embeddings, ids = extractor.extract(accepted, cfg['candidates']['padding_value'])
            summary['dino_tensor_shapes_last_batch'] = extractor.shapes
            del extractor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary['embedding_dimension'] = dimension
        if cfg['output']['save_embeddings']:
            (output / 'embeddings').mkdir(exist_ok=True)
            torch.save(dict(embeddings=embeddings, instance_ids=ids, model=backbone['model'],
                            backbone=backbone['name'], normalization_mean=backbone['mean'],
                            normalization_std=backbone['std'], patch_size=backbone['patch_size'],
                            embedding_type=cfg['dino']['embedding_type'], input_size=cfg['dino']['input_size'],
                            crop_mode=cfg['candidates']['crop_mode']),
                       output / 'embeddings' / 'features.pt')
        timings['feature_extraction'] = time.perf_counter() - start
        start = time.perf_counter()
        result, reason = cluster(embeddings.numpy(), ids, cfg['clustering'], output / 'clustering')
        timings['clustering'] = time.perf_counter() - start
        start = time.perf_counter()
        visual = output / 'visualizations'
        visual.mkdir(exist_ok=True)
        rejected_debug(records, visual)
        if result is None:
            summary['status'] = 'clustering_skipped'
            summary['warnings'].append(reason)
        else:
            labels = result['labels']
            summary.update(status='complete', number_of_clusters=len(set(labels)),
                           instances_per_cluster=result['counts'])
            summary['warnings'].extend(result['warnings'])
            if cfg['visualization']['save_annotated_images']:
                annotate(processed, accepted, dict(zip(ids, map(int, labels))), cfg['detector']['image_dir'], visual)
            if cfg['visualization']['save_cluster_galleries']:
                galleries(accepted, labels, cfg['clustering']['k'], cfg['visualization'],
                          visual, cfg['candidates']['padding_value'])
            summary['warnings'].extend(projections(embeddings.numpy(), ids, labels,
                                                   cfg['visualization'], cfg['clustering']['random_state'], visual,
                                                   model_name=backbone['model']))
        timings['visualization'] = time.perf_counter() - start
        write_json(output / 'summary.json', summary)
        print(f'Results: {output}\nAccepted: {len(accepted)}; rejected: {len(rejected)}; status: {summary["status"]}', flush=True)
        return summary
    except Exception as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output / 'summary.json', summary)
        raise


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    run()

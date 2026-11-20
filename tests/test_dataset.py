import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from dataset_analysis import MAR20Dataset, evaluate_clusters
from main import run
from runtime import write_image


class DatasetTests(unittest.TestCase):
    def make_dataset(self, root):
        annotations = root / 'Annotations' / 'Horizontal Bounding Boxes'
        annotations.mkdir(parents=True)
        for index in range(2):
            write_image(root / 'JPEGImages' / f'{index}.jpg', np.full((40, 60, 3), 100, np.uint8))
            (annotations / f'{index}.xml').write_text('''<annotation><filename>wrong.xml</filename>
                <object><name>A2</name><bndbox><xmin>1</xmin><ymin>1</ymin>
                <xmax>20</xmax><ymax>20</ymax></bndbox></object>
                <object><name>A10</name><bndbox><xmin>10</xmin><ymin>10</ymin>
                <xmax>30</xmax><ymax>30</ymax></bndbox></object></annotation>''', encoding='utf-8')

    def test_mar20_pairing_coordinates_and_missing_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root)
            dataset = MAR20Dataset(root)
            detections = dataset.detections(dataset.paths[0])
            self.assertEqual(len(dataset.paths), 2)
            self.assertEqual(dataset.class_names, ['A10', 'A2'])
            np.testing.assert_array_equal(detections[0], [0, 0, 20, 20, 1, 1])
            (root / 'Annotations' / 'Horizontal Bounding Boxes' / '0.xml').unlink()
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                MAR20Dataset(root)

    def test_metrics_are_permutation_invariant_and_align_by_instance_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = [dict(instance_id='a', class_id=0), dict(instance_id='b', class_id=1),
                       dict(instance_id='c', class_id=0), dict(instance_id='d', class_id=1)]
            metrics = evaluate_clusters(records, ['d', 'c', 'b', 'a'], [0, 1, 0, 1],
                                        ['A1', 'A2'], Path(tmp))
            for key in ('adjusted_rand_index', 'normalized_mutual_information', 'matched_accuracy'):
                self.assertAlmostEqual(metrics[key], 1)
            self.assertEqual(metrics['cluster_to_class'], {'0': 'A2', '1': 'A1'})
            imperfect = evaluate_clusters(records, ['a', 'b', 'c', 'd'], [0, 0, 0, 0],
                                          ['A1', 'A2'], Path(tmp))
            self.assertEqual(imperfect['matched_accuracy'], .5)
            self.assertAlmostEqual(imperfect['adjusted_rand_index'], 0)

    def test_dataset_pipeline_reuses_extraction_and_skips_detector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_dataset(root / 'MAR20')
            cfg = yaml.safe_load(Path('config.example.yaml').read_text(encoding='utf-8'))
            cfg.update(analyze_dataset=True, dataset_path='MAR20')
            cfg['dino']['device'] = 'cpu'
            cfg['detector'].update(workspace='missing', weights='missing.pt', image_dir='missing', device='cuda:99')
            cfg['candidates'].update(target_class=999, crop_mode='contextual', reject_bbox_overlap=True)
            cfg['clustering']['k'] = 99
            cfg['output'].update(directory='results', save_crops=False)
            for key in ('save_pca', 'save_tsne', 'save_cluster_galleries', 'save_annotated_images'):
                cfg['visualization'][key] = False
            path = root / 'config.yaml'
            path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
            def extract(records, padding_value):
                self.assertEqual(len(records), 4)
                self.assertTrue(all(r['crop_mode'] == 'bbox' and r['valid'] for r in records))
                return torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]), [r['instance_id'] for r in records]
            with patch('main.Detector') as detector, patch('main.FeatureExtractor') as extractor:
                extractor.return_value.extract.side_effect = extract
                extractor.return_value.shapes = {}
                summary = run(path)
                detector.assert_not_called()
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(summary['requested_clusters'], 2)
            self.assertEqual(summary['accepted_candidates'], 4)
            metrics = json.loads((root / 'results' / 'clustering' / 'metrics.json').read_text())
            self.assertEqual(metrics['matched_accuracy'], 1)


if __name__ == '__main__':
    unittest.main()

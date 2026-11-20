import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from candidate_extractor import bbox_geometry, crop_geometry, extract_candidates, intersects, materialize_crop
from clustering import cluster
from feature_extractor import FeatureExtractor
from runtime import backbone_settings, input_shape, load_config, read_image, write_image
from visualization import annotate, galleries, projections, rejected_debug


class CandidateTests(unittest.TestCase):
    def test_null_target_class_includes_all_detected_classes(self):
        image = np.zeros((100, 100, 3), np.uint8)
        detections = np.array([[10, 10, 20, 20, .9, 2],
                               [60, 60, 70, 70, .8, 4]], np.float32)
        cfg = dict(target_class=None, crop_mode='bbox', padding_value=114)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = extract_candidates(image, root / 'image.png', Path('image.png'),
                                         0, detections, cfg, root, False)
            self.assertEqual([r['class_id'] for r in records], [2, 4])
            self.assertTrue(all(r['valid'] for r in records))
            cfg['target_class'] = 2
            records = extract_candidates(image, root / 'image.png', Path('image.png'),
                                         0, detections, cfg, root, False)
            self.assertEqual([r['class_id'] for r in records], [2])

    def test_square_contains_fractional_box_and_padding(self):
        image = np.full((80, 100, 3), 200, dtype=np.uint8)
        for bbox in ([0.1, 0.2, 20.8, 21.7], [1.1, 5.2, 99.8, 35.4], [20, 20, 25, 25]):
            coords, clipped, padding = crop_geometry(bbox, 32, image.shape)
            self.assertEqual(coords[2] - coords[0], coords[3] - coords[1])
            self.assertLessEqual(coords[0], bbox[0])
            self.assertLessEqual(coords[1], bbox[1])
            self.assertGreaterEqual(coords[2], bbox[2])
            self.assertGreaterEqual(coords[3], bbox[3])
            crop = materialize_crop(image, dict(clipped_crop_coordinates=clipped, padding=padding), 114)
            self.assertEqual(crop.shape[:2], (coords[2] - coords[0],) * 2)
        self.assertFalse(intersects([0, 0, 10, 10], [10, 0, 20, 10]))

    def test_other_classes_self_exclusion_and_rejection_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'source.png'
            image = np.full((100, 100, 3), 200, np.uint8)
            write_image(path, image)
            det = np.array([[40, 40, 50, 50, .9, 4], [55, 40, 60, 50, .8, 2]], np.float32)
            cfg = dict(target_class=4, crop_size=32, reject_other_objects=True, padding_value=114)
            rejected = extract_candidates(image, path, Path(path.name), 0, det, cfg, root, True)
            self.assertFalse(rejected[0]['valid'])
            self.assertEqual(rejected[0]['interfering_detection_indices'], [1])
            rejected_debug(rejected, root)
            self.assertTrue((root / 'rejected_candidates' / 'i000000_d0000.png').is_file())
            accepted = extract_candidates(image, path, Path(path.name), 1, det[:1], cfg, root, True)
            self.assertTrue(accepted[0]['valid'])
            self.assertTrue(Path(accepted[0]['crop_path']).is_file())
            cfg['reject_other_objects'] = False
            allowed = extract_candidates(image, path, Path(path.name), 2, det, cfg, root, False)
            self.assertTrue(allowed[0]['valid'])
            self.assertEqual(allowed[0]['crop_path'], '')
            self.assertEqual(len(extract_candidates(image, path, Path(path.name), 3,
                                                    np.empty((0, 6)), cfg, root, False)), 0)

    def test_invalid_bbox(self):
        with self.assertRaises(ValueError):
            crop_geometry([10, 10, 2, 3], 112, (100, 100, 3))


    def test_bbox_mode_uses_tight_region_for_pixels_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
            path = root / 'image.png'
            write_image(path, image)
            detections = np.array([[40.2, 40.1, 50.5, 45.8, .9, 4],
                                   [55, 40, 60, 50, .8, 2]], np.float32)
            cfg = dict(target_class=4, crop_mode='bbox', crop_size=32,
                       reject_other_objects=True, padding_value=114)
            record = extract_candidates(image, path, Path(path.name), 0, detections, cfg, root, True)[0]
            self.assertTrue(record['valid'])
            self.assertEqual(record['crop_coordinates'], [40, 40, 51, 46])
            np.testing.assert_array_equal(read_image(record['crop_path']), image[40:46, 40:51])
            np.testing.assert_array_equal(materialize_crop(image, record, 114), image[40:46, 40:51])
            cfg['crop_mode'] = 'contextual'
            self.assertFalse(extract_candidates(image, path, Path(path.name), 1,
                                               detections, cfg, root, False)[0]['valid'])
            cfg['crop_mode'] = 'bbox'
            detections[1, :4] = [50.6, 41, 60, 50]
            self.assertTrue(extract_candidates(image, path, Path(path.name), 2,
                                              detections, cfg, root, False)[0]['valid'])
            detections[1, :4] = [50.5, 41, 60, 50]
            self.assertTrue(extract_candidates(image, path, Path(path.name), 3,
                                              detections, cfg, root, False)[0]['valid'])
            detections[1, :4] = [49, 41, 60, 50]
            cfg['reject_other_objects'] = False
            overlapping = extract_candidates(image, path, Path(path.name), 4,
                                             detections, cfg, root, False)[0]
            self.assertFalse(overlapping['valid'])
            self.assertEqual(overlapping['rejection_reason'], 'bbox_overlaps_other_detection')
            cfg['reject_bbox_overlap'] = False
            cfg['reject_other_objects'] = True
            allowed = extract_candidates(image, path, Path(path.name), 5,
                                         detections, cfg, root, False)[0]
            self.assertTrue(allowed['valid'])
            self.assertEqual(allowed['rejection_reason'], '')
            self.assertEqual(allowed['interfering_detection_indices'], [1])
            cfg['crop_mode'] = 'contextual'
            self.assertFalse(extract_candidates(image, path, Path(path.name), 6,
                                               detections, cfg, root, False)[0]['valid'])
            coords, clipped, padding = bbox_geometry([-5, -3, 30, 12], image.shape)
            self.assertEqual(coords, [0, 0, 30, 12])
            self.assertEqual(sum(padding.values()), 0)


class PreprocessingTests(unittest.TestCase):
    def test_backbone_profiles_and_patch_sizes(self):
        normal = backbone_settings(dict(backbone='normal', normal=dict(model='dinov2_vitb14')))
        satellite = backbone_settings(dict(backbone='satellite'))
        self.assertEqual((normal['dimension'], normal['patch_size']), (768, 14))
        self.assertEqual((satellite['dimension'], satellite['patch_size']), (1024, 16))
        self.assertEqual(input_shape([112, 224], satellite['patch_size']), [112, 224])
        with self.assertRaises(ValueError):
            input_shape(126, satellite['patch_size'])
        with self.assertRaises(ValueError):
            backbone_settings(dict(backbone='unknown'))
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(backbone='satellite', satellite=dict(local_directory=Path(tmp)))
            with self.assertRaisesRegex(RuntimeError, 'Satellite DINOv3 is unavailable'):
                FeatureExtractor.prepare(config)

    def test_satellite_local_loading_pooling_and_normalization(self):
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = DINOv3ViTModel(DINOv3ViTConfig(hidden_size=1024, intermediate_size=128,
                                                num_hidden_layers=1, num_attention_heads=16,
                                                num_register_tokens=4, image_size=112))
            model.save_pretrained(root)
            del model
            config = dict(backbone='satellite', device='cpu', input_size=[112, 224],
                          embedding_type='cls', satellite=dict(local_directory=root))
            extractor = FeatureExtractor(config)
            crop = np.zeros((30, 50, 3), np.uint8)
            crop[:, :, 2] = 255
            batch = extractor.preprocess(crop).unsqueeze(0)
            expected = torch.tensor([(1 - .430) / .213, -.411 / .156, -.296 / .143])
            torch.testing.assert_close(batch[0, :, 0, 0], expected)
            with torch.inference_mode():
                tokens = extractor.model(pixel_values=batch).last_hidden_state
            self.assertEqual(tuple(tokens.shape), (1, 1 + 4 + 98, 1024))
            for mode in ('cls', 'mean', 'cls_mean'):
                config['embedding_type'] = mode
                result = extractor.encode(batch)
                cls, mean = tokens[:, 0], tokens[:, 5:].mean(dim=1)
                expected_embedding = cls if mode == 'cls' else mean if mode == 'mean' else torch.cat([cls, mean], dim=1)
                expected_embedding = torch.nn.functional.normalize(expected_embedding, dim=1)
                torch.testing.assert_close(result, expected_embedding)
                self.assertEqual(tuple(result.shape), (1, 2048 if mode == 'cls_mean' else 1024))
                self.assertEqual(extractor.shapes['patch_tokens'], [1, 98, 1024])

    def test_scalar_and_rectangular_sizes_and_normalization(self):
        extractor = FeatureExtractor.__new__(FeatureExtractor)
        crop = np.zeros((15, 60, 3), np.uint8)
        crop[:, :, 2] = 255
        expected = torch.tensor([(1 - .485) / .229, -.456 / .224, -.406 / .225])
        for size, shape in [(112, [112, 112]), ([112, 224], [112, 224])]:
            self.assertEqual(input_shape(size), shape)
            extractor.config = dict(input_size=size)
            result = extractor.preprocess(crop)
            self.assertEqual(list(result.shape), [3] + shape)
            torch.testing.assert_close(result[:, 0, 0], expected)
        for invalid in ([112, 225], [112], 0, True, '224', [112, -14]):
            with self.assertRaises(ValueError):
                input_shape(invalid)


class ClusteringVisualizationTests(unittest.TestCase):
    def test_mapping_and_visual_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.array([[1, 0, 0], [.99, .01, 0], [0, 1, 0], [0, .99, .01]], np.float32)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            ids = [f'id{i}' for i in range(4)]
            result, reason = cluster(x, ids, dict(k=2, random_state=42, n_init=10), root / 'clustering')
            self.assertIsNone(reason)
            with (root / 'clustering' / 'assignments.csv').open(encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r['instance_id'] for r in rows], ids)
            self.assertEqual([int(r['cluster_id']) for r in rows], list(result['labels']))
            config = dict(save_pca=True, save_tsne=True, tsne_perplexity=30, dpi=60,
                          gallery_page_size=2, gallery_columns=2)
            notes = projections(x, ids, result['labels'], config, 42, root)
            self.assertIn('3.0', notes[0])
            for name in ('pca', 'tsne'):
                with (root / f'clustering_{name}.csv').open(encoding='utf-8') as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual([r['instance_id'] for r in rows], ids)
                self.assertTrue((root / f'clustering_{name}.png').is_file())
            image_path = root / 'original.png'
            write_image(image_path, np.full((100, 100, 3), 120, np.uint8))
            records = [dict(instance_id=i, image_filename=image_path.name, image_path=str(image_path),
                            bbox=[30, 30, 60, 60], confidence=.9, crop_path='',
                            clipped_crop_coordinates=[20, 20, 80, 80], padding=dict(left=0, top=0, right=0, bottom=0))
                       for i in ids]
            empty_path = root / 'no_candidates.png'
            rejected_path = root / 'rejected_only.png'
            write_image(empty_path, np.zeros((100, 100, 3), np.uint8))
            write_image(rejected_path, np.zeros((100, 100, 3), np.uint8))
            invalid_record = dict(records[0], image_path=str(rejected_path), valid=False)
            annotate([image_path, empty_path, rejected_path], records + [invalid_record],
                     dict(zip(ids, result['labels'])), root, root / 'visual')
            self.assertEqual([p.name for p in (root / 'visual' / 'annotated_images').iterdir()],
                             [image_path.name])
            galleries(records, result['labels'], 2, config, root / 'visual', 114)
            for record, cid in zip(records, result['labels']):
                self.assertTrue((root / 'visual' / f'cluster_{cid}' / f"{record['instance_id']}.png").is_file())
            self.assertEqual(read_image(image_path)[0, 0, 0], 120)

    def test_small_and_degenerate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dict(save_pca=True, save_tsne=True, tsne_perplexity=30, dpi=60)
            for n in (0, 1, 2, 3):
                x = np.ones((n, 4), np.float32) / 2
                ids = [str(i) for i in range(n)]
                labels = np.zeros(n, int)
                notes = projections(x, ids, labels, cfg, 42, root / str(n))
                self.assertEqual(len(notes), 2)
                result, reason = cluster(x, ids, dict(k=3, random_state=42, n_init=10), root / str(n))
                if n < 3:
                    self.assertIsNone(result)
                    self.assertIn('Insufficient', reason)
                else:
                    self.assertEqual(len(set(result['labels'])), 1)
                    self.assertTrue(result['warnings'])
            projections(np.array([[1, 0], [0, 1]], np.float32), ['a', 'b'],
                        np.array([0, 1]), cfg, 42, root / 'two_distinct')


if __name__ == '__main__':
    unittest.main()

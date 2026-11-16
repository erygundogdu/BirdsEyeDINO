import math

import cv2
import numpy as np

from runtime import write_image


MANIFEST_FIELDS = ['instance_id', 'image_filename', 'image_path', 'image_relative_path',
                   'detection_index', 'class_id', 'confidence', 'bbox', 'crop_mode', 'crop_coordinates',
                   'clipped_crop_coordinates', 'padding', 'crop_path', 'valid',
                   'rejection_reason', 'interfering_detection_indices', 'interfering_detections']


def intersects(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def crop_geometry(bbox, crop_size, shape):
    x1, y1, x2, y2 = map(float, bbox)
    if not np.isfinite(bbox).all() or x2 <= x1 or y2 <= y1:
        raise ValueError('invalid_bbox')
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(crop_size, math.ceil(x2 - x1), math.ceil(y2 - y1))
    left, top = math.floor(cx - side / 2), math.floor(cy - side / 2)
    side = max(side, math.ceil(x2 - left), math.ceil(y2 - top))
    right, bottom = left + side, top + side
    h, w = shape[:2]
    clipped = [max(0, left), max(0, top), min(w, right), min(h, bottom)]
    padding = dict(left=max(0, -left), top=max(0, -top),
                   right=max(0, right - w), bottom=max(0, bottom - h))
    return [left, top, right, bottom], clipped, padding


def materialize_crop(image, record, padding_value):
    x1, y1, x2, y2 = record['clipped_crop_coordinates']
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError('empty_crop')
    p = record['padding']
    return cv2.copyMakeBorder(crop, p['top'], p['bottom'], p['left'], p['right'],
                              cv2.BORDER_CONSTANT, value=(padding_value,) * 3)


def bbox_geometry(bbox, shape):
    x1, y1, x2, y2 = map(float, bbox)
    if not np.isfinite(bbox).all() or x2 <= x1 or y2 <= y1:
        raise ValueError('invalid_bbox')
    h, w = shape[:2]
    coords = [max(0, min(w, math.floor(x1))), max(0, min(h, math.floor(y1))),
              max(0, min(w, math.ceil(x2))), max(0, min(h, math.ceil(y2)))]
    return coords, coords.copy(), dict(left=0, top=0, right=0, bottom=0)


def extract_candidates(image, path, relative, image_index, detections, config, output, save_crops):
    mode = config.get('crop_mode', 'contextual')
    if mode not in ('contextual', 'bbox'):
        raise ValueError('crop_mode must be contextual or bbox.')
    records = []
    for index, detection in enumerate(detections):
        if int(detection[5]) != config['target_class']:
            continue
        instance = f'i{image_index:06d}_d{index:04d}'
        record = dict(instance_id=instance, image_filename=path.name, image_path=str(path),
                      image_relative_path=str(relative), detection_index=index,
                      class_id=int(detection[5]), confidence=float(detection[4]),
                      bbox=detection[:4].tolist(), crop_mode=mode, crop_coordinates=[], clipped_crop_coordinates=[],
                      padding={}, crop_path='', valid=False, rejection_reason='',
                      interfering_detection_indices=[], interfering_detections=[])
        try:
            if mode == 'bbox':
                coords, clipped, padding = bbox_geometry(detection[:4], image.shape)
            else:
                coords, clipped, padding = crop_geometry(detection[:4], config['crop_size'], image.shape)
            record.update(crop_coordinates=coords, clipped_crop_coordinates=clipped, padding=padding)
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                raise ValueError('empty_crop')
            overlap_region = detection[:4] if mode == 'bbox' else coords
            interfering = [j for j, other in enumerate(detections)
                           if j != index and intersects(overlap_region, other[:4])]
            record['interfering_detection_indices'] = interfering
            record['interfering_detections'] = [dict(index=j, bbox=detections[j, :4].tolist(),
                                                    class_id=int(detections[j, 5]),
                                                    confidence=float(detections[j, 4])) for j in interfering]
            reject_overlap = config.get('reject_bbox_overlap', True) if mode == 'bbox' else config['reject_other_objects']
            if interfering and reject_overlap:
                raise ValueError('bbox_overlaps_other_detection' if mode == 'bbox'
                                 else 'crop_intersects_other_detection')
            crop = materialize_crop(image, record, config['padding_value'])
            if save_crops:
                destination = output / 'candidates' / f'{instance}.png'
                write_image(destination, crop)
                record['crop_path'] = str(destination)
            record['valid'] = True
        except ValueError as error:
            record['rejection_reason'] = str(error)
        records.append(record)
    return records

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


def device_for(name):
    device = torch.device(name)
    if device.type == 'cuda' and (not torch.cuda.is_available() or
                                 (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f'Configured device {name} is unavailable; select cpu in config.yaml.')
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices are cpu and cuda:N.')
    return device


def backbone_settings(config):
    name = config.get('backbone', 'normal')
    if name not in ('normal', 'satellite'):
        raise ValueError('dino.backbone must be normal or satellite.')
    if name == 'normal':
        settings = config.get('normal', config)
        if settings.get('model', 'dinov2_vitb14') != 'dinov2_vitb14':
            raise ValueError('The normal profile supports dinov2_vitb14.')
        return dict(name=name, settings=settings, model='dinov2_vitb14', patch_size=14,
                    dimension=768, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    settings = config.get('satellite', {})
    model = 'facebook/dinov3-vitl16-pretrain-sat493m'
    if settings.get('model', model) != model:
        raise ValueError(f'The satellite profile supports {model}.')
    return dict(name=name, settings=settings, model=model, patch_size=16,
                dimension=1024, mean=[0.430, 0.411, 0.296], std=[0.213, 0.156, 0.143])


def input_shape(value, patch_size=14):
    shape = [value, value] if isinstance(value, int) and not isinstance(value, bool) else value
    if (not isinstance(shape, (list, tuple)) or len(shape) != 2 or
            any(isinstance(v, bool) or not isinstance(v, int) or v < 1 or v % patch_size for v in shape)):
        raise ValueError(f'dino.input_size must be a positive multiple of {patch_size} or '
                         f'[height, width], each divisible by {patch_size} for the selected backbone.')
    return list(shape)


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    for section in ('detector', 'candidates', 'dino', 'clustering', 'visualization', 'output'):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f'Missing configuration section: {section}')
    cfg.setdefault('analyze_dataset', False)
    if not isinstance(cfg['analyze_dataset'], bool):
        raise ValueError('analyze_dataset must be true or false.')
    if cfg['analyze_dataset']:
        if not cfg.get('dataset_path'):
            raise ValueError('dataset_path is required when analyze_dataset is true.')
        cfg['dataset_path'] = (path.parent / Path(cfg['dataset_path']).expanduser()).resolve()
        if not cfg['dataset_path'].is_dir():
            raise ValueError('dataset_path directory does not exist.')
    for section, keys in {
        'detector': ('workspace', 'weights', 'image_dir'),
        'dino': ('repository', 'weights', 'hub_directory'),
        'output': ('directory',),
    }.items():
        for key in keys:
            value = cfg[section].get(key)
            if value is not None:
                cfg[section][key] = (path.parent / Path(value).expanduser()).resolve()
    for profile, keys in {'normal': ('repository', 'weights'),
                          'satellite': ('local_directory', 'cache_directory')}.items():
        for key in keys:
            value = cfg['dino'].get(profile, {}).get(key)
            if value is not None:
                cfg['dino'][profile][key] = (path.parent / Path(value).expanduser()).resolve()
    if not cfg['analyze_dataset']:
        for key in ('workspace', 'image_dir'):
            if not cfg['detector'][key].is_dir():
                raise ValueError(f'Detector {key} directory does not exist.')
        if not cfg['detector']['weights'].is_file():
            raise ValueError('Detector checkpoint does not exist.')
    backbone = backbone_settings(cfg['dino'])
    settings = backbone['settings']
    for key in ('repository', 'weights', 'local_directory'):
        value = settings.get(key)
        if value is not None and not value.exists():
            raise ValueError(f'DINO {key} does not exist: {value}')
    for section, key in (('detector', 'image_size'), ('candidates', 'crop_size'),
                         ('dino', 'batch_size'),
                         ('clustering', 'k'), ('clustering', 'n_init'),
                         ('visualization', 'dpi'), ('visualization', 'gallery_columns'),
                         ('visualization', 'gallery_page_size')):
        if not isinstance(cfg[section][key], int) or cfg[section][key] < 1:
            raise ValueError(f'{section}.{key} must be a positive integer.')
    input_shape(cfg['dino']['input_size'], backbone['patch_size'])
    cfg['candidates'].setdefault('crop_mode', 'contextual')
    cfg['candidates'].setdefault('reject_bbox_overlap', True)
    if not isinstance(cfg['candidates']['reject_bbox_overlap'], bool):
        raise ValueError('candidates.reject_bbox_overlap must be true or false.')
    if cfg['candidates']['crop_mode'] not in ('contextual', 'bbox'):
        raise ValueError('candidates.crop_mode must be contextual or bbox.')
    if cfg['dino']['embedding_type'] not in ('cls', 'mean', 'cls_mean'):
        raise ValueError('embedding_type must be cls, mean, or cls_mean.')
    for key in ('confidence_threshold', 'iou_threshold'):
        if not 0 <= cfg['detector'][key] <= 1:
            raise ValueError(f'{key} must be in [0, 1].')
    limit = cfg['detector'].get('max_images')
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError('max_images must be null or a positive integer.')
    if cfg['visualization']['tsne_perplexity'] <= 0:
        raise ValueError('tsne_perplexity must be positive.')
    if not 0 <= cfg['candidates']['padding_value'] <= 255:
        raise ValueError('padding_value must be in [0, 255].')
    for section in ('detector', 'dino'):
        if section == 'detector' and cfg['analyze_dataset']:
            continue
        device_for(cfg[section]['device'])
    output = cfg['output']['directory']
    workspace = cfg['detector']['workspace']
    if output == workspace or workspace in output.parents:
        raise ValueError('Output must be outside the detector workspace.')
    if output == path.parent or output in path.parent.parents:
        raise ValueError('Output must be a dedicated result directory.')
    if cfg['analyze_dataset'] and (output == cfg['dataset_path'] or cfg['dataset_path'] in output.parents):
        raise ValueError('Output must be outside the dataset directory.')
    return cfg


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f'Cannot encode image: {path}')
    encoded.tofile(str(path))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                             for k, v in row.items() if k in fields})


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')

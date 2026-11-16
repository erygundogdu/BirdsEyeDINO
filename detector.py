import importlib
import sys
from pathlib import Path

import numpy as np
import torch

from runtime import device_for


class Detector:
    def __init__(self, config):
        self.config = config
        self.device = device_for(config['device'])
        workspace = config['workspace']
        sys.dont_write_bytecode = True
        for name in ('models', 'utils'):
            module = sys.modules.get(name)
            if module is not None and workspace not in Path(module.__file__).resolve().parents:
                raise RuntimeError(f'{name} already imported from another repository; use a fresh process.')
        sys.path.insert(0, str(workspace))
        experimental = importlib.import_module('models.experimental')
        general = importlib.import_module('utils.general')
        datasets = importlib.import_module('utils.datasets')
        self.letterbox = datasets.letterbox
        self.nms = general.non_max_suppression
        self.scale_coords = general.scale_coords
        self.formats = set('.' + s for s in datasets.img_formats)
        self.model = experimental.attempt_load(str(config['weights']), map_location=self.device)
        common = importlib.import_module('models.common')
        self.compatibility_repairs = []
        for name, module in self.model.named_modules():
            if type(module).__name__ == 'SAM' and not hasattr(module, 'max_kernel'):
                if not all(hasattr(module, key) for key in ('max_like', 'avg_w', 'conv', 'act')):
                    raise RuntimeError(f'Unsupported historical SAM checkpoint layout at {name}.')
                if module.max_like is None or module.avg_w is None or module.max_like.shape != module.avg_w.shape:
                    raise RuntimeError(f'Incomplete historical SAM parameters at {name}.')
                module.__class__ = common.SpatialAttention1
                self.compatibility_repairs.append(f'{name}: SAM -> SpatialAttention1 (max_like/avg_w checkpoint)')
        self.model.eval().requires_grad_(False)
        self.stride = int(self.model.stride.max())
        self.size = general.check_img_size(config['image_size'], s=self.stride)
        self.attention_modules = sorted({type(m).__name__ for m in self.model.modules()
                                         if 'attention' in type(m).__name__.lower() or type(m).__name__.lower().startswith('sam')})
        print(f'Detector attention modules: {self.attention_modules}', flush=True)

    def image_paths(self):
        root = self.config['image_dir']
        paths = sorted(p for p in (root.rglob('*') if self.config.get('recursive', True) else root.glob('*'))
                       if p.is_file() and p.suffix.lower() in self.formats)
        limit = self.config.get('max_images')
        return paths[:limit] if limit else paths

    @torch.inference_mode()
    def predict(self, original):
        padded, ratio, pad = self.letterbox(original, self.size, stride=self.stride)
        rgb = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
        tensor = torch.from_numpy(rgb).to(self.device).float().unsqueeze(0) / 255.0
        prediction = self.model(tensor, augment=False)[0]
        detections = self.nms(prediction, self.config['confidence_threshold'],
                              self.config['iou_threshold'], classes=None, agnostic=False)[0]
        if len(detections):
            detections[:, :4] = self.scale_coords(tensor.shape[2:], detections[:, :4],
                                                 original.shape, ratio_pad=(ratio, pad))
        return detections.detach().cpu().numpy().astype(np.float32)

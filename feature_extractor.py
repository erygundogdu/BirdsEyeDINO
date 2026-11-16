from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from candidate_extractor import materialize_crop
from runtime import backbone_settings, device_for, input_shape, read_image


class FeatureExtractor:
    @staticmethod
    def prepare(config):
        spec = backbone_settings(config)
        if spec['name'] == 'normal':
            return
        settings = spec['settings']
        try:
            from huggingface_hub import snapshot_download
            from transformers import DINOv3ViTModel
            if settings.get('local_directory'):
                source = settings['local_directory']
                source_path = Path(source)
                if not (source_path / 'config.json').is_file() or not any(source_path.glob('*.safetensors')):
                    raise ValueError('Local satellite directory needs config.json and safetensors weights.')
            else:
                source = snapshot_download(
                    repo_id=spec['model'], cache_dir=str(settings['cache_directory']),
                    local_files_only=settings.get('local_files_only', False),
                    allow_patterns=['config.json', '*.safetensors', '*.safetensors.index.json'])
            config['_satellite_source'] = str(source)
        except (ImportError, OSError, ValueError) as error:
            raise RuntimeError(
                'Satellite DINOv3 is unavailable. Install requirements.txt, accept access at '
                'https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m and run '
                '`hf auth login`, or set dino.satellite.local_directory to a downloaded '
                'Hugging Face model folder containing config.json and safetensors weights. '
                'No detector inference has been started.') from error

    def __init__(self, config):
        self.config = config
        spec = backbone_settings(config)
        self.device = device_for(config['device'])
        if spec['name'] == 'satellite':
            if '_satellite_source' not in config:
                self.prepare(config)
            from transformers import DINOv3ViTModel
            self.model = DINOv3ViTModel.from_pretrained(config['_satellite_source'], local_files_only=True)
            if self.model.config.hidden_size != spec['dimension'] or self.model.config.patch_size != spec['patch_size']:
                raise ValueError('The local satellite checkpoint is not the expected DINOv3 ViT-L/16 architecture.')
        else:
            self.model = self._load_normal(config, spec['settings'])
        self.model.to(self.device).eval().requires_grad_(False)
        self.dimension = spec['dimension'] * (2 if config['embedding_type'] == 'cls_mean' else 1)
        self.shapes = {}

    @staticmethod
    def _load_normal(config, settings):
        torch.hub.set_dir(str(config['hub_directory']))
        repository = settings.get('repository')
        if repository:
            model = torch.hub.load(str(repository), 'dinov2_vitb14', source='local', pretrained=False)
        else:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14',
                                        pretrained=False, trust_repo=True)
        weights = settings.get('weights')
        if weights:
            state = torch.load(str(weights), map_location='cpu', weights_only=True)
        else:
            state = torch.hub.load_state_dict_from_url(
                'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
                map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        return model

    def preprocess(self, crop):
        spec = backbone_settings(self.config)
        tensor = TF.to_tensor(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        interpolation = InterpolationMode.BILINEAR if spec['name'] == 'satellite' else InterpolationMode.BICUBIC
        tensor = TF.resize(tensor, input_shape(self.config['input_size'], spec['patch_size']),
                           interpolation=interpolation, antialias=True)
        return TF.normalize(tensor, spec['mean'], spec['std'])

    @torch.inference_mode()
    def encode(self, batch):
        if backbone_settings(self.config)['name'] == 'satellite':
            tokens = self.model(pixel_values=batch.to(self.device)).last_hidden_state
            cls = tokens[:, 0]
            patches = tokens[:, 1 + self.model.config.num_register_tokens:]
        else:
            features = self.model.forward_features(batch.to(self.device))
            cls, patches = features['x_norm_clstoken'], features['x_norm_patchtokens']
        mean = patches.mean(dim=1)
        mode = self.config['embedding_type']
        embedding = cls if mode == 'cls' else mean if mode == 'mean' else torch.cat((cls, mean), dim=1)
        embedding = F.normalize(embedding.float(), dim=1)
        if not torch.isfinite(embedding).all() or (embedding.norm(dim=1) < 0.99).any():
            raise ValueError('DINO produced nonfinite or zero embeddings.')
        self.shapes = dict(input=list(batch.shape), patch_tokens=list(patches.shape),
                           cls=list(cls.shape), embeddings=list(embedding.shape))
        return embedding.cpu()

    def extract(self, records, padding_value):
        batches, pending = [], []
        for index, record in enumerate(records):
            crop = read_image(record['crop_path']) if record['crop_path'] else materialize_crop(
                read_image(record['image_path']), record, padding_value)
            pending.append(self.preprocess(crop))
            if len(pending) == self.config['batch_size'] or index == len(records) - 1:
                batches.append(self.encode(torch.stack(pending)))
                pending.clear()
        embeddings = torch.cat(batches) if batches else torch.empty((0, self.dimension))
        return embeddings, [r['instance_id'] for r in records]

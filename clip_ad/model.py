"""Assemble upstream blocks and optional additions into one forward pass."""

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from DAPO.adapter import LinearAdapter
from DAPO.learnable_prompt import LearnablePrompt
from DAPO.visual_prompt_tuning import VisualPromptTuning

from .blocks.spectral import SpectralFusion
from .readout import DenseReadout
from .vendor import open_clip


@dataclass
class ModelConfig:
    model_name: str = 'ViT-L-14-336-quickgelu'
    pretrained: str = 'openai'
    image_size: int = 518
    feature_layers: tuple = (6, 12, 18, 24)
    depth: int = 24
    prefix_token_cnt: int = 4
    text_depth: int = 12
    normal_token_cnt: int = 5
    abnormal_token_cnt: int = 5
    prompt_count: int = 10
    layer_token_cnt: int = 4
    temperature: float = 0.07
    spectral: bool = False
    spectral_window: int = 28
    spectral_bins: int = 16
    spectral_hidden_dim: int = 128


class SpatialSpectralCLIP(nn.Module):
    def __init__(self, backbone, visual_prompt, text_prompt, spatial_adapter,
                 feature_layers, fusion=None, readout=None):
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        self.visual_prompt = visual_prompt
        self.text_prompt = text_prompt
        self.spatial_adapter = spatial_adapter
        self.fusion = fusion
        self.readout = readout if readout is not None else DenseReadout()
        self.feature_layers = tuple(feature_layers)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images):
        # Frozen weights still need autograd to train prompts at their inputs.
        image_features, tokens, _ = self.backbone.encode_image(
            [images, self.visual_prompt()], list(self.feature_layers)
        )
        spatial = self.spatial_adapter(tokens)
        patches = self.fusion(spatial, images) if self.fusion is not None else spatial
        prompts, learned_tokens, template_tokens = self.text_prompt(F.normalize(image_features, dim=-1))
        text_features, _ = self.backbone.encode_learn_prompts(prompts, learned_tokens, template_tokens)
        return self.readout(image_features, patches, text_features, images.shape[-2:])

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def block_state_dict(self):
        """Save learned blocks without duplicating frozen CLIP weights."""
        return {
            name: getattr(self, name).state_dict()
            for name in ('visual_prompt', 'text_prompt', 'spatial_adapter', 'fusion')
            if getattr(self, name) is not None
        }

    def load_block_state_dict(self, state):
        expected = set(self.block_state_dict())
        if set(state) != expected:
            raise ValueError(f'Checkpoint blocks {set(state)} do not match model blocks {expected}')
        for name, weights in state.items():
            getattr(self, name).load_state_dict(weights)

    def load_dapo_checkpoint(self, checkpoint):
        """Import original DAPO learned blocks; optional fusion remains initialized."""
        for name, keys in {
            'visual_prompt': ('visual_prompt', 'visual_prompt_state_dict'),
            'text_prompt': ('prompt_learner', 'prompt_learner_state_dict'),
            'spatial_adapter': ('linear_adapter', 'trainable_adapter_state_dict'),
        }.items():
            state = next((checkpoint[key] for key in keys if checkpoint.get(key)), None)
            if state is None:
                raise ValueError(f'Missing DAPO checkpoint block: {name}')
            getattr(self, name).load_state_dict(state)


def build_model(config, defect_names, device='cpu'):
    """Return model and matching preprocessing; normal is always channel zero."""
    if not defect_names or len(set(defect_names)) != len(defect_names):
        raise ValueError('Provide at least one unique defect name')
    if not config.feature_layers:
        raise ValueError('At least one CLIP feature layer is required')
    clip = open_clip()
    from torchvision import transforms
    backbone, _, preprocess = clip.create_model_and_transforms(
        config.model_name, pretrained=config.pretrained, force_image_size=config.image_size
    )
    backbone = backbone.to(device).eval().requires_grad_(False)
    preprocess.transforms[0] = transforms.Resize(
        (config.image_size, config.image_size), interpolation=transforms.InterpolationMode.BICUBIC,
        max_size=None, antialias=None
    )
    preprocess.transforms[1] = transforms.CenterCrop((config.image_size, config.image_size))
    text_prompt = LearnablePrompt(
        clip_model=backbone, prompt_state=list(defect_names),
        normal_token_count=config.normal_token_cnt, abnormal_token_count=config.abnormal_token_cnt,
        prompt_count=config.prompt_count, token_depth=config.text_depth,
        learnable_token_length=config.layer_token_cnt, device=device,
        tokenizer=clip.get_tokenizer(config.model_name)
    )
    # Upstream caches template tensors as plain attributes. Nonpersistent buffers
    # make device transfers safe while allowing unseen candidate names at inference.
    for name, value in list(vars(text_prompt).items()):
        if isinstance(value, torch.Tensor):
            delattr(text_prompt, name)
            text_prompt.register_buffer(name, value, persistent=False)
    visual_prompt = VisualPromptTuning(backbone, config.depth, config.prefix_token_cnt, device)
    model_config = clip.get_model_config(config.model_name)
    dim = model_config['embed_dim']
    adapter = LinearAdapter(model_config['vision_cfg']['width'], dim, len(config.feature_layers))
    fusion = SpectralFusion(
        dim, len(config.feature_layers), config.spectral_window,
        config.spectral_bins, config.spectral_hidden_dim
    ) if config.spectral else None
    model = SpatialSpectralCLIP(
        backbone, visual_prompt, text_prompt, adapter, config.feature_layers,
        fusion=fusion, readout=DenseReadout(config.temperature)
    ).to(device)
    model.config = config
    model.defect_names = list(defect_names)
    return model, preprocess


def checkpoint_payload(model, **extra):
    return dict(format_version=1, model_config=asdict(model.config),
                defect_names=model.defect_names, blocks=model.block_state_dict(), **extra)

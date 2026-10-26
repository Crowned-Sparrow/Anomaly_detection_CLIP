"""Text-similarity readout, independent of feature extraction and losses."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class DenseReadout(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        self.temperature = temperature

    def forward(self, image_features, patch_features, text_features, output_size):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        binary_text = torch.cat([text_features[:1], F.normalize(text_features[1:].mean(0, keepdim=True), dim=-1)])
        maps, binary_maps, embeddings = [], [], []
        for tokens in patch_features:
            tokens = F.normalize(tokens, dim=-1)
            side = math.isqrt(tokens.shape[1])
            if side * side != tokens.shape[1]:
                raise ValueError('Dense readout requires a square patch grid')
            def upsample(anchors):
                scores = ((tokens @ anchors.T) / self.temperature).softmax(-1)
                scores = scores.transpose(1, 2).reshape(tokens.shape[0], anchors.shape[0], side, side)
                return F.interpolate(scores, size=output_size, mode='bilinear', align_corners=True)
            maps.append(upsample(text_features))
            binary_maps.append(upsample(binary_text))
            embeddings.append(tokens)
        if not maps:
            raise ValueError('At least one feature layer is required')
        return {
            'image_logits': (image_features @ binary_text.T) / self.temperature,
            'layer_maps': maps,
            'type_probabilities': torch.stack(maps).mean(0),
            'binary_probabilities': torch.stack(binary_maps).mean(0),
            'patch_embeddings': embeddings,
            'text_embeddings': text_features,
            'image_embeddings': image_features,
        }

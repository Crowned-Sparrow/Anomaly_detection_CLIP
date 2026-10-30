"""Training-only pooling of patch embeddings using source defect masks."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class MaskedRegionPooling(nn.Module):
    """One region per image and positive type ID; disconnected parts are pooled together.

    Area coverage weights preserve small defects when reducing masks to the patch
    grid. Background is excluded. Returns features, image indices and type IDs.
    """

    def forward(self, patches, type_mask):
        if patches.ndim != 3:
            raise ValueError('Patches must have shape [batch, patches, channels]')
        if type_mask.ndim == 4 and type_mask.shape[1] == 1:
            type_mask = type_mask[:, 0]
        if type_mask.ndim != 3 or type_mask.shape[0] != patches.shape[0]:
            raise ValueError('Type mask must have shape [batch, height, width]')
        side = math.isqrt(patches.shape[1])
        if side == 0 or side * side != patches.shape[1]:
            raise ValueError('Region pooling requires a square patch grid')
        regions, images, labels = [], [], []
        for image in range(patches.shape[0]):
            for label in type_mask[image].unique():
                if label.item() <= 0:
                    continue
                mask = (type_mask[image] == label).float()[None, None]
                weights = F.adaptive_avg_pool2d(mask, (side, side)).flatten().to(patches.dtype)
                regions.append((patches[image] * weights[:, None]).sum(0) / weights.sum())
                images.append(image)
                labels.append(label)
        features = torch.stack(regions) if regions else patches.reshape(-1, patches.shape[-1])[:0]
        return (features, torch.tensor(images, device=patches.device, dtype=torch.long),
                torch.stack(labels).long() if labels else torch.empty(0, device=patches.device, dtype=torch.long))

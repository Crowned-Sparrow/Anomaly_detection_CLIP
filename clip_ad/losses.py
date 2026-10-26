"""Training-only objectives. The model never consumes labels or masks."""

import torch
from torch import nn
from torch.nn import functional as F

from DAPO.loss import FocalLoss, BinaryDiceLoss


class DenseSegmentationLoss(nn.Module):
    """DAPO's per-layer typed Focal + normal Dice + anomaly Dice."""
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = BinaryDiceLoss()

    def forward(self, outputs, targets):
        typed = targets['type_mask']
        if typed.ndim == 3:
            typed = typed.unsqueeze(1)
        binary = (typed > 0).to(outputs['layer_maps'][0].dtype)
        loss = outputs['image_logits'].new_zeros(())
        for probabilities in outputs['layer_maps']:
            loss = loss + self.focal(probabilities, typed)
            loss = loss + self.dice(probabilities[:, :1].contiguous(), 1 - binary)
            loss = loss + self.dice(probabilities[:, 1:].sum(1, keepdim=True), binary)
        return loss


class ImageClassificationLoss(nn.Module):
    def forward(self, outputs, targets):
        return F.cross_entropy(outputs['image_logits'], targets['anomaly'].long())


class LossComposer(nn.Module):
    """Named objectives share (outputs, targets); new objectives need no model edits."""
    def __init__(self, objectives=None, weights=None):
        super().__init__()
        self.objectives = nn.ModuleDict(objectives if objectives is not None else {
            'segmentation': DenseSegmentationLoss(), 'classification': ImageClassificationLoss()
        })
        self.weights = dict(weights if weights is not None else {'segmentation': 4.0, 'classification': 1.0})
        if set(self.objectives) != set(self.weights):
            raise ValueError('Supply exactly one weight per objective')

    def forward(self, outputs, targets):
        values = {name: objective(outputs, targets) for name, objective in self.objectives.items()}
        values['total'] = sum(self.weights[name] * value for name, value in values.items())
        return values


def typed_targets(mask, anomaly, defect_types, label_ids):
    """Convert source binary masks and defect labels to candidate-channel IDs."""
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if len(defect_types) != mask.shape[0] or len(anomaly) != mask.shape[0]:
        raise ValueError('Mask, anomaly and defect-label batch sizes must agree')
    ids = []
    for is_anomalous, name in zip(anomaly, defect_types):
        if not bool(is_anomalous):
            ids.append(0)
        elif name not in label_ids or label_ids[name] <= 0:
            raise ValueError(f'No positive candidate-channel ID for source defect {name!r}')
        else:
            ids.append(label_ids[name])
    ids = torch.tensor(ids, device=mask.device).view(-1, 1, 1, 1)
    # Preserve upstream training's integer conversion before thresholding.
    return {'type_mask': (mask.long() > 0).long() * ids, 'anomaly': anomaly.to(mask.device)}

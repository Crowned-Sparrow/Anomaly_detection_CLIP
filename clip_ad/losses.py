"""Training-only objectives. The model never consumes labels or masks."""

import torch
from torch import nn
from torch.nn import functional as F

from DAPO.loss import FocalLoss, BinaryDiceLoss
from .blocks.region_pooling import MaskedRegionPooling


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


class RegionTextAlignmentLoss(nn.Module):
    """Cosine-softmax classification of defect regions against all text anchors.

    Mean over regions, then feature layers. Normal images contribute no regions;
    normal text remains a negative candidate. This is not a query decoder or an
    implementation of UniSpector's angular-margin objective.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        if not 0 < temperature < float('inf'):
            raise ValueError('Region temperature must be finite and positive')
        self.temperature = temperature
        self.pool = MaskedRegionPooling()

    def forward(self, outputs, targets):
        text = F.normalize(outputs['text_embeddings'].float(), dim=-1)
        patches = outputs['patch_embeddings']
        if not patches:
            raise ValueError('Region alignment requires at least one feature layer')
        losses = []
        for layer in patches:
            regions, _, labels = self.pool(layer.float(), targets['type_mask'])
            if not labels.numel():
                losses.append(layer.sum() * 0 + text.sum() * 0)
                continue
            if labels.max().item() >= text.shape[0]:
                raise ValueError('Region type IDs must index text anchors')
            logits = F.normalize(regions, dim=-1) @ text.T / self.temperature
            losses.append(F.cross_entropy(logits, labels))
        return torch.stack(losses).mean()


def build_loss_composer(experiment):
    """Enable the optional objective only when its configured weight is nonzero."""
    weights = dict(experiment['loss_weights'])
    objectives = {'segmentation': DenseSegmentationLoss(), 'classification': ImageClassificationLoss()}
    region_weight = weights.pop('region_alignment', 0.0)
    if not 0 <= region_weight < float('inf'):
        raise ValueError('Region alignment weight must be finite and nonnegative')
    if region_weight:
        objectives['region_alignment'] = RegionTextAlignmentLoss(
            **experiment.get('region_alignment', {}))
        weights['region_alignment'] = region_weight
    return LossComposer(objectives=objectives, weights=weights)


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

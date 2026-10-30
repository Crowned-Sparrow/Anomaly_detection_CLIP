"""Small synthetic checks, requiring PyTorch but no CLIP weights or datasets."""

import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from DAPO.adapter import LinearAdapter
from DAPO.loss import FocalLoss, BinaryDiceLoss
from clip_ad.blocks.spectral import SpectralFusion
from clip_ad.losses import LossComposer, build_loss_composer, typed_targets
from clip_ad.model import SpatialSpectralCLIP


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8))

    def encode_image(self, inputs, layers):
        image, prompt = inputs
        value = image.mean((1, 2, 3))[:, None, None] + prompt.mean()
        tokens = value.expand(image.shape[0], 17, 8) + self.weight[None, None]
        return tokens[:, 0, :6], [tokens.clone() for _ in layers], None

    def encode_learn_prompts(self, prompt, learned_tokens, template_tokens):
        return prompt + self.weight[:6], None


class FakeVisualPrompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(2, 1, 8))

    def forward(self):
        return self.tokens


class FakeTextPrompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchors = nn.Parameter(torch.randn(3, 6))

    def forward(self, image_features):
        return self.anchors, None, None


def small_model(spectral=False):
    torch.manual_seed(12)
    return SpatialSpectralCLIP(
        FakeBackbone(), FakeVisualPrompt(), FakeTextPrompt(), LinearAdapter(8, 6, 2),
        (1, 2), fusion=SpectralFusion(6, 2, window_size=16) if spectral else None
    )


class ModelLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.image = torch.randn(2, 3, 32, 32)
        mask = torch.zeros(2, 1, 32, 32)
        mask[1, :, 4:12, 4:12] = 1
        self.targets = typed_targets(mask, torch.tensor([0, 1]), ['good', 'scratch'], {'scratch': 2})

    def test_zero_gate_composed_model_matches_spatial_model(self):
        baseline, spectral = small_model(), small_model(True)
        expected, actual = baseline(self.image), spectral(self.image)
        for name in ('image_logits', 'type_probabilities', 'binary_probabilities'):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        self.assertEqual(actual['type_probabilities'].shape, (2, 3, 32, 32))
        torch.testing.assert_close(actual['type_probabilities'].sum(1), torch.ones(2, 32, 32))

    def test_frozen_clip_preserves_prompt_gradient_flow(self):
        model = small_model(True).train()
        self.assertFalse(model.backbone.training)
        losses = LossComposer()(model(self.image), self.targets)
        losses['total'].backward()
        self.assertIsNone(model.backbone.weight.grad)
        self.assertFalse(model.backbone.weight.requires_grad)
        for parameter in (model.visual_prompt.tokens, model.text_prompt.anchors,
                          model.spatial_adapter.fc[0].weight, model.fusion.gates):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_loss_matches_original_dapo_formula(self):
        outputs = small_model()(self.image)
        actual = LossComposer()(outputs, self.targets)
        focal, dice = FocalLoss(), BinaryDiceLoss()
        gt = self.targets['type_mask']
        binary = (gt > 0).float()
        local = 0
        for scores in outputs['layer_maps']:
            local = local + focal(scores, gt)
            local = local + dice(scores[:, :1].contiguous(), 1 - binary)
            local = local + dice(scores[:, 1:].sum(1, keepdim=True), binary)
        image = F.cross_entropy(outputs['image_logits'], self.targets['anomaly'])
        torch.testing.assert_close(actual['segmentation'], local)
        torch.testing.assert_close(actual['classification'], image)
        torch.testing.assert_close(actual['total'], 4 * local + image)

    def test_region_objective_integrates_with_composed_model(self):
        model = small_model(True)
        criterion = build_loss_composer({'loss_weights': {
            'segmentation': 4., 'classification': 1., 'region_alignment': 0.1}})
        values = criterion(model(self.image), self.targets)
        torch.testing.assert_close(values['total'], 4 * values['segmentation']
                                   + values['classification'] + 0.1 * values['region_alignment'])
        values['total'].backward()
        self.assertIsNone(model.backbone.weight.grad)
        for parameter in (model.text_prompt.anchors, model.spatial_adapter.fc[0].weight,
                          model.fusion.gates):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_original_checkpoint_import_and_modular_roundtrip(self):
        baseline, spectral = small_model(), small_model(True)
        original = {'visual_prompt': baseline.visual_prompt.state_dict(),
                    'prompt_learner': baseline.text_prompt.state_dict(),
                    'linear_adapter': baseline.spatial_adapter.state_dict()}
        spectral.load_dapo_checkpoint(original)
        torch.testing.assert_close(spectral(self.image)['type_probabilities'], baseline(self.image)['type_probabilities'])
        with torch.no_grad():
            spectral.fusion.gates.fill_(0.3)
        state = copy.deepcopy(spectral.block_state_dict())
        self.assertNotIn('backbone', state)
        restored = small_model(True)
        restored.load_block_state_dict(state)
        torch.testing.assert_close(restored(self.image)['type_probabilities'], spectral(self.image)['type_probabilities'])
        with self.assertRaisesRegex(ValueError, 'do not match'):
            baseline.load_block_state_dict(state)

    def test_unknown_training_labels_fail(self):
        with self.assertRaisesRegex(ValueError, 'No positive'):
            typed_targets(torch.ones(1, 1, 8, 8), torch.tensor([1]), ['unmapped'], {})
        self.assertEqual(self.targets['type_mask'][0].max().item(), 0)
        self.assertEqual(self.targets['type_mask'][1].max().item(), 2)

    def test_fractional_masks_preserve_upstream_integer_conversion(self):
        mask = torch.tensor([[[[0.0, 0.8, 1.0]]]])
        targets = typed_targets(mask, torch.tensor([1]), ['scratch'], {'scratch': 2})
        torch.testing.assert_close(targets['type_mask'], torch.tensor([[[[0, 0, 2]]]]))


if __name__ == '__main__':
    unittest.main()

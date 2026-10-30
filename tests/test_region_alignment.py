"""Synthetic region checks requiring PyTorch, without datasets or CLIP weights."""

import unittest

import torch
from torch.nn import functional as F

from clip_ad.blocks.region_pooling import MaskedRegionPooling
from clip_ad.losses import RegionTextAlignmentLoss, build_loss_composer


class RegionAlignmentTests(unittest.TestCase):
    def test_pooling_uses_coverage_and_keeps_image_and_type_ids(self):
        patches = torch.tensor([[[2., 0.], [0., 4.], [6., 0.], [0., 8.]],
                                [[10., 0.], [0., 12.], [14., 0.], [0., 16.]]])
        mask = torch.zeros(2, 1, 4, 4, dtype=torch.long)
        mask[0, 0, 0, 0] = 1
        mask[0, 0, :2, 2:] = 1
        mask[0, 0, 2:, :2] = 2
        mask[1, 0, 3, 3] = 2
        features, images, labels = MaskedRegionPooling()(patches, mask)
        torch.testing.assert_close(features, torch.tensor([[0.4, 3.2], [6., 0.], [0., 16.]]))
        self.assertEqual(images.tolist(), [0, 0, 1])
        self.assertEqual(labels.tolist(), [1, 2, 2])

    def test_small_region_preserved_and_background_has_no_gradient(self):
        patches = torch.randn(1, 4, 3, requires_grad=True)
        mask = torch.zeros(1, 8, 8, dtype=torch.long)
        mask[0, 0, 0] = 1
        regions, _, labels = MaskedRegionPooling()(patches, mask)
        self.assertEqual(labels.tolist(), [1])
        torch.testing.assert_close(regions[0], patches[0, 0])
        regions.sum().backward()
        torch.testing.assert_close(patches.grad[0, 0], torch.ones(3))
        self.assertEqual(patches.grad[0, 1:].abs().sum().item(), 0)

    def test_alignment_matches_known_cosine_logits_and_layer_mean(self):
        text = torch.eye(3)
        layers = [torch.tensor([[[0., 1., 0.]]]), torch.tensor([[[1., 0., 0.]]])]
        outputs = {'patch_embeddings': layers, 'text_embeddings': text}
        targets = {'type_mask': torch.ones(1, 1, 2, 2, dtype=torch.long)}
        actual = RegionTextAlignmentLoss(temperature=0.5)(outputs, targets)
        expected = (F.cross_entropy(torch.tensor([[0., 2., 0.]]), torch.tensor([1]))
                    + F.cross_entropy(torch.tensor([[2., 0., 0.]]), torch.tensor([1]))) / 2
        torch.testing.assert_close(actual, expected)

    def test_gradient_reaches_patch_and_text_features(self):
        torch.manual_seed(14)
        patches = torch.randn(2, 4, 6, requires_grad=True)
        text = torch.randn(3, 6, requires_grad=True)
        mask = torch.zeros(2, 1, 4, 4, dtype=torch.long)
        mask[1, :, :2, :2] = 2
        loss = RegionTextAlignmentLoss()(
            {'patch_embeddings': [patches], 'text_embeddings': text}, {'type_mask': mask})
        loss.backward()
        for gradient in (patches.grad, text.grad):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertEqual(patches.grad[0].abs().sum().item(), 0)

    def test_empty_regions_return_differentiable_zero(self):
        patches = torch.randn(2, 4, 6, requires_grad=True)
        text = torch.randn(3, 6, requires_grad=True)
        loss = RegionTextAlignmentLoss()(
            {'patch_embeddings': [patches], 'text_embeddings': text},
            {'type_mask': torch.zeros(2, 1, 4, 4, dtype=torch.long)})
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(patches.grad.abs().sum().item(), 0)
        self.assertEqual(text.grad.abs().sum().item(), 0)

    def test_optional_loss_composes_and_defaults_to_disabled(self):
        weights = {'segmentation': 4., 'classification': 1.}
        for optional in ({}, {'region_alignment': 0.}):
            criterion = build_loss_composer({'loss_weights': dict(weights, **optional)})
            self.assertEqual(set(criterion.objectives), set(weights))
            self.assertEqual(criterion.weights, weights)
        criterion = build_loss_composer({
            'loss_weights': dict(weights, region_alignment=0.1),
            'region_alignment': {'temperature': 0.2}})
        self.assertEqual(criterion.weights['region_alignment'], 0.1)
        self.assertEqual(criterion.objectives['region_alignment'].temperature, 0.2)

    def test_invalid_grid_labels_and_settings_fail(self):
        with self.assertRaisesRegex(ValueError, 'square'):
            MaskedRegionPooling()(torch.randn(1, 3, 4), torch.ones(1, 2, 2))
        with self.assertRaisesRegex(ValueError, 'index text'):
            RegionTextAlignmentLoss()(
                {'patch_embeddings': [torch.randn(1, 4, 3)], 'text_embeddings': torch.eye(3)},
                {'type_mask': torch.full((1, 1, 4, 4), 3)})
        for temperature in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                RegionTextAlignmentLoss(temperature)
        with self.assertRaisesRegex(ValueError, 'weight'):
            build_loss_composer({'loss_weights': {'region_alignment': -1}})


if __name__ == '__main__':
    unittest.main()

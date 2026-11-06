"""Upstream vocabulary parity and numerical binary inference regression checks."""

import ast
from pathlib import Path
import unittest

from clip_ad.data import DATASET_NAMES
from clip_ad.evaluation_protocol import DAPO_DEFECT_NAMES, dapo_anomaly_map

try:
    import torch
except ImportError:
    torch = None


class VocabularyTests(unittest.TestCase):
    def test_defaults_match_vendored_dapo_for_every_supported_dataset(self):
        path = Path(__file__).resolve().parents[1] / 'DAPO' / 'test.py'
        tree = ast.parse(path.read_text())
        assignment = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Attribute) and target.attr == 'dataset_configs'
                for target in node.targets))
        upstream = ast.literal_eval(assignment.value)
        self.assertEqual(set(DAPO_DEFECT_NAMES), set(DATASET_NAMES))
        for dataset in DATASET_NAMES:
            with self.subTest(dataset=dataset):
                self.assertEqual(list(DAPO_DEFECT_NAMES[dataset]),
                                 upstream[dataset]['prompt_states'])


@unittest.skipIf(torch is None, 'PyTorch required for numerical inference checks')
class PixelScoringTests(unittest.TestCase):
    def test_readout_matches_upstream_binary_inference(self):
        from torch.nn import functional as F
        from clip_ad.readout import DenseReadout

        torch.manual_seed(17)
        images = torch.randn(2, 8)
        patches = [torch.randn(2, 9, 8) for _ in range(4)]
        texts = torch.randn(5, 8)
        outputs = DenseReadout()(images, patches, texts, (8, 8))

        # Independently follow DAPO/test.py forward_pass and train.py's
        # compute_similarity_map, including its channel-last summation order.
        texts = texts / texts.norm(dim=-1, keepdim=True)
        upstream_maps = []
        for patch in patches:
            patch = patch / patch.norm(dim=-1, keepdim=True)
            probabilities = ((patch @ texts.T) / 0.07).softmax(dim=-1)
            grid = probabilities.reshape(2, 3, 3, 5).permute(0, 3, 1, 2)
            maps = F.interpolate(grid, (8, 8), mode='bilinear', align_corners=True)
            upstream_maps.append(maps.permute(0, 2, 3, 1)[..., 1:].sum(dim=-1))
        expected = torch.stack(upstream_maps).sum(dim=0)
        actual = dapo_anomaly_map(outputs)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (2, 8, 8))
        # Guard against accidentally restoring the former two-anchor readout.
        self.assertFalse(torch.allclose(actual / len(patches),
                                        outputs['binary_probabilities'][:, 1]))

        binary_text = torch.cat([texts[:1], F.normalize(texts[1:].mean(0, keepdim=True), dim=-1)])
        expected_image = (F.normalize(images, dim=-1) @ binary_text.T) / 0.07
        torch.testing.assert_close(outputs['image_logits'], expected_image)


if __name__ == '__main__':
    unittest.main()

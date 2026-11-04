"""CPU metric tests: no PyTorch, CLIP weights, or dataset downloads required."""

import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from DAPO.metrics import cal_pro_score
from clip_ad.metrics import binary_metrics, pixel_aupro, summarize_records


class MetricTests(unittest.TestCase):
    def test_binary_metrics_perfect_and_tied_predictions(self):
        metrics = binary_metrics([0, 1, 1], [0.1, 0.8, 0.9])
        self.assertEqual(metrics, dict(auroc=1., ap=1., f1_max=1.))
        tied = binary_metrics([0, 1, 1], [0.5, 0.5, 0.5])
        self.assertAlmostEqual(tied['f1_max'], 0.8)
        self.assertEqual(tied['auroc'], 0.5)

    def test_existing_auroc_and_ap_are_unchanged(self):
        truth = np.array([[0, 1], [1, 0]])
        scores = np.array([[0.8, 0.7], [0.4, 0.1]])
        result = binary_metrics(truth, scores)
        self.assertEqual(result['auroc'], roc_auc_score(truth.ravel(), scores.ravel()))
        self.assertEqual(result['ap'], average_precision_score(truth.ravel(), scores.ravel()))

    def test_aupro_matches_upstream_with_different_region_sizes_and_ties(self):
        masks = np.zeros((3, 16, 16), dtype=bool)
        masks[0, 0, 0] = True
        masks[0, 4:9, 4:9] = True
        masks[1, 1:4, 1:8] = True
        # DAPO uses full connectivity: these pixels belong to one component.
        masks[1, 10, 10] = masks[1, 11, 11] = True
        for seed in (1, 2, 3):
            maps = np.random.default_rng(seed).random(masks.shape)
            for scores in (maps, np.round(maps, 2)):
                expected = cal_pro_score(masks, scores)
                self.assertAlmostEqual(pixel_aupro(masks, scores), expected, places=12)

    def test_perfect_region_ranking_has_unit_aupro(self):
        masks = np.zeros((2, 16, 16), dtype=bool)
        masks[0, 2:4, 2:4] = True
        maps = np.linspace(0, 0.7, masks.size).reshape(masks.shape)
        maps[masks] = 0.95
        self.assertAlmostEqual(pixel_aupro(masks, maps), 1.)

    def test_degenerate_inputs_are_null_not_nan(self):
        for truth in ([0, 0], [1, 1], []):
            self.assertEqual(binary_metrics(truth, np.zeros(len(truth))),
                             dict(auroc=None, ap=None, f1_max=None))
        masks = np.zeros((1, 4, 4), dtype=bool)
        scores = np.ones(masks.shape)
        self.assertIsNone(pixel_aupro(masks, scores))
        masks[0, 0, 0] = True
        self.assertIsNone(pixel_aupro(masks, scores))
        # Two-valued maps have only zero FPR among the retained thresholds.
        self.assertIsNone(pixel_aupro(masks, masks.astype(float)))
        self.assertIsNone(pixel_aupro(np.ones_like(masks), scores))

    def test_macro_averages_categories_and_excludes_nulls(self):
        masks = np.zeros((2, 16, 16), dtype=bool)
        masks[1, 3:5, 3:5] = True
        maps = np.linspace(0, 0.7, masks.size).reshape(masks.shape)
        maps[masks] = 0.95
        records = {
            'perfect': dict(labels=[0, 1], scores=[0.1, 0.9], masks=masks, maps=maps),
            'reverse': dict(labels=[0, 1], scores=[0.9, 0.1], masks=masks, maps=maps),
            'normal_only': dict(labels=[0], scores=[0.5], masks=masks[:1], maps=maps[:1]),
        }
        categories, macro = summarize_records(records)
        self.assertIsNone(categories['normal_only']['pixel']['aupro'])
        self.assertEqual(macro['image']['auroc'], 0.5)
        self.assertAlmostEqual(macro['image']['f1_max'], (1 + 2 / 3) / 2)
        self.assertAlmostEqual(macro['pixel']['aupro'], 1.)
        self.assertEqual(set(macro['pixel']), {'auroc', 'ap', 'f1_max', 'aupro'})

    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):
            binary_metrics([0, 1], [0.5, float('nan')])
        with self.assertRaises(ValueError):
            pixel_aupro(np.zeros((2, 2)), np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, 'empty'):
            summarize_records({})


if __name__ == '__main__':
    unittest.main()

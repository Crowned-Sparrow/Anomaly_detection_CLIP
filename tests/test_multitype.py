"""Regression checks for upstream typed labels, averaging, and shared inference."""

import ast
import json
from pathlib import Path
import tempfile
import unittest

from clip_ad.data import validate_dataset
from clip_ad.multitype import (PROTOCOLS, MultitypeRecords, defect_id, multitype_maps,
                               validate_multitype_partition)

try:
    import numpy as np
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    from clip_ad.metrics import multitype_metrics
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None


class TypedProtocolTests(unittest.TestCase):
    def test_protocol_data_matches_vendored_evaluator(self):
        path = Path(__file__).resolve().parents[1] / 'DAPO' / 'test_multi_defect.py'
        tree = ast.parse(path.read_text())
        products = {n.targets[0].id.removeprefix('product_type2defect_type_'): ast.literal_eval(n.value)
                    for n in tree.body if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id.startswith('product_type2defect_type_')}
        checked = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
                continue
            values = {n.targets[0].id: ast.literal_eval(n.value) for n in node.body
                      if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                      and n.targets[0].id in ('defects', 'gt_defect')}
            if 'gt_defect' not in values:
                continue
            dataset = ast.literal_eval(node.test.comparators[0])
            protocol = PROTOCOLS[dataset.replace('_', '-')]
            self.assertEqual(protocol['defect_names'], values['defects'][1:])
            self.assertEqual(protocol['products'], products[dataset.replace('-', '_')])
            self.assertEqual(protocol['label_aliases'],
                             {key: values['defects'][index] for key, index in values['gt_defect'].items()})
            checked.add(dataset.replace('_', '-'))
        self.assertEqual(checked, set(PROTOCOLS))

    def test_all_bundled_metadata_maps_without_silently_dropping_labels(self):
        root = Path(__file__).resolve().parents[1] / 'DAPO' / 'data'
        for dataset, folder in (('mvtec', 'mvtec'), ('visa', 'visa'), ('mpdd', 'mpdd'),
                                ('mad-real', 'MAD-Real'), ('mad-sim', 'MAD-Sim'), ('real-iad', 'Real-IAD')):
            with self.subTest(dataset=dataset):
                validate_multitype_partition(dataset, validate_dataset(root / folder, dataset))

    def test_aliases_and_invalid_labels(self):
        self.assertEqual(defect_id('mpdd', 'metal_plate', 'major_rust', 1), 1)
        self.assertEqual(defect_id('mad-real', 'Bear', 'stains', 1), 1)
        self.assertEqual(defect_id('visa', 'candle', 'weird wick', 1), 2)
        self.assertEqual(defect_id('mpdd', 'metal_plate', 'good', 0), 0)
        with self.assertRaisesRegex(ValueError, 'Cannot map'):
            defect_id('mpdd', 'metal_plate', 'hole', 1)
        with self.assertRaisesRegex(ValueError, 'use --tasks binary'):
            validate_multitype_partition('mvtec_loco', {'breakfast_box': []})


@unittest.skipIf(np is None, 'NumPy, SciPy and sklearn required')
class TypedMetricTests(unittest.TestCase):
    def test_metrics_match_sklearn_including_normal_and_absent_classes(self):
        for truth, scores in (
            (np.array([0, 0, 1, 1]), np.array([[.8, .2], [.7, .3], [.4, .6], [.1, .9]])),
            (np.array([0, 1, 2, 0]), np.array([[.8, .1, .1], [.3, .2, .5], [.1, .6, .3], [.7, .2, .1]])),
            (np.array([0, 1, 0, 1]), np.array([[.7, .3, 0], [.2, .8, 0], [.6, .4, 0], [.3, .7, 0]])),
        ):
            labels = [str(i) for i in range(scores.shape[1])]
            actual, per_class = multitype_metrics(truth, scores, labels)
            one_hot = np.eye(len(labels))[truth]
            self.assertAlmostEqual(actual['ap'], average_precision_score(one_hot, scores, average='macro'))
            self.assertAlmostEqual(actual['f1'], f1_score(truth, scores.argmax(1), average='macro'))
            valid = [i for i in range(len(labels)) if 0 < one_hot[:, i].sum() < len(truth)]
            expected_auc = np.mean([roc_auc_score(one_hot[:, i], scores[:, i]) for i in valid])
            self.assertAlmostEqual(actual['auroc'], expected_auc)
            self.assertEqual(set(per_class), set(labels))

    def test_disk_records_average_products_equally_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MultitypeRecords(directory, {'connector': 1, 'tubes': 2}, 'mpdd')
            truth = np.array([[0, 1]], dtype=np.uint8)
            correct = np.array([[[.9, .1], [.1, .9]]])
            store.append('connector', truth, correct)
            for _ in range(2):
                store.append('tubes', truth, correct[..., ::-1])
            result = store.summarize()
            self.assertEqual(result['macro']['pixel']['auroc'], .5)
            self.assertEqual(result['macro']['pixel']['f1'], .5)
            store.close()
            self.assertFalse(store.records)


@unittest.skipIf(torch is None, 'PyTorch required')
class TypedReadoutTests(unittest.TestCase):
    def test_product_softmax_matches_upstream_and_shares_binary_features(self):
        from torch.nn import functional as F
        from clip_ad.evaluation_protocol import dapo_anomaly_map
        from clip_ad.readout import DenseReadout

        torch.manual_seed(9)
        images, texts = torch.randn(1, 8), torch.randn(6, 8)
        patches = [torch.randn(1, 9, 8) for _ in range(4)]
        readout = DenseReadout()
        outputs = readout(images, patches, texts, (8, 8))
        indices = [0, 3, 1]
        actual = multitype_maps(outputs, indices, (8, 8), .07)
        anchors = F.normalize(texts[indices], dim=-1)
        expected = []
        for patch in patches:
            scores = ((F.normalize(patch, dim=-1) @ anchors.T) / .07).softmax(-1)
            grid = scores.transpose(1, 2).reshape(1, 3, 3, 3)
            expected.append(F.interpolate(grid, (8, 8), mode='bilinear', align_corners=True))
        torch.testing.assert_close(actual, torch.stack(expected).sum(0))
        self.assertFalse(torch.allclose(actual / 4, outputs['type_probabilities'][:, indices]))
        binary_indices = [0, 1, 2, 4]
        shared = readout(outputs['image_embeddings'], outputs['patch_embeddings'],
                         outputs['text_embeddings'][binary_indices], (8, 8))
        standalone = readout(images, patches, texts[binary_indices], (8, 8))
        torch.testing.assert_close(dapo_anomaly_map(shared), dapo_anomaly_map(standalone))
        torch.testing.assert_close(shared['image_logits'], standalone['image_logits'])

    @unittest.skipIf(np is None, 'NumPy, SciPy and sklearn required')
    def test_combined_cli_reports_both_tasks_with_one_model_call_per_image(self):
        from unittest.mock import patch
        from clip_ad import evaluate
        from clip_ad.readout import DenseReadout

        class FakeModel(torch.nn.Module):
            def __init__(self, names):
                super().__init__()
                self.names = ['normal'] + names
                self.readout = DenseReadout()
                self.calls = 0

            def load_block_state_dict(self, state):
                pass

            def forward(self, image):
                self.calls += 1
                offset = image.mean()
                global_features = (torch.arange(8).float()[None] + offset).sin()
                patches = [(torch.arange(32).float().reshape(1, 4, 8) / 9 + offset).cos()]
                texts = torch.stack([(torch.arange(8).float() + sum(map(ord, name))).sin()
                                     for name in self.names])
                return self.readout(global_features, patches, texts, (4, 4))

        samples, rows = [], []
        for anomaly in (0, 1):
            mask = torch.zeros(1, 4, 4)
            if anomaly:
                mask[:, 1:3, 1:3] = 1
            label = 'parts_mismatch' if anomaly else 'good'
            samples.append(dict(img=torch.full((3, 4, 4), float(anomaly)), mask=mask,
                                class_name='connector', anomaly=anomaly, defect_type=label))
            rows.append(dict(cls_name='connector', anomaly=anomaly, specie_name=label))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'meta.json').write_text(json.dumps({'test': {'connector': rows}}))
            reports = {}
            for task in ('both', 'binary', 'multi-type'):
                models = []

                def build(config, names, device):
                    model = FakeModel(names)
                    models.append(model)
                    return model, None

                argv = ['evaluate', '--checkpoint', 'unused.pth', '--dataset', 'mpdd',
                        '--data-root', directory, '--output', str(root / f'{task}.json'),
                        '--tasks', task, '--device', 'cpu', '--scratch-dir', directory]
                with patch('sys.argv', argv), patch.object(evaluate.torch, 'load', return_value={
                    'blocks': {}, 'model_config': {'image_size': 4}, 'epoch': 2,
                }), patch.object(evaluate, 'build_model', side_effect=build), patch.object(
                    evaluate, 'build_dataset', return_value=samples), patch('builtins.print'):
                    evaluate.main()
                self.assertEqual(models[0].calls, len(samples))
                reports[task] = json.loads((root / f'{task}.json').read_text())
                self.assertFalse(list(root.glob('clip-ad-eval-*')))
            self.assertEqual(reports['both']['binary'], reports['binary']['binary'])
            self.assertEqual(reports['both']['multi_type'], reports['multi-type']['multi_type'])
            self.assertEqual(reports['both']['macro'], reports['both']['binary']['macro'])
            self.assertTrue(reports['both']['multi_type']['includes_normal'])
            self.assertNotIn('multi_type', reports['binary'])
            self.assertNotIn('binary', reports['multi-type'])


if __name__ == '__main__':
    unittest.main()

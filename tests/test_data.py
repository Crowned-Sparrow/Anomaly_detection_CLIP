"""Metadata checks run without PyTorch or image/model dependencies."""

import json
from pathlib import Path
import tempfile
import unittest

from clip_ad.data import normalize_sample, validate_dataset


class DatasetMetadataTests(unittest.TestCase):
    def test_defect_and_product_aliases_preserve_source_record(self):
        sample = dict(img_path='Parrot/test/scratch/1.jpg', anomaly=1,
                      product_cls='Parrot', defect_cls='scratch')
        result = normalize_sample(sample, 'Parrot')
        self.assertEqual(result['specie_name'], 'scratch')
        self.assertEqual(result['cls_name'], 'Parrot')
        self.assertNotIn('specie_name', sample)
        self.assertNotIn('cls_name', sample)

    def test_canonical_fields_take_precedence(self):
        sample = dict(cls_name='bottle', specie_name='broken', anomaly=1,
                      product_cls='other', defect_cls='scratch')
        self.assertEqual(normalize_sample(sample, 'fallback'), sample)

    def test_missing_anomalous_defect_is_not_silently_labeled_normal(self):
        with self.assertRaisesRegex(ValueError, 'specie_name.*defect_cls'):
            normalize_sample(dict(img_path='bad.png', anomaly=1), 'candle')
        self.assertEqual(normalize_sample(dict(anomaly=0), 'candle')['specie_name'], 'good')

    def test_all_upstream_datasets_normalize_required_fields(self):
        root = Path(__file__).resolve().parents[1] / 'DAPO' / 'data'
        for folder, name in (('visa', 'visa'), ('mpdd', 'mpdd'), ('mvtec', 'mvtec'),
                             ('MAD-Sim', 'mad-sim'), ('MAD-Real', 'mad-real'), ('Real-IAD', 'real-iad')):
            with self.subTest(dataset=name):
                partition = validate_dataset(root / folder, name)
                for class_name, rows in partition.items():
                    for sample in rows:
                        record = normalize_sample(sample, class_name)
                        self.assertTrue(record['cls_name'])
                        self.assertTrue(record['specie_name'])

    def test_rejects_experiment_name_before_opening_metadata(self):
        with self.assertRaisesRegex(ValueError, 'Use --output'):
            validate_dataset('/missing', 'baseline-visa-4')

    def test_missing_test_partition_reports_available_keys(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'meta.json').write_text(json.dumps({'train': {}}))
            with self.assertRaisesRegex(ValueError, r"Found: \['train'\]"):
                validate_dataset(root, 'visa')

    def test_rejects_invalid_partition_shape(self):
        for partition in ([], {'candle': {}}):
            with self.subTest(partition=partition), tempfile.TemporaryDirectory() as root:
                (Path(root) / 'meta.json').write_text(json.dumps({'test': partition}))
                with self.assertRaisesRegex(ValueError, 'map class names to lists'):
                    validate_dataset(root, 'visa')

    def test_flat_and_nested_metadata_return_same_records(self):
        partition = {'candle': [{'img_path': 'candle/test/good/1.png'}]}
        for metadata in (partition, {'train': {}, 'test': partition}):
            with self.subTest(metadata=metadata), tempfile.TemporaryDirectory() as root:
                (Path(root) / 'meta.json').write_text(json.dumps(metadata))
                self.assertEqual(validate_dataset(root, 'visa'), partition)

    def test_accepts_upstream_metadata(self):
        root = Path(__file__).resolve().parents[1] / 'DAPO' / 'data'
        for name in ('visa', 'mpdd', 'mvtec'):
            with self.subTest(dataset=name):
                validate_dataset(root / name, name)

"""Metadata checks run without PyTorch or image/model dependencies."""

import json
from pathlib import Path
import tempfile
import unittest

from clip_ad.data import validate_dataset


class DatasetMetadataTests(unittest.TestCase):
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

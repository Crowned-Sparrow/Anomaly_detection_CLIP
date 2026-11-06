"""DAPO multi-type vocabulary, label mapping, and disk-backed predictions."""

import json
from pathlib import Path

from .data import normalize_sample


PROTOCOLS = json.loads(Path(__file__).with_name('dapo_multitype.json').read_text())


def product_labels(dataset, product):
    if dataset not in PROTOCOLS:
        raise ValueError(f'No DAPO multi-type protocol for {dataset!r}; use --tasks binary')
    try:
        return PROTOCOLS[dataset]['products'][product]
    except KeyError:
        raise ValueError(f'Unknown DAPO product {product!r} for {dataset!r}') from None


def defect_id(dataset, product, label, anomaly):
    labels = product_labels(dataset, product)
    if not anomaly:
        return 0
    aliases = PROTOCOLS[dataset]['label_aliases']
    canonical = aliases.get(label)
    if canonical is None:
        # Accept canonical labels and case variants from canonical data exports.
        matches = {value for key, value in aliases.items()
                   if key.casefold() == label.casefold() or value.casefold() == label.casefold()}
        if len(matches) == 1:
            canonical = matches.pop()
    if canonical not in labels[1:]:
        raise ValueError(f'Cannot map anomalous label {label!r} for {dataset}/{product}; '
                         f'expected one of {labels[1:]} or its DAPO aliases')
    return labels.index(canonical)


def validate_multitype_partition(dataset, partition):
    """Reject unsupported products and labels before loading model weights."""
    for product, rows in partition.items():
        product_labels(dataset, product)
        for sample in rows:
            record = normalize_sample(sample, product)
            if record['cls_name'] != product:
                raise ValueError(f'Metadata product {product!r} conflicts with {record["cls_name"]!r}')
            defect_id(dataset, product, record['specie_name'], record['anomaly'])


def multitype_maps(outputs, indices, output_size, temperature):
    """Product-specific softmax BEFORE upsampling, then sum feature layers.

    Selecting channels from a dataset-wide probability map would retain the
    wrong softmax denominator and would not reproduce test_multi_defect.py.
    """
    import torch
    from .readout import DenseReadout

    readout = DenseReadout(temperature)
    typed = readout(outputs['image_embeddings'], outputs['patch_embeddings'],
                    outputs['text_embeddings'][indices], output_size)
    return torch.stack(typed['layer_maps']).sum(dim=0)


class MultitypeRecords:
    """Store dense typed maps on temporary disk rather than all in RAM."""

    def __init__(self, directory, counts, dataset):
        self.directory = Path(directory)
        self.counts = counts
        self.dataset = dataset
        self.records = {}

    def append(self, product, truth, scores):
        import numpy as np

        if product not in self.records:
            index = len(self.records)
            shape = (self.counts[product], *truth.shape)
            masks = np.lib.format.open_memmap(self.directory / f'{index}-truth.npy',
                                            mode='w+', dtype=np.uint8, shape=shape)
            maps = np.lib.format.open_memmap(self.directory / f'{index}-scores.npy',
                                           mode='w+', dtype=np.float32,
                                           shape=(*shape, len(product_labels(self.dataset, product))))
            self.records[product] = dict(masks=masks, maps=maps, count=0)
        record = self.records[product]
        index = record['count']
        record['masks'][index] = truth
        record['maps'][index] = scores
        record['count'] += 1

    def summarize(self):
        from .metrics import multitype_metrics

        classes = {}
        for product, record in self.records.items():
            if record['count'] != self.counts[product]:
                raise ValueError(f'Incomplete multi-type predictions for {product}')
            pixel, per_defect = multitype_metrics(
                record['masks'], record['maps'], product_labels(self.dataset, product))
            classes[product] = dict(pixel=pixel, per_defect=per_defect)
        macro = {}
        for metric in ('auroc', 'ap', 'f1'):
            values = [result['pixel'][metric] for result in classes.values()
                      if result['pixel'][metric] is not None]
            macro[metric] = sum(values) / len(values) if values else None
        return dict(classes=classes, macro=dict(pixel=macro))

    def close(self):
        # Release mmap handles before TemporaryDirectory removes the files.
        for record in self.records.values():
            record['masks']._mmap.close()
            record['maps']._mmap.close()
        self.records.clear()

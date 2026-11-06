"""Binary inference protocol from the vendored DAPO/test.py.

Keep these literal target names, including upstream spelling, for reproduction.
They are independent of the vocabulary used to train the shared context tokens.
"""

DAPO_DEFECT_NAMES = {
    'mvtec': (
        'bent', 'broken', 'color', 'combined', 'contamination', 'crack', 'cut',
        'fabric', 'faulty imprint', 'glue', 'hole', 'missing', 'poke', 'rough',
        'scratch', 'squeeze', 'thread', 'liquid', 'misplaced', 'damaged',
    ),
    'visa': (
        'damage', 'scratch', 'breakage', 'burnt', 'irregular wick', 'stuck',
        'crack', 'wrong place', 'partical', 'bubble', 'melded', 'hole', 'melt',
        'bent', 'spot', 'extra', 'chip', 'missing',
    ),
    'mpdd': ('hole', 'scratch', 'bent', 'rust', 'flattening', 'mismatch', 'defective painting'),
    'mvtec_loco': ('misplaced', 'damaged'),
    'mad-real': ('stains', 'missing'),
    'mad-sim': ('missing', 'stains', 'burrs'),
    'real-iad': ('hole', 'deformed', 'scratch', 'damage', 'missing parts', 'contamination'),
}


def dapo_anomaly_map(outputs):
    """Sum upsampled abnormal probabilities over defect channels and layers.

    Each layer uses softmax over normal and every individual defect anchor.
    The two-anchor binary_probabilities output is a different scoring rule.
    Returns [batch, height, width], on the same device as the layer maps.
    """
    import torch

    return torch.stack([layer[:, 1:].sum(dim=1)
                        for layer in outputs['layer_maps']]).sum(dim=0)

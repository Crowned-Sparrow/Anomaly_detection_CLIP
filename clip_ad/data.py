"""Reuse DAPO's metadata-driven dataset without importing its trainer."""

import json
from pathlib import Path


DATASET_NAMES = ('mvtec', 'visa', 'mpdd', 'mad-real', 'mad-sim', 'mvtec_loco', 'real-iad')


def validate_dataset(root, dataset_name):
    """Validate and return class records from nested or flat DAPO metadata."""
    if dataset_name not in DATASET_NAMES:
        raise ValueError(
            f'Unknown dataset {dataset_name!r}. Choose one of: {", ".join(DATASET_NAMES)}. '
            'Use --output for experiment names; --dataset identifies the dataset at --data-root.'
        )
    path = Path(root) / 'meta.json'
    with path.open() as handle:
        metadata = json.load(handle)
    if isinstance(metadata, dict) and 'test' in metadata:
        partition = metadata['test']
    elif isinstance(metadata, dict) and metadata and all(isinstance(rows, list) for rows in metadata.values()):
        # Upstream VisA stores the evaluation classes directly at the top level.
        partition = metadata
    else:
        keys = list(metadata) if isinstance(metadata, dict) else type(metadata).__name__
        raise ValueError(
            f'{path} must contain a "test" partition or flat evaluation class records. '
            f'Found: {keys}. Check the dataset export; expected '
            '{"test": {"class_name": [sample, ...]}} or a flat class-to-list mapping. '
            'Do not substitute a training partition for held-out evaluation.'
        )
    if not isinstance(partition, dict) or any(not isinstance(rows, list) for rows in partition.values()):
        raise ValueError(f'{path}: "test" must map class names to lists of samples')
    return partition


def build_dataset(root, dataset_name, preprocess, image_size):
    partition = validate_dataset(root, dataset_name)
    from .vendor import open_clip
    open_clip()  # DAPO.dataset uses an absolute OpenCLIP import.
    from DAPO.dataset import Dataset, get_class_names
    from torchvision import transforms

    class MetadataDataset(Dataset):
        """Reuse DAPO's sample reader with either metadata layout."""

        def __init__(self, root, transform, target_transform, dataset_name):
            self.dataset_root = root
            self.transform = transform
            self.target_transform = target_transform
            self.dataset_name = dataset_name
            self.k_shot = 0
            self.class_names = list(partition)
            self.class_name = get_class_names(dataset_name)
            self.all_data = [sample for rows in partition.values() for sample in rows]
            self.length = len(self.all_data)

    return MetadataDataset(
        root=root, transform=preprocess,
        target_transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size), transforms.ToTensor()
        ]), dataset_name=dataset_name
    )

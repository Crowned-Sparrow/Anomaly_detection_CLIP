"""Reuse DAPO's metadata-driven dataset without importing its trainer."""


def build_dataset(root, dataset_name, preprocess, image_size):
    from .vendor import open_clip
    open_clip()  # DAPO.dataset uses an absolute OpenCLIP import.
    from DAPO.dataset import Dataset
    from torchvision import transforms
    return Dataset(
        root=root, transform=preprocess,
        target_transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size), transforms.ToTensor()
        ]), dataset_name=dataset_name, mode='test', k_shot=0
    )

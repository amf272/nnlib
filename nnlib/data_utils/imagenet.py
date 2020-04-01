from .base import StandardVisionDataset
from torchvision import transforms, datasets

import torch


class ImageNet(StandardVisionDataset):
    def __init__(self, data_augmentation: bool = False):
        super(ImageNet, self).__init__()
        self.data_augmentation = data_augmentation

    @property
    def dataset_name(self) -> str:
        return "imagenet"

    @property
    def means(self):
        return torch.tensor([0.485, 0.456, 0.406])

    @property
    def stds(self):
        return torch.tensor([0.229, 0.224, 0.225])

    @property
    def train_transforms(self):
        if not self.data_augmentation:
            return self.test_transforms

        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def test_transforms(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    def raw_dataset(self, data_dir: str, download: bool, train: bool, transform):
        return datasets.ImageNet(data_dir, download=download, train=train, transform=transform)
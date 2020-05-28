import torch.utils.data
import torch

from .abstract import StandardVisionDataset
from .base import log_call_parameters

from .aif360_extensions import compas_dataset


# TODO: for now use standard vision dataset, down the line change to standard tabular dataset or something
class Compas(StandardVisionDataset):
    @log_call_parameters
    def __init__(self, **kwargs):
        super(Compas, self).__init__(**kwargs)

    @property
    def dataset_name(self) -> str:
        return "compas"

    @property
    def means(self):
        return torch.tensor([0])

    @property
    def stds(self):
        return torch.tensor([1])

    @property
    def train_transforms(self):
        return []

    @property
    def test_transforms(self):
        return []

    def raw_dataset(self, data_dir: str, download: bool, train: bool, transform):
        return compas_dataset.CompasDataset(data_dir, download=download, train=train)

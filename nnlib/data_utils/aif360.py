import os
from abc import abstractmethod, ABC

import aif360.datasets
import numpy as np
import torch
import torch.utils.data

from .abstract import StandardVisionDataset


class StandardAIF360Dataset(torch.utils.data.Dataset, ABC):

    @abstractmethod
    def raw_data_class(self, root, **kwargs) -> aif360.datasets.StandardDataset:
        """
        Override this with an AIF360 dataset class e.g. aif360.datasets.
        """
        raise NotImplementedError("raw data loader is not implemented")

    def __init__(self, root, train: bool, transform=None, target_transform=None, download: bool = False, seed=6174,
                 **kwargs):
        super(StandardAIF360Dataset, self).__init__()
        raise NotImplementedError("currently AIF360 datasets are not yet implemented")
        self.raw_dataset = self.raw_data_class(root, **kwargs)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return [self.features[idx], self.sensitive_labels[idx]], [self.labels[idx], self.sensitive_labels[idx]]


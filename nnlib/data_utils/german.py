import os

import torch
import torch.utils.data
from torchvision.datasets.utils import download_url
import numpy as np

from .abstract import StandardVisionDataset
from .base import print_loaded_dataset_shapes, log_call_parameters


class GermanDataset(torch.utils.data.Dataset):

    base_folder = "german"

    relevant_files = ["german.data-numeric",
                      'random_indices.txt']

    # define train/test indices
    def __init__(self, root, train: bool = True, transform=None, target_transform=None, download: bool = False,):
        super(GermanDataset, self).__init__()
        assert transform is None and target_transform is None, "transforms not implemented"

        self.root = root
        self.base_dir = os.path.join(self.root, self.base_folder, 'raw')
        data_file = os.path.join(self.base_dir, "german.data-numeric")
        if download:
            self.download()

        data = np.loadtxt(data_file)

        random_indices_file = os.path.join(self.base_dir, "random_indices.txt")

        num_entries, num_features = data.shape
        if not os.path.exists(random_indices_file):
            print(f"{random_indices_file} does not exist, generating random indices")
            rng = np.random.default_rng(seed=6174)
            random_indices = np.arange(num_entries)
            rng.shuffle(random_indices)
            np.savetxt(fname=random_indices_file, X=random_indices, fmt="%d")
        else:
            random_indices = np.loadtxt(random_indices_file, dtype=int)

        # do postprocessing of previous work
        data[:, 24] = data[:, 24] - 1
        index = (data[:, 6] == 1) | (data[:, 6] == 3) | (data[:, 6] == 4)
        data[:, 6] = (index).astype(int)

        if train:
            relevant_indices = random_indices[:700]
        else:
            relevant_indices = random_indices[700:]

        self.features = torch.from_numpy(np.concatenate((data[relevant_indices, 0:8], data[relevant_indices, 9:24]), axis=1)).float()
        self.target_labels = torch.from_numpy(data[relevant_indices, 24]).long()
        self.sensitive_labels = torch.from_numpy(data[relevant_indices, 6]).long()

    def download(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        download_url("https://raw.githubusercontent.com/human-analysis/MaxEnt-ARL/master/data/german/german.data-numeric",
                     self.base_dir)

    def _check_integrity(self):
        if not os.path.exists(os.path.join(self.base_dir, "german.data-numeric")):
            return False
        return True

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return [self.features[idx], self.sensitive_labels[idx]], [self.target_labels[idx], self.sensitive_labels[idx]]


# TODO: for now use standard vision dataset, down the line change to standard tabular dataset or something
class German(StandardVisionDataset):
    @log_call_parameters
    def __init__(self, **kwargs):
        super(German, self).__init__(**kwargs)

    @property
    def dataset_name(self) -> str:
        return "german"

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
        return GermanDataset(data_dir, download=download, train=train)


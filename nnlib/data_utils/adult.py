import os

import torch.utils.data
from torchvision.datasets.utils import download_url
import torch
import numpy as np
from sklearn import preprocessing
import pandas as pd

from .abstract import StandardVisionDataset
from .base import print_loaded_dataset_shapes, log_call_parameters


def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column].astype(str))
    return result, encoders


class AdultDataset(torch.utils.data.Dataset):
    base_folder = 'Adult'

    relevant_files = {"adult.names": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
                      "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                      "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"}

    def __init__(self, root, train: bool = True, download: bool = False):
        self.name = "adult"
        super(AdultDataset, self).__init__()
        self.root = root

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if train:
            data_file = os.path.join(self.base_dir, "adult.data")
        else:
            data_file = os.path.join(self.base_dir, "adult.data")

        raw_data = pd.read_csv(
            data_file,
            names=[
                "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?")
        raw_data.tail()

        raw_data, encoders = number_encode_features(raw_data)
        raw_data = np.asarray(raw_data)
        features = np.concatenate((raw_data[:, 0:9], raw_data[:, 10:14]), axis=1)
        target_labels = raw_data[:, 14]
        sensitive_labels = raw_data[:, 9]

        self.features = torch.from_numpy(features)
        self.target_labels = torch.from_numpy(target_labels)
        self.sensitive_labels = torch.from_numpy(sensitive_labels)

    @property
    def base_dir(self):
        return os.path.join(self.root, self.base_folder, 'raw')

    def download(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        for fname, url in self.relevant_files.items():
            download_url(url, self.base_dir, fname)

    def _check_integrity(self):
        for file_name in self.relevant_files.keys():
            file_path = os.path.join(self.base_dir, file_name)
            if not os.path.exists(file_path):
                return False
        return True

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return [self.features[idx], self.sensitive_labels[idx]], [self.target_labels[idx], self.sensitive_labels[idx]]


# TODO: for now use standard vision dataset, down the line change to standard tabular dataset or something
class Adult(StandardVisionDataset):
    @log_call_parameters
    def __init__(self, **kwargs):
        super(Adult, self).__init__(**kwargs)

    @property
    def dataset_name(self) -> str:
        return "adult"

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
        return AdultDataset(data_dir, download=download, train=train)

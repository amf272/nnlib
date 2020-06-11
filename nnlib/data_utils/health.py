import os

import pandas as pd
import torch
import torch.utils.data
from torchvision.datasets.utils import download_url
import numpy as np

from .abstract import StandardVisionDataset
from .base import print_loaded_dataset_shapes, log_call_parameters


def gather_labels(df):
    labels = []
    for j in range(df.shape[1]):
        if type(df[0, j]) is str:
            labels.append(np.unique(df[:, j]).tolist())
        else:
            labels.append(np.median(df[:, j]))
    return labels


class HealthDataset(torch.utils.data.Dataset):
    base_folder = "health"

    relevant_files = ["health.csv",
                      'random_indices.txt']

    # define train/test indices
    def __init__(self, root, train: bool = True, pct_train=0.8, transform=None, target_transform=None, download: bool = False, ):
        super(HealthDataset, self).__init__()
        assert transform is None and target_transform is None, "transforms not implemented"

        self.root = root
        self.base_dir = os.path.join(self.root, self.base_folder, 'raw')
        data_file = os.path.join(self.base_dir, "health.csv")
        if download:
            self.download()

        raw_df = pd.read_csv(data_file)
        raw_df = raw_df[raw_df['YEAR_t'] == 'Y3']
        sex = raw_df['sexMISS'] == 0
        age = raw_df['age_MISS'] == 0
        raw_df = raw_df.drop(['DaysInHospital', 'MemberID_t', 'YEAR_t'], axis=1)
        raw_df = raw_df[sex & age]
        ages = raw_df[[f'age_{i}5' for i in range(0, 9)]]
        sexs = raw_df[['sexMALE', 'sexFEMALE']]
        charlson = raw_df['CharlsonIndexI_max']

        x = raw_df.drop(
            [f'age_{i}5' for i in range(0, 9)] + ['sexMALE', 'sexFEMALE', 'CharlsonIndexI_max',
                                                  'CharlsonIndexI_min',
                                                  'CharlsonIndexI_ave', 'CharlsonIndexI_range',
                                                  'CharlsonIndexI_stdev',
                                                  'trainset'], axis=1).to_numpy()

        labels = gather_labels(x)
        xs = np.zeros_like(x)
        for i in range(len(labels)):
            xs[:, i] = x[:, i] > labels[i]
        x = xs[:, np.nonzero(np.mean(xs, axis=0) > 0.05)[0]].astype(np.float32)

        # u = np.expand_dims(sexs.to_numpy()[:, 0], 1)
        u = ages.to_numpy().argmax(axis=1)
        # u = np.concatenate([v, u], axis=1).astype(np.float32)
        y = (charlson.to_numpy() > 0).astype(np.float32)
        # from IPython import embed; import sys; embed(); sys.exit(1)
        sensitive_labels = u
        target_labels = y
        features = x

        random_indices_file = os.path.join(self.base_dir, "random_indices.txt")

        num_entries, num_features = features.shape
        if not os.path.exists(random_indices_file):
            print(f"{random_indices_file} does not exist, generating random indices")
            rng = np.random.default_rng(seed=6174)
            random_indices = np.arange(num_entries)
            rng.shuffle(random_indices)
            np.savetxt(fname=random_indices_file, X=random_indices, fmt="%d")
        else:
            random_indices = np.loadtxt(random_indices_file, dtype=int)

        split_location = int(pct_train * len(random_indices))

        if train:
            relevant_indices = random_indices[:split_location]
        else:
            relevant_indices = random_indices[split_location:]

        self.features = torch.from_numpy(features[relevant_indices]).float()
        self.target_labels = torch.from_numpy(target_labels[relevant_indices]).long()
        self.sensitive_labels = torch.from_numpy(sensitive_labels[relevant_indices]).long()

    def download(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        download_url("https://raw.githubusercontent.com/ermongroup/lag-fairness/master/health.csv",
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
class Health(StandardVisionDataset):
    @log_call_parameters
    def __init__(self, **kwargs):
        super(Health, self).__init__(**kwargs)

    @property
    def dataset_name(self) -> str:
        return "health"

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
        return HealthDataset(data_dir, download=download, train=train)

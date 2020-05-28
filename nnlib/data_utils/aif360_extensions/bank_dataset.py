import os

import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive
import torch

from aif360.datasets import StandardDataset


class BankDataset(StandardDataset):
    """Bank marketing Dataset.

    See :file:`aif360/data/raw/bank/README.md`.
    """

    base_folder = 'bank'

    relevant_files = [os.path.join("bank-additional", "bank-additional-full.csv"),
                      "random_indices.txt"]

    def __init__(self, root, train=True, pct_train=0.8, download=True, protected_att_name="age",
                 label_name='y', favorable_classes=['yes'],
                 protected_attribute_names=['age'],
                 privileged_classes=[lambda x: x >= 25],
                 instance_weights_name=None,
                 categorical_features=['job', 'marital', 'education', 'default',
                                       'housing', 'loan', 'contact', 'month', 'day_of_week',
                                       'poutcome'],
                 features_to_keep=[], features_to_drop=[],
                 na_values=["unknown"], custom_preprocessing=None,
                 metadata=None):
        """See :obj:`StandardDataset` for a description of the arguments.

        By default, this code converts the 'age' attribute to a binary value
        where privileged is `age >= 25` and unprivileged is `age < 25` as in
        :obj:`GermanDataset`.
        """

        self.root = root
        self.base_dir = os.path.join(self.root, self.base_folder, 'raw')
        if download:
            self.download()

        df = pd.read_csv(os.path.join(self.base_dir, "bank-additional", "bank-additional-full.csv"),
                         sep=';', na_values=na_values)

        super(BankDataset, self).__init__(
            df=df,
            label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)

        num_entries, num_features = self.features.shape

        random_indices_file = os.path.join(self.base_dir, "random_indices.txt")

        if not os.path.exists(random_indices_file):
            print(f"{random_indices_file} does not exist, generating random indices")
            rng = np.random.default_rng(seed=hash("compas") % 100000)
            random_indices = np.arange(num_entries)
            rng.shuffle(random_indices)
            np.savetxt(fname=random_indices_file, X=random_indices, fmt="%d")
        else:
            random_indices = np.loadtxt(random_indices_file, dtype=int)

        split_location = int(pct_train * len(random_indices))
        if train:
            selected_indices = random_indices[:split_location]
        else:
            selected_indices = random_indices[split_location:]

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        used_data = self.subset(selected_indices)

        target_labels = (used_data.labels == used_data.favorable_label).astype(int).flatten()
        sensitive_col_idx = used_data.protected_attribute_names.index(protected_att_name)
        sensitive_labels = used_data.protected_attributes[:, sensitive_col_idx].flatten()

        self.features = torch.from_numpy(used_data.features).float()
        self.target_labels = torch.from_numpy(target_labels).long()
        self.sensitive_labels = torch.from_numpy(sensitive_labels).long()

    def download(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        if not os.path.exists(os.path.join(self.base_dir, "bank-additional", "bank-additional-full.csv")):
            download_and_extract_archive(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip",
                self.base_dir)

    def _check_integrity(self):
        for file_name in self.relevant_files:
            file_path = os.path.join(self.base_dir, file_name)
            if not os.path.exists(file_path):
                return False
        return True

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return [self.features[idx], self.sensitive_labels[idx]], [self.target_labels[idx],
                                                                  self.sensitive_labels[idx]]

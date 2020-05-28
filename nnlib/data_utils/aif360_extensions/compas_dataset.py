import os

import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_url
import torch

from aif360.datasets import StandardDataset

default_mappings = {
    'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}],
    'protected_attribute_maps': [{0.0: 'Male', 1.0: 'Female'},
                                 {1.0: 'Caucasian', 0.0: 'Not Caucasian'}]
}


def default_preprocessing(df):
    """Perform the same preprocessing as the original analysis:
    https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    """
    return df[(df.days_b_screening_arrest <= 30)
              & (df.days_b_screening_arrest >= -30)
              & (df.is_recid != -1)
              & (df.c_charge_degree != 'O')
              & (df.score_text != 'N/A')]


class CompasDataset(StandardDataset):
    """ProPublica COMPAS Dataset.

    See :file:`aif360/data/raw/compas/README.md`.
    """

    base_folder = 'compas'

    relevant_files = {"compas-scores-two-years.csv":
                          "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two"
                          "-years.csv",
                      "random_indices.txt": None}

    def __init__(self, root, train=True, pct_train=0.8, protected_att_name="race", label_name='two_year_recid',
                 favorable_classes=[0],
                 protected_attribute_names=['sex', 'race'],
                 privileged_classes=[['Female'], ['Caucasian']],
                 instance_weights_name=None,
                 categorical_features=['age_cat', 'c_charge_degree',
                                       'c_charge_desc'],
                 features_to_keep=['sex', 'age', 'age_cat',
                                   'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                                   'priors_count', 'c_charge_degree', 'c_charge_desc',
                                   'two_year_recid'],
                 features_to_drop=[], na_values=[],
                 custom_preprocessing=default_preprocessing,
                 metadata=default_mappings,
                 download=True):
        """See :obj:`StandardDataset` for a description of the arguments.

        Note: The label value 0 in this case is considered favorable (no
        recidivism).

        Examples:
            In some cases, it may be useful to keep track of a mapping from
            `float -> str` for protected attributes and/or labels. If our use
            case differs from the default, we can modify the mapping stored in
            `metadata`:

            >>> label_map = {1.0: 'Did recid.', 0.0: 'No recid.'}
            >>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            >>> cd = CompasDataset(protected_attribute_names=['sex'],
            ... privileged_classes=[['Male']], metadata={'label_map': label_map,
            ... 'protected_attribute_maps': protected_attribute_maps})

            Now this information will stay attached to the dataset and can be
            used for more descriptive visualizations.
        """

        self.root = root
        self.base_dir = os.path.join(self.root, self.base_folder, 'raw')
        if download:
            self.download()

        df = pd.read_csv(os.path.join(self.base_dir, "compas-scores-two-years.csv"),
                         index_col='id', na_values=na_values)

        super(CompasDataset, self).__init__(df=df, label_name=label_name,
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
        for fname, url in self.relevant_files.items():
            if url is not None and not os.path.exists(os.path.join(self.base_dir, fname)):
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
        return [self.features[idx], self.sensitive_labels[idx]], [self.target_labels[idx],
                                                                  self.sensitive_labels[idx]]

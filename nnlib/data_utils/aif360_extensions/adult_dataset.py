import os

import pandas as pd
from torchvision.datasets.utils import download_url

from aif360.datasets import StandardDataset

default_mappings = {
    'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
    'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'},
                                 {1.0: 'Male', 0.0: 'Female'}]
}


class AdultDataset(StandardDataset):
    """Adult Census Income Dataset.

    See :file:`aif360/data/raw/adult/README.md`.
    """

    base_folder = 'Adult'

    relevant_files = {"adult.names": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
                      "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                      "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"}

    def __init__(self, root, split="train", label_name='income-per-year',
                 favorable_classes=['>50K', '>50K.'],
                 protected_attribute_names=['race', 'sex'],
                 privileged_classes=[['White'], ['Male']],
                 instance_weights_name=None,
                 categorical_features=['workclass', 'education',
                                       'marital-status', 'occupation', 'relationship',
                                       'native-country'],
                 features_to_keep=[], features_to_drop=['fnlwgt'],
                 na_values=['?'], custom_preprocessing=None,
                 metadata=default_mappings,
                 download=True):
        """See :obj:`StandardDataset` for a description of the arguments.

        Examples:
            The following will instantiate a dataset which uses the `fnlwgt`
            feature:

            >>> from aif360.datasets import AdultDataset
            >>> ad = AdultDataset(instance_weights_name='fnlwgt',
            ... features_to_drop=[])
            WARNING:root:Missing Data: 3620 rows removed from dataset.
            >>> not np.all(ad.instance_weights == 1.)
            True

            To instantiate a dataset which utilizes only numerical features and
            a single protected attribute, run:

            >>> single_protected = ['sex']
            >>> single_privileged = [['Male']]
            >>> ad = AdultDataset(protected_attribute_names=single_protected,
            ... privileged_classes=single_privileged,
            ... categorical_features=[],
            ... features_to_keep=['age', 'education-num'])
            >>> print(ad.feature_names)
            ['education-num', 'age', 'sex']
            >>> print(ad.label_names)
            ['income-per-year']

            Note: the `protected_attribute_names` and `label_name` are kept even
            if they are not explicitly given in `features_to_keep`.

            In some cases, it may be useful to keep track of a mapping from
            `float -> str` for protected attributes and/or labels. If our use
            case differs from the default, we can modify the mapping stored in
            `metadata`:

            >>> label_map = {1.0: '>50K', 0.0: '<=50K'}
            >>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            >>> ad = AdultDataset(protected_attribute_names=['sex'],
            ... categorical_features=['workclass', 'education', 'marital-status',
            ... 'occupation', 'relationship', 'native-country', 'race'],
            ... privileged_classes=[['Male']], metadata={'label_map': label_map,
            ... 'protected_attribute_maps': protected_attribute_maps})

            Note that we are now adding `race` as a `categorical_features`.
            Now this information will stay attached to the dataset and can be
            used for more descriptive visualizations.
        """

        self.root = root
        self.base_dir = os.path.join(self.root, self.base_folder, 'raw')

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        assert split in ["train", "test", "all"], 'split must be one of "train", "test", "all"'
        train_path = os.path.join(self.base_dir, 'adult.data')
        test_path = os.path.join(self.base_dir, 'adult.test')
        # as given by adult.names
        column_names = ['age', 'workclass', 'fnlwgt', 'education',
                        'education-num', 'marital-status', 'occupation', 'relationship',
                        'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country', 'income-per-year']
        if split in ["train", "all"]:
            df = pd.read_csv(train_path, header=None, names=column_names,
                             skipinitialspace=True, na_values=na_values)
        elif split in ["test", "all"]:
            df = pd.read_csv(test_path, header=0, names=column_names,
                             skipinitialspace=True, na_values=na_values)
        elif split == "all":
            train = pd.read_csv(train_path, header=None, names=column_names,
                                skipinitialspace=True, na_values=na_values)
            test = pd.read_csv(test_path, header=0, names=column_names,
                               skipinitialspace=True, na_values=na_values)
            df = pd.concat([test, train], ignore_index=True)
        else:
            raise ValueError(f"split {split} must be one of train, test, or all")

        super(AdultDataset, self).__init__(df=df, label_name=label_name,
                                           favorable_classes=favorable_classes,
                                           protected_attribute_names=protected_attribute_names,
                                           privileged_classes=privileged_classes,
                                           instance_weights_name=instance_weights_name,
                                           categorical_features=categorical_features,
                                           features_to_keep=features_to_keep,
                                           features_to_drop=features_to_drop, na_values=na_values,
                                           custom_preprocessing=custom_preprocessing, metadata=metadata)

    def download(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        for fname, url in self.relevant_files.items():
            download_url(url, self.base_dir, fname)

    def _check_integrity(self):
        root = self.root
        for file_name in self.relevant_files.keys():
            file_path = os.path.join(root, self.base_folder, file_name)
            if not os.path.exists(file_path):
                return False
        return True

"""
The structure of this file is similar to
https://github.com/GMvandeVen/brain-inspired-replay/blob/master/data/available.py

2021 Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

Some settings for datasets.
"""

from torchvision import datasets, transforms

# Specify available data-sets
AVAILABLE_DATASETS = {
    'ncaltech12': datasets.DatasetFolder,
    'ncaltech256': datasets.DatasetFolder,
}

# Specify available transforms
AVAILABLE_TRANSFORMS = {
    'ncaltech12': [
        transforms.ToTensor(),
    ],
    'ncaltech256': [
        transforms.ToTensor(),
    ],
}

# Specify configurations of available data-sets
DATASET_CONFIGS = {
    'ncaltech12': {'size': 40, 'channels': 1, 'classes': 12},  # TODO
    'ncaltech256': {'size': 40, 'channels': 1, 'classes': 257},  # TODO
}
"""
Constants and methods for manipulating different datasets.
2021 Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).
"""

from enum import Enum

# Number of classes:
NCALTECH12_OUTPUT_CLASSES_NUMBER = 12
NCALTECH256_OUTPUT_CLASSES_NUMBER = 257


class DatasetNames(Enum):
    """
    Enums for the names of datasets.
    """
    NCALTECH12 = "ncaltech12"
    NCALTECH256 = "ncaltech256"


def get_output_classes_number(dataset):
    """
    Returns the number of output classes for the given dataset.
    :param dataset: name of the dataset.
    :return: the number of output classes for the given dataset.
    """
    dataset = dataset.lower()
    if dataset == DatasetNames.NCALTECH12.value:
        return NCALTECH12_OUTPUT_CLASSES_NUMBER
    if dataset == DatasetNames.NCALTECH256.value:
        return NCALTECH256_OUTPUT_CLASSES_NUMBER




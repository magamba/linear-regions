# -*- coding: utf-8 -*-

"""
Counting strategies -- For use with line counting.
"""
from functools import partial

def closed_augmented_path(dataset_split, data_manager):
    """Note: closed paths are generated natively as a Dataset in core/data.py
    """
    if dataset_split == "train":
        dloader = data_manager.dloader
    elif dataset_split == "test":
        dloader = data_manager.tloader
    elif dataset_split == "val":
        dloader = data_manager.vloader
    else:
        raise ValueError("No split named {}.".format(dataset_split))
    return dataset_split, dloader


GEN_STRATEGIES = {
    "closed-path-train": partial(closed_augmented_path, "train"),
    "closed-path-test": partial(closed_augmented_path, "test"),
    "closed-path-val": partial(closed_augmented_path, "val"),
}


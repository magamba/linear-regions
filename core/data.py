# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision import datasets as torch_dsets, transforms as tx
from typing import Tuple, Union
from types import SimpleNamespace
import logging

from core import utils
from core.random_dataset import RANDOM_DATASETS_MAP, RANDOM_DATASETS, RANDOM_DATASETS_INFO_MAP

logger = logging.getLogger(__name__)

DATASETS = ("cifar10", "cifar100") + RANDOM_DATASETS
Datasets = SimpleNamespace(**{ds: ds for ds in DATASETS})

DATASET_INFO_MAP = {
    "cifar10": utils.DatasetInfo("cifar10", (3, 32, 32), 10),
    "cifar100": utils.DatasetInfo("cifar100", (3, 32, 32), 100),
    **RANDOM_DATASETS_INFO_MAP,
}

DatasetInfos = SimpleNamespace(**DATASET_INFO_MAP)

def with_indices(datasetclass):
    """
        Wraps a DataSet class, so that it returns (data, target, index).
    """
    def __getitem__(self, index):
        data, target = datasetclass.__getitem__(self, index)
        return data, target, index
        
    return type(datasetclass.__name__, (datasetclass,), {
        '__getitem__': __getitem__,
    })


class PathTransform:
    """Generate paths close to the support of the data distribution
       with @num_anchor anchor points from a sample @x

    For a torchvision.datasets.vision.VisionDataset, we provide a transform that:
        - Upscales image @x (copying the values at the edges)
        - Copies the image K times
        - Applies a translation to each of the K images
        - Crops the image back to its original size

    @retun path: torch.Tensor of shape [K, C, H, W], denoting the K=num_anchor
                 augmented images that form a path.

    Note: the returned label is a scalar Tensor, which holds for all K
          images, so that batches of N paths correctly contain N labels.

          If needed, target_transform can be passed to replicate a label
          K times.

    This way, a torch.utils.data.DataLoader can return batches of
    [N, K, C, H, W] images.

    Note: this design allows to iterate over a whole Dataset and generate one
          path per image within the Dataset. In order to restrict the number of
          paths returned, one should provide a sampler argument to the
          DataLoader. This is also useful for making sure that the paths
          considered contain samples with noisy and clean labels for instance.
    """

    def __init__(self, input_shape, mean=None, std=None, num_anchor=10, radius=4):
        """Set up path generation
        @param input_shape: tuple shape input images (C, H, W)
        @param mean: list list of per-channel means
        @param std: list of per-channel stds
        @param num_anchor: int number of anchor points to be generated for each path
        @param radius: float radius of the rotation used to generate the anchor points
        Note: the affine translation used on PIL images by default aligns the translation
              to the nearest pixel
        """
        self.mean = mean
        self.std = std
        self.normalize = mean is not None and std is not None
        self.num_anchor = num_anchor
        self.radius = radius
        self.output_shape = torch.Size([num_anchor] + [_ for _ in input_shape])
        self.ts = np.linspace(0, 2 * np.pi, num_anchor, endpoint=False)

    def __call__(self, x):
        """
        @param x: PIL Image to be transformed
        @return torch.Tensor : of shape [K, C, H, W], where K=@self.num_points
                               is the number of anchor points in each path
        """
        output = torch.zeros(self.output_shape)
        x = tx.functional.pad(x, padding=2 * self.radius, padding_mode="edge")

        for k, t in enumerate(self.ts):
            translated_x = tx.functional.affine(
                x,
                angle=0,
                translate=(self.radius * np.cos(t), self.radius * np.sin(t)),
                scale=1,
                shear=0,
            )
            translated_x = tx.functional.center_crop(
                translated_x, output_size=list(self.output_shape[2:])
            )
            output[k] = tx.functional.to_tensor(translated_x)

            if self.normalize:
                output = utils.normalize(output, mean=self.mean, std=self.std)
        return output.type(torch.float64)


class PathTransformOpen(PathTransform):

    def __init__(self, input_shape, mean=None, std=None, num_anchor=10, radius=4):
        super(PathTransformOpen, self).__init__(
            input_shape=input_shape, 
            mean=mean, 
            std=std, 
            num_anchor=num_anchor, 
            radius=radius
        )
        self.ts = np.linspace(0, 1, num_anchor, endpoint=False)
        
    def __call__(self, x):
        """
        @param x: PIL Image to be transformed
        @return torch.Tensor : of shape [K, C, H, W], where K=@self.num_points
                               is the number of anchor points in each path
        """
        output = torch.zeros(self.output_shape)
        x = tx.functional.pad(x, padding=2 * self.radius, padding_mode="edge")
        
        for k, t in enumerate(self.ts):
            s = 0.5 * np.pi * t + (1. - t) * 1.5 * np.pi
            translated_x = tx.functional.affine(
                x,
                angle=0,
                translate=(self.output_shape[-1] / self.num_anchor, self.radius * np.sin(s)),
                scale=1,
                shear=0,
            )
            translated_x = tx.functional.center_crop(
                translated_x, output_size=list(self.output_shape[2:])
            )
            output[k] = tx.functional.to_tensor(translated_x)

            if self.normalize:
                output = utils.normalize(output, mean=self.mean, std=self.std)
        return output.type(get_default_dtype())


def corrupt_labels(dset, noise_percent, seed=None):
    from numpy.random import default_rng
    rng = default_rng(seed)
    num_labels_to_corrupt = int(round(len(dset) * noise_percent))
    if num_labels_to_corrupt == 0:
        return
    if isinstance(dset, Subset):
        all_targets = dset.dataset.targets
        dset.dataset._targets_orig = dset.dataset.targets.copy()
        if isinstance(all_targets, list):
            all_targets = torch.tensor(all_targets)
        targets = all_targets[dset.indices]
    else:
        targets = dset.targets
        dset._targets_orig = dset.targets.copy()
        if isinstance(targets, list):
            targets = torch.tensor(targets)

    num_classes = targets.unique().shape[0]

    noise = torch.zeros_like(targets)
    if num_classes == 2:
        noise[0:num_labels_to_corrupt] = 1
    else:
        noise[0:num_labels_to_corrupt] = torch.from_numpy(
            rng.integers(1, num_classes, (num_labels_to_corrupt,))
        )
    shuffle = torch.from_numpy(rng.permutation(noise.shape[0]))
    noise = noise[shuffle]
    if isinstance(dset, Subset):
        all_noisy_targets = (targets + noise) % num_classes
        if isinstance(dset.dataset.targets, list):
            for idx, noisy_label in enumerate(all_noisy_targets.tolist()):
                dset.dataset.targets[dset.indices[idx]] = noisy_label
        else:
            dset.dataset.targets[dset.indices] = all_noisy_targets
    else:
        dset.targets = (targets + noise) % num_classes


def _create_transforms(normalize, mean, std, **kwargs):
    crop_size = kwargs.pop("crop_size", 0)
    hflip = kwargs.pop("hflip", False)

    transform_funcs = []
    if crop_size > 0:
        transform_funcs.append(tx.RandomCrop(crop_size, padding=4))
    if hflip:
        transform_funcs.append(tx.RandomHorizontalFlip())

    transform_funcs.append(tx.ToTensor())
    if normalize:
        transform_funcs.append(tx.Normalize(mean, std))
    return tx.Compose(transform_funcs)


def create_dataset(
    args,
    train=True,
    normalize=False,
    augment=False,
    subset_pct=None,
    gen_paths=False,
    validation=False,
    override_dset_class=None
):
    if args.data not in DATASETS:
        raise ValueError("{} is not a valid dataset".format(args.data))

    dset = None
    if args.data == Datasets.cifar10:
        if args.l_random_dataset:
            dclass = RANDOM_DATASETS_MAP["random_" + str(args.data)]
        else:
            CIFAR10 = torch_dsets.CIFAR10
        
        if override_dset_class is not None:
            CIFAR10 = override_dset_class(dclass)
        else:
            CIFAR10 = dclass    
        
        if gen_paths:
            if args.l_open_path:
                PathTransformCls = PathTransformOpen
            else:
                PathTransformCls = PathTransform
            transforms = PathTransformCls(
                DatasetInfos.cifar10.input_shape,
                mean=(0.4914, 0.4822, 0.4465) if normalize else None,
                std=(0.2023, 0.1994, 0.2010) if normalize else None,
                num_anchor=args.l_num_anchors,
                radius=args.l_closed_path_radius,
            )
            CIFAR10 = with_indices(CIFAR10)
        elif train and augment:
            transforms = _create_transforms(
                normalize=normalize,
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
                crop_size=32,
                hflip=True,
            )
        else:
            transforms = _create_transforms(
                normalize,
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            )
        dset = CIFAR10(
            args.e_data_dir, transform=transforms, train=train, download=False
        )
    elif args.data == Datasets.cifar100:
        if args.l_random_dataset:
            dclass = RANDOM_DATASETS_MAP["random_" + str(args.data)]
        else:
            CIFAR100 = torch_dsets.CIFAR100
            
        if override_dset_class is not None:
            CIFAR100 = override_dset_class(dclass)
        else:
            CIFAR100 = dclass    
            
        if gen_paths:
            if args.l_open_path:
                PathTransformCls = PathTransformOpen
            else:
                PathTransformCls = PathTransform
            transforms = PathTransformCls(
                DatasetInfos.cifar100.input_shape,
                mean=(0.5071, 0.4865, 0.4409) if normalize else None,
                std=(0.2009, 0.1984, 0.2023) if normalize else None,
                num_anchor=args.l_num_anchors,
                radius=args.l_closed_path_radius,
            )
            CIFAR100 = with_indices(CIFAR100)
        elif train and augment:
            transforms = _create_transforms(
                normalize=normalize,
                mean=(0.5071, 0.4865, 0.4409),
                std=(0.2009, 0.1984, 0.2023),
                crop_size=32,
                hflip=True,
            )
        else:
            transforms = _create_transforms(
                normalize,
                (0.5071, 0.4865, 0.4409),
                (0.2009, 0.1984, 0.2023),
            )
        dset = CIFAR100(
            args.e_data_dir, transform=transforms, train=train, download=False
        )
    if subset_pct is not None and 1 > subset_pct > 0:
        rng = np.random.default_rng(args.data_split_seed)
        shuffle = rng.permutation(len(dset))
        split_index = int(subset_pct * len(dset))
        if validation:
            rand_indices = torch.from_numpy(shuffle)[split_index:]
        else:
            rand_indices = torch.from_numpy(shuffle)[:split_index]
        dset = Subset(dset, rand_indices)
    return dset


@dataclass
class DataManager:
    dset: Dataset
    dloader: DataLoader
    tloader: DataLoader
    vloader: DataLoader
    vset: Dataset
    tset: Dataset


def create_data_manager(
    args,
    noise,
    seed=None,
    normalize=True,
    augment=False,
    train_validation_split=(None, None),
    train_subset_pct=None,
    test_subset_pct=None,
    gen_paths=False,
    override_dset_class=None
):
    if override_dset_class is None and gen_paths:
        override_dset_class=with_indices
    dset = create_dataset(
        args,
        train=True,
        normalize=normalize,
        augment=augment,
        subset_pct=train_subset_pct,
        gen_paths=gen_paths,
        override_dset_class=override_dset_class
    )
    tset = create_dataset(
        args,
        train=False,
        normalize=normalize,
        augment=False,
        subset_pct=train_subset_pct,
        gen_paths=gen_paths,
        override_dset_class=override_dset_class
    )
    vset, vloader = None, None
    if train_validation_split != (None, None):
        vset = create_dataset(
            args,
            train=True,
            normalize=normalize,
            augment=False,
            subset_pct=train_subset_pct,
            gen_paths=gen_paths,
            validation=True,
            override_dset_class=override_dset_class
        )
        if seed is not None:
            torch.manual_seed(seed)
        rng_state = torch.get_rng_state()
        logger.info("Splitting training set into train: {}, val: {}.".format(
                train_validation_split[0], train_validation_split[1]
            )
        )
        _, vset = random_split(
            vset, train_validation_split, generator=torch.Generator("cpu").manual_seed(
                args.data_split_seed
            )
        )
        dset, _ = random_split(
            dset, train_validation_split, generator=torch.Generator("cpu").manual_seed(
                args.data_split_seed
            )
        )
        torch.set_rng_state(rng_state)
    corrupt_labels(dset, noise, args.label_seed)
    num_workers = args.e_workers
    pin_memory=True
    if args.e_device != "cpu" and torch.cuda.device_count() > 1 and gen_paths:
        logger.warn("Setting data-loader workers to 0 to avoid conflict with torch.multiprocessing and GPU workers.")
        num_workers = 0
        pin_memory=False
    
    logger.info("Running with {} cpu workers.".format(num_workers))
    kwargs_train = {
        "batch_size": args.batch_size,
        "num_workers" : num_workers,
        "shuffle": True,
        "pin_memory": pin_memory,
    }
    kwargs_no_train = {
        "batch_size": args.batch_size,
        "num_workers" : num_workers,
        "shuffle": gen_paths,
        "pin_memory": pin_memory,
    }

    if gen_paths:
        logger.info("Shuffling test/val splits. Remember to seed torch right before creating any iterator of your dataloader to ensure paths are visited in the same order.")

    if train_validation_split != (None, None):
        vloader = DataLoader(vset, **kwargs_no_train)
    dloader = DataLoader(dset, **kwargs_train)
    tloader = DataLoader(tset, **kwargs_no_train)

    return DataManager(
        dset,
        dloader,
        tloader,
        vloader,
        vset,
        tset,
    )

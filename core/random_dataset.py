"""
Ablating trajectories

    1. Proximity to support of the data distribution: generate random dataset with CIFAR-10 pixel-wise statistics
    2. Circular trajectories: open-path strategy with corresponding transform

    Compute density and abs_deviation for VGG8 on clean and 20% noisy labels using the three strategies
    Plot the norm-based ECDFs using train data
    
    Compute KS test to check whether the samples from the different trajectories come from the same distribution
    for close-path vs closed-path-random, and closed-path vs open-path.

"""
from PIL import Image
from typing import Any, Callable, Optional, Tuple
import numpy as np
from torchvision.datasets import VisionDataset
from functools import partialmethod

from core.utils import DatasetInfo

def build_dataset(cls, *args, **kwargs):
    class DatasetClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return DatasetClass

class RandomDataset(VisionDataset):
    
    def __init__(
        self,
        train_size : int,
        test_size: int,
        num_classes: int,
        data_shape : Tuple[int],
        train_mean: Tuple[float],
        train_std: Tuple[float],
        data_sample_seed: int,
        download: bool = None,
        train: bool = True,
         *args,
         **kwargs
    ) -> None:
    
        super(RandomDataset, self).__init__(*args, **kwargs)
        self.data_shape = data_shape
        self.data_sample_seed = data_sample_seed
        self.num_classes = num_classes
        self.test_size = test_size
        self.train = train
        self.train_size = train_size
        self.train_mean = np.asarray(train_mean)
        self.train_std = np.asarray(train_std)
        
        if download:
            raise NotImplementedError("Random datasets are generated on the fly and cannot be downloaded.")
            
        self.data: Any = []
        self.targets = []
        
        # init PRNG
        rng = np.random.default_rng(data_sample_seed)
        
        # generate train data
        self.data = rng.uniform(
            low=(self.train_mean - self.train_std), 
            high=(self.train_mean + self.train_std),
            size=(self.train_size * np.prod(self.data_shape[1:]), self.data_shape[0])
        )
        self.data = np.vstack(
            tuple(self.data[:,i].reshape((-1,) + self.data_shape[1:]) for i in range(len(self.train_mean)))
        )
        self.data = self.data.reshape((self.data_shape[0], -1,) + self.data_shape[1:]).transpose(1,2,3,0) # HWC
        
        # generate train labels
        self.targets = list(rng.integers(low=0, high=self.num_classes, size=self.train_size, dtype=np.long))
        
        # generate test data
        if not self.train:
            self.data = rng.uniform(
                low=(self.train_mean - self.train_std), 
                high=(self.train_mean + self.train_std),
                size=(self.test_size * np.prod(self.data_shape[1:]), self.data_shape[0])
            )
            self.data = np.vstack(
                tuple(self.data[:,i].reshape((-1,) + self.data_shape[1:]) for i in range(len(self.train_mean)))
            )
            self.data = self.data.reshape((self.data_shape[0], -1,) + self.data_shape[1:]).transpose(1,2,3,0) # HWC
            self.targets = list(rng.integers(low=0, high=self.num_classes, size=self.test_size, dtype=np.long))
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
            
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img, target = self.data[index], self.targets[index]
        
        img = Image.fromarray((img * 255).astype(np.uint8))
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
        
    def __len__(self) -> int:
        return len(self.data)
    
    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


RANDOM_DATASETS_MAP = {
    "random_mnist": build_dataset(
        RandomDataset, train_size=60000, test_size=10000, num_classes=10, data_shape=(1,28,28), train_mean=(0.1307,), train_std=(0.3081,), data_sample_seed=1234
    ),
    "random_cifar10": build_dataset(
        RandomDataset, train_size=50000, test_size=10000, num_classes=10, data_shape=(3,32,32), train_mean=(0.4914, 0.4822, 0.4465), train_std=(0.2023, 0.1994, 0.2010), data_sample_seed=1234
    ),
    "random_cifar100": build_dataset(
        RandomDataset, train_size=50000, test_size=10000, num_classes=100, data_shape=(3,32,32), train_mean=(0.5071, 0.4865, 0.4409), train_std=(0.2009, 0.1984, 0.2023), data_sample_seed=1234
    ),
}

RANDOM_DATASETS = (
    "random_mnist",
    "random_cifar10",
    "random_cifar100",
)

RANDOM_DATASETS_INFO_MAP = {
    "random_mnist" : DatasetInfo("random_mnist", (1, 28, 28), 10),
    "random_cifar10" :DatasetInfo("random_cifar10", (3, 32, 32), 10),
    "random_cifar100" : DatasetInfo("random_cifar100", (3, 32, 32), 100),
}

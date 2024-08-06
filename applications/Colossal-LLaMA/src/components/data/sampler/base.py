from typing import Iterator
import math
from abc import ABC, abstractmethod

import torch.distributed as dist

from datasets import Dataset as HuggingFaceDataset

from ...utils.generic import get_rank, get_world_size, sync_random_seed


class Sampler(ABC):
    """Abstract base class of data sampler for both distributed & non-distributed training."""

    def __init__(self,
                 dataset: HuggingFaceDataset,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 seed: int = None,
                 process_group: dist.ProcessGroup = None
                 ) -> None:
        self.dataset = dataset
        self.rank = get_rank(group=process_group)
        self.num_replicas = get_world_size(group=process_group)
        self.epoch = 0  # `int`, initial epoch.
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )  # `int`, number of training samples for current process.
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed(seed=42, group=process_group)
        self.seed = seed
        self.process_group = process_group

    @abstractmethod
    def __iter__(self) -> Iterator[int]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        """
        assert isinstance(epoch, int) and epoch >= 0, \
            f"Invalid epoch ({epoch}), it must be a non-negative integer."
        self.epoch = epoch

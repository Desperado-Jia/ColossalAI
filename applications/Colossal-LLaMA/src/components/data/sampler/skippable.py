from typing import Iterator

import math

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from datasets import Dataset as HFDataset

from .base import Sampler


class SkippableSampler(Sampler):
    """A class used to represent the skippable distributed sampler for checkpoint recovery."""

    def __init__(self,
                 dataset: HFDataset,
                 shuffle: bool = True,
                 drop_last: bool = True,
                 seed: int = None,
                 process_group: dist.ProcessGroup = None
                 ) -> None:
        super().__init__(
            dataset=dataset,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            process_group=process_group
        )
        self.num_skipped_samples = 0  # `int`, number of skipped samples for current process.
        # Note that it represent sample, not batch !

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed.
            g = torch.Generator(device="cpu")
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # Add extra samples to make it evenly divisible.
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[: padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[: padding_size]
        else:
            # Remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size, f"Invalid indices length from dataset."

        # Sub-sample
        indices = indices[self.rank: self.total_size: self.num_replicas][self.num_skipped_samples:]
        assert len(indices) == self.num_samples - self.num_skipped_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.num_skipped_samples

    def set_num_skipped_samples(self, num_skipped_samples: int) -> None:
        if not (isinstance(num_skipped_samples, int) and 0 <= num_skipped_samples < self.num_samples):
            raise RuntimeError(
                f"Invalid ``num_skipped_samples`` ({num_skipped_samples}), it must be an non-negative integer "
                f"[0, {self.num_samples}] representing the number of skipped first samples for current process."
            )
        self.num_skipped_samples = num_skipped_samples


def skip_first_batches(dataloader: DataLoader, num_batches: int) -> DataLoader:
    """
    Create a `torch.utils.data.DataLoader` that will efficiently skip the first ``num_batches``.
    """
    if num_batches == 0:
        return dataloader

    dataset = dataloader.dataset  # `Union[datasets.arrow_dataset.IterableDataset, datasets.arrow_dataset.Dataset]`
    if isinstance(dataset, HFDataset):
        if not isinstance(dataloader.sampler, SkippableSampler):
            raise RuntimeError(
                f"`dataloader.sampler` is expected to be an instance of `{SkippableSampler.__name__}`, "
                f"but got `{dataloader.sampler.__class__.__name__}` now."
            )
        sampler = SkippableSampler(
            dataset=dataset,
            shuffle=dataloader.sampler.shuffle,
            drop_last=dataloader.sampler.drop_last,
            seed=dataloader.sampler.seed,
            process_group=dataloader.sampler.process_group
        )
        sampler.num_skipped_samples = num_batches * dataloader.batch_size
        sampler.set_epoch(epoch=dataloader.sampler.epoch)
    else:
        raise RuntimeError(f"Unsupported type for `{dataset.__class__.__name__}` yet.")

    # TODO(Tong Jia): The default kwargs for `torch.utils.data.Dataloader`
    #  is related to the version of PyTorch.
    default_kwargs = {
        "batch_size": 1,
        "shuffle": None,
        "sampler": None, "batch_sampler": None,
        "num_workers": 0,
        "collate_fn": None,
        "pin_memory": False,
        "drop_last": False,
        "timeout": 0,
        "worker_init_fn": None,
        "multiprocessing_context": None,
        "generator": None,
        "prefetch_factor": None,
        "persistent_workers": False,
        "pin_memory_device": ""
    }  # Default kwargs for `Dataloader` instantiation. PyTorch version: 2.3.0

    kwargs = {
        "dataset": dataset,
        "batch_size": dataloader.batch_size,
        "shuffle": False,  # # It must be False, because whether to shuffle dataset is actually
        # set through `SkippableDistributedSampler`, an error will be reported if it's True.
        "sampler": sampler, "batch_sampler": None
    }
    kwargs.update(
        {k: getattr(dataloader, k, v) for k, v in default_kwargs.items() if k not in kwargs}
    )
    return DataLoader(**kwargs)

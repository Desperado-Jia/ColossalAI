from typing import List, Optional, Union

import os
from pathlib import Path

from datasets import Dataset as HFDataset
from datasets import Split, concatenate_datasets, load_dataset

from .schema.raw import RawSample
from .schema.tokenized import TokenizedSample
from .packer.base import Packer
from ..models.base import ChatLLM
from ..utils.generic import IGNORE_INDEX


SUFFIX_OF_RAW_SAMPLE_PATH = ".jsonl"  # File path of a collection of `RawSample`.
SUFFIX_OF_TOKENIZED_SAMPLE_PATH = ".tk"  # Directory path of a collection of `TokenizedSample`.


def load_hf_dataset(filenames: Union[str, List[str]]) -> HFDataset:
    """Load collections of `TokenizedSamples` from disk,
    get the huggingface dataset like `datasets.Dataset[TokenizedSamples]`.
    """
    batched = True
    if not isinstance(filenames, List):
        batched = False
        filenames = [filenames]

    list_of_dataset = []  # `List[HFDataset]`
    for path in filenames:
        if not (Path(path).is_dir() and Path(path).suffix == SUFFIX_OF_TOKENIZED_SAMPLE_PATH):
            raise RuntimeError(
                f"Find invalid directory, it must be an existed directory with suffix "
                f"\"{SUFFIX_OF_TOKENIZED_SAMPLE_PATH}\" containing '.arrow' data files & '.json' configs."
            )
        dataset = HFDataset.load_from_disk(dataset_path=path, keep_in_memory=False)
        list_of_dataset.append(dataset)
    if batched:
        return concatenate_datasets(dsets=list_of_dataset)
    return list_of_dataset.pop(0)


def prepare_hf_dataset(filenames: Union[str, List[str]],
                       llm: ChatLLM,
                       packer: Optional[Packer] = None,
                       shuffle: bool = True,
                       num_workers: int = None,
                       cache_dir: str = "./cache"
                       ) -> HFDataset:
    """Load collections of `RawSample` and prepare a huggingface dataset like
    `datasets.Dataset[TokenizedSamples]`.
    """
    if num_workers is None:
        num_workers = os.cpu_count()
    num_workers = min(num_workers, os.cpu_count())

    # Verification of the raw files containing `RawSample`.
    if not isinstance(filenames, List):
        filenames = [filenames]
    for file in filenames:
        if not (Path(file).is_file() and Path(file).suffix == SUFFIX_OF_RAW_SAMPLE_PATH):
            raise RuntimeError(f"Invalid raw filename, must be a existed \"{SUFFIX_OF_RAW_SAMPLE_PATH}\" file.")

    dataset = load_dataset(
        path="json", data_files=filenames, split=Split.TRAIN, cache_dir=cache_dir,
        keep_in_memory=False, num_proc=num_workers
    )  # A collection of dict objs with `RawSample` schema.
    dataset = dataset.filter(
        function=RawSample.verify,
        fn_kwargs={},
        with_indices=False, with_rank=False,
        batched=False,
        keep_in_memory=False,
        num_proc=min(num_workers, len(dataset)),
        desc=f"Filtering samples that do not follow schema `{RawSample.__name__}`..."
    )  # Automatically filter out samples that do not follow the specified `ChatSample` schema.
    tokenized_dataset = dataset.map(
        function=llm.tokenize,
        fn_kwargs={
            "training": True, "return_dict": True
        },
        with_indices=False, with_rank=False,
        batched=False,
        remove_columns=list(dataset.column_names),
        keep_in_memory=False,
        num_proc=min(num_workers, len(dataset)),
        desc=f"Tokenizing from `{RawSample.__name__}` to `{TokenizedSample.__name__}`..."
    )  # `HFDataset[TokenizedSample]`.
    tokenized_dataset = tokenized_dataset.filter(
        function=lambda item: not set(item["labels"]) == {IGNORE_INDEX},
        fn_kwargs={},
        with_indices=False, with_rank=False,
        batched=False,
        keep_in_memory=False,
        num_proc=min(num_workers, len(dataset)),
        desc=f"Filtering invalid training `{TokenizedSample.__name__}`..."
    )
    if packer:
        tokenized_dataset = tokenized_dataset.map(
            function=packer,
            fn_kwargs={},
            with_indices=False, with_rank=False,
            batched=True, batch_size=5000,
            keep_in_memory=False,
            num_proc=min(num_workers, len(tokenized_dataset)),
            desc="Packing multi token sequences into one..."
        )
    if shuffle:
        tokenized_dataset = tokenized_dataset.shuffle(keep_in_memory=False)
    return tokenized_dataset

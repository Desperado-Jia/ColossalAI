from typing import Any, Dict, List, Optional, Union

from abc import ABC, abstractmethod
from enum import unique

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel

from ..schema.tokenized import TokenizedSample, TokenizedSamples
from ...utils.generic import ExplicitEnum, IGNORE_INDEX


@unique
class PaddingSide(ExplicitEnum):
    """Possible values for padding side of token sequences."""

    LEFT = "left"
    RIGHT = "right"


@unique
class PaddingStrategy(ExplicitEnum):
    """Possible values for the ``padding`` argument in [`DataCollator`]."""

    LONGEST = "longest"
    MAX_LENGTH = "max_length"


class DataCollator(ABC):
    """
    Abstract base class for all data collators that will dynamically pad the
    inputs received (input_ids, labels and attention_mask).
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 model: Optional[PreTrainedModel] = None,
                 padding: Union[str, PaddingStrategy] = PaddingStrategy.MAX_LENGTH,
                 max_length: Optional[int] = None,
                 ignore_index: int = None
                 ) -> None:
        if not (isinstance(tokenizer.pad_token_id, int) and tokenizer.pad_token_id >= 0):
            raise RuntimeError(
                f"Invalid `tokenizer.pad_token_id` ({tokenizer.pad_token_id}), "
                f"must be a non-negative integer."
            )
        self.tokenizer = tokenizer  # `transformers.PreTrainedTokenizer`
        self.model = model  # `Optional[transformers.PreTrainedModel]`
        self.padding = PaddingStrategy(padding)
        if self.padding == PaddingStrategy.MAX_LENGTH and not max_length:
            raise RuntimeError(
                f"Invalid or unset `max_length`, it must be a positive integer representing "
                f"the maximum (packed) sequence length when the \"{PaddingStrategy.MAX_LENGTH.value}\" "
                f"padding strategy is selected."
            )
        self.max_length = max_length
        self.padding_side = PaddingSide(self.tokenizer.padding_side)
        self.pad_token_id = self.tokenizer.pad_token_id  # `int`

        if ignore_index is None:
            ignore_index = IGNORE_INDEX
        self.ignore_index = ignore_index  # Label pad token id.

    @abstractmethod
    def _collate(self, inputs: List[TokenizedSample]) -> TokenizedSamples:
        pass

    def __call__(self, inputs: List[Dict[str, Union[List[int], List[List[float]]]]]) -> Dict[str, torch.Tensor]:
        # `List[Dict[str, Any]]` -> `List[TokenizedSample]`.
        inputs = [TokenizedSample.model_validate(obj) for obj in inputs]
        return self._collate(inputs=inputs).tensorify()

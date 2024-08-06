from typing import Dict, List, Tuple, Union
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from ..schema.tokenized import TokenizedSamples


class Packer(ABC):
    """
    Abstract base class for packing multi tokenized sequences into one.
    Note that different packers maybe have distinct workflows for packing.
    """

    def __init__(self, max_length: int = 4096) -> None:
        self.max_length = max_length

    @abstractmethod
    def _pack(self, inputs: TokenizedSamples) -> TokenizedSamples:
        """Pack a mini-batch sequences into a mini-batch packed sequences."""
        pass

    def pack(self, inputs: TokenizedSamples) -> TokenizedSamples:
        # Validation for the input batch (before packing).
        if inputs.attention_mask:
            raise RuntimeError(
                f"Expected to be a None for `{TokenizedSamples.__name__}.attention_mask` before packing, "
                f"attention_mask needs to be generated in the packing workflow."
            )
        # Only one of input_ids and input_embeds can be effective.
        if (not inputs.input_ids and not inputs.input_embeds) or (inputs.input_ids and inputs.input_embeds):
            raise RuntimeError(
                "Unexpected of input_ids & input_embeds, only one of them can be effective."
            )
        xs = inputs.input_ids if inputs.input_ids else inputs.input_embeds
        bsz = len(xs)

        # Ensure the label exists and is aligned correctly with the input for training.
        if not (inputs.labels and len(inputs.labels) == bsz):
            raise RuntimeError(
                "Label must be present and aligned correctly with the input"
            )
        for i, (x, y) in enumerate(zip(xs, inputs.labels)):
            if len(x) != len(y):
                raise RuntimeError(
                    "Length of token sequence is expected be equal to label sequence, "
                    "confirm the correctness of the argument ``inputs``."
                )
        outputs = self._pack(inputs=inputs)
        # Validation for the output batch (after packing).
        if not (outputs.attention_mask and outputs.labels and outputs.attention_mask):
            raise RuntimeError(
                "Expected to be a `List[List[int]]` for attention_mask & labels after packing."
            )
        xs = outputs.input_ids if outputs.input_ids else outputs.input_embeds
        for i, (x, y, attn_mask) in enumerate(zip(xs, outputs.labels, outputs.attention_mask)):
            if len(x) != len(y) or len(x) != len(attn_mask) or len(y) != len(attn_mask):
                raise RuntimeError(
                    f"Expected to be the same length of input_ids/input_embeds ({len(x)}), "
                    f"labels ({len(y)}) and attention_mask ({len(attn_mask)})."
                )
        return outputs

    def __call__(self,
                 inputs: Dict[str, Union[List[int], List[List[float]]]]
                 ) -> Dict[str, Union[List[int], List[List[float]]]]:
        # `Dict[str, Union[List[int], List[List[float]]]]` -> `TokenizedSamples`
        inputs = TokenizedSamples.model_validate(inputs)
        return self.pack(inputs=inputs).model_dump(exclude_none=True)

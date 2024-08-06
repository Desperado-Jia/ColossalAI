from typing import List

import torch
import torch.nn.functional as F

from .base import DataCollator, PaddingSide, PaddingStrategy
from ..schema.tokenized import TokenizedSample, TokenizedSamples


class DataCollatorForLanguageModeling(DataCollator):
    """
    A class used to represent the data collator used for LLMs.
    Inputs are dynamically padded to the maximum length of a batch if they are all not of the same length.
    """

    def _collate(self, inputs: List[TokenizedSample]) -> TokenizedSamples:
        input_ids, attention_mask, labels = tuple(
            [
                [
                    torch.LongTensor(getattr(item, key)[: self.max_length])  # Automatically truncated,
                    # make sure that the length of each sample does not exceed the maximum limit.
                    for item in inputs
                ]
                for key in ("input_ids", "attention_mask", "labels")
                # Only input_ids, attention_mask & labels are needed for language model.
            ]
        )  # Tuple of `List[torch.LongTensor]`
        if self.padding_side == PaddingSide.LEFT:
            # Reverse the sequence, process them all using the workflow for right side padding,
            # then reverse them again.
            input_ids, attention_mask, labels = tuple(
                [
                    [sequence.flip(dims=(0,)) for sequence in sequences]
                    for sequences in (input_ids, attention_mask, labels)
                ]
            )  # Tuple of `List[torch.LongTensor]`
            # Perform padding on the right side uniformly.
        input_ids, attention_mask, labels = tuple(
            [
                torch.nn.utils.rnn.pad_sequence(sequences=seqs, batch_first=True, padding_value=pad_token_id)
                for seqs, pad_token_id in [
                    (input_ids, self.pad_token_id),
                    (attention_mask, 0),
                    (labels, self.ignore_index)
                ]
            ]
        )  # Tuple of `torch.Tensor` (Size = (bsz, seqlen))
        if self.padding == PaddingStrategy.MAX_LENGTH:
            padding_length = self.max_length - input_ids.size(1)
            input_ids, attention_mask, labels = tuple(
                F.pad(input=tensor, pad=(0, padding_length), value=pad_token_id)
                for tensor, pad_token_id in [
                    (input_ids, self.pad_token_id), (attention_mask, 0), (labels, self.ignore_index)
                ]
            )  # (bsz, seqlen) -> (bsz, max_seqlen)
        if self.padding_side == PaddingSide.LEFT:
            # Restore the previously reversed sequences.
            input_ids, attention_mask, labels = tuple(
                [
                    torch.flip(tensor, dims=(1,)) for tensor in (input_ids, attention_mask, labels)
                ]
            )
        # Verification based on tensor shape.
        if (input_ids.shape != attention_mask.shape
                or input_ids.shape != labels.shape
                or attention_mask.shape != labels.shape):
            raise RuntimeError(
                f"Unexpected tensor shape of ``input_ids``({input_ids.shape}), "
                f"``attention_mask``({attention_mask.shape}) and ``labels``({labels.shape}), "
                f"all three must be the same size."
            )
        return TokenizedSamples(
            input_ids=input_ids.tolist(), attention_mask=attention_mask.tolist(), labels=labels.tolist()
        )

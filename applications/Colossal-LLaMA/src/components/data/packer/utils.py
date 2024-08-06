from typing import Tuple

import torch
import torch.nn.functional as F

import transformers
from transformers.models.auto import AutoConfig


def _get_seq_len_in_batch(attention_mask: torch.LongTensor) -> torch.Tensor:
    """Get the length of each valid original sequence (before packing)
    in the mini-batch packed sequences.

    Args:
        attention_mask (`torch.LongTensor`): Size = (bsz, max_len),
            attention mask for mini-batch packed sequences after padding.
    Returns:
        (`torch.Tensor`): Size = (n,), `n` means the number of valid sequences within the mini-batch.

    Examples:
        .. code-block:: python
            attn_mask = torch.LongTensor(
                [
                    [1, 1, 1, 2, 2, 2, 2, 2, 3, 0],
                    [1, 1, 2, 3, 3, 3, 4, 4, 5, 5]
                ]
            )
            print(get_seq_len_in_batch(attn_mask))
            # tensor([3, 5, 1, 2, 1, 3, 2, 2], dtype=torch.int32)
    """
    max_num = torch.max(attention_mask)  # The maximum number of sequences been packed
    # in each packed sequence within the batch.
    counts = [
        torch.sum(torch.BoolTensor(attention_mask == i), dim=-1) for i in range(1, max_num + 1)
    ]  # `List[torch.Tensor]`, size of each item: (bsz,)
    # Count the length of sequence masked with i within each packed sequence.
    # Note that 0 means padding for a sequence in `attention_mask`.
    lens = torch.stack(counts, dim=1)  # Size = (bsz, max_num)
    # e.g.,
    # tensor([[3, 5, 1, 0, 0],
    #         [2, 1, 3, 2, 2]])
    lens = lens.flatten()
    lens = lens[lens.nonzero()].squeeze(dim=-1).to(dtype=torch.int32)  # Size = (n,),
    # n means the number of sequences with length greater than 0 in the batch.
    return lens


def get_unpad_data(attention_mask: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Modified from:
    https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_flash_attention_utils.py#L33

    Args:
        attention_mask (`torch.LongTensor`): Size = (bsz, max_len)

    Returns:
        (`torch.Tensor`): Size = (n,), n means the total number of tokens within batch with non-zero mask.
        (`torch.Tensor`): Size = (m,), m means the total number of sequences (before packing) within batch.
            Cumulative length after flattening all valid sequences.
        (`int`): Maximum length of sequences (before packing) within batch.

    Examples:
        .. code-block:: python
            attention_mask = torch.LongTensor(
                [
                    [1, 1, 1, 2, 2, 2, 2, 2, 3, 0],
                    [1, 1, 2, 3, 3, 3, 4, 4, 5, 5]
                ]
            )
            print(get_unpad_data(attention_mask))
            # (tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
            #  tensor([ 0,  3,  8,  9, 11, 12, 15, 17, 18], dtype=torch.int32),
            #  5)
    """
    seqlens_in_batch = _get_seq_len_in_batch(attention_mask=attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), pad=(1, 0)
    )
    return indices, cu_seqlens, max_seqlen_in_batch


def monkey_patch_for_packing_flash_attention() -> None:
    if not (hasattr(transformers, "modeling_flash_attention_utils")
            and hasattr(transformers.modeling_flash_attention_utils, "_get_unpad_data")):
        raise RuntimeError()
    transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data

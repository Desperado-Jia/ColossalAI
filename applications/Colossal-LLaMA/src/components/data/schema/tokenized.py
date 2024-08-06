from typing import Dict, List, Optional
from pydantic import Field

import torch

from .base import Schema


class TokenizedSample(Schema):
    """A class used to represent a standardized tokenized sample of LLMs/VLMs.
    Note that all candidate fields need to be consistent with the `TokenizedSamples`.
    """

    input_ids: Optional[List[int]] = Field(
        default=None,
        description="Size = (seqlen), a sequence of token ids, commonly used for LLMs."
    )
    attention_mask: Optional[List[int]] = Field(
        default=None,
        description="Size = (seqlen), a binary mask that designates which tokens should be attended to "
                    "(assigned non-zero weights) and which should be ignored (assigned zero weights)."
    )
    input_embeds: Optional[List[List[float]]] = Field(
        default=None,
        description="Size = (seqlen, ndim), a sequence of token embeddings. "
                    "Commonly used for VLMs and ``input_ids`` doesn't exist at the same time."
    )
    token_type_ids: Optional[List[int]] = Field(
        default=None,
        description="Size = (seqlen), a sequence of token type ids. Commonly used for VLMs."
    )
    position_ids: Optional[List[int]] = Field(
        default=None,
        description="Size = (seqlen), a sequence of indices used by the model to "
                    "identify each token's position in the list of tokens."
    )
    labels: Optional[List[int]] = Field(
        default=None,
        description="Size = (seqlen), a sequence of token ids used by a model "
                    "to identify label of each token in `input_ids`."
    )


class TokenizedSamples(Schema):
    """A class used to represent a standardized tokenized mini-batch samples of LLMs/VLMs.
    Note that all fields need to be consistent with the `TokenizedSample`.
    """

    input_ids: Optional[List[List[int]]] = Field(
        default=None,
        description="Size = (bsz, seqlen), a mini-batch sequences of token ids, commonly used for LLMs."
    )
    attention_mask: Optional[List[List[int]]] = Field(
        default=None,
        description="Size = (bsz, seqlen), a mini-batch binary masks that designates which tokens should be attended "
                    "to (assigned non-zero weights) and which should be ignored (assigned zero weights)."
    )
    input_embeds: Optional[List[List[List[float]]]] = Field(
        default=None,
        description="Size = (bsz, seqlen, ndim), a mini-batch sequences of token embeddings. "
                    "Commonly used for VLMs and ``input_ids`` doesn't exist at the same time."
    )
    token_type_ids: Optional[List[List[int]]] = Field(
        default=None,
        description="Size = (bsz, seqlen), a mini-batch sequences of token type ids. Commonly used for VLMs."
    )
    position_ids: Optional[List[List[int]]] = Field(
        default=None,
        description="Size = (bsz, seqlen), a mini-batch sequences of indices used by the model to "
                    "identify each token's position in the list of tokens."
    )
    labels: Optional[List[List[int]]] = Field(
        default=None,
        description="Size = (bsz, seqlen), a mini-batch sequences of token ids used by a model "
                    "to identify label of each token in `input_ids`."
    )

    def tensorify(self) -> Dict[str, torch.Tensor]:
        output = self.model_dump(exclude_none=True)
        ret = {}
        for k, v in output.items():
            if k in {"input_ids", "attention_mask", "token_type_ids", "position_ids", "labels"}:
                ret[k] = torch.LongTensor(v)
            else:  # For `input_embeds`.
                ret[k] = torch.Tensor(v)
        return ret

from .base import Packer
from ..schema.tokenized import TokenizedSamples


class IntegrityPacker(Packer):

    def __init__(self, max_length: int) -> None:
        super().__init__(max_length=max_length)
        self.buffer = {
            "input_ids": (),  # `Tuple[int]`
            "labels": (),  # `Tuple[int]`
            "attention_mask": ()  # `Tuple[int]`
        }

    def _pack(self, inputs: TokenizedSamples) -> TokenizedSamples:
        outputs = TokenizedSamples(
            input_ids=[], attention_mask=[], labels=[]
        )
        mask = 1  # `int`, identify the attention mask index of a sequence in each concatenated sample.
        bsz = len(inputs.input_ids)
        for i in range(bsz):
            if len(self.buffer["input_ids"]) + len(inputs.input_ids[i]) <= self.max_length:
                length = None  # `int`, length of sequence.
                for k in {"input_ids", "labels"}:
                    seq = getattr(inputs, k)[i]
                    if length is None:
                        length = len(seq)
                    self.buffer[k] += tuple(seq)
                self.buffer["attention_mask"] += tuple([mask for _ in range(length)])
                mask += 1
            else:
                mask = 1  # Reset the attention mask index for a new packed sequence.
                length = None  # `int`, length of sequence.
                for k in ["input_ids", "labels", "attention_mask"]:
                    # Update the outputs firstly.
                    v = getattr(outputs, k)  # `List[List[int]]`
                    v.append(list(self.buffer[k]))
                    setattr(outputs, k, v)
                    # Update the buffer secondly.
                    if k != "attention_mask":
                        seq = getattr(inputs, k)[i]
                        if length is None:
                            length = len(seq)
                        self.buffer[k] = tuple(seq)
                    else:
                        assert length is not None
                        self.buffer[k] = tuple([mask for _ in range(length)])
        # Add the remaining packed sample in the buffer.
        if len(self.buffer["input_ids"]) > 0:
            for k, v in self.buffer.items():
                vv = getattr(outputs, k)
                vv.append(list(v))
                setattr(outputs, k, vv)
        return outputs

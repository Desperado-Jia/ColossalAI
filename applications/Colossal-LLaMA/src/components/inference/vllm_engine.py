from typing import List, Union, Tuple

from vllm import LLM, SamplingParams

from .base import InferenceEngine, Usage
from ..data.schema import RawSample, TokenizedSample
from ..data.schema.raw import Message
from ..models import ChatLLM


class VLLMInferenceEngine(InferenceEngine):

    def __init__(self, llm: ChatLLM) -> None:
        self.engine = None  # `vllm.LLM`
        super().__init__(llm=llm)

    def _initialize(self) -> None:
        raise NotImplementedError()

    def _generate(self,
                  prompts: Union[List[str], List[TokenizedSample]],
                  **kwargs
                  ) -> List[Tuple[str, Usage]]:
        raise NotImplementedError()

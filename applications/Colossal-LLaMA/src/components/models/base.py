from typing import Any, Dict, List, Optional, Union

from abc import ABC, abstractmethod
from pydantic import Field
from pydantic_settings import BaseSettings

from ..data.schema.raw import RawSample
from ..data.schema.tokenized import TokenizedSample
from ..utils.generic import IGNORE_INDEX


class ChatLLMConfig(BaseSettings):
    """A class used to represent the configurations for all LLMs & VLMs."""

    pretrained_model_name_or_path: Optional[str] = Field(
        default=None,
        description="A path to a directory containing configuration files, vocabulary files and "
                    "model weights used for chat LLM."
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Whether or not to allow for custom models defined on the Hub in their own modeling files. "
                    "This option should only be set to `True` for repositories you trust and in which "
                    "you have read the code, as it will execute code present on the Hub on your local machine."
    )
    ignore_index: int = Field(
        default=IGNORE_INDEX,
        description="Specifies a target value that is ignored and does not contribute to the "
                    "gradient of loss function in PyTorch."
    )


class ChatLLM(ABC):
    """Abstract base (wrapper) class used to represent the naive transformers-based LLMs/VLMs.
    Note that different LLMs and VLMs have distinct workflows for data processing and
    model/tokenizer initialization.
    """

    def __init__(self, config: ChatLLMConfig) -> None:
        self.config = config
        self.tokenizer = None  # `transformers.models.auto.AutoTokenizer`
        self.model = None  # `transformers.models.auto.AutoModelForCausalLM`

    @abstractmethod
    def init_tokenizer(self) -> None:
        """Initialize the transformers-based tokenizer for LLMs/VLMs."""
        pass

    @abstractmethod
    def init_model(self) -> None:
        """Initialize the transformers-based model for LLMs/VLMs."""
        pass

    @abstractmethod
    def _tokenize(self, sample: RawSample, training: bool) -> TokenizedSample:
        """Tokenize a raw multi-turn conversation sample and get the tokenized input sample
        for both training & inference of transformers-based model.
        """
        pass

    def tokenize(self,
                 sample: Union[RawSample, Dict[str, Any]],
                 training: bool = True,
                 return_dict: bool = False
                 ) -> Union[TokenizedSample, Dict[str, Any]]:
        """Tokenize a raw multi-turn conversation sample and get the tokenized input sample
        for both training & inference of transformers-based model.
        """
        tokenized_sample = self._tokenize(
            sample=RawSample.model_validate(sample),  # `RawSample`
            training=training
        )
        if return_dict:
            return tokenized_sample.model_dump(exclude_none=True)
        return tokenized_sample

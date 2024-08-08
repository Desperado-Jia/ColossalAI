from typing import Any, ContextManager, Dict, Optional, Union

from abc import ABC, abstractmethod
from contextlib import nullcontext
from enum import unique
from pydantic import Field
from pydantic_settings import BaseSettings

from ..data.schema.raw import RawSample, Role, Message
from ..data.schema.tokenized import TokenizedSample
from ..utils.generic import IGNORE_INDEX, Language


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
    language: Union[str, Language] = Field(
        default=Language.CHINESE,
        description="Default language setting for automatic completion of prompts in conversations."
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
        self.ignore_index = self.config.ignore_index

    @abstractmethod
    def init_tokenizer(self) -> None:
        """Initialization (instantiation & verification) of the transformers-based tokenizer for LLMs/VLMs.
        Note that the reason for separating the initialization of the tokenizer and model
        is to minimize the initialization cost in various usage scenarios.
        """
        pass

    @abstractmethod
    def init_model(self, context: ContextManager = nullcontext()) -> None:
        """Initialization (instantiation & verification) of the transformers-based model for LLMs/VLMs."""
        pass

    @abstractmethod
    def _tokenize(self, sample: RawSample, training: bool) -> TokenizedSample:
        """Tokenize a raw multi-turn conversation sample and get the tokenized input sample
        for both training & inference of transformers-based model.
        """
        pass

    @abstractmethod
    def prepare_response_message(self, text: str) -> Message:
        """Postprocess the response string after model (engine) generation.
        e.g., handle the situation where a tool call occurs.
        Note that different models have distinct workflows for post-processing response content.
        This method only used in inference mode.
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
        if not self.tokenizer:
            raise RuntimeError(
                f"Uninstantiated `{self.__class__.__name__}.tokenizer`, "
                f"please execute 'init_tokenizer' method first."
            )
        tokenized_sample = self._tokenize(
            sample=RawSample.model_validate(sample),  # `RawSample`
            training=training
        )
        if return_dict:
            return tokenized_sample.model_dump(exclude_none=True)
        return tokenized_sample

    @classmethod
    def _is_trainable_msg(cls, msg: Message, training: bool) -> bool:
        """Determine whether the current message needs to calculate loss."""
        if not training:
            return False
        if msg.role != Role.ASSISTANT:
            return False
        if msg.loss is False:
            return False
        return True

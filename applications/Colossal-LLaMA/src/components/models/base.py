from typing import Any, ContextManager, Dict, List, NoReturn, Optional, Union

from abc import ABC, abstractmethod
from contextlib import nullcontext
from enum import unique
from pydantic import Field, ValidationError
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
    def parse_response(self, text: str) -> Message:
        """Postprocess the response text after generation, e.g., handle function calling.
        Note that different models have distinct workflows for post-processing, and the method
        is only used under the inference mode.
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
        sample = RawSample.model_validate(sample)  # `RawSample`
        self._check_sample(sample=sample, training=training)
        tokenized_sample = self._tokenize(
            sample=sample,
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

    @classmethod
    def _check_sample(cls, sample: RawSample, training: bool) -> None:
        """Basic check of raw conversation sample. All error must be `ValidationError`.

        Rules:
            * There is only one system in the conversation sequence, and it can only appear at the beginning;
            * The definition of a tool can only appear in the system and can only appear once.
            * Tool calls can only appear in the assistant.
        """
        if len(sample.messages) == 0:
            raise ValidationError("Invalid conversation, find an empty sequence of messages.")
        if not training and (sample.messages[-1].role != Role.USER):
            raise ValidationError(
                "Invalid conversation for inference mode, expected to be a user message at the last."
            )
        num_sys = 0
        for i, msg in enumerate(sample.messages):
            if msg.role == Role.SYSTEM:
                num_sys += 1
            if i == 0 and msg.role != Role.SYSTEM:
                raise ValidationError("Invalid conversation, the conversation sequence must begin with a system message.")
            if msg.tools and i != 0:
                raise ValidationError(
                    "Invalid conversation, tools definition is only allowed to appear "
                    "in the first and only system message."
                )

            if msg.tool_calls and msg.role != Role.ASSISTANT:
                raise ValidationError(
                    "Invalid conversation, tools calling is only allowed to appear "
                    "in the assistant message."
                )
            if msg.tool_calls and msg.content:
                raise ValidationError(
                    "Invalid conversation, tool calling and normal content cannot appear at the same time."
                )
        if num_sys != 1:
            raise ValidationError("Invalid conversation, expected to be only one system message.")

    @classmethod
    def basic_check_sample(cls, sample: Union[Dict[str, Any], RawSample], training: bool) -> bool:
        try:
            if not isinstance(sample, RawSample):
                sample = RawSample.model_validate(sample)
            cls._check_sample(sample=sample, training=training)
            return True
        except ValidationError as e:
            return False

    @abstractmethod
    def custom_check_sample(self, sample: Union[Dict[str, Any], RawSample], training: bool) -> bool:
        pass

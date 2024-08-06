from typing import Union
from enum import unique

from .base import ChatLLM, ChatLLMConfig
from .llama3 import Llama3ChatLLM
from ..utils.generic import ExplicitEnum


__all__ = [
    "ChatLLM", "ChatLLMConfig",
    "Llama3ChatLLM",
    "ChatLLMType", "setup_chat_llm"
]


@unique
class ChatLLMType(ExplicitEnum):
    """Acceptable values for chat LLMs/VLMs' types."""

    LLAMA3 = "llama3"


_CHAT_LLM_MAPPING = {
    ChatLLMType.LLAMA3: Llama3ChatLLM
}  # `Dict[ChatLLMType, Type[ChatLLM]]`
_DEFAULT_KEY = ChatLLMType.LLAMA3


def setup_chat_llm(llm_type: Union[str, ChatLLMType] = None, **kwargs) -> ChatLLM:
    if not llm_type:
        llm_type = _DEFAULT_KEY
    llm_type = ChatLLMType(llm_type)
    if llm_type not in _CHAT_LLM_MAPPING:
        raise KeyError(
            f"Invalid ``llm_type`` ({llm_type.value}) for instantiation `{ChatLLM.__name__}`."
        )
    cls = _CHAT_LLM_MAPPING[llm_type]
    config = ChatLLMConfig(**kwargs)
    return cls(config=config)

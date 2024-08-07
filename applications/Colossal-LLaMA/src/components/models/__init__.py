from typing import Union
from typing_extensions import Type
from enum import unique

from .base import ChatLLM, ChatLLMConfig
from .llama3 import Llama3ChatLLM
from .qwen2 import Qwen2ChatLLM
from ..utils.generic import ExplicitEnum


__all__ = [
    "ChatLLM", "ChatLLMConfig",
    "Llama3ChatLLM",
    "Qwen2ChatLLM",
    "ChatLLMType",
    "setup_llm_cls"
]


@unique
class ChatLLMType(ExplicitEnum):
    """Acceptable values for chat LLMs/VLMs' types."""

    LLAMA3 = "llama3"
    QWEN2 = "qwen2"


_CHAT_LLM_MAPPING = {
    ChatLLMType.LLAMA3: Llama3ChatLLM,
    ChatLLMType.QWEN2: Qwen2ChatLLM
}  # `Dict[ChatLLMType, Type[ChatLLM]]`
_DEFAULT_KEY = ChatLLMType.LLAMA3


def setup_llm_cls(key: Union[str, ChatLLMType] = None) -> Type[ChatLLM]:
    if not key:
        key = _DEFAULT_KEY
    key = ChatLLMType(key)
    if key not in _CHAT_LLM_MAPPING:
        raise KeyError(
            f"Invalid ``key`` ({key.value}) for instantiation `{ChatLLM.__name__}`."
        )
    return _CHAT_LLM_MAPPING[key]

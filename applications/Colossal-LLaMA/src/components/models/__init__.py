from typing import Union
from enum import unique

from .base import ChatLLM, ChatLLMConfig
from .llama3 import Llama3ChatLLM
from ..utils.generic import ExplicitEnum


__all__ = [
    "ChatLLM", "ChatLLMConfig",
    "Llama3ChatLLM",
    "ChatLLMName", "setup_chat_llm"
]


@unique
class ChatLLMName(ExplicitEnum):
    """Acceptable values for chat LLMs/VLMs' names."""

    LLAMA3 = "llama3"


_CHAT_LLM_MAPPING = {
    ChatLLMName.LLAMA3: Llama3ChatLLM
}  # `Dict[ChatLLMName, Type[ChatLLM]]`


def setup_chat_llm(name: Union[str, ChatLLMName], **kwargs) -> ChatLLM:
    name = ChatLLMName(name)
    if name not in _CHAT_LLM_MAPPING:
        raise KeyError(f"Invalid name ({name.value}) of `{ChatLLM.__name__}` for instantiation of LLMs/VLMs.")
    cls = _CHAT_LLM_MAPPING[name]
    config = ChatLLMConfig(**kwargs)
    return cls(config=config)

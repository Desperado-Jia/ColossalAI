from typing import Any, Dict, List, Optional, Union

from enum import unique
from pydantic import Field

from .base import Schema
from .tool import Tool, ToolCall
from ...utils.generic import ExplicitEnum


@unique
class ContentType(ExplicitEnum):
    """Acceptable values of content type."""

    TEXT = "text"
    IMAGE = "image"


@unique
class Role(ExplicitEnum):
    """Acceptable values of role (i.e., information sender) for chat message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    EXECUTOR = "executor"  # Tool executor.


class Content(Schema):
    """A class used to represent a standardized content (item) for chat completion."""

    value: str = Field(
        default=...,
        description="A content string, it can be either a textual string or a base64 encoded string of an image."
    )
    type: ContentType = Field(
        default=...,
        description="The type of the content item."
    )


class Message(Schema):
    """A class used to represent a standardized chat message."""

    role: Role = Field(
        default=...,
        description="Sender role of the chat message."
    )
    content: Optional[Union[str, Content, List[Content]]] = Field(
        default=None,
        description="Content item(s) of the chat message."
    )
    tools: Optional[Union[Tool, List[Tool]]] = Field(
        default=None,
        description="A collection of tool definitions. "
                    "Note that only the user and system have the authority to publish tool definitions, "
                    "the user can republish new tool definitions in a conversation, but it will affect "
                    "the range of tool call options that the assistant can use when complete the chat later."
    )
    tool_calls: Optional[Union[ToolCall, List[ToolCall]]] = Field(
        default=None,
        description="A collection of tool call commands to be executed later based on the contextual messages."
    )
    loss: Optional[bool] = Field(
        default=None,
        description="Whether to calculate the loss for the current message, only used for training."
    )


class RawSample(Schema):
    """A class used to represent a standardized chat conversation sample."""

    messages: List[Message] = Field(
        default=...,
        description="A sequence of chat messages representing a multi-turn conversation."
    )

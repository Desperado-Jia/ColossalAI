import typing
from typing import Any, Dict, List, Optional
from enum import unique
from pydantic import Field

from .base import Schema
from ...utils.generic import ExplicitEnum


@unique
class ToolType(ExplicitEnum):
    """Acceptable values of tool types."""

    FUNCTION = "function"


class FuncParams(Schema):
    """A class used to represent a standardized input & output parameters description of a function."""

    type: str = Field(
        default=...,
        description="Type of a function parameter, e.g., 'object'."
    )
    properties: Dict[str, Any] = Field(
        default=...,
        description="All properties of the function."
    )
    required: List[str] = Field(
        default=...,
        description="All required input parameters' name."
    )


class Func(Schema):
    """A class used to represent a standardized function description for chat completion."""

    name: str = Field(
        default=...,
        description="The unique name of the function."
    )
    desc: str = Field(
        default=...,
        description="A natural language description of the function."
    )
    params: FuncParams = Field(
        default=...,
        description="Input & output parameters of the function."
    )


class Tool(Schema):
    """A class used to represent a standardized tool description."""

    type: ToolType = Field(
        default=ToolType.FUNCTION,
        description="Type of the tool item."
    )
    func: Optional[Func] = Field(
        default=None,
        description="Description of the function definition."
    )


class FuncCall(Schema):
    """A class used to represent a standardized function call description."""

    name: str = Field(
        default=...,
        description="The unique name of the function to be called by executor. "
                    "It must be a valid function (name) registered in a previous conversation sequence."
    )
    args: Dict[str, Any] = Field(
        default=...,
        description="The JSON-format arguments of the function being called."
    )


class ToolCall(Schema):
    """A class used to represent a standardized tool call description."""

    type: ToolType = Field(
        default=ToolType.FUNCTION,
        description="Type of the tool call item."
    )
    func: Optional[FuncCall] = Field(
        default=None,
        description="Description of the function call."
    )


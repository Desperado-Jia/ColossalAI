from typing import Any, Dict, Union
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, ValidationError


class Schema(BaseModel):
    """A class that represents the base schema of all pydantic models."""

    model_config = ConfigDict(extra="forbid", validate_default=True, use_enum_values=True)

    @classmethod
    def verify(cls, obj: Any) -> bool:
        """Perform schema validation on the specified object to determine whether it complies with."""
        try:
            _ = cls.model_validate(obj)
            return True
        except ValidationError as err:
            return False

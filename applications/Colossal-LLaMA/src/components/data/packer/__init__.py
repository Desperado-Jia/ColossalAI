from typing import Union
from enum import unique

from .base import Packer
from .integrity import IntegrityPacker
from ...utils.generic import ExplicitEnum


__all__ = [
    "Packer",
    "IntegrityPacker",
    "PackerType", "setup_packer"
]


@unique
class PackerType(ExplicitEnum):
    """Acceptable values for `Packer` names."""

    INTEGRITY = "integrity"


_PACKER_MAPPING = {
    PackerType.INTEGRITY: IntegrityPacker
}  # `Dict[PackerType, Type[Packer]]`
_DEFAULT_KEY = PackerType.INTEGRITY


def setup_packer(packer_type: Union[str, PackerType] = None, **kwargs) -> Packer:
    if not packer_type:
        packer_type = _DEFAULT_KEY
    packer_type = PackerType(packer_type)
    if packer_type not in _PACKER_MAPPING:
        raise KeyError(
            f"Invalid ``packer_type`` ({packer_type.value}) for instantiation `{Packer.__name__}`."
        )
    cls = _PACKER_MAPPING[packer_type]
    return cls(**kwargs)

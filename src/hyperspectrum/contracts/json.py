"""Recursively immutable, JSON-safe metadata values for public contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from math import isfinite
from types import MappingProxyType
from typing import NoReturn, TypeAlias, Union

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
FrozenJsonValue: TypeAlias = Union[JsonScalar, tuple["FrozenJsonValue", ...], "FrozenJsonMapping"]


class FrozenJsonMapping(Mapping[str, FrozenJsonValue]):
    """An immutable mapping containing only recursively frozen JSON values."""

    __slots__ = ("_values",)

    _values: Mapping[str, FrozenJsonValue]

    def __init__(self, values: Mapping[object, object]) -> None:
        frozen_values: dict[str, FrozenJsonValue] = {}
        for key, value in values.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            frozen_values[key] = freeze_json_value(value)
        object.__setattr__(self, "_values", MappingProxyType(frozen_values))

    def __getitem__(self, key: str) -> FrozenJsonValue:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise TypeError("FrozenJsonMapping instances are immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise TypeError("FrozenJsonMapping instances are immutable")


def freeze_json_mapping(value: object) -> FrozenJsonMapping:
    """Validate and recursively detach a JSON object from caller-owned inputs."""
    if not isinstance(value, Mapping):
        raise TypeError("JSON metadata must be an object")
    if isinstance(value, FrozenJsonMapping):
        return value
    return FrozenJsonMapping(value)


def freeze_json_value(value: object) -> FrozenJsonValue:
    """Convert a JSON-like value to its recursively immutable representation."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, FrozenJsonMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenJsonMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json_value(item) for item in value)
    raise ValueError(f"unsupported non-JSON value: {type(value).__name__}")


def thaw_json_mapping(value: FrozenJsonMapping) -> dict[str, JsonValue]:
    """Produce a standard JSON-compatible object for Pydantic serialization."""
    return {key: thaw_json_value(item) for key, item in value.items()}


def thaw_json_value(value: FrozenJsonValue) -> JsonValue:
    """Produce mutable JSON-compatible output without exposing stored values."""
    if isinstance(value, FrozenJsonMapping):
        return thaw_json_mapping(value)
    if isinstance(value, tuple):
        return [thaw_json_value(item) for item in value]
    return value

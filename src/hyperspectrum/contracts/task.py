"""Contracts for declared evaluation tasks and metrics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class MetricSpec(BaseModel):
    """A metric and its deterministic aggregation rule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    direction: Literal["min", "max"]
    aggregation: Literal["sample_mean", "group_mean", "grouped_bootstrap"]
    primary: bool = False


class TaskSpec(BaseModel):
    """Immutable task declaration that later components must honor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["hyperspectrum-task/v1"]
    id: str
    modality: str
    task_type: str
    input_roles: tuple[str, ...]
    output_kind: str
    ground_truth_roles: tuple[str, ...]
    split_group_keys: tuple[str, ...]
    metrics: tuple[MetricSpec, ...]

"""Staged orchestration executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import Field

from nlp_stock_prediction.contracts import AuditArtifact, ContractModel, JsonObject, NonEmptyStr
from nlp_stock_prediction.orchestration.context import RunContext
from nlp_stock_prediction.orchestration.tools import ToolRegistry, ToolRunResult

T = TypeVar("T")


class ToolRunRecord(ContractModel):
    """Serializable summary of one completed tool invocation."""

    tool_id: NonEmptyStr
    stage: NonEmptyStr
    updated_keys: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    artifact_ids: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    metadata: JsonObject = Field(default_factory=dict)


@dataclass
class OrchestrationState:
    """Mutable state passed between staged tools."""

    values: dict[str, object] = field(default_factory=dict)
    artifacts: list[AuditArtifact] = field(default_factory=list)
    records: list[ToolRunRecord] = field(default_factory=list)

    def require(self, key: str, expected_type: type[T]) -> T:
        try:
            value = self.values[key]
        except KeyError as exc:
            raise KeyError(f"orchestration state is missing required key: {key}") from exc
        if not isinstance(value, expected_type):
            raise TypeError(
                f"orchestration state key {key!r} expected "
                f"{expected_type.__name__}, got {type(value).__name__}"
            )
        return value

    def apply(self, result: ToolRunResult, record: ToolRunRecord) -> None:
        duplicate_keys = sorted(set(result.updates).intersection(self.values))
        if duplicate_keys:
            joined = ", ".join(duplicate_keys)
            raise ValueError(f"tool result would overwrite state keys: {joined}")
        self.values.update(result.updates)
        self.artifacts.extend(result.artifacts)
        self.records.append(record)


@dataclass(frozen=True)
class StagedExecutionResult:
    """Completed state and run records for a staged execution."""

    state: OrchestrationState

    @property
    def artifacts(self) -> tuple[AuditArtifact, ...]:
        return tuple(self.state.artifacts)

    @property
    def tool_records(self) -> tuple[ToolRunRecord, ...]:
        return tuple(self.state.records)


@dataclass(frozen=True)
class StagedExecutor:
    """Execute registered tools in stage order, sorted by tool ID within each stage."""

    registry: ToolRegistry
    stage_order: tuple[str, ...]

    def run(self, context: RunContext) -> StagedExecutionResult:
        state = OrchestrationState()
        for tool in self.registry.ordered_for_stages(self.stage_order):
            self._validate_inputs(tool.spec.input_keys, state)
            result = tool.run(context, state)
            if result.tool_id != tool.spec.tool_id:
                raise ValueError(f"tool {tool.spec.tool_id} returned result for {result.tool_id}")
            self._validate_outputs(tool.spec.output_keys, result)
            record = ToolRunRecord(
                tool_id=tool.spec.tool_id,
                stage=tool.spec.stage,
                updated_keys=tuple(sorted(result.updates)),
                artifact_ids=tuple(artifact.artifact_id for artifact in result.artifacts),
                metadata=result.metadata,
            )
            state.apply(result, record)
        return StagedExecutionResult(state=state)

    @staticmethod
    def _validate_inputs(input_keys: tuple[str, ...], state: OrchestrationState) -> None:
        missing = [key for key in input_keys if key not in state.values]
        if missing:
            joined = ", ".join(missing)
            raise KeyError(f"tool inputs missing from orchestration state: {joined}")

    @staticmethod
    def _validate_outputs(output_keys: tuple[str, ...], result: ToolRunResult) -> None:
        missing = [key for key in output_keys if key not in result.updates]
        if missing:
            joined = ", ".join(missing)
            raise KeyError(f"tool result missing declared output keys: {joined}")


__all__ = [
    "OrchestrationState",
    "StagedExecutionResult",
    "StagedExecutor",
    "ToolRunRecord",
]

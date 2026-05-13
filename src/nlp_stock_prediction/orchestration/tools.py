"""Tool specifications and deterministic registry for orchestration runs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from pydantic import Field

from nlp_stock_prediction.contracts import AuditArtifact, ContractModel, JsonObject, NonEmptyStr

if TYPE_CHECKING:
    from nlp_stock_prediction.orchestration.context import RunContext
    from nlp_stock_prediction.orchestration.runtime import OrchestrationState


class ToolSpec(ContractModel):
    """Static metadata for one orchestration tool."""

    tool_id: NonEmptyStr
    stage: NonEmptyStr
    description: NonEmptyStr
    input_keys: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)
    output_keys: tuple[NonEmptyStr, ...] = Field(default_factory=tuple)


@dataclass(frozen=True)
class ToolRunResult:
    """State updates and artifacts produced by one tool invocation."""

    tool_id: str
    updates: Mapping[str, object] = field(default_factory=dict)
    artifacts: tuple[AuditArtifact, ...] = ()
    metadata: JsonObject = field(default_factory=dict)


class OrchestrationTool(Protocol):
    """Callable tool interface used by the staged executor."""

    @property
    def spec(self) -> ToolSpec: ...

    def run(self, context: RunContext, state: OrchestrationState) -> ToolRunResult: ...


class ToolRegistry:
    """In-memory registry that provides deterministic tool ordering."""

    def __init__(self, tools: Iterable[OrchestrationTool] = ()) -> None:
        self._tools: dict[str, OrchestrationTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: OrchestrationTool) -> None:
        tool_id = tool.spec.tool_id
        if tool_id in self._tools:
            raise ValueError(f"tool is already registered: {tool_id}")
        self._tools[tool_id] = tool

    def get(self, tool_id: str) -> OrchestrationTool:
        try:
            return self._tools[tool_id]
        except KeyError as exc:
            raise KeyError(f"tool is not registered: {tool_id}") from exc

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self.ordered_by_id())

    def ordered_by_id(self) -> tuple[OrchestrationTool, ...]:
        return tuple(self._tools[tool_id] for tool_id in sorted(self._tools))

    def ordered_for_stages(self, stage_order: Sequence[str]) -> tuple[OrchestrationTool, ...]:
        stages = tuple(stage_order)
        if len(set(stages)) != len(stages):
            raise ValueError("stage_order entries must be unique")
        stage_index = {stage: index for index, stage in enumerate(stages)}
        unknown_stages = sorted(
            {tool.spec.stage for tool in self._tools.values() if tool.spec.stage not in stage_index}
        )
        if unknown_stages:
            joined = ", ".join(unknown_stages)
            raise ValueError(f"registered tools use stages missing from stage_order: {joined}")
        return tuple(
            sorted(
                self._tools.values(),
                key=lambda tool: (stage_index[tool.spec.stage], tool.spec.tool_id),
            )
        )


__all__ = ["OrchestrationTool", "ToolRegistry", "ToolRunResult", "ToolSpec"]

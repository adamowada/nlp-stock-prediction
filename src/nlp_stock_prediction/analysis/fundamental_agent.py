"""Bridge validated fundamental-agent output into report analysis."""

from __future__ import annotations

from nlp_stock_prediction.analysis._metrics import analysis_metric
from nlp_stock_prediction.contracts import (
    AnalysisSignal,
    EvidenceReference,
    FundamentalAgentSignal,
    FundamentalAnalysis,
    FundamentalNlpAnalysisResponse,
    ProviderResult,
)


def apply_fundamental_agent_result(
    analysis: FundamentalAnalysis,
    result: ProviderResult[FundamentalNlpAnalysisResponse],
) -> FundamentalAnalysis:
    """Attach validated agent output while preserving provider degradation warnings."""

    warnings = analysis.warnings + result.warnings
    if result.data is None:
        return analysis.model_copy(update={"warnings": warnings})

    response = result.data
    agent_signal = FundamentalAgentSignal(
        request_id=response.request_id,
        provider_name=result.provider_name,
        runner_name=response.audit.runner_name,
        raw_response_id=response.audit.raw_response_id or result.raw_snapshot_id,
        signal=response.signal,
        confidence=response.confidence,
        summary=response.summary,
        source_evidence_ids=response.source_evidence_ids,
        claim_count=len(response.claims),
        risk_count=len(response.risks),
        contradictions=response.contradictions,
        warning_ids=_warning_ids(result),
        confidence_inputs=response.confidence_inputs,
    )
    return analysis.model_copy(
        update={
            "summary": f"{analysis.summary} Fundamental agent: {response.summary}",
            "signal": _combine_signal(analysis, response),
            "confidence": max(analysis.confidence, response.confidence),
            "metrics": (
                *analysis.metrics,
                analysis_metric(
                    "fundamental-agent-confidence",
                    response.confidence,
                    unit="score",
                    as_of=response.as_of,
                    metadata={
                        "provider_name": result.provider_name,
                        "request_id": response.request_id,
                    },
                ),
            ),
            "evidence": _merge_evidence_refs(analysis.evidence, response.evidence),
            "warnings": warnings,
            "assumptions": tuple(
                dict.fromkeys(
                    (
                        *analysis.assumptions,
                        *response.assumptions,
                        "Fundamental agent output is cited interpretation, not observed fact.",
                    )
                )
            ),
            "agent_signal": agent_signal,
        }
    )


def _combine_signal(
    analysis: FundamentalAnalysis,
    response: FundamentalNlpAnalysisResponse,
) -> AnalysisSignal:
    if analysis.signal == response.signal:
        return analysis.signal
    if response.confidence >= analysis.confidence + 0.15:
        return response.signal
    return analysis.signal


def _merge_evidence_refs(
    first: tuple[EvidenceReference, ...],
    second: tuple[EvidenceReference, ...],
) -> tuple[EvidenceReference, ...]:
    refs: list[EvidenceReference] = []
    seen: set[str] = set()
    for reference in (*first, *second):
        if reference.evidence_id in seen:
            continue
        seen.add(reference.evidence_id)
        refs.append(reference)
    return tuple(refs)


def _warning_ids(result: ProviderResult[FundamentalNlpAnalysisResponse]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            f"{warning.provider_name or result.provider_name}:{warning.code.value}"
            for warning in result.warnings
        )
    )


__all__ = ["apply_fundamental_agent_result"]

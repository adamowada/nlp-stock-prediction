from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from nlp_stock_prediction.agents import FixtureFundamentalAgentProvider
from nlp_stock_prediction.analysis import apply_fundamental_agent_result
from nlp_stock_prediction.contracts import AnalysisSignal, FundamentalAnalysis
from nlp_stock_prediction.contracts.analysis import FundamentalNlpAnalysisRequest
from nlp_stock_prediction.contracts.enums import (
    FreshnessStatus,
    ProviderStatus,
    RetrievalMethod,
    SourceKind,
    WarningCode,
)
from nlp_stock_prediction.contracts.evidence import SourceEvidence
from nlp_stock_prediction.contracts.provenance import SourceProvenance

pytestmark = pytest.mark.unit

RUN_DATE = date(2026, 5, 11)
NOW = datetime(2026, 5, 11, 16, 0, tzinfo=UTC)


def _provenance(
    evidence_id: str,
    *,
    freshness_status: FreshnessStatus = FreshnessStatus.FRESH,
) -> SourceProvenance:
    return SourceProvenance(
        provider_name="fixture-news",
        source_kind=SourceKind.NEWS_ARTICLE,
        retrieval_method=RetrievalMethod.FIXTURE,
        fetched_at=NOW,
        observed_at=NOW,
        source_url=f"https://example.test/news/{evidence_id}",
        raw_identifier=f"raw-{evidence_id}",
        raw_snapshot_id=f"raw-snapshot-{evidence_id}",
        freshness_status=freshness_status,
        freshness_seconds=300 if freshness_status == FreshnessStatus.FRESH else 172_800,
        provider_metadata={"fixture": True},
    )


def _evidence(
    evidence_id: str,
    text: str,
    *,
    freshness_status: FreshnessStatus = FreshnessStatus.FRESH,
    stance: str = "positive",
) -> SourceEvidence:
    return SourceEvidence(
        evidence_id=evidence_id,
        source_kind=SourceKind.NEWS_ARTICLE,
        ticker="nvda",
        title=f"{evidence_id} title",
        text=text,
        created_at=NOW,
        permalink=f"https://example.test/news/{evidence_id}",
        matched_tickers=("nvda",),
        provenance=_provenance(evidence_id, freshness_status=freshness_status),
        metadata={"stance": stance},
    )


def _request(
    evidence: tuple[SourceEvidence, ...] = (
        _evidence(
            "news-nvda-growth",
            "NVDA reported revenue growth of 62% and net margin of 48% in the latest quarter.",
        ),
    ),
) -> FundamentalNlpAnalysisRequest:
    return FundamentalNlpAnalysisRequest(
        request_id="fundamental-agent-nvda-2026-05-11",
        ticker="nvda",
        run_date=RUN_DATE,
        as_of=NOW,
        evidence=evidence,
        prompt_version="fundamental-agent-prompt-v1",
        schema_version="fundamental-agent-response-v1",
        focus_areas=("valuation", "profitability", "growth", "risk"),
    )


def _citation(
    evidence_id: str = "news-nvda-growth",
    quote: str = "revenue growth of 62%",
) -> dict[str, Any]:
    return {"evidence_id": evidence_id, "quote": quote, "relevance": 0.92}


def _valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "request_id": "fundamental-agent-nvda-2026-05-11",
        "ticker": "NVDA",
        "as_of": NOW,
        "summary": "NVDA growth and profitability are supported by cited fixture evidence.",
        "signal": "supports",
        "confidence": 0.76,
        "source_evidence_ids": ["news-nvda-growth"],
        "citations": [_citation()],
        "claims": [
            {
                "claim_id": "claim-growth",
                "claim_type": "observed",
                "text": "NVDA reported revenue growth of 62%.",
                "citations": [_citation()],
                "confidence": 0.86,
            },
            {
                "claim_id": "claim-quality",
                "claim_type": "interpretation",
                "text": "Growth and margin support a quality fundamental read.",
                "citations": [_citation(quote="net margin of 48%")],
                "confidence": 0.71,
            },
        ],
        "risks": [
            {
                "risk_id": "risk-growth-deceleration",
                "text": "If growth slows, valuation risk could rise.",
                "severity": "medium",
                "citations": [_citation()],
            }
        ],
        "assumptions": ("Fixture evidence is representative of the provider packet.",),
        "confidence_inputs": {
            "claim_count": 2,
            "cited_source_count": 1,
            "stale_source_count": 0,
        },
    }
    payload.update(overrides)
    return payload


def test_fixture_fundamental_agent_accepts_cited_audit_ready_output() -> None:
    request = _request()
    provider = FixtureFundamentalAgentProvider(payload=_valid_payload(), fetched_at=NOW)

    result = provider.analyze_fundamentals(request)

    assert result.status == ProviderStatus.OK
    assert result.data is not None
    assert result.data.ticker == "NVDA"
    assert result.data.source_evidence_ids == ("news-nvda-growth",)
    assert result.data.validation_warnings == ()
    assert result.data.evidence[0].evidence_id == "news-nvda-growth"
    assert result.data.audit.raw_response_id == result.raw_snapshot_id
    assert result.data.audit.prompt_sha256 is not None
    assert result.cache_key is not None


def test_apply_fundamental_agent_result_attaches_report_sidecar() -> None:
    provider = FixtureFundamentalAgentProvider(
        payload=_valid_payload(),
        provider_name="fundamental-agent",
        fetched_at=NOW,
    )
    result = provider.analyze_fundamentals(_request())
    analysis = FundamentalAnalysis(
        ticker="NVDA",
        summary="Baseline fundamental analysis is mixed.",
        signal=AnalysisSignal.MIXED,
        confidence=0.52,
    )

    integrated = apply_fundamental_agent_result(analysis, result)

    assert integrated.agent_signal is not None
    assert integrated.agent_signal.provider_name == "fundamental-agent"
    assert integrated.agent_signal.claim_count == 2
    assert integrated.agent_signal.risk_count == 1
    assert integrated.agent_signal.source_evidence_ids == ("news-nvda-growth",)
    assert integrated.signal == AnalysisSignal.SUPPORTS
    assert integrated.confidence == 0.76
    assert [reference.evidence_id for reference in integrated.evidence] == ["news-nvda-growth"]
    assert any(metric.name == "fundamental-agent-confidence" for metric in integrated.metrics)
    assert "Fundamental agent:" in integrated.summary


def test_apply_fundamental_agent_result_marks_close_conflict_mixed_without_confidence_boost() -> (
    None
):
    provider = FixtureFundamentalAgentProvider(
        payload=_valid_payload(signal="conflicts", confidence=0.62),
        provider_name="fundamental-agent",
        fetched_at=NOW,
    )
    result = provider.analyze_fundamentals(_request())
    analysis = FundamentalAnalysis(
        ticker="NVDA",
        summary="Baseline fundamental analysis supports the setup.",
        signal=AnalysisSignal.SUPPORTS,
        confidence=0.56,
    )

    integrated = apply_fundamental_agent_result(analysis, result)

    assert integrated.signal == AnalysisSignal.MIXED
    assert integrated.confidence == 0.6


def test_fixture_fundamental_agent_rejects_missing_claim_citations() -> None:
    payload = _valid_payload(
        claims=[
            {
                "claim_id": "claim-uncited",
                "claim_type": "observed",
                "text": "NVDA reported revenue growth of 62%.",
                "citations": [],
                "confidence": 0.86,
            }
        ]
    )
    provider = FixtureFundamentalAgentProvider(payload=payload, fetched_at=NOW)

    result = provider.analyze_fundamentals(_request())

    assert result.status == ProviderStatus.MALFORMED
    assert result.data is None
    assert result.warnings[0].code == WarningCode.LLM_EVIDENCE_MISMATCH
    assert "missing citations" in result.warnings[0].message


def test_fixture_fundamental_agent_rejects_unsupported_observed_claims() -> None:
    payload = _valid_payload(
        claims=[
            {
                "claim_id": "claim-invented-margin",
                "claim_type": "observed",
                "text": "NVDA reported profit margin of 99%.",
                "citations": [_citation()],
                "confidence": 0.86,
            }
        ]
    )
    provider = FixtureFundamentalAgentProvider(payload=payload, fetched_at=NOW)

    result = provider.analyze_fundamentals(_request())

    assert result.status == ProviderStatus.MALFORMED
    assert result.data is None
    assert any(warning.code == WarningCode.UNSUPPORTED_CLAIM for warning in result.warnings)


def test_fixture_fundamental_agent_surfaces_stale_and_contradictory_evidence() -> None:
    request = _request(
        evidence=(
            _evidence(
                "news-nvda-growth",
                "NVDA reported revenue growth of 62% and net margin of 48%.",
                stance="positive",
            ),
            _evidence(
                "news-nvda-guide-cut",
                "NVDA guidance was cut and free cash flow declined.",
                freshness_status=FreshnessStatus.STALE,
                stance="negative",
            ),
        )
    )
    payload = _valid_payload(
        source_evidence_ids=["news-nvda-growth", "news-nvda-guide-cut"],
        citations=[
            _citation(),
            _citation("news-nvda-guide-cut", "guidance was cut"),
        ],
        risks=[
            {
                "risk_id": "risk-guide-cut",
                "text": "A guidance cut conflicts with the growth setup.",
                "severity": "high",
                "citations": [_citation("news-nvda-guide-cut", "guidance was cut")],
            }
        ],
    )
    provider = FixtureFundamentalAgentProvider(payload=payload, fetched_at=NOW)

    result = provider.analyze_fundamentals(request)

    assert result.status == ProviderStatus.PARTIAL
    assert result.data is not None
    warning_codes = {warning.code for warning in result.warnings}
    assert WarningCode.STALE_DATA in warning_codes
    assert WarningCode.PARTIAL_DATA in warning_codes
    assert result.data.validation_warnings == result.warnings


def test_fixture_fundamental_agent_rejects_malformed_json() -> None:
    provider = FixtureFundamentalAgentProvider(payload="{not-json", fetched_at=NOW)

    result = provider.analyze_fundamentals(_request())

    assert result.status == ProviderStatus.MALFORMED
    assert result.data is None
    assert result.warnings[0].code == WarningCode.LLM_SCHEMA_INVALID


def test_fixture_fundamental_agent_no_agent_fallback_returns_empty_result() -> None:
    provider = FixtureFundamentalAgentProvider(payload=None, fetched_at=NOW)

    result = provider.analyze_fundamentals(_request())

    assert result.status == ProviderStatus.EMPTY
    assert result.data is None
    assert result.warnings[0].code == WarningCode.NO_DATA
    assert result.health.status == ProviderStatus.EMPTY

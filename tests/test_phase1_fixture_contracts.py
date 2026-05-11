from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from nlp_stock_prediction.contracts import (
    FixtureManifest,
    NormalizedFixture,
    ProviderRequest,
    RawProviderFixture,
)


def _recorded_at() -> datetime:
    return datetime(2026, 5, 11, 16, 30, tzinfo=UTC)


def _provider_request(*, request_id: str = "reddit-devvit-card-2026-05-11") -> ProviderRequest:
    return ProviderRequest(
        request_id=request_id,
        run_date=date(2026, 5, 11),
        tickers=("TSLA", "NVDA", "AMD", "AAPL", "MU", "SPY"),
        query="r/wallstreetbets Devvit daily ticker card",
        options={"fixture_mode": True, "include_raw_html": True},
    )


def _raw_fixture(
    *,
    provider_name: str = "fixture-reddit",
    scenario: str = "normal_six_ticker_day",
    response_path: str = "raw/reddit/normal_six_ticker_day.html",
    request_id: str = "reddit-devvit-card-2026-05-11",
) -> RawProviderFixture:
    return RawProviderFixture(
        provider_name=provider_name,
        scenario=scenario,
        recorded_at=_recorded_at(),
        request=_provider_request(request_id=request_id),
        response_path=response_path,
        content_type="text/html",
        provider_metadata={
            "status_code": 200,
            "source": "recorded_fixture",
            "redaction_policy": {
                "secrets_removed": True,
                "raw_user_ids_hashed": True,
            },
        },
        redactions=("authorization_header", "session_cookie", "raw_user_id"),
    )


def _normalized_fixture(
    *,
    layer: str = "provider_result",
    path: str = "normalized/provider_results/reddit/normal_six_ticker_day.json",
    source_fixture_ids: tuple[str, ...] = ("raw/reddit/normal_six_ticker_day.html",),
    expected_record_count: int | None = 6,
) -> NormalizedFixture:
    return NormalizedFixture.model_validate(
        {
            "scenario": "normal_six_ticker_day",
            "layer": layer,
            "path": path,
            "source_fixture_ids": source_fixture_ids,
            "expected_record_count": expected_record_count,
            "metadata": {
                "contract": "ProviderResult",
                "inherits_raw_redactions": True,
            },
        }
    )


@pytest.mark.schema
def test_raw_provider_fixture_captures_request_snapshot_and_redaction_metadata() -> None:
    raw_fixture = _raw_fixture()

    assert raw_fixture.fixture_version == 1
    assert raw_fixture.provider_name == "fixture-reddit"
    assert raw_fixture.request.run_date == date(2026, 5, 11)
    assert raw_fixture.request.tickers == ("TSLA", "NVDA", "AMD", "AAPL", "MU", "SPY")
    assert raw_fixture.response_path == "raw/reddit/normal_six_ticker_day.html"

    dumped = raw_fixture.model_dump(mode="json")

    assert dumped["recorded_at"] == "2026-05-11T16:30:00Z"
    assert dumped["content_type"] == "text/html"
    assert dumped["request"]["options"]["include_raw_html"] is True
    assert dumped["provider_metadata"]["redaction_policy"]["secrets_removed"] is True
    assert dumped["redactions"] == ["authorization_header", "session_cookie", "raw_user_id"]


@pytest.mark.schema
def test_normalized_fixture_records_layer_path_source_ids_and_counts() -> None:
    normalized_fixture = _normalized_fixture(
        layer="evidence",
        path="normalized/evidence/normal_six_ticker_day.json",
        source_fixture_ids=(
            "normalized/provider_results/reddit/normal_six_ticker_day.json",
            "normalized/provider_results/news/normal_six_ticker_day.json",
        ),
        expected_record_count=18,
    )

    dumped = normalized_fixture.model_dump(mode="json")

    assert dumped["fixture_version"] == 1
    assert dumped["scenario"] == "normal_six_ticker_day"
    assert dumped["layer"] == "evidence"
    assert dumped["path"] == "normalized/evidence/normal_six_ticker_day.json"
    assert dumped["source_fixture_ids"] == [
        "normalized/provider_results/reddit/normal_six_ticker_day.json",
        "normalized/provider_results/news/normal_six_ticker_day.json",
    ]
    assert dumped["expected_record_count"] == 18
    assert dumped["metadata"]["inherits_raw_redactions"] is True


@pytest.mark.schema
def test_fixture_manifest_composes_raw_normalized_and_expected_report_artifacts() -> None:
    reddit_raw = _raw_fixture()
    news_raw = _raw_fixture(
        provider_name="fixture-news",
        response_path="raw/news/normal_six_ticker_day.json",
        request_id="news-search-2026-05-11",
    )
    provider_result = _normalized_fixture()
    evidence = _normalized_fixture(
        layer="evidence",
        path="normalized/evidence/normal_six_ticker_day.json",
        source_fixture_ids=(
            "raw/reddit/normal_six_ticker_day.html",
            "raw/news/normal_six_ticker_day.json",
        ),
        expected_record_count=18,
    )

    manifest = FixtureManifest(
        scenario="normal_six_ticker_day",
        run_date=date(2026, 5, 11),
        raw_fixtures=(reddit_raw, news_raw),
        normalized_fixtures=(provider_result, evidence),
        expected_report_json_path="reports/normal_six_ticker_day/expected_report.json",
        expected_report_markdown_path="reports/normal_six_ticker_day/expected_report.md",
        notes=("Phase 1 fixture harness scenario",),
    )

    assert [fixture.provider_name for fixture in manifest.raw_fixtures] == [
        "fixture-reddit",
        "fixture-news",
    ]
    assert [fixture.layer for fixture in manifest.normalized_fixtures] == [
        "provider_result",
        "evidence",
    ]
    assert manifest.expected_report_json_path == (
        "reports/normal_six_ticker_day/expected_report.json"
    )
    assert manifest.expected_report_markdown_path == (
        "reports/normal_six_ticker_day/expected_report.md"
    )
    assert manifest.notes == ("Phase 1 fixture harness scenario",)


@pytest.mark.schema
def test_fixture_manifest_json_serialization_round_trips_nested_fixtures() -> None:
    manifest = FixtureManifest(
        scenario="normal_six_ticker_day",
        run_date=date(2026, 5, 11),
        raw_fixtures=(_raw_fixture(),),
        normalized_fixtures=(
            _normalized_fixture(),
            _normalized_fixture(
                layer="report",
                path="reports/normal_six_ticker_day/expected_report.json",
                source_fixture_ids=("normalized/evidence/normal_six_ticker_day.json",),
                expected_record_count=None,
            ),
        ),
        expected_report_json_path="reports/normal_six_ticker_day/expected_report.json",
        expected_report_markdown_path="reports/normal_six_ticker_day/expected_report.md",
    )

    serialized = manifest.model_dump_json(indent=2)
    round_tripped = FixtureManifest.model_validate_json(serialized)

    assert '"run_date": "2026-05-11"' in serialized
    assert '"recorded_at": "2026-05-11T16:30:00Z"' in serialized
    assert '"redactions": [' in serialized
    assert round_tripped == manifest


@pytest.mark.schema
def test_redaction_metadata_survives_raw_fixture_json_round_trip() -> None:
    raw_fixture = _raw_fixture()

    round_tripped = RawProviderFixture.model_validate_json(raw_fixture.model_dump_json())
    dumped = round_tripped.model_dump(mode="json")

    assert dumped["provider_metadata"]["redaction_policy"] == {
        "secrets_removed": True,
        "raw_user_ids_hashed": True,
    }
    assert dumped["redactions"] == ["authorization_header", "session_cookie", "raw_user_id"]


@pytest.mark.schema
def test_fixture_contracts_reject_invalid_shapes() -> None:
    raw_payload = _raw_fixture().model_dump(mode="json")
    raw_payload["unexpected_field"] = "not allowed"
    with pytest.raises(ValidationError):
        RawProviderFixture.model_validate(raw_payload)

    invalid_metadata_payload = _raw_fixture().model_dump(mode="json")
    invalid_metadata_payload["provider_metadata"] = {"not_json": object()}
    with pytest.raises(ValidationError):
        RawProviderFixture.model_validate(invalid_metadata_payload)

    with pytest.raises(ValidationError):
        RawProviderFixture.model_validate(
            {
                "provider_name": " ",
                "scenario": "normal_six_ticker_day",
                "recorded_at": "2026-05-11T16:30:00Z",
                "request": _provider_request().model_dump(mode="json"),
                "response_path": "raw/reddit/normal_six_ticker_day.html",
            }
        )

    with pytest.raises(ValidationError):
        NormalizedFixture.model_validate(
            {
                "scenario": "normal_six_ticker_day",
                "layer": "unsupported_layer",
                "path": "normalized/evidence/normal_six_ticker_day.json",
            }
        )

    with pytest.raises(ValidationError):
        NormalizedFixture.model_validate(
            {
                "scenario": "normal_six_ticker_day",
                "layer": "evidence",
                "path": "normalized/evidence/normal_six_ticker_day.json",
                "expected_record_count": -1,
            }
        )

    with pytest.raises(ValidationError):
        FixtureManifest.model_validate(
            {
                "fixture_version": 0,
                "scenario": "normal_six_ticker_day",
                "run_date": "2026-05-11",
            }
        )

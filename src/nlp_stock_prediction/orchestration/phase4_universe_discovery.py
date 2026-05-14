"""First-class Phase 4 instrument universe discovery tool."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import quote

from nlp_stock_prediction.contracts.base import JsonObject
from nlp_stock_prediction.contracts.enums import AssetClass, TradabilityStatus
from nlp_stock_prediction.contracts.instruments import (
    Instrument,
    InstrumentDataAvailability,
    InstrumentQuery,
    InstrumentUniverse,
    InstrumentUniverseRequest,
    ProviderInstrumentId,
    TradabilityEvidence,
)
from nlp_stock_prediction.contracts.report import AuditArtifact
from nlp_stock_prediction.instruments.registry import InstrumentRegistry, instrument_to_record
from nlp_stock_prediction.orchestration.artifacts import ArtifactIndex
from nlp_stock_prediction.orchestration.context import RunContext
from nlp_stock_prediction.orchestration.phase4_execution import safe_phase4_tool_execution
from nlp_stock_prediction.orchestration.report_data_modes import (
    report_data_mode_metadata_for_run_id,
)
from nlp_stock_prediction.storage.records import (
    InstrumentRecord,
    SourceQueryRecord,
    ToolRunRecord,
)
from nlp_stock_prediction.storage.sqlite import SQLiteStore

PHASE4_UNIVERSE_SCHEMA_VERSION = "phase4.instrument-universe.v1"
PHASE4_FIXTURE_PROVIDER = "phase4-fixture-directory"
PHASE4_TOOL_NAME = "phase4_universe_discovery"
PHASE4_TOOL_VERSION = "phase4.fixture.v1"


@dataclass(frozen=True)
class UniverseDiscoverySourceQuery:
    """Provider query provenance observed while discovering instrument candidates."""

    provider: str
    query: str
    url: str | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class UniverseDiscoveryProviderResult:
    """Candidate instruments and source queries returned by a discovery provider."""

    instruments: tuple[Instrument, ...] = ()
    source_queries: tuple[UniverseDiscoverySourceQuery, ...] = ()
    warnings: tuple[str, ...] = ()


class UniverseDiscoveryProvider(Protocol):
    """Adapter seam for fixture-backed and future live universe discovery providers."""

    @property
    def provider_name(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def discover(
        self,
        request: InstrumentUniverseRequest,
        *,
        retrieved_at: datetime,
    ) -> UniverseDiscoveryProviderResult: ...


@dataclass(frozen=True)
class Phase4UniverseDiscoveryToolResult:
    """Persisted Phase 4 universe discovery output."""

    universe: InstrumentUniverse
    artifact: AuditArtifact
    tool_run_record: ToolRunRecord
    source_query_records: tuple[SourceQueryRecord, ...]
    instrument_records: tuple[InstrumentRecord, ...]

    @property
    def artifact_id(self) -> str:
        return self.artifact.artifact_id

    @property
    def tool_run_id(self) -> str:
        return self.tool_run_record.tool_run_id

    @property
    def instrument_ids(self) -> tuple[str, ...]:
        return self.universe.instrument_ids


@dataclass(frozen=True)
class Phase4UniverseDiscoveryTool:
    """Resolve and persist an instrument universe through the Phase 3 registry."""

    provider: UniverseDiscoveryProvider = field(
        default_factory=lambda: Phase4FixtureUniverseProvider()
    )

    def run(
        self,
        *,
        request: InstrumentUniverseRequest,
        context: RunContext,
        store: SQLiteStore,
        repo_root: Path,
    ) -> Phase4UniverseDiscoveryToolResult:
        validated = InstrumentUniverseRequest.model_validate(request)
        retrieved_at = context.generated_at
        tool_run_id = _tool_run_id(context.run_id, validated.request_id)
        artifact_id = _artifact_id(context.run_id, validated.request_id)
        mode_metadata = report_data_mode_metadata_for_run_id(store, context.run_id)
        inputs: JsonObject = {
            "request": cast(JsonObject, validated.model_dump(mode="json")),
            "provider": self.provider.provider_name,
            **mode_metadata,
        }
        with safe_phase4_tool_execution(
            store=store,
            artifact_roots=(context.audit_dir,),
            tool_run_id=tool_run_id,
            run_id=context.run_id,
            tool_name=PHASE4_TOOL_NAME,
            tool_version=PHASE4_TOOL_VERSION,
            started_at=retrieved_at,
            inputs=inputs,
        ):
            provider_result = self.provider.discover(validated, retrieved_at=retrieved_at)
            source_query_records = _source_query_records(
                tool_run_id=tool_run_id,
                retrieved_at=retrieved_at,
                source_queries=provider_result.source_queries,
            )
            source_query_ids_by_instrument = _source_query_ids_by_instrument(
                source_queries=provider_result.source_queries,
                records=source_query_records,
            )
            candidate_records = tuple(
                instrument_record_from_contract(
                    instrument,
                    provider_name=self.provider.provider_name,
                    source_query_ids=tuple(
                        source_query_ids_by_instrument.get(instrument.instrument_id, ())
                    ),
                )
                for instrument in provider_result.instruments
            )

            store.initialize()
            registry = InstrumentRegistry(store, initialize=False)
            for record in candidate_records:
                store.upsert_instrument(record)

            resolved_universe = registry.resolve_universe(validated)
            universe = _phase4_universe(
                request=validated,
                resolved_universe=resolved_universe,
                generated_at=retrieved_at,
                provider_result=provider_result,
                source_query_records=source_query_records,
                provider_name=self.provider.provider_name,
                provider_version=self.provider.provider_version,
            )
            warnings = _dedupe_strings((*provider_result.warnings, *universe.warnings))
            tool_run_record = ToolRunRecord(
                tool_run_id=tool_run_id,
                run_id=context.run_id,
                tool_name=PHASE4_TOOL_NAME,
                tool_version=PHASE4_TOOL_VERSION,
                status="partial" if warnings else "successful",
                started_at=retrieved_at,
                completed_at=retrieved_at,
                inputs={
                    "request": cast(JsonObject, validated.model_dump(mode="json")),
                    "provider": self.provider.provider_name,
                    **mode_metadata,
                },
                warnings=warnings,
            )
            store.record_tool_run(tool_run_record)
            for source_query_record in source_query_records:
                store.record_source_query(source_query_record)

            selected_records = tuple(
                instrument_record_from_contract(
                    instrument,
                    provider_name=self.provider.provider_name,
                    source_query_ids=tuple(
                        source_query_ids_by_instrument.get(instrument.instrument_id, ())
                    ),
                )
                for instrument in universe.instruments
            )
            for record in selected_records:
                store.upsert_instrument(record)

            artifact = ArtifactIndex.for_directory(
                store=store,
                repo_root=repo_root,
                base_dir=context.audit_dir,
                created_at=retrieved_at,
                produced_by=PHASE4_TOOL_NAME,
                tool_run_id=tool_run_id,
                schema_version=PHASE4_UNIVERSE_SCHEMA_VERSION,
                default_metadata=mode_metadata,
            ).write_json(
                artifact_id=artifact_id,
                artifact_type="instrument_universe",
                filename=_artifact_filename(validated.request_id, context.run_id),
                payload=phase4_universe_artifact_payload(
                    run_id=context.run_id,
                    universe=universe,
                ),
                record_count=len(universe.resolutions),
                metadata={
                    "universe_id": universe.request_id,
                    "instrument_ids": list(universe.instrument_ids),
                    "resolution_status_counts": _resolution_status_counts(universe),
                    "source_query_ids": [record.source_query_id for record in source_query_records],
                    "warnings": list(warnings),
                },
            )
            return Phase4UniverseDiscoveryToolResult(
                universe=universe,
                artifact=artifact,
                tool_run_record=tool_run_record,
                source_query_records=source_query_records,
                instrument_records=selected_records,
            )


@dataclass(frozen=True)
class Phase4FixtureUniverseProvider:
    """Deterministic offline instrument discovery provider for Phase 4 tests."""

    provider_name: str = PHASE4_FIXTURE_PROVIDER
    provider_version: str = PHASE4_TOOL_VERSION

    def discover(
        self,
        request: InstrumentUniverseRequest,
        *,
        retrieved_at: datetime,
    ) -> UniverseDiscoveryProviderResult:
        validated = InstrumentUniverseRequest.model_validate(request)
        catalog = _fixture_catalog(retrieved_at)
        instruments: list[Instrument] = []
        source_queries: list[UniverseDiscoverySourceQuery] = []

        for index, target in enumerate(_iter_request_targets(validated)):
            matches = _fixture_matches(
                target.query,
                catalog,
                provider_name=self.provider_name,
                requested_instrument_id=target.requested_instrument_id,
            )
            instruments.extend(matches)
            source_queries.append(
                UniverseDiscoverySourceQuery(
                    provider=self.provider_name,
                    query=target.query.query,
                    url=_fixture_query_url(target.query.query),
                    metadata={
                        "source": "fixture",
                        "provider_version": self.provider_version,
                        "request_id": validated.request_id,
                        "request_query_index": index,
                        "request_query": cast(JsonObject, target.query.model_dump(mode="json")),
                        "requested_instrument_id": target.requested_instrument_id,
                        "matched_instrument_ids": [
                            instrument.instrument_id for instrument in matches
                        ],
                    },
                )
            )

        return UniverseDiscoveryProviderResult(
            instruments=_dedupe_instruments(tuple(instruments)),
            source_queries=tuple(source_queries),
            warnings=(),
        )


def phase4_universe_artifact_payload(*, run_id: str, universe: InstrumentUniverse) -> JsonObject:
    """Serialize a Phase 4 universe artifact with a stable top-level envelope."""

    universe_payload = universe.model_dump(mode="json")
    universe_payload["instrument_ids"] = list(universe.instrument_ids)
    return cast(
        JsonObject,
        {
            "schema_version": PHASE4_UNIVERSE_SCHEMA_VERSION,
            "run_id": run_id,
            "universe_id": universe.request_id,
            "universe": universe_payload,
        },
    )


def instrument_record_from_contract(
    instrument: Instrument,
    *,
    provider_name: str = PHASE4_FIXTURE_PROVIDER,
    source_query_ids: tuple[str, ...] = (),
) -> InstrumentRecord:
    """Convert a discovered instrument into a registry persistence record."""

    metadata: JsonObject = {
        "phase4_universe_discovery": True,
        "discovery_provider": provider_name,
    }
    if source_query_ids:
        metadata["source_query_ids"] = list(source_query_ids)
    return instrument_to_record(instrument, metadata=metadata)


@dataclass(frozen=True)
class _RequestTarget:
    query: InstrumentQuery
    requested_instrument_id: str | None = None


def _phase4_universe(
    *,
    request: InstrumentUniverseRequest,
    resolved_universe: InstrumentUniverse,
    generated_at: datetime,
    provider_result: UniverseDiscoveryProviderResult,
    source_query_records: tuple[SourceQueryRecord, ...],
    provider_name: str,
    provider_version: str,
) -> InstrumentUniverse:
    metadata: JsonObject = dict(request.metadata)
    metadata.update(
        {
            "schema_version": PHASE4_UNIVERSE_SCHEMA_VERSION,
            "provider": provider_name,
            "provider_version": provider_version,
            "source": "fixture" if provider_name == PHASE4_FIXTURE_PROVIDER else "provider",
            "discovered_candidate_count": len(provider_result.instruments),
            "source_query_ids": [record.source_query_id for record in source_query_records],
            "resolution_status_counts": _resolution_status_counts(resolved_universe),
        }
    )
    return InstrumentUniverse(
        request_id=resolved_universe.request_id,
        generated_at=generated_at,
        resolutions=resolved_universe.resolutions,
        instruments=resolved_universe.instruments,
        warnings=_dedupe_strings((*provider_result.warnings, *resolved_universe.warnings)),
        metadata=metadata,
    )


def _fixture_catalog(generated_at: datetime) -> tuple[Instrument, ...]:
    return (
        _fixture_instrument(
            instrument_id="instrument:equity:us:tsla",
            symbol="TSLA",
            display_name="Tesla Inc.",
            asset_class=AssetClass.STOCK,
            venue="NASDAQ",
            provider_identifier="TSLA",
            generated_at=generated_at,
            aliases=("TESLA", "TSLA.US"),
            notes="Fixture record for a retail-accessible U.S. equity.",
        ),
        _fixture_instrument(
            instrument_id="instrument:etf:us:spy",
            symbol="SPY",
            display_name="SPDR S&P 500 ETF Trust",
            asset_class=AssetClass.ETF,
            venue="NYSEARCA",
            provider_identifier="SPY",
            generated_at=generated_at,
            aliases=("SPY.US",),
            notes="Fixture record for broad-market ETF proxy context.",
        ),
        _fixture_instrument(
            instrument_id="instrument:etf:us:qqq",
            symbol="QQQ",
            display_name="Invesco QQQ Trust",
            asset_class=AssetClass.ETF,
            venue="NASDAQ",
            provider_identifier="QQQ",
            generated_at=generated_at,
            aliases=("QQQ.US",),
            notes="Fixture record for technology-heavy index ETF context.",
        ),
        _fixture_instrument(
            instrument_id="instrument:crypto:btc-usd",
            symbol="BTC/USD",
            display_name="Bitcoin versus U.S. dollar",
            asset_class=AssetClass.CRYPTO,
            venue="fixture-crypto-venues",
            provider_identifier="BTC/USD",
            generated_at=generated_at,
            aliases=("BTC", "BTC-USD", "BTC:USD"),
            notes="Fixture crypto pair with provider data availability only.",
        ),
        _fixture_instrument(
            instrument_id="instrument:futures:cme:esm6",
            symbol="ESM6",
            display_name="E-mini S&P 500 June 2026 futures context",
            asset_class=AssetClass.FUTURES,
            venue="CME",
            provider_identifier="ESM6",
            generated_at=generated_at,
            aliases=("ESM26",),
            tradability_status=TradabilityStatus.RESTRICTED,
            notes=(
                "Fixture futures record is macro/proxy context only; availability does not "
                "imply retail trading access."
            ),
        ),
        _fixture_instrument(
            instrument_id="instrument:equity:us:ai",
            symbol="AI",
            display_name="C3.ai Inc.",
            asset_class=AssetClass.STOCK,
            venue="NYSE",
            provider_identifier="AI",
            generated_at=generated_at,
            aliases=("C3AI",),
            notes="Fixture equity side of the ambiguous AI query.",
        ),
        _fixture_instrument(
            instrument_id="instrument:crypto:ai-usd",
            symbol="AI/USD",
            display_name="AI token versus U.S. dollar fixture",
            asset_class=AssetClass.CRYPTO,
            venue="fixture-crypto-venues",
            provider_identifier="AI/USD",
            generated_at=generated_at,
            aliases=("AI", "AI-USD"),
            notes="Fixture crypto side of the ambiguous AI query.",
        ),
    )


def _fixture_instrument(
    *,
    instrument_id: str,
    symbol: str,
    display_name: str,
    asset_class: AssetClass,
    venue: str | None,
    provider_identifier: str,
    generated_at: datetime,
    aliases: tuple[str, ...] = (),
    tradability_status: TradabilityStatus = TradabilityStatus.UNKNOWN,
    notes: str,
) -> Instrument:
    source_url = _fixture_instrument_url(provider_identifier)
    return Instrument(
        instrument_id=instrument_id,
        symbol=symbol,
        display_name=display_name,
        asset_class=asset_class,
        venue=venue,
        aliases=aliases,
        provider_ids=(
            ProviderInstrumentId(
                provider=PHASE4_FIXTURE_PROVIDER,
                identifier=provider_identifier.upper(),
                namespace="fixture-symbol",
                url=source_url,
                metadata={"source": "phase4_fixture"},
            ),
        ),
        tradability_evidence=(
            TradabilityEvidence(
                provider=PHASE4_FIXTURE_PROVIDER,
                status=tradability_status,
                retrieved_at=generated_at,
                source_url=source_url,
                raw_identifier=f"{provider_identifier.upper()}:phase4-fixture",
                notes=notes,
                metadata={"source": "phase4_fixture"},
            ),
        ),
        data_availability=(
            InstrumentDataAvailability(
                provider=PHASE4_FIXTURE_PROVIDER,
                data_type="fixture_research_context",
                status=TradabilityStatus.AVAILABLE,
                checked_at=generated_at,
                provider_identifier=provider_identifier.upper(),
                notes="Fixture data is deterministic and offline.",
                metadata={"source": "phase4_fixture"},
            ),
        ),
        metadata={"phase": "phase4", "fixture": True},
    )


def _fixture_matches(
    query: InstrumentQuery,
    catalog: tuple[Instrument, ...],
    *,
    provider_name: str,
    requested_instrument_id: str | None,
) -> tuple[Instrument, ...]:
    if query.asset_class == AssetClass.UNKNOWN:
        return ()
    if query.provider is not None and _normalize(query.provider) != _normalize(provider_name):
        return ()

    matches: list[Instrument] = []
    if requested_instrument_id is not None:
        matches.extend(
            instrument
            for instrument in catalog
            if instrument.instrument_id == requested_instrument_id
            and _instrument_matches_query_or_provider_id(instrument, query)
        )

    lookup_values = (query.query, *query.aliases)
    for instrument in catalog:
        if any(_instrument_matches_lookup(instrument, value) for value in lookup_values):
            matches.append(instrument)
            continue
        if query.provider_identifier is not None and any(
            _provider_identifier_matches(
                provider_id,
                query.provider_identifier,
                provider=query.provider,
                namespace=query.provider_namespace,
            )
            for provider_id in instrument.provider_ids
        ):
            matches.append(instrument)
    return _dedupe_instruments(tuple(matches))


def _instrument_matches_query_or_provider_id(
    instrument: Instrument,
    query: InstrumentQuery,
) -> bool:
    lookup_values = (query.query, *query.aliases)
    if any(_instrument_matches_lookup(instrument, value) for value in lookup_values):
        return True
    if query.provider_identifier is None:
        return False
    return any(
        _provider_identifier_matches(
            provider_id,
            query.provider_identifier,
            provider=query.provider,
            namespace=query.provider_namespace,
        )
        for provider_id in instrument.provider_ids
    )


def _instrument_matches_lookup(instrument: Instrument, value: str) -> bool:
    normalized = _normalize(value)
    return normalized in {
        _normalize(instrument.symbol),
        *(_normalize(alias) for alias in instrument.aliases),
        *(_normalize(provider_id.identifier) for provider_id in instrument.provider_ids),
    }


def _provider_identifier_matches(
    provider_id: ProviderInstrumentId,
    value: str,
    *,
    provider: str | None,
    namespace: str | None,
) -> bool:
    if provider is not None and _normalize(provider_id.provider) != _normalize(provider):
        return False
    if namespace is not None and _normalize(provider_id.namespace or "") != _normalize(namespace):
        return False
    return _normalize(provider_id.identifier) == _normalize(value)


def _iter_request_targets(request: InstrumentUniverseRequest) -> tuple[_RequestTarget, ...]:
    targets = [_RequestTarget(query=query) for query in request.queries]
    targets.extend(
        _RequestTarget(
            query=entry.query,
            requested_instrument_id=entry.requested_instrument_id,
        )
        for watchlist in request.watchlists
        for entry in watchlist.entries
    )
    return tuple(targets)


def _source_query_records(
    *,
    tool_run_id: str,
    retrieved_at: datetime,
    source_queries: tuple[UniverseDiscoverySourceQuery, ...],
) -> tuple[SourceQueryRecord, ...]:
    records: list[SourceQueryRecord] = []
    for index, source_query in enumerate(source_queries):
        records.append(
            SourceQueryRecord(
                source_query_id=_source_query_id(tool_run_id, source_query, index),
                tool_run_id=tool_run_id,
                provider=source_query.provider,
                query=source_query.query,
                url=source_query.url,
                retrieved_at=retrieved_at,
                metadata=source_query.metadata,
            )
        )
    return tuple(records)


def _source_query_ids_by_instrument(
    *,
    source_queries: tuple[UniverseDiscoverySourceQuery, ...],
    records: tuple[SourceQueryRecord, ...],
) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, list[str]] = {}
    for source_query, record in zip(source_queries, records, strict=True):
        raw_matches = source_query.metadata.get("matched_instrument_ids")
        if not isinstance(raw_matches, list):
            continue
        for raw_instrument_id in raw_matches:
            if isinstance(raw_instrument_id, str):
                mapping.setdefault(raw_instrument_id, []).append(record.source_query_id)
    return {
        instrument_id: tuple(dict.fromkeys(source_query_ids))
        for instrument_id, source_query_ids in mapping.items()
    }


def _resolution_status_counts(universe: InstrumentUniverse) -> JsonObject:
    counts = Counter(resolution.status.value for resolution in universe.resolutions)
    return dict(sorted(counts.items()))


def _dedupe_instruments(instruments: tuple[Instrument, ...]) -> tuple[Instrument, ...]:
    records: dict[str, Instrument] = {}
    for instrument in instruments:
        records.setdefault(instrument.instrument_id, instrument)
    return tuple(records.values())


def _dedupe_strings(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _tool_run_id(run_id: str, request_id: str) -> str:
    return f"tool-phase4-universe-{_stable_digest(f'{run_id}:{request_id}')}"


def _artifact_id(run_id: str, request_id: str) -> str:
    return f"artifact-phase4-universe-{_stable_digest(f'{run_id}:{request_id}')}"


def _source_query_id(
    tool_run_id: str,
    source_query: UniverseDiscoverySourceQuery,
    index: int,
) -> str:
    digest = _stable_digest(
        "|".join([tool_run_id, str(index), source_query.provider, source_query.query])
    )
    return f"query-phase4-universe-{digest}"


def _artifact_filename(request_id: str, run_id: str) -> str:
    return f"instrument-universe-{_slug(request_id)}-{_stable_digest(run_id)}.json"


def _fixture_query_url(query: str) -> str:
    return f"https://example.com/phase4-fixtures/universe?q={quote(query, safe='')}"


def _fixture_instrument_url(provider_identifier: str) -> str:
    return (
        "https://example.com/phase4-fixtures/instruments/"
        f"{quote(provider_identifier.lower(), safe='')}"
    )


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-._")
    return slug or "universe"


def _normalize(value: str) -> str:
    return value.strip().casefold()


__all__ = [
    "PHASE4_FIXTURE_PROVIDER",
    "PHASE4_TOOL_NAME",
    "PHASE4_TOOL_VERSION",
    "PHASE4_UNIVERSE_SCHEMA_VERSION",
    "Phase4FixtureUniverseProvider",
    "Phase4UniverseDiscoveryTool",
    "Phase4UniverseDiscoveryToolResult",
    "UniverseDiscoveryProvider",
    "UniverseDiscoveryProviderResult",
    "UniverseDiscoverySourceQuery",
    "instrument_record_from_contract",
    "phase4_universe_artifact_payload",
]

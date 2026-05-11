"""FRED macro-series provider adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime

from nlp_stock_prediction.contracts import (
    CredentialState,
    MacroRequest,
    MacroSeries,
    MacroSnapshot,
    ProviderHealth,
    ProviderMetric,
    ProviderResult,
    ProviderStatus,
    ProviderWarning,
    WarningCode,
    WarningSeverity,
)
from nlp_stock_prediction.providers._base import (
    JsonFetch,
    JsonTransport,
    MalformedProviderResponse,
    ProviderCache,
    ProviderTransportError,
    UrllibJsonTransport,
    append_query_params,
    build_cache_key,
    fetch_json,
    malformed_result,
    missing_credentials_result,
    no_data_result,
    parse_decimal,
    parse_provider_date,
    provider_health,
    provider_result,
    provider_warning,
    raw_snapshot_id_for_payload,
    transport_error_result,
    utc_now,
)

FRED_OBSERVATIONS_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"


@dataclass(frozen=True)
class FredSeriesDefinition:
    series_id: str
    name: str
    unit: str | None


DEFAULT_FRED_SERIES: Mapping[str, FredSeriesDefinition] = {
    "FEDFUNDS": FredSeriesDefinition("FEDFUNDS", "Effective Federal Funds Rate", "percent"),
    "CPIAUCSL": FredSeriesDefinition("CPIAUCSL", "Consumer Price Index", "index"),
    "UNRATE": FredSeriesDefinition("UNRATE", "Unemployment Rate", "percent"),
    "GDP": FredSeriesDefinition("GDP", "Gross Domestic Product", "billions_usd"),
    "DGS10": FredSeriesDefinition("DGS10", "10-Year Treasury Constant Maturity Rate", "percent"),
    "DGS2": FredSeriesDefinition("DGS2", "2-Year Treasury Constant Maturity Rate", "percent"),
    "T10Y2Y": FredSeriesDefinition("T10Y2Y", "10-Year Minus 2-Year Treasury Spread", "percent"),
}


class FredMacroProvider:
    """FRED observations adapter for macro context inputs."""

    provider_name = "fred"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str = FRED_OBSERVATIONS_ENDPOINT,
        series_definitions: Mapping[str, FredSeriesDefinition] = DEFAULT_FRED_SERIES,
        transport: JsonTransport | None = None,
        cache: ProviderCache | None = None,
        now: Callable[[], datetime] = utc_now,
        timeout: float = 10.0,
        stale_after_days: int = 90,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._series_definitions = {
            series_id.upper(): definition for series_id, definition in series_definitions.items()
        }
        self._transport = transport or UrllibJsonTransport()
        self._cache = cache
        self._now = now
        self._timeout = timeout
        self._stale_after_days = stale_after_days

    def fetch_macro(self, request: MacroRequest) -> ProviderResult[MacroSnapshot]:
        fetched_at = self._now()
        if not self._api_key:
            return missing_credentials_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                credential_name="FRED API key",
            )
        series_ids = request.series_ids or tuple(self._series_definitions)
        series: list[MacroSeries] = []
        warnings: list[ProviderWarning] = []
        raw_snapshot_ids: list[str] = []
        cache_keys: list[str] = []
        for series_id in series_ids:
            result = self._fetch_one_series(request, series_id, fetched_at)
            if isinstance(result, ProviderWarning):
                warnings.append(result)
                continue
            raw_snapshot_ids.append(result.raw_snapshot_id)
            cache_keys.append(result.cache_key)
            mapped_series, stale_warning = self._map_series(series_id, result, request.run_date)
            series.append(mapped_series)
            if stale_warning is not None:
                warnings.append(stale_warning)
        if not series:
            if warnings:
                return provider_result(
                    provider_name=self.provider_name,
                    status=ProviderStatus.FAILED,
                    request=request,
                    fetched_at=fetched_at,
                    credential_state=CredentialState.CONFIGURED,
                    warnings=tuple(warnings),
                )
            return no_data_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message="FRED returned no macro series",
                credential_state=CredentialState.CONFIGURED,
            )
        status = ProviderStatus.OK
        if any(warning.code == WarningCode.STALE_DATA for warning in warnings):
            status = ProviderStatus.STALE
        elif warnings:
            status = ProviderStatus.PARTIAL
        raw_snapshot_id = raw_snapshot_id_for_payload(
            "fred-combined",
            {"raw_snapshot_ids": raw_snapshot_ids, "series_ids": list(series_ids)},
        )
        return provider_result(
            provider_name=self.provider_name,
            status=status,
            request=request,
            fetched_at=fetched_at,
            credential_state=CredentialState.CONFIGURED,
            data=MacroSnapshot(series=tuple(series)),
            warnings=tuple(warnings),
            raw_snapshot_id=raw_snapshot_id,
            cache_key=";".join(cache_keys) if cache_keys else None,
        )

    def health(self) -> ProviderHealth:
        checked_at = self._now()
        if not self._api_key:
            warning = provider_warning(
                provider_name=self.provider_name,
                code=WarningCode.MISSING_CREDENTIALS,
                severity=WarningSeverity.ERROR,
                message="FRED API key is not configured",
                occurred_at=checked_at,
            )
            return provider_health(
                provider_name=self.provider_name,
                status=ProviderStatus.UNCONFIGURED,
                checked_at=checked_at,
                credential_state=CredentialState.MISSING,
                warnings=(warning,),
            )
        return provider_health(
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            checked_at=checked_at,
            credential_state=CredentialState.CONFIGURED,
        )

    def _fetch_one_series(
        self,
        request: MacroRequest,
        series_id: str,
        fetched_at: datetime,
    ) -> JsonFetch | ProviderWarning:
        normalized_series_id = series_id.upper()
        url = append_query_params(
            self._endpoint,
            {
                "series_id": normalized_series_id,
                "api_key": self._api_key,
                "file_type": "json",
                "observation_end": request.run_date.isoformat(),
                "sort_order": "desc",
                "limit": 100,
            },
        )
        cache_key = build_cache_key(
            provider_name=self.provider_name,
            source="fred-series",
            run_date=request.run_date,
            tickers=(normalized_series_id,),
            query=normalized_series_id,
            url=url,
        )
        try:
            return fetch_json(
                transport=self._transport,
                url=url,
                run_date=request.run_date,
                ticker=normalized_series_id,
                source="fred-series",
                cache_key=cache_key,
                fetched_at=fetched_at,
                cache=self._cache,
                timeout=self._timeout,
            )
        except ProviderTransportError as exc:
            error_envelope: ProviderResult[MacroSnapshot] = transport_error_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                error=exc,
                credential_state=CredentialState.CONFIGURED,
            )
            return error_envelope.warnings[0]
        except MalformedProviderResponse as exc:
            malformed_envelope: ProviderResult[MacroSnapshot] = malformed_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message=str(exc),
                credential_state=CredentialState.CONFIGURED,
                cache_key=cache_key,
            )
            return malformed_envelope.warnings[0]

    def _map_series(
        self,
        series_id: str,
        fetched: JsonFetch,
        run_date: date,
    ) -> tuple[MacroSeries, ProviderWarning | None]:
        normalized_series_id = series_id.upper()
        observations = fetched.payload.get("observations")
        if not isinstance(observations, list):
            raise MalformedProviderResponse("FRED response missing observations list")
        definition = self._series_definitions.get(
            normalized_series_id,
            FredSeriesDefinition(normalized_series_id, normalized_series_id, None),
        )
        values: list[ProviderMetric] = []
        for raw_observation in observations:
            if not isinstance(raw_observation, dict):
                raise MalformedProviderResponse("FRED observation must be an object")
            observation_date = parse_provider_date(raw_observation.get("date"))
            if observation_date is None:
                continue
            values.append(
                ProviderMetric(
                    name="observation",
                    value=parse_decimal(raw_observation.get("value")),
                    unit=definition.unit,
                    as_of=observation_date,
                    metadata={
                        "series_id": normalized_series_id,
                        "realtime_start": raw_observation.get("realtime_start"),
                        "realtime_end": raw_observation.get("realtime_end"),
                        "raw_snapshot_id": fetched.raw_snapshot_id,
                    },
                )
            )
        if not values:
            raise MalformedProviderResponse(f"FRED series {normalized_series_id} had no values")
        latest_date = max(metric.as_of for metric in values if isinstance(metric.as_of, date))
        stale_warning = None
        if (run_date - latest_date).days > self._stale_after_days:
            stale_warning = provider_warning(
                provider_name=self.provider_name,
                code=WarningCode.STALE_DATA,
                severity=WarningSeverity.WARNING,
                message=f"FRED series {normalized_series_id} is stale: {latest_date.isoformat()}",
                occurred_at=self._now(),
                raw_snapshot_id=fetched.raw_snapshot_id,
                metadata={
                    "series_id": normalized_series_id,
                    "latest_date": latest_date.isoformat(),
                },
            )
        return MacroSeries(
            series_id=normalized_series_id,
            name=definition.name,
            values=tuple(values),
        ), stale_warning


__all__ = ["DEFAULT_FRED_SERIES", "FredMacroProvider", "FredSeriesDefinition"]

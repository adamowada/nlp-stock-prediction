"""SEC EDGAR supplemental fundamentals provider."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

from nlp_stock_prediction.contracts import (
    CredentialState,
    FundamentalsRequest,
    FundamentalsSnapshot,
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
    build_cache_key,
    fetch_json,
    first_ticker,
    malformed_result,
    no_data_result,
    parse_decimal,
    parse_provider_date,
    provider_health,
    provider_result,
    provider_warning,
    raw_snapshot_id_for_payload,
    snake_case,
    transport_error_result,
    utc_now,
)

SEC_COMPANY_FACTS_ENDPOINT = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_ENDPOINT = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANY_TICKERS_EXCHANGE_ENDPOINT = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_USER_AGENT_ENV = "NLP_STOCK_PREDICTION_SEC_USER_AGENT"
LIVE_USER_AGENT_ENV = "NLP_STOCK_PREDICTION_LIVE_USER_AGENT"
DEFAULT_SEC_USER_AGENT = ""


class SecEdgarFundamentalsProvider:
    """Supplemental SEC company-facts and recent-filings fundamentals adapter."""

    provider_name = "sec-edgar"

    def __init__(
        self,
        *,
        company_facts_endpoint: str = SEC_COMPANY_FACTS_ENDPOINT,
        submissions_endpoint: str = SEC_SUBMISSIONS_ENDPOINT,
        company_tickers_endpoint: str = SEC_COMPANY_TICKERS_EXCHANGE_ENDPOINT,
        user_agent: str | None = None,
        transport: JsonTransport | None = None,
        cache: ProviderCache | None = None,
        now: Callable[[], datetime] = utc_now,
        timeout: float = 10.0,
    ) -> None:
        self._company_facts_endpoint = company_facts_endpoint
        self._submissions_endpoint = submissions_endpoint
        self._company_tickers_endpoint = company_tickers_endpoint
        self._user_agent = (
            user_agent
            or os.environ.get(SEC_USER_AGENT_ENV)
            or os.environ.get(LIVE_USER_AGENT_ENV, "")
        ).strip()
        self._transport = transport or UrllibJsonTransport()
        self._cache = cache
        self._now = now
        self._timeout = timeout

    def fetch_fundamentals(
        self, request: FundamentalsRequest
    ) -> ProviderResult[FundamentalsSnapshot]:
        fetched_at = self._now()
        user_agent_warning = self._user_agent_warning(fetched_at)
        if user_agent_warning is not None:
            return provider_result(
                provider_name=self.provider_name,
                status=ProviderStatus.UNCONFIGURED,
                request=request,
                fetched_at=fetched_at,
                credential_state=CredentialState.MISSING,
                warnings=(user_agent_warning,),
            )
        ticker = first_ticker(request)
        if ticker is None:
            return no_data_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message="SEC EDGAR fundamentals request did not include a ticker",
            )
        headers = {"User-Agent": self._user_agent, "Accept": "application/json"}
        try:
            company_tickers = self._fetch_company_tickers(
                request=request,
                fetched_at=fetched_at,
                headers=headers,
            )
            cik = _cik_for_ticker(company_tickers.payload, ticker)
        except ProviderTransportError as exc:
            return transport_error_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                error=exc,
                credential_state=CredentialState.NOT_REQUIRED,
            )
        except MalformedProviderResponse as exc:
            return malformed_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message=str(exc),
                credential_state=CredentialState.NOT_REQUIRED,
            )
        if not cik:
            warning = provider_warning(
                provider_name=self.provider_name,
                code=WarningCode.NO_DATA,
                severity=WarningSeverity.ERROR,
                message=(
                    f"SEC company tickers exchange dataset did not resolve CIK for {ticker}; "
                    "SEC fundamentals cannot run without an official ticker-to-CIK match."
                ),
                occurred_at=fetched_at,
                provider_error_type="cik_lookup_failed",
                raw_snapshot_id=company_tickers.raw_snapshot_id,
                source_url=self._company_tickers_endpoint,
                metadata={"ticker": ticker},
            )
            return provider_result(
                provider_name=self.provider_name,
                status=ProviderStatus.FAILED,
                request=request,
                fetched_at=fetched_at,
                credential_state=CredentialState.NOT_REQUIRED,
                warnings=(warning,),
                raw_snapshot_id=company_tickers.raw_snapshot_id,
                cache_key=company_tickers.cache_key,
            )
        normalized_cik = _normalize_cik(cik)
        facts_url = self._company_facts_endpoint.format(cik=normalized_cik)
        submissions_url = self._submissions_endpoint.format(cik=normalized_cik)
        facts_cache_key = build_cache_key(
            provider_name=self.provider_name,
            source="sec-edgar-companyfacts",
            run_date=request.run_date,
            tickers=(ticker,),
            query=normalized_cik,
            url=facts_url,
        )
        submissions_cache_key = build_cache_key(
            provider_name=self.provider_name,
            source="sec-edgar-submissions",
            run_date=request.run_date,
            tickers=(ticker,),
            query=normalized_cik,
            url=submissions_url,
        )
        facts: JsonFetch | None = None
        submissions: JsonFetch | None = None
        try:
            facts = fetch_json(
                transport=self._transport,
                url=facts_url,
                run_date=request.run_date,
                ticker=ticker,
                source="sec-edgar-companyfacts",
                cache_key=facts_cache_key,
                fetched_at=fetched_at,
                cache=self._cache,
                headers=headers,
                timeout=self._timeout,
            )
            submissions = fetch_json(
                transport=self._transport,
                url=submissions_url,
                run_date=request.run_date,
                ticker=ticker,
                source="sec-edgar-submissions",
                cache_key=submissions_cache_key,
                fetched_at=fetched_at,
                cache=self._cache,
                headers=headers,
                timeout=self._timeout,
            )
            snapshot = self._map_payload(ticker, normalized_cik, facts, submissions)
        except ProviderTransportError as exc:
            return transport_error_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                error=exc,
                credential_state=CredentialState.NOT_REQUIRED,
            )
        except MalformedProviderResponse as exc:
            raw_snapshot_id = raw_snapshot_id_for_payload(
                "sec-edgar-combined",
                {
                    "facts": facts.raw_snapshot_id if facts is not None else None,
                    "submissions": submissions.raw_snapshot_id if submissions is not None else None,
                },
            )
            return malformed_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message=str(exc),
                credential_state=CredentialState.NOT_REQUIRED,
                raw_snapshot_id=raw_snapshot_id,
                cache_key=facts.cache_key if facts is not None else facts_cache_key,
            )
        assert facts is not None
        assert submissions is not None
        raw_snapshot_id = raw_snapshot_id_for_payload(
            "sec-edgar-combined",
            {"facts": facts.raw_snapshot_id, "submissions": submissions.raw_snapshot_id},
        )
        if not snapshot.metrics:
            return no_data_result(
                provider_name=self.provider_name,
                request=request,
                fetched_at=fetched_at,
                message=f"SEC EDGAR returned no supplemental metrics for {ticker}",
                raw_snapshot_id=raw_snapshot_id,
                cache_key=facts.cache_key,
            )
        return provider_result(
            provider_name=self.provider_name,
            status=ProviderStatus.OK,
            request=request,
            fetched_at=fetched_at,
            credential_state=CredentialState.NOT_REQUIRED,
            data=snapshot,
            raw_snapshot_id=raw_snapshot_id,
            cache_key=facts.cache_key,
        )

    def health(self) -> ProviderHealth:
        checked_at = self._now()
        user_agent_warning = self._user_agent_warning(checked_at)
        return provider_health(
            provider_name=self.provider_name,
            status=ProviderStatus.UNCONFIGURED
            if user_agent_warning is not None
            else ProviderStatus.OK,
            checked_at=checked_at,
            credential_state=CredentialState.MISSING
            if user_agent_warning is not None
            else CredentialState.CONFIGURED,
            warnings=() if user_agent_warning is None else (user_agent_warning,),
        )

    def _user_agent_warning(self, occurred_at: datetime) -> ProviderWarning | None:
        if self._user_agent:
            return None
        return provider_warning(
            provider_name=self.provider_name,
            code=WarningCode.MISSING_CREDENTIALS,
            severity=WarningSeverity.ERROR,
            message=(
                "SEC EDGAR requires a contact User-Agent; set "
                f"{SEC_USER_AGENT_ENV}, {LIVE_USER_AGENT_ENV}, or pass user_agent explicitly"
            ),
            occurred_at=occurred_at,
            metadata={"credential_name": SEC_USER_AGENT_ENV},
        )

    def _fetch_company_tickers(
        self,
        *,
        request: FundamentalsRequest,
        fetched_at: datetime,
        headers: Mapping[str, str],
    ) -> JsonFetch:
        cache_key = build_cache_key(
            provider_name=self.provider_name,
            source="sec-company-tickers-exchange",
            run_date=request.run_date,
            url=self._company_tickers_endpoint,
        )
        return fetch_json(
            transport=self._transport,
            url=self._company_tickers_endpoint,
            run_date=request.run_date,
            ticker=None,
            source="sec-company-tickers-exchange",
            cache_key=cache_key,
            fetched_at=fetched_at,
            cache=self._cache,
            headers=headers,
            timeout=self._timeout,
        )

    def _map_payload(
        self,
        ticker: str,
        cik: str,
        facts: JsonFetch,
        submissions: JsonFetch,
    ) -> FundamentalsSnapshot:
        company_name = _optional_text(facts.payload.get("entityName")) or _optional_text(
            submissions.payload.get("name")
        )
        metrics = list(_company_fact_metrics(facts))
        metrics.extend(_recent_filing_metrics(cik, submissions))
        return FundamentalsSnapshot(
            ticker=ticker,
            company_name=company_name,
            metrics=tuple(metrics),
        )


def _cik_for_ticker(payload: Mapping[str, Any], ticker: str) -> str | None:
    normalized_ticker = ticker.strip().upper()
    fields = payload.get("fields")
    data = payload.get("data")
    if isinstance(fields, Sequence) and not isinstance(fields, str | bytes):
        return _cik_for_exchange_dataset(
            fields=tuple(fields),
            data=data,
            ticker=normalized_ticker,
        )
    return _cik_for_company_tickers_object(payload, normalized_ticker)


def _cik_for_exchange_dataset(
    *,
    fields: Sequence[object],
    data: object,
    ticker: str,
) -> str | None:
    field_names = tuple(str(field).strip().lower() for field in fields)
    required_fields = {"cik", "ticker"}
    if not required_fields.issubset(field_names):
        raise MalformedProviderResponse(
            "SEC company tickers exchange response missing cik/ticker fields"
        )
    if not isinstance(data, Sequence) or isinstance(data, str | bytes):
        raise MalformedProviderResponse("SEC company tickers exchange response missing data rows")
    cik_index = field_names.index("cik")
    ticker_index = field_names.index("ticker")
    for row in data:
        if not isinstance(row, Sequence) or isinstance(row, str | bytes):
            raise MalformedProviderResponse("SEC company tickers exchange row must be an array")
        if max(cik_index, ticker_index) >= len(row):
            raise MalformedProviderResponse(
                "SEC company tickers exchange row missing cik/ticker values"
            )
        if str(row[ticker_index]).strip().upper() == ticker:
            return _cik_text(row[cik_index])
    return None


def _cik_for_company_tickers_object(payload: Mapping[str, Any], ticker: str) -> str | None:
    saw_company_record = False
    for record in payload.values():
        if not isinstance(record, Mapping):
            continue
        if "ticker" not in record or "cik_str" not in record:
            continue
        saw_company_record = True
        if str(record["ticker"]).strip().upper() == ticker:
            return _cik_text(record["cik_str"])
    if not saw_company_record:
        raise MalformedProviderResponse(
            "SEC company tickers response missing ticker/cik_str records"
        )
    return None


def _cik_text(value: object) -> str:
    normalized = "".join(character for character in str(value) if character.isdigit())
    if not normalized:
        raise MalformedProviderResponse("SEC company tickers response included an empty CIK")
    return normalized


def _company_fact_metrics(fetched: JsonFetch) -> tuple[ProviderMetric, ...]:
    facts = fetched.payload.get("facts")
    if not isinstance(facts, dict):
        raise MalformedProviderResponse("SEC company facts response missing facts object")
    metrics: list[ProviderMetric] = []
    for taxonomy, taxonomy_facts in facts.items():
        if not isinstance(taxonomy, str) or not isinstance(taxonomy_facts, dict):
            continue
        for concept, raw_fact in taxonomy_facts.items():
            if not isinstance(concept, str) or not isinstance(raw_fact, dict):
                continue
            units = raw_fact.get("units")
            if not isinstance(units, dict):
                continue
            for unit, observations in units.items():
                if not isinstance(unit, str) or not isinstance(observations, list):
                    continue
                latest = _latest_fact_observation(observations)
                if latest is None:
                    continue
                value = _fact_value(latest.get("val"))
                if value is None:
                    continue
                metrics.append(
                    ProviderMetric(
                        name=f"sec_{snake_case(concept)}",
                        value=value,
                        unit=unit,
                        as_of=parse_provider_date(latest.get("end")),
                        metadata={
                            "taxonomy": taxonomy,
                            "label": raw_fact.get("label"),
                            "description": raw_fact.get("description"),
                            "form": latest.get("form"),
                            "filed": latest.get("filed"),
                            "accession_number": latest.get("accn"),
                            "fiscal_year": latest.get("fy"),
                            "fiscal_period": latest.get("fp"),
                            "raw_snapshot_id": fetched.raw_snapshot_id,
                        },
                    )
                )
                break
    return tuple(metrics)


def _recent_filing_metrics(cik: str, fetched: JsonFetch) -> tuple[ProviderMetric, ...]:
    filings = fetched.payload.get("filings")
    if not isinstance(filings, dict):
        raise MalformedProviderResponse("SEC submissions response missing filings object")
    recent = filings.get("recent")
    if not isinstance(recent, dict):
        return ()
    accession_numbers = _list_field(recent.get("accessionNumber"))
    filing_dates = _list_field(recent.get("filingDate"))
    forms = _list_field(recent.get("form"))
    documents = _list_field(recent.get("primaryDocument"))
    count = min(len(accession_numbers), len(filing_dates), len(forms), len(documents), 5)
    metrics: list[ProviderMetric] = []
    for index in range(count):
        form = forms[index]
        accession_number = accession_numbers[index]
        primary_document = documents[index]
        filing_date = parse_provider_date(filing_dates[index])
        metrics.append(
            ProviderMetric(
                name=f"sec_recent_filing_{snake_case(form)}",
                value=form,
                unit=None,
                as_of=filing_date,
                metadata={
                    "accession_number": accession_number,
                    "primary_document": primary_document,
                    "source_url": _filing_url(cik, accession_number, primary_document),
                    "raw_snapshot_id": fetched.raw_snapshot_id,
                },
            )
        )
    return tuple(metrics)


def _latest_fact_observation(observations: list[object]) -> dict[str, object] | None:
    candidates = [item for item in observations if isinstance(item, dict)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            str(item.get("end") or ""),
            str(item.get("filed") or ""),
            str(item.get("accn") or ""),
        ),
    )


def _fact_value(value: object) -> Decimal | str | None:
    decimal_value = parse_decimal(value)
    if decimal_value is not None:
        return decimal_value
    text = _optional_text(value)
    return text


def _list_field(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _filing_url(cik: str, accession_number: str, primary_document: str) -> str:
    accession_path = accession_number.replace("-", "")
    cik_path = str(int(cik))
    return f"https://www.sec.gov/Archives/edgar/data/{cik_path}/{accession_path}/{primary_document}"


def _normalize_cik(value: str) -> str:
    digits = "".join(character for character in value if character.isdigit())
    return digits.zfill(10)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "LIVE_USER_AGENT_ENV",
    "SEC_COMPANY_TICKERS_EXCHANGE_ENDPOINT",
    "SEC_USER_AGENT_ENV",
    "SecEdgarFundamentalsProvider",
]

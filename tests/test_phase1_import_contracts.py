from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_MODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "analysis": (
        "AnalysisBundle",
        "AnalysisComponent",
        "FundamentalAnalysis",
        "MacroContext",
        "MetricValue",
        "SectorContext",
        "TechnicalAnalysis",
    ),
    "base": (
        "Confidence",
        "ContractModel",
        "JsonObject",
        "JsonValue",
        "NonEmptyStr",
        "PositiveDecimal",
        "PositiveInt",
        "Score",
        "TickerSymbol",
        "ensure_utc_timestamp",
    ),
    "discovery": ("TickerCandidate", "TickerDiscoveryResult"),
    "enums": (
        "AnalysisSignal",
        "CredentialState",
        "Direction",
        "FreshnessStatus",
        "InstrumentType",
        "PositionType",
        "ProviderStatus",
        "RecommendationAction",
        "RetrievalMethod",
        "RiskProfile",
        "SourceKind",
        "TickerDiscoveryStatus",
        "TimeHorizon",
        "WarningCode",
        "WarningSeverity",
    ),
    "evidence": ("SourceEvidence", "TextSpan"),
    "extraction": ("StrategyCluster", "StrategyExtraction"),
    "fixtures": ("FixtureManifest", "NormalizedFixture", "RawProviderFixture"),
    "provenance": (
        "DataReference",
        "EvidenceReference",
        "ProviderHealth",
        "ProviderWarning",
        "SourceProvenance",
    ),
    "providers": (
        "DateWindow",
        "EvidenceRequest",
        "ExtractionRequest",
        "FundamentalsProvider",
        "FundamentalsRequest",
        "FundamentalsSnapshot",
        "LLMExtractor",
        "MacroProvider",
        "MacroRequest",
        "MacroSeries",
        "MacroSnapshot",
        "MarketDataProvider",
        "MarketDataRequest",
        "MarketSnapshot",
        "NewsProvider",
        "PriceBar",
        "ProviderMetric",
        "ProviderRequest",
        "ProviderResult",
        "RedditProvider",
        "RunConfig",
        "TickerDiscoveryRequest",
        "XProvider",
    ),
    "recommendation": (
        "RiskAssessment",
        "ScoreBreakdown",
        "ScoreComponent",
        "TradeCandidate",
    ),
    "report": (
        "AuditArtifact",
        "AuditManifest",
        "DailyReport",
        "DataFreshnessSummary",
        "Disclaimer",
        "TickerReportSection",
    ),
}

EXPECTED_CONTRACT_NAMESPACE_EXPORTS = (
    "AnalysisBundle",
    "AnalysisComponent",
    "AnalysisSignal",
    "AuditArtifact",
    "AuditManifest",
    "Confidence",
    "ContractModel",
    "CredentialState",
    "DailyReport",
    "DataFreshnessSummary",
    "DataReference",
    "DateWindow",
    "Direction",
    "Disclaimer",
    "EvidenceReference",
    "EvidenceRequest",
    "ExtractionRequest",
    "FixtureManifest",
    "FreshnessStatus",
    "FundamentalAnalysis",
    "FundamentalsProvider",
    "FundamentalsRequest",
    "FundamentalsSnapshot",
    "InstrumentType",
    "JsonObject",
    "JsonValue",
    "LLMExtractor",
    "MacroContext",
    "MacroProvider",
    "MacroRequest",
    "MacroSeries",
    "MacroSnapshot",
    "MarketDataProvider",
    "MarketDataRequest",
    "MarketSnapshot",
    "MetricValue",
    "NewsProvider",
    "NonEmptyStr",
    "NormalizedFixture",
    "PositionType",
    "PositiveDecimal",
    "PositiveInt",
    "PriceBar",
    "ProviderHealth",
    "ProviderMetric",
    "ProviderRequest",
    "ProviderResult",
    "ProviderStatus",
    "ProviderWarning",
    "RawProviderFixture",
    "RecommendationAction",
    "RedditProvider",
    "RetrievalMethod",
    "RiskAssessment",
    "RiskProfile",
    "RunConfig",
    "Score",
    "ScoreBreakdown",
    "ScoreComponent",
    "SectorContext",
    "SourceEvidence",
    "SourceKind",
    "SourceProvenance",
    "StrategyCluster",
    "StrategyExtraction",
    "TechnicalAnalysis",
    "TextSpan",
    "TickerCandidate",
    "TickerDiscoveryRequest",
    "TickerDiscoveryResult",
    "TickerDiscoveryStatus",
    "TickerReportSection",
    "TickerSymbol",
    "TimeHorizon",
    "TradeCandidate",
    "WarningCode",
    "WarningSeverity",
    "XProvider",
)

EXPECTED_NAMESPACE_CANONICAL_MODULES: dict[str, str] = {
    export_name: module_name
    for module_name, exports in EXPECTED_MODULE_EXPORTS.items()
    for export_name in exports
    if export_name != "ensure_utc_timestamp"
}


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path if existing_pythonpath is None else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )
    return env


def test_contract_namespace_all_matches_phase1_public_surface() -> None:
    contracts = importlib.import_module("nlp_stock_prediction.contracts")
    exported_names = tuple(contracts.__all__)

    assert len(exported_names) == len(set(exported_names))
    assert exported_names == tuple(sorted(exported_names))
    assert exported_names == EXPECTED_CONTRACT_NAMESPACE_EXPORTS


def test_contract_namespace_star_import_matches_public_all() -> None:
    namespace: dict[str, object] = {}

    exec("from nlp_stock_prediction.contracts import *", namespace)

    imported_names = {
        name for name in namespace if not name.startswith("__") and name != "annotations"
    }
    assert imported_names == set(EXPECTED_CONTRACT_NAMESPACE_EXPORTS)


@pytest.mark.parametrize("export_name", EXPECTED_CONTRACT_NAMESPACE_EXPORTS)
def test_contract_namespace_reexports_canonical_objects(export_name: str) -> None:
    contracts = importlib.import_module("nlp_stock_prediction.contracts")
    module_name = EXPECTED_NAMESPACE_CANONICAL_MODULES[export_name]
    canonical_module = importlib.import_module(f"nlp_stock_prediction.contracts.{module_name}")

    assert getattr(contracts, export_name) is getattr(canonical_module, export_name)


@pytest.mark.parametrize(
    ("module_name", "expected_exports"),
    tuple(EXPECTED_MODULE_EXPORTS.items()),
)
def test_contract_submodule_all_matches_expected_exports(
    module_name: str, expected_exports: tuple[str, ...]
) -> None:
    module = importlib.import_module(f"nlp_stock_prediction.contracts.{module_name}")
    exported_names = tuple(module.__all__)

    assert len(exported_names) == len(set(exported_names))
    assert exported_names == tuple(sorted(exported_names))
    assert exported_names == expected_exports
    for export_name in expected_exports:
        assert hasattr(module, export_name)


def test_top_level_package_and_module_entrypoint_import_cleanly() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import nlp_stock_prediction; "
                "import nlp_stock_prediction.__main__ as entrypoint; "
                "import nlp_stock_prediction.cli as cli; "
                "assert nlp_stock_prediction.__version__ == '0.1.0'; "
                "assert entrypoint.main is cli.main"
            ),
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_python_module_entrypoint_exposes_help_without_import_errors() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "nlp_stock_prediction", "--help"],
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Generate an evidence-grounded daily stock opportunity report." in completed.stdout
    assert completed.stderr == ""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from nlp_stock_prediction.cli import main
from nlp_stock_prediction.contracts import RunConfig, WsbBatchRunConfig
from nlp_stock_prediction.orchestration.orchestration_common import symbol_slug
from nlp_stock_prediction.orchestration.report_bundle import ReportBundle, WsbBatchReportBundle
from nlp_stock_prediction.orchestration.wsb_trending import (
    DEFAULT_WSB_SOURCE_URL,
    discover_wsb_trending_stocks,
    generate_wsb_batch_research_reports,
)
from nlp_stock_prediction.providers.reddit_scrape import StaticHtmlTransport
from nlp_stock_prediction.reporting.fixtures import build_offline_fixture_bundle

pytestmark = pytest.mark.unit

RUN_DATE = date(2026, 5, 12)
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "reddit"


def test_wsb_trending_discovery_counts_public_reddit_mentions() -> None:
    discovery = discover_wsb_trending_stocks(
        WsbBatchRunConfig(
            run_date=RUN_DATE,
            output_dir=Path("reports"),
            offline=True,
            source_mode="offline",
            limit=10,
            max_discussion_pages=2,
        ),
        transport=_fixture_transport(),
    )

    assert discovery.generated_at.isoformat() == "2026-05-12T21:00:00+00:00"
    assert discovery.trending_stocks[0].symbol == "NVDA"
    assert discovery.trending_stocks[0].mention_count >= 8
    assert len(discovery.trending_stocks) == 10
    assert "HOOD" not in {stock.symbol for stock in discovery.trending_stocks}
    assert discovery.raw_snapshot_ids
    assert discovery.source_urls[0] == DEFAULT_WSB_SOURCE_URL


def test_wsb_trending_discovery_normalizes_sentence_punctuation_suffixes() -> None:
    html = """
    <html>
      <body>
        <div
          class="thing id-t3_punct001"
          data-type="link"
          data-fullname="t3_punct001"
          data-subreddit="wallstreetbets"
          data-permalink="/r/wallstreetbets/comments/punct001/hpe_watch/"
        >
          <a class="title" href="/r/wallstreetbets/comments/punct001/hpe_watch/">
            Watching $HPE. HPE. BRK.B.
          </a>
          <div class="usertext-body">
            HPE stock and $HPE. into earnings.
          </div>
        </div>
      </body>
    </html>
    """

    discovery = discover_wsb_trending_stocks(
        WsbBatchRunConfig(
            run_date=RUN_DATE,
            output_dir=Path("reports"),
            offline=True,
            source_mode="offline",
            limit=10,
            max_discussion_pages=0,
        ),
        transport=StaticHtmlTransport({DEFAULT_WSB_SOURCE_URL: html}),
    )

    hpe = next(stock for stock in discovery.trending_stocks if stock.symbol == "HPE")
    symbols = {stock.symbol for stock in discovery.trending_stocks}

    assert "HPE." not in symbols
    assert hpe.mention_count == 4


def test_wsb_batch_workflow_discovers_then_batch_analyzes(tmp_path: Path) -> None:
    seen_symbols: list[str] = []

    def fake_generator(config: RunConfig) -> ReportBundle:
        seen_symbols.append(config.symbol)
        return _fake_bundle(config)

    bundle = generate_wsb_batch_research_reports(
        WsbBatchRunConfig(
            run_date=RUN_DATE,
            output_dir=tmp_path / "reports",
            offline=True,
            source_mode="offline",
            limit=3,
            max_discussion_pages=2,
            max_workers=1,
        ),
        report_generator=fake_generator,
        transport=_fixture_transport(),
    )

    assert seen_symbols == ["NVDA", "AMD", "TSLA"]
    assert bundle.discovery_json_path.exists()
    assert bundle.discovery_markdown_path.exists()
    assert bundle.batch_bundle.ranking_json_path.exists()
    assert bundle.discovery_report.trending_stocks[0].symbol == "NVDA"
    assert [target.symbol for target in bundle.batch_bundle.ranking_report.ranked_targets] == [
        "NVDA",
        "AMD",
        "TSLA",
    ]


def test_cli_wsb_batch_prints_discovery_and_ranking_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "reports"

    def fake_workflow(config: WsbBatchRunConfig) -> WsbBatchReportBundle:
        assert config.limit == 10
        return generate_wsb_batch_research_reports(
            config,
            report_generator=_fake_bundle,
            transport=_fixture_transport(),
        )

    monkeypatch.setattr(
        "nlp_stock_prediction.cli.generate_wsb_trending_research_reports",
        fake_workflow,
    )

    exit_code = main(
        [
            "research-wsb-batch",
            "--date",
            "2026-05-12",
            "--output",
            str(output_dir),
            "--offline",
            "--max-workers",
            "2",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "Wrote WSB discovery Markdown:" in captured.out
    assert "Discovered WSB symbols: NVDA" in captured.out
    assert "Wrote batch ranking JSON:" in captured.out


def _fixture_transport() -> StaticHtmlTransport:
    return StaticHtmlTransport(
        {
            DEFAULT_WSB_SOURCE_URL: _html("public_wsb_trending.html"),
            "wsbtrending001/daily_moves": _html("public_wsb_trending_post_1.html"),
            "wsbtrending002/earnings_watch": _html("public_wsb_trending_post_2.html"),
        }
    )


def _html(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def _fake_bundle(config: RunConfig) -> ReportBundle:
    fixture = build_offline_fixture_bundle(
        RunConfig(
            run_date=config.run_date,
            output_dir=config.output_dir,
            symbol="TSLA",
            offline=True,
            source_mode="offline",
        )
    )
    report_dir = (
        config.output_dir.resolve() / config.run_date.isoformat() / symbol_slug(config.symbol)
    )
    audit_dir = report_dir / "audit"
    audit_dir.mkdir(parents=True)
    markdown_path = report_dir / "report.md"
    json_path = report_dir / "report.json"
    audit_manifest_path = audit_dir / "audit-manifest.json"
    markdown_path.write_text("# report\n", encoding="utf-8")
    json_path.write_text(fixture.report.model_dump_json(indent=2), encoding="utf-8")
    audit_manifest_path.write_text("{}\n", encoding="utf-8")
    return ReportBundle(
        report_dir=report_dir,
        markdown_path=markdown_path,
        json_path=json_path,
        audit_dir=audit_dir,
        audit_manifest_path=audit_manifest_path,
        report=fixture.report,
        tool_records=(),
    )

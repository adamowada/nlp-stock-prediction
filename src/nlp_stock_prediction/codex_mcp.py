"""Local MCP server exposing Phase 2 research tools to Codex."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from nlp_stock_prediction.orchestration.phase2_service import Phase2McpService


def build_server(repo_root: Path | None = None, database_path: Path | None = None) -> Any:
    """Build the optional MCP server.

    The `mcp` package is an optional dependency so the default CLI and test suite do not import it.
    """

    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found, unused-ignore]
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without optional extra
        raise RuntimeError(
            "The Phase 2 Codex MCP server requires the optional `codex-smoke` extra: "
            'python -m pip install -e ".[codex-smoke]"'
        ) from exc

    service = Phase2McpService(
        repo_root=repo_root or Path.cwd(),
        database_path=database_path or Path("data/prediction-research.sqlite3"),
    )
    server = FastMCP("nlp-stock-prediction")

    def start_research_run(
        run_date: str,
        output_dir: str,
        symbol: str,
        objective: str | None = None,
    ) -> dict[str, object]:
        """Start a Phase 2 smoke research run and return run paths."""

        return dict(
            service.start_research_run(
                run_date=run_date,
                output_dir=output_dir,
                symbol=symbol,
                objective=objective,
            )
        )

    def list_research_tool_plan() -> dict[str, object]:
        """List the staged dummy tools Codex should call for Phase 2 smoke."""

        return dict(service.list_research_tool_plan())

    def record_codex_search_evidence(
        run_id: str,
        symbol: str,
        title: str,
        url: str,
        claim: str,
        query: str,
        published_at: str | None = None,
        stance: str | None = None,
    ) -> dict[str, object]:
        """Record one live-search source as normalized evidence.

        stance may be supports, contradicts, or neutral relative to the candidate thesis.
        """

        return dict(
            service.record_codex_search_evidence(
                run_id=run_id,
                symbol=symbol,
                title=title,
                url=url,
                claim=claim,
                query=query,
                published_at=published_at,
                stance=stance,
            )
        )

    def run_dummy_universe_tool(run_id: str, symbol: str) -> dict[str, object]:
        """Write deterministic dummy instrument-universe artifacts."""

        return dict(service.run_dummy_universe_tool(run_id=run_id, symbol=symbol))

    def run_dummy_analysis_tool(run_id: str, symbol: str) -> dict[str, object]:
        """Write deterministic dummy analysis artifacts."""

        return dict(service.run_dummy_analysis_tool(run_id=run_id, symbol=symbol))

    def synthesize_prediction_candidates(run_id: str, symbol: str) -> dict[str, object]:
        """Synthesize conservative prediction candidates from stored evidence."""

        return dict(service.synthesize_prediction_candidates(run_id=run_id, symbol=symbol))

    def render_prediction_report(run_id: str, symbol: str) -> dict[str, object]:
        """Render Markdown, JSON, and audit manifest files."""

        return dict(service.render_prediction_report(run_id=run_id, symbol=symbol))

    def inspect_research_run(run_id: str) -> dict[str, object]:
        """Inspect stored run graph counts for smoke verification."""

        return dict(service.inspect_research_run(run_id=run_id))

    server.tool()(start_research_run)
    server.tool()(list_research_tool_plan)
    server.tool()(record_codex_search_evidence)
    server.tool()(run_dummy_universe_tool)
    server.tool()(run_dummy_analysis_tool)
    server.tool()(synthesize_prediction_candidates)
    server.tool()(render_prediction_report)
    server.tool()(inspect_research_run)
    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/prediction-research.sqlite3"),
        help="Research SQLite database path, relative to --repo-root unless absolute.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_server(repo_root=args.repo_root, database_path=args.database).run()


if __name__ == "__main__":
    main()

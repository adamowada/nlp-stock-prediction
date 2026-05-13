"""Local MCP server exposing Phase 2 research tools to Codex."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from nlp_stock_prediction.orchestration.phase2_mcp_registration import register_phase2_mcp_tools
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

    register_phase2_mcp_tools(server, service)
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

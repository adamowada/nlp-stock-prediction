"""Local MCP server exposing research and evaluation tools to Codex."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from nlp_stock_prediction.environment import load_local_dotenv
from nlp_stock_prediction.orchestration.evaluation_mcp_registration import (
    register_evaluation_mcp_tools,
)
from nlp_stock_prediction.orchestration.evaluation_service import EvaluationService
from nlp_stock_prediction.orchestration.research_mcp_registration import register_research_mcp_tools
from nlp_stock_prediction.orchestration.research_service import ResearchService


def build_server(repo_root: Path | None = None, database_path: Path | None = None) -> Any:
    """Build the optional MCP server.

    The `mcp` package is an optional dependency so the default CLI and test suite do not import it.
    """

    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found, unused-ignore]
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without optional extra
        raise RuntimeError(
            "The research/evaluation Codex MCP server requires the optional `codex-smoke` extra: "
            'python -m pip install -e ".[codex-smoke]"'
        ) from exc

    resolved_repo_root = (repo_root or Path.cwd()).resolve()
    load_local_dotenv(resolved_repo_root)
    if database_path is None:
        raise ValueError("database_path is required for the Codex MCP server")
    resolved_database_path = (
        database_path if database_path.is_absolute() else resolved_repo_root / database_path
    )
    if not resolved_database_path.exists():
        raise ValueError(
            "database_path must reference an existing research SQLite database: "
            f"{resolved_database_path}"
        )
    research_service = ResearchService(
        repo_root=resolved_repo_root,
        database_path=resolved_database_path,
    )
    evaluation_service = EvaluationService(
        repo_root=resolved_repo_root,
        database_path=resolved_database_path,
    )
    server = FastMCP("nlp-stock-prediction")

    register_research_mcp_tools(server, research_service)
    register_evaluation_mcp_tools(server, evaluation_service)
    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--database",
        type=Path,
        required=True,
        help="Research SQLite database path, relative to --repo-root unless absolute.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_server(repo_root=args.repo_root, database_path=args.database).run()


if __name__ == "__main__":
    main()

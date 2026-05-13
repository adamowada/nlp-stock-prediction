from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from nlp_stock_prediction.orchestration import codex_smoke

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_phase2_codex_smoke.py"
REPO_ROOT = SCRIPT_PATH.parents[1]


def _load_script_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_phase2_codex_smoke", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_codex_smoke_command_exposes_mcp_server_and_search(tmp_path: Path) -> None:
    smoke = codex_smoke
    config = smoke.CodexSmokeConfig(
        run_date=date(2026, 5, 13),
        output_dir=tmp_path / "reports",
        symbol="tsla",
        repo_root=tmp_path,
        python_executable=Path("python"),
    )

    command = smoke.build_codex_command(config)
    rendered = " ".join(command)

    assert command[:4] == ["codex", "--ask-for-approval", "never", "--search"]
    assert "mcp_servers.nlp-stock-prediction.command" in rendered
    assert "-B" in rendered
    assert "nlp_stock_prediction.codex_mcp" in rendered
    assert "--repo-root" in rendered
    assert "--database" in rendered
    assert "phase2-codex-smoke-2026-05-13-tsla.sqlite3" in rendered
    assert "danger-full-access" in command
    assert str(config.final_message_path) in command
    assert "Do not import project modules directly" in command[-1]
    assert "record_codex_search_evidence" in command[-1]
    assert "stance set to" in command[-1]
    assert "render_prediction_report" in command[-1]


@pytest.mark.unit
def test_codex_smoke_output_verification_requires_search_evidence(tmp_path: Path) -> None:
    smoke = codex_smoke
    config = smoke.CodexSmokeConfig(
        run_date=date(2026, 5, 13),
        output_dir=tmp_path / "reports",
        symbol="TSLA",
        repo_root=tmp_path,
        python_executable=Path("python"),
    )
    run_dir = config.run_dir
    (run_dir / "audit").mkdir(parents=True)
    (run_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    (run_dir / "audit" / "audit-manifest.json").write_text("{}\n", encoding="utf-8")
    config.final_message_path.write_text("done\n", encoding="utf-8")
    (run_dir / "report.json").write_text(
        """
{
  "evidence_sources": [
    {
      "metadata": {"codex_search": true},
      "provenance": {"provider_name": "codex-web-search"}
    }
  ],
  "prediction_candidates": [{"candidate_id": "candidate-tsla"}]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    smoke.verify_smoke_outputs(config, require_sqlite=False)

    (run_dir / "report.json").write_text(
        '{"evidence_sources": [], "prediction_candidates": []}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="evidence_sources"):
        smoke.verify_smoke_outputs(config, require_sqlite=False)

    (run_dir / "report.json").write_text(
        """
{
  "evidence_sources": [
    {
      "metadata": {"codex_search": true},
      "provenance": {"provider_name": "codex-web-search"}
    }
  ],
  "prediction_candidates": [{"candidate_id": "candidate-tsla"}]
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    config.final_message_path.write_text(
        "FastMCP stdio transport was blocked; using Phase2McpService.\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="MCP fallback"):
        smoke.verify_smoke_outputs(config, require_sqlite=False)


@pytest.mark.unit
def test_codex_smoke_prepares_clean_ignored_run_dir(tmp_path: Path) -> None:
    smoke = codex_smoke
    config = smoke.CodexSmokeConfig(
        run_date=date(2026, 5, 13),
        output_dir=tmp_path / "reports" / "phase2-codex-smoke",
        symbol="TSLA",
        repo_root=tmp_path,
        python_executable=Path("python"),
    )
    stale = config.run_dir / "report.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"stale": true}\n', encoding="utf-8")

    smoke.prepare_clean_run_dir(config)

    assert config.run_dir.exists()
    assert not stale.exists()

    unsafe = smoke.CodexSmokeConfig(
        run_date=date(2026, 5, 13),
        output_dir=tmp_path / "unsafe",
        symbol="TSLA",
        repo_root=tmp_path,
        python_executable=Path("python"),
    )
    with pytest.raises(RuntimeError, match="outside ignored roots"):
        smoke.prepare_clean_run_dir(unsafe)


@pytest.mark.unit
def test_codex_smoke_prepares_isolated_database(tmp_path: Path) -> None:
    smoke = codex_smoke
    config = smoke.CodexSmokeConfig(
        run_date=date(2026, 5, 13),
        output_dir=tmp_path / "reports" / "phase2-codex-smoke",
        symbol="TSLA",
        repo_root=tmp_path,
        python_executable=Path("python"),
    )
    database_path = config.database_path
    database_path.parent.mkdir(parents=True)
    database_path.write_text("stale\n", encoding="utf-8")
    database_path.with_name(f"{database_path.name}-wal").write_text("stale\n", encoding="utf-8")

    smoke.prepare_clean_database(config)

    assert not database_path.exists()
    assert not database_path.with_name(f"{database_path.name}-wal").exists()


@pytest.mark.unit
def test_codex_smoke_uses_safe_symbol_slug_for_database_and_run_id(tmp_path: Path) -> None:
    smoke = codex_smoke
    config = smoke.CodexSmokeConfig(
        run_date=date(2026, 5, 13),
        output_dir=tmp_path / "reports" / "phase2-codex-smoke",
        symbol="BTC:USD",
        repo_root=tmp_path,
        python_executable=Path("python"),
    )

    assert config.database_arg_path == Path("data/phase2-codex-smoke-2026-05-13-btc-usd.sqlite3")
    assert smoke.expected_run_id(config) == "codex-smoke-2026-05-13-btc-usd"


@pytest.mark.unit
def test_codex_smoke_detects_restricted_ignored_file_changes(tmp_path: Path) -> None:
    smoke = codex_smoke
    (tmp_path / ".env").write_text("before=true\n", encoding="utf-8")
    (tmp_path / "reports").mkdir()

    baseline = smoke.snapshot_restricted_paths(tmp_path)
    (tmp_path / "reports" / "allowed.txt").write_text("allowed\n", encoding="utf-8")
    smoke.ensure_restricted_paths_unchanged(tmp_path, baseline)

    (tmp_path / ".env").write_text("after=true\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="outside allowed artifact roots"):
        smoke.ensure_restricted_paths_unchanged(tmp_path, baseline)


@pytest.mark.unit
def test_codex_smoke_disables_bytecode_writes_for_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    smoke = codex_smoke
    config = smoke.CodexSmokeConfig(
        run_date=date(2026, 5, 13),
        output_dir=tmp_path / "reports" / "phase2-codex-smoke",
        symbol="TSLA",
        repo_root=tmp_path,
        python_executable=Path("python"),
    )
    captured_env: dict[str, str] = {}

    def fake_run(*_args: object, env: dict[str, str] | None = None, **_kwargs: object) -> object:
        assert env is not None
        captured_env.update(env)
        return subprocess.CompletedProcess(args=["codex"], returncode=0)

    monkeypatch.setenv("NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE", "1")
    monkeypatch.setattr(
        "nlp_stock_prediction.orchestration.codex_smoke.shutil.which",
        lambda _name: "codex",
    )
    monkeypatch.setattr(
        "nlp_stock_prediction.orchestration.codex_smoke.importlib.util.find_spec",
        lambda _name: object(),
    )
    monkeypatch.setattr(smoke, "prepare_clean_run_dir", lambda _config: None)
    monkeypatch.setattr(smoke, "prepare_clean_database", lambda _config: None)
    monkeypatch.setattr(smoke, "require_clean_tracked_status", lambda _repo_root: "")
    monkeypatch.setattr(smoke, "snapshot_restricted_paths", lambda _repo_root: {})
    monkeypatch.setattr(
        "nlp_stock_prediction.orchestration.codex_smoke.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(smoke, "verify_smoke_outputs", lambda _config: None)
    monkeypatch.setattr(smoke, "ensure_tracked_status_unchanged", lambda *_args: None)
    monkeypatch.setattr(smoke, "ensure_restricted_paths_unchanged", lambda *_args: None)

    smoke.run_codex_smoke(config)

    assert captured_env["PYTHONDONTWRITEBYTECODE"] == "1"


@pytest.mark.codex_smoke
def test_phase2_real_codex_smoke_runner_is_opt_in() -> None:
    if os.environ.get("NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE") != "1":
        pytest.skip("Set NLP_STOCK_PREDICTION_RUN_CODEX_SMOKE=1 to run real Codex smoke.")
    if shutil.which("codex") is None:
        pytest.skip("Codex CLI is not available on PATH.")
    if importlib.util.find_spec("mcp") is None:
        pytest.skip('Install the optional smoke extra: python -m pip install -e ".[codex-smoke]"')

    smoke = _load_script_module()
    exit_code = smoke.main(
        [
            "--date",
            "2026-05-13",
            "--output",
            "reports/phase2-codex-smoke-pytest",
            "--symbol",
            "TSLA",
            "--repo-root",
            str(REPO_ROOT),
            "--python",
            sys.executable,
        ]
    )

    assert exit_code == 0

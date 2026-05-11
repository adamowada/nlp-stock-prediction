from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from nlp_stock_prediction.cli import (
    CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE,
    build_parser,
    build_run_config,
    main,
)
from nlp_stock_prediction.contracts import RiskProfile, RunConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def _module_env() -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(SRC_DIR) if not existing_pythonpath else f"{SRC_DIR}{os.pathsep}{existing_pythonpath}"
    )
    return env


@pytest.mark.unit
def test_parser_accepts_minimal_run_command_and_defaults() -> None:
    args = build_parser().parse_args(["run", "--date", "2026-05-11", "--output", "reports/"])

    assert args.command == "run"
    assert args.run_date == date(2026, 5, 11)
    assert args.output_dir == Path("reports")
    assert args.capital is None
    assert args.risk_profile == RiskProfile.EXPLORATORY.value
    assert args.fixture_dir is None
    assert args.cache_dir is None
    assert args.offline is False


@pytest.mark.unit
def test_main_without_subcommand_prints_help_and_succeeds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "usage: python -m nlp_stock_prediction" in captured.out
    assert "run" in captured.out
    assert captured.err == ""


@pytest.mark.unit
def test_module_help_subprocess_exposes_canonical_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "nlp_stock_prediction", "--help"],
        cwd=PROJECT_ROOT,
        env=_module_env(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage: python -m nlp_stock_prediction" in result.stdout
    assert "Generate an evidence-grounded daily stock opportunity report." in result.stdout
    assert "run" in result.stdout
    assert result.stderr == ""


@pytest.mark.unit
def test_run_command_phase1_exit_behavior_after_contract_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "reports"

    exit_code = main(
        [
            "run",
            "--date",
            "2026-05-11",
            "--output",
            str(output_dir),
            "--capital",
            "1000",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == CONTRACT_GATE_NOT_IMPLEMENTED_EXIT_CODE
    assert captured.out == ""
    assert "run command contract is available" in captured.err
    assert not output_dir.exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["run", "--date", "05/11/2026", "--output", "reports/"],
            "expected YYYY-MM-DD",
        ),
        (
            [
                "run",
                "--date",
                "2026-05-11",
                "--output",
                "reports/",
                "--capital",
                "not-money",
            ],
            "expected a decimal number",
        ),
        (
            [
                "run",
                "--date",
                "2026-05-11",
                "--output",
                "reports/",
                "--capital",
                "-1",
            ],
            "capital must be non-negative",
        ),
    ],
)
def test_parser_rejects_invalid_date_and_capital(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(argv)

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert message in captured.err


@pytest.mark.unit
@pytest.mark.parametrize("risk_profile", [profile.value for profile in RiskProfile])
def test_parser_accepts_all_public_risk_profile_choices(risk_profile: str) -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--date",
            "2026-05-11",
            "--output",
            "reports/",
            "--risk-profile",
            risk_profile,
        ]
    )

    assert args.risk_profile == risk_profile


@pytest.mark.unit
def test_parser_rejects_unknown_risk_profile(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(
            [
                "run",
                "--date",
                "2026-05-11",
                "--output",
                "reports/",
                "--risk-profile",
                "aggressive",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "invalid choice" in captured.err
    assert "aggressive" in captured.err


@pytest.mark.unit
def test_parser_accepts_fixture_cache_and_offline_options(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixtures"
    cache_dir = tmp_path / "cache"

    args = build_parser().parse_args(
        [
            "run",
            "--date",
            "2026-05-11",
            "--output",
            str(tmp_path / "reports"),
            "--fixture-dir",
            str(fixture_dir),
            "--cache-dir",
            str(cache_dir),
            "--offline",
        ]
    )

    assert args.fixture_dir == fixture_dir
    assert args.cache_dir == cache_dir
    assert args.offline is True


@pytest.mark.unit
def test_build_run_config_constructs_public_contract(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports"
    fixture_dir = tmp_path / "fixtures"
    cache_dir = tmp_path / "cache"
    args = build_parser().parse_args(
        [
            "run",
            "--date",
            "2026-05-11",
            "--output",
            str(output_dir),
            "--capital",
            "1250.50",
            "--risk-profile",
            RiskProfile.BALANCED.value,
            "--fixture-dir",
            str(fixture_dir),
            "--cache-dir",
            str(cache_dir),
            "--offline",
        ]
    )

    config = build_run_config(args)

    assert isinstance(config, RunConfig)
    assert config.run_date == date(2026, 5, 11)
    assert config.output_dir == output_dir
    assert config.capital == Decimal("1250.50")
    assert config.risk_profile is RiskProfile.BALANCED
    assert config.fixture_dir == fixture_dir
    assert config.cache_dir == cache_dir
    assert config.offline is True

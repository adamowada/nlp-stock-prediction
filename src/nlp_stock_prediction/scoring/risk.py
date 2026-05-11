"""Retail risk controls for recommendation candidates."""

from __future__ import annotations

from decimal import Decimal

from nlp_stock_prediction.contracts import (
    InstrumentType,
    PositionType,
    RiskAssessment,
    RiskProfile,
)

DEFAULT_MAX_ACCOUNT_RISK_PCT = Decimal("0.01")

OPTION_INSTRUMENTS = {
    InstrumentType.CALL_OPTION,
    InstrumentType.PUT_OPTION,
    InstrumentType.OPTION_SPREAD,
    InstrumentType.CASH_SECURED_PUT,
    InstrumentType.COVERED_CALL,
}


def assess_risk(
    *,
    instrument: InstrumentType,
    position_type: PositionType,
    risk_profile: RiskProfile = RiskProfile.EXPLORATORY,
    account_capital: Decimal | None = None,
    max_loss_estimate: Decimal | None = None,
    max_account_risk_pct: Decimal = DEFAULT_MAX_ACCOUNT_RISK_PCT,
    uses_margin: bool = False,
) -> RiskAssessment:
    """Apply default no-margin, no-naked-options, one-percent risk controls."""

    failed_gates: list[str] = []
    if uses_margin:
        failed_gates.append("margin-not-allowed")
    if instrument == InstrumentType.UNKNOWN:
        failed_gates.append("unsupported-instrument")
    if _is_naked_option(instrument, position_type):
        failed_gates.append("naked-options-not-allowed")

    defined_risk = _is_defined_risk(instrument, position_type)
    if not defined_risk:
        failed_gates.append("defined-risk-required")

    position_size_pct: Decimal | None = None
    sizing_basis: str
    if account_capital is None:
        position_size_pct = max_account_risk_pct
        sizing_basis = (
            "percentage-only sizing: no account capital was provided, so max risk is expressed "
            "as a percent of account value."
        )
    elif account_capital <= Decimal("0"):
        failed_gates.append("account-capital-must-be-positive")
        sizing_basis = "account capital must be positive before dollar risk can be evaluated"
    elif max_loss_estimate is None:
        failed_gates.append("missing-max-loss-estimate")
        sizing_basis = "max loss estimate is required when account capital is provided"
    else:
        position_size_pct = max_loss_estimate / account_capital
        sizing_basis = (
            f"maximum loss is {position_size_pct:.2%} of account capital; "
            f"policy limit is {max_account_risk_pct:.2%}"
        )
        if position_size_pct > max_account_risk_pct:
            failed_gates.append("max-account-risk-exceeded")

    return RiskAssessment(
        risk_profile=risk_profile,
        defined_risk=defined_risk,
        margin_required=uses_margin,
        max_account_risk_pct=max_account_risk_pct,
        account_capital=account_capital,
        max_loss_estimate=max_loss_estimate,
        position_size_pct=position_size_pct,
        passed=not failed_gates,
        failed_gates=tuple(failed_gates),
        sizing_basis=sizing_basis,
    )


def _is_naked_option(instrument: InstrumentType, position_type: PositionType) -> bool:
    return instrument in OPTION_INSTRUMENTS and position_type == PositionType.SHORT


def _is_defined_risk(instrument: InstrumentType, position_type: PositionType) -> bool:
    if instrument == InstrumentType.SHARES:
        return position_type == PositionType.LONG
    if instrument in {InstrumentType.CALL_OPTION, InstrumentType.PUT_OPTION}:
        return position_type == PositionType.LONG
    if instrument == InstrumentType.OPTION_SPREAD:
        return position_type == PositionType.DEFINED_RISK
    if instrument in {InstrumentType.CASH_SECURED_PUT, InstrumentType.COVERED_CALL}:
        return position_type in {PositionType.INCOME, PositionType.DEFINED_RISK}
    return False


__all__ = ["DEFAULT_MAX_ACCOUNT_RISK_PCT", "assess_risk"]

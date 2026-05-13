"""Legacy recommendation scoring helpers.

Retained for the current compatibility `run` command. New scoring work should produce prediction
candidate confidence, uncertainty, and evidence-status outputs instead of trade recommendations.
"""

from nlp_stock_prediction.scoring.recommendations import (
    RecommendationSignals,
    build_no_trade_summary,
    score_strategy_cluster,
    select_qualified_candidates,
)
from nlp_stock_prediction.scoring.risk import DEFAULT_MAX_ACCOUNT_RISK_PCT, assess_risk

__all__ = [
    "DEFAULT_MAX_ACCOUNT_RISK_PCT",
    "RecommendationSignals",
    "assess_risk",
    "build_no_trade_summary",
    "score_strategy_cluster",
    "select_qualified_candidates",
]

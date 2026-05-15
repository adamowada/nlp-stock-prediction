from __future__ import annotations

import pytest

from nlp_stock_prediction.contracts.live_validation import require_live_metadata, text_is_non_live


@pytest.mark.unit
def test_live_metadata_rejects_tokenized_non_live_markers_without_substring_false_positives() -> (
    None
):
    assert text_is_non_live("fixture-market-data")
    assert text_is_non_live("dummy provider")
    assert not text_is_non_live("Smokehouse Capital live market data")

    require_live_metadata(
        "source query",
        "query-live-smokehouse",
        {"provider_name": "Smokehouse Capital", "report_data_mode": "live"},
    )
    with pytest.raises(ValueError, match="non-live"):
        require_live_metadata(
            "source query",
            "query-fixture-provider",
            {"provider_name": "fixture-market-data", "report_data_mode": "live"},
        )

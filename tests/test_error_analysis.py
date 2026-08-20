import pandas as pd
import pytest

from src.analysis.error_analysis import ranked_error_cases


def test_ranked_error_cases_returns_auc_relevant_extremes() -> None:
    frame = pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b", "c", "d"],
            "fold": [0, 1, 0, 1],
            "ACL__gold": [0.0, 0.0, 1.0, 1.0],
            "ACL__prediction": [0.2, 0.8, 0.1, 0.9],
        }
    )

    result = ranked_error_cases(frame, ["ACL"], top_k=1)

    assert result["ACL"]["highest_ranked_negatives"][0]["StudyInstanceUID"] == "b"
    assert result["ACL"]["lowest_ranked_positives"][0]["StudyInstanceUID"] == "c"


def test_ranked_error_cases_rejects_invalid_predictions() -> None:
    frame = pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b"],
            "ACL__gold": [0.0, 1.0],
            "ACL__prediction": [0.2, 1.1],
        }
    )

    with pytest.raises(ValueError, match="invalid for ACL"):
        ranked_error_cases(frame, ["ACL"])

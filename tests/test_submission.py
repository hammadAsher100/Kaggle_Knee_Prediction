"""Tests for strict submission construction."""

import numpy as np
import pandas as pd
import pytest

from src.inference.submission import build_submission, validate_submission


def test_submission_schema_and_range_contract() -> None:
    sample = pd.DataFrame({"StudyInstanceUID": ["b", "a"], "ACL": [0.5, 0.5]})
    submission = build_submission(["a", "b"], np.asarray([[0.2], [0.8]]), sample)
    assert submission["ACL"].tolist() == [0.8, 0.2]
    validate_submission(submission, sample)
    invalid = submission.copy()
    invalid.loc[0, "ACL"] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_submission(invalid, sample)

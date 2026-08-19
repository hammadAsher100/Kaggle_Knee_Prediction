"""Tests for OOF uncertainty and subgroup summaries."""

import numpy as np
import pandas as pd

from src.analysis.oof_analysis import bootstrap_macro_auc, subgroup_auc


def test_bootstrap_and_subgroup_analysis_are_explicit_about_valid_targets() -> None:
    truth = np.asarray([[0], [0], [1], [1]], dtype=float)
    probabilities = np.asarray([[0.1], [0.2], [0.8], [0.9]])
    interval = bootstrap_macro_auc(truth, probabilities, ["ACL"], iterations=20, seed=7)
    assert interval["point_estimate"] == 1.0
    assert 0 < interval["valid_bootstraps"] <= 20
    frame = pd.DataFrame(
        {
            "PatientSex": ["Female", "Female", "Male", "Male"],
            "ACL__gold": truth[:, 0],
            "ACL__prediction": probabilities[:, 0],
        }
    )
    groups = subgroup_auc(frame, ["ACL"], subgroup_column="PatientSex")
    assert groups["Female"]["macro_auc"] is None
    assert groups["Male"]["valid_target_count"] == 0

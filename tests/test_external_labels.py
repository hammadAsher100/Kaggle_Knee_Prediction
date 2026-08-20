import pandas as pd
import pytest

from src.labeling.external_labels import normalize_external_labels


def test_normalize_external_labels_aligns_ids_and_schema() -> None:
    train = pd.DataFrame({"StudyInstanceUID": ["a", "b"], "ACL": [1.0, None]})
    external = pd.DataFrame({"StudyInstanceUID": ["b", "a"], "ACL": [0.2, 0.8]})

    result = normalize_external_labels(train, external, ["ACL"])

    assert result["StudyInstanceUID"].tolist() == ["a", "b"]
    assert result["ACL__semantic_probability"].tolist() == [0.8, 0.2]
    assert result["ACL__rule_probability"].tolist() == [0.8, 0.2]


def test_normalize_external_labels_rejects_id_mismatch() -> None:
    train = pd.DataFrame({"StudyInstanceUID": ["a", "b"], "ACL": [1.0, None]})
    external = pd.DataFrame({"StudyInstanceUID": ["a", "c"], "ACL": [0.8, 0.2]})

    with pytest.raises(ValueError, match="study IDs differ"):
        normalize_external_labels(train, external, ["ACL"])


def test_normalize_external_labels_rejects_bad_probability() -> None:
    train = pd.DataFrame({"StudyInstanceUID": ["a"], "ACL": [1.0]})
    external = pd.DataFrame({"StudyInstanceUID": ["a"], "ACL": [1.2]})

    with pytest.raises(ValueError, match="invalid for ACL"):
        normalize_external_labels(train, external, ["ACL"])

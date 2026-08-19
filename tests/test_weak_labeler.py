"""Tests for multilingual weak-label polarity and uncertainty."""

from __future__ import annotations

from src.labeling.weak_labeler import label_report

TARGETS = ("ACL", "Effusion", "Fracture")


def test_positive_and_negative_evidence_are_distinguished() -> None:
    labels = label_report(
        "ACL rupture is present. No fracture. Moderate joint effusion.",
        target_columns=TARGETS,
    )

    assert labels.targets["ACL"].binary_label == 1
    assert labels.targets["Fracture"].binary_label == 0
    assert labels.targets["Effusion"].binary_label == 1


def test_intact_ligament_is_negative_and_possible_finding_is_uncertain() -> None:
    labels = label_report(
        "The ACL is intact. Possible small joint effusion.",
        target_columns=TARGETS,
    )

    assert labels.targets["ACL"].binary_label == 0
    assert labels.targets["Effusion"].binary_label is None
    assert labels.targets["Effusion"].status == "uncertain"


def test_spanish_negation_is_supported() -> None:
    labels = label_report(
        "Hallazgos de la rodilla sin fractura. Derrame articular moderado.",
        target_columns=TARGETS,
    )

    assert labels.language.language == "es"
    assert labels.targets["Fracture"].binary_label == 0
    assert labels.targets["Effusion"].binary_label == 1

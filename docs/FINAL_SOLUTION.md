# Final Solution

Status: full-data pipeline executed and locally validated; no competition
submission made.

The current reproducible candidate is a compact 2.5D DINOv2 system. For the
best auditable sagittal, coronal, and axial series, four uniform centers are
sampled and each center is represented by its previous/current/next slices.
DICOM geometry determines ordering; robust percentile normalization handles
scanner intensity variation. DINOv2-small is loaded from an offline
Apache-licensed Kaggle model and frozen. Its 384-dimensional slice embeddings
are cached once in FP16.

Five patient-grouped folds train small plane-aware attention heads over those
embeddings. Targets are soft multilingual report probabilities with hard gold
overrides and fold-nested blend selection. Selection uses gold-only macro ROC
AUC. Final probabilities average the five fold heads and apply conservative
positive-slope Platt calibration fitted on OOF gold predictions.

The original checkpoint selection repeatedly inspected the evaluated fold's
gold AUC and was replaced by fixed-epoch training. The corrected original-teacher
image model scores 0.6595. Replacing the training teacher with validated public
CC0 hybrid report labels raises strict image-only OOF macro ROC AUC to 0.7292.
An equal-probability ensemble of both independently valid image models scores
0.7466. Reports are used only as training supervision; they are unavailable at
test inference. Target-specific attention (0.7045), rank averaging (0.7212),
and a ridge probe (0.6067) were rejected for the macro submission.

Because only 58 studies have gold labels, all local estimates have high
variance and are not leaderboard scores. A credited private fork of a leading
public 20-member DINO/RadImageNet rank ensemble also produced a strictly valid
example-test submission artifact, but it has not been submitted.

The in-house two-model inference kernel also completed on the example test:
3 studies, 15 geometry-ordered series, 557 slices, zero header failures, and a
1.0 valid-stack fraction. Its `submission.csv` exactly matches the required
3-row, 13-column sample schema. During scoring, Kaggle replaces the example test
with the hidden test. Neither candidate has been submitted automatically.

The final code kernel independently orders hidden-test DICOMs, extracts the
same features, loads all five checkpoints with Internet disabled, and writes a
strictly validated `submission.csv`. It does not submit automatically.

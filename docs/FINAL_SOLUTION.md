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

On the corrected complete feature artifact, the selected image head scores
0.6678 gold-only OOF macro ROC AUC. A leakage-safe nested simplex blend of image,
semantic-report, and rule-report predictions scores 0.7471. Blend weights are
selected using only the development portion of each outer fold. Because only
58 studies have gold labels, this estimate has high variance and is not a
leaderboard score. A target-specific blend (0.7233) and nested ridge probe
(0.6067) were rejected.

The final code kernel independently orders hidden-test DICOMs, extracts the
same features, loads all five checkpoints with Internet disabled, and writes a
strictly validated `submission.csv`. It does not submit automatically.

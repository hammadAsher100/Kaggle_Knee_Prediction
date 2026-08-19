# Final Solution

Status: provisional pipeline implemented; full-data image CV awaiting the
running metadata dependency.

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

The final code kernel independently orders hidden-test DICOMs, extracts the
same features, loads all five checkpoints with Internet disabled, and writes a
strictly validated `submission.csv`. It does not submit automatically.

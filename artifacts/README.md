# Artifact policy

Artifact contents are ignored by Git except placeholder files and the lightweight
experiment registry.

- Tier A — preserve: fold definitions, OOF predictions, best checkpoints,
  target schema, experiment configs/registry, final ensemble weights.
- Tier B — cache when practical: DICOM metadata, weak labels, selected-series
  manifests, preprocessed representations.
- Tier C — disposable: temporary tensors, debug images, intermediate batches,
  failed or redundant checkpoints.

Tier A/B binaries live in Kaggle outputs or a dedicated artifact store, not Git.
Only compact review artifacts should be copied to this laptop.

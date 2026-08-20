# Research Audit — 2026-08-20

## Executive finding

The original performance gap was not primarily a backbone-size problem. Three
methodological defects had to be resolved first:

1. The first image artifact contained 34 of 177 metadata parts and real features
   for only 850 of 4,407 studies. Its 0.5317 result is invalid.
2. The corrected image run selected a checkpoint after repeatedly observing the
   evaluated fold's 11–12 gold studies. Its 0.6678 result is optimistically biased.
3. Report/image OOF fusion is not deployable because reports are not present in
   the test table. Its 0.7471 score is a diagnostic, not a submission estimate.

The corrected fixed-epoch, image-only OOF macro AUC is 0.65946 with a bootstrap
95% interval of 0.60355–0.71342. This is the trustworthy internal baseline.

## Validation

- Split unit: study. DICOM `PatientID` is complete but has 4,407 unique values for
  4,407 studies, so it creates the same groups as `StudyInstanceUID`.
- Five folds are retained. With only 58 gold studies, more folds would make
  per-fold positive counts even less stable.
- All slices and series from one study stay together.
- Checkpoint duration is fixed before held-out evaluation. Gold labels in the
  outer fold are evaluated once.
- OOF contains every study exactly once, while AUC uses only finite gold labels.
- Submission construction is sample-driven and fails on any row, ID, column,
  order, finiteness, or probability-range mismatch.

## Data and representation audit

- 4,407 training studies
- 24,371 MRI series
- 819,078 DICOM instances
- 177 complete metadata parts
- 58 gold studies per target
- reports present for all training studies
- corrected feature tensor: `(4407, 12, 384)`
- valid DINOv2 stack fraction: 0.999962

The current image representation samples four 2.5D stacks from one selected
series in each of three planes. It records plane but not pulse-sequence identity.
The shared attention pool must use one study vector for all twelve findings.
This is medically restrictive: ACL, menisci, collateral ligaments, OA,
effusion, Baker's cyst, contusion, and fracture are best demonstrated by
different planes, sequences, and slice locations.

The strict model's prediction standard deviation is only 0.021–0.062 depending
on target. Spearman agreement with the stronger report teacher ranges from
0.031 to 0.528. This indicates underfitting and insufficient visual coverage,
not merely poor calibration.

## Supervision audit

The original rule/MiniLM report teacher scores 0.6881 on the sparse gold audit.
A public CC0 hybrid report-label table covers all 4,407 studies and scores
0.8991 on the same audit. Per-target AUC is strongest for ACL (0.9871), MCL
(0.9683), medial meniscus (0.9483), medial OA (0.9318), PF OA (0.9015), and
Baker's cyst (0.9438). It is weakest for synovitis (0.7903).

This audit is not an unbiased estimate because public teacher development may
have used the 58 labels. It is nevertheless a much stronger training target.
External labels are validated for exact study coverage, unique IDs, target
schema, finite values, and the `[0, 1]` range before training.

## Public frontier reverse engineering

The live leaderboard leader was 0.952 when queried on 2026-08-20. Leading
shared systems consistently use:

- graded multilingual report-derived pseudo-labels;
- six clinically defined plane/sequence slots rather than one series per plane;
- substantially more central-band slice coverage;
- geometry-based ordering and physical knee cropping;
- target-specific slice or slot aggregation;
- DINO and RadImageNet diversity;
- overlapping-window test-time inference;
- fold, architecture, resolution, and seed ensembles;
- per-target rank averaging for ROC AUC.

A private credited fork of the leading public ensemble completed successfully.
It produced a strictly valid example-test `submission.csv` from 20 members. It
has not been submitted and therefore has no score under this account.

## Controlled experiment order

1. Fixed-epoch shared attention with original teacher: 0.65946 — baseline.
2. Fixed-epoch shared attention with hybrid teacher: running.
3. Fixed-epoch target-specific attention with hybrid teacher: prepared.
4. Rank/probability OOF ensemble of independently valid image models.
5. If incremental gains justify the cost, extract six-slot, higher-coverage
   features and train target-aware heads.
6. Fine-tune the last DINO blocks only after cached-head experiments establish
   that label quality and aggregation are no longer the dominant bottlenecks.

## Realistic score ladder

- 0.70: stronger teacher plus target-specific pooling may be sufficient.
- 0.80: likely requires six-slot sequence modeling, many more slices, and
  partial encoder fine-tuning.
- 0.85: likely requires architecture/seed diversity and robust target-wise
  ensembling.
- 0.90: requires a public-frontier-class system with high-quality pseudo-labels,
  strong pretrained encoders, physical preprocessing, TTA, and rank ensembles.
- 0.93+: likely requires multiple independently strong model families and
  leaderboard-tested blend decisions. It cannot be promised from 58 local gold
  studies.

## Research basis

- MRNet established study-level knee MRI modeling over complete series and
  multi-plane examination evidence:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/>
- DINOv2 provides robust frozen and fine-tunable visual representations:
  <https://arxiv.org/abs/2304.07193>
- BiomedCLIP provides biomedical image-text pretraining and is a justified
  diversity encoder:
  <https://www.microsoft.com/en-us/research/publication/biomedclip-a-multimodal-biomedical-foundation-model-pretrained-from-fifteen-million-scientific-image-text-pairs/>

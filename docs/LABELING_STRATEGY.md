# Labeling Strategy

Stage 3 is complete. The report corpus contains 4,407 studies, with 58 gold
labels per target. The privacy-preserving language audit found Latin, Cyrillic,
and Greek scripts and substantial English, Spanish, Turkish, Greek, Croatian,
German, Bulgarian, Dutch, and French content.

Missing target cells remain unknown and are never converted to negatives. The
current labeler combines multilingual sentence embeddings from the offline
Apache-licensed `paraphrase-multilingual-MiniLM-L12-v2` artifact with explicit
terminology, negation, and uncertainty rules. Gold labels always override weak
targets.

The rules-only gold macro AUC was 0.6693. The 50/50 semantic/rule blend reached
0.6881 on the same 58-study development subset. This is a development metric,
not image-model OOF. To prevent that blend selection from leaking validation
gold labels, every image CV fold reselects its rule weight using only the other
folds' gold rows. No raw report text is sent to hosted inference services or
included in downloaded logs.

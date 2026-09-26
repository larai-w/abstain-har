# HAR input and split inspection

Dataset kind: **synthetic**.
Feature positions: 561; unique source names: 559.

| Role | Source | Subjects | Rows | Missing classes |
| --- | --- | --- | ---: | --- |
| fit | train | 1, 2, 5, 6 | 24 | none |
| calibration | train | 4 | 6 | none |
| threshold | train | 3 | 6 | none |
| test | test | 7, 8 | 12 | none |

The original test subjects remain exclusively in test. Internal roles use only original training subjects.
This is an input-contract and split-planning result. No model performance has been measured.

Features are addressed by position (`f001` through `f561`), retaining repeated source names as metadata.
Missing classes are reported without changing the seed or silently moving subjects.

See the accompanying report.json for row assignments, class counts, file hashes, and adapter hash.

# HARTH input preparation

Research recordings; no model trained.

Input rows: 6461328. Retained windows: 25531.

| Role | Participants | Retained windows | Missing classes |
| --- | ---: | ---: | --- |
| test | 4 | 5738 | none |
| calibration | 3 | 2961 | none |
| threshold | 3 | 3569 | [140] |
| fit | 12 | 13263 | none |

250 samples per window, no overlap, 30 explicitly named features.
See report.json for subject-level retention, timestamp breaks, mixed labels and hashes.
windows.csv is a local derived dataset and must remain outside version control.

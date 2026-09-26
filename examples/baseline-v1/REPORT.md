# HARTH baseline comparison

Research results on a fixed participant split; no clinical validation.

| Model | Fit-only CV macro-F1 | Test macro-F1 (12 classes) | Test balanced accuracy | Test accuracy |
| --- | ---: | ---: | ---: | ---: |
| dummy | 0.0439 | 0.0638 | 0.0833 | 0.6199 |
| logistic | 0.5808 | 0.5945 | 0.6654 | 0.8865 |
| random_forest | 0.5954 | 0.6407 | 0.6381 | 0.9259 |

Candidate chosen using fitting-participant CV only: **random_forest**.
All three models use the same 30 features, fitting participants, and held-out test windows.

## Test participant variation

| Model | Participant | Windows | Accuracy | Macro-F1 (12 classes) |
| --- | --- | ---: | ---: | ---: |
| dummy | S012 | 1513 | 0.8209 | 0.0751 |
| dummy | S013 | 1452 | 0.6129 | 0.0633 |
| dummy | S017 | 1451 | 0.5975 | 0.0623 |
| dummy | S022 | 1322 | 0.4221 | 0.0495 |
| logistic | S012 | 1513 | 0.9584 | 0.5175 |
| logistic | S013 | 1452 | 0.8650 | 0.5968 |
| logistic | S017 | 1451 | 0.8463 | 0.4804 |
| logistic | S022 | 1322 | 0.8722 | 0.5963 |
| random_forest | S012 | 1513 | 0.9775 | 0.5752 |
| random_forest | S013 | 1452 | 0.9063 | 0.5680 |
| random_forest | S017 | 1451 | 0.8980 | 0.5573 |
| random_forest | S022 | 1322 | 0.9191 | 0.5824 |

## Interpretation boundaries

- No calibration or abstention threshold has been fitted; every test window receives a prediction.
- Fixed 12-class macro-F1 assigns zero to absent/unpredicted classes; balanced accuracy averages recall over classes with true support.
- High accuracy can coexist with low rare-class recall. Inspect the confusion matrices and class support in report.json.
- Participant bootstrap intervals use only four people and are unstable. Window count is not participant count.
- Test results must not guide changes to this protocol. Further adaptation using these results is exploratory.
- Runtime measurements describe this local CPU run, not a serving latency guarantee.

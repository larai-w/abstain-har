# Calibration and abstention

**Exploratory extension: the baseline test split was already viewed.**

| Variant | Log loss | Brier (0–2) | ECE (10 bins) | Threshold | Test coverage | Accepted errors / answers | Test selective risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.2486 | 0.1097 | 0.0388 | 0.43 | 0.9502 | 250 / 5452 | 0.0459 |
| sigmoid | 0.2514 | 0.1096 | 0.0435 | 0.64 | 0.9441 | 246 / 5417 | 0.0454 |

Thresholds were selected on separate participants for maximum coverage with empirical risk ≤5%, coverage ≥50%, and ≥100 accepted windows.
These are selection targets, not test or population guarantees. No test-based method selection was performed.

## Participant operating points

| Variant | Participant | Coverage | Accepted | Errors | Risk |
| --- | --- | ---: | ---: | ---: | ---: |
| raw | S012 | 0.9841 | 1489 | 20 | 0.0134 |
| raw | S013 | 0.9477 | 1376 | 89 | 0.0647 |
| raw | S017 | 0.9345 | 1356 | 79 | 0.0583 |
| raw | S022 | 0.9312 | 1231 | 62 | 0.0504 |
| sigmoid | S012 | 0.9769 | 1478 | 21 | 0.0142 |
| sigmoid | S013 | 0.9304 | 1351 | 73 | 0.0540 |
| sigmoid | S017 | 0.9331 | 1354 | 76 | 0.0561 |
| sigmoid | S022 | 0.9334 | 1234 | 76 | 0.0616 |

## Limits

- Class 140 is absent from threshold selection. Inspect classwise counts before interpreting aggregate risk.
- Some calibration classes have only a few examples. Sigmoid calibration can worsen results.
- Four test participants yield unstable bootstrap intervals; no clinical or population safety guarantee follows.
- Confidence is not correctness. Abstained predictions have no asserted label.
- See report.json for full risk–coverage grids, reliability bins, class support, bootstrap intervals and provenance.

# Frozen-model feature stress results

**Exploratory, paired feature-level perturbations on the previously viewed test split.**

Models and thresholds are unchanged. Missing inputs are rejected before prediction; this is not learned abstention.

| Scenario | Variant | Input rejected | Model abstained | Answered | Coverage (all windows) | Errors / answers | Newly wrong accepted |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | raw | 0 | 286 | 5452 | 0.9502 | 250/5452 (0.0459) | 0 |
| clean | sigmoid | 0 | 321 | 5417 | 0.9441 | 246/5417 (0.0454) | 0 |
| missing_thigh_10 | raw | 574 | 263 | 4901 | 0.8541 | 223/4901 (0.0455) | 0 |
| missing_thigh_10 | sigmoid | 574 | 295 | 4869 | 0.8486 | 220/4869 (0.0452) | 0 |
| missing_thigh_30 | raw | 1722 | 203 | 3813 | 0.6645 | 167/3813 (0.0438) | 0 |
| missing_thigh_30 | sigmoid | 1722 | 223 | 3793 | 0.6610 | 166/3793 (0.0438) | 0 |
| missing_thigh_100 | raw | 5738 | 0 | 0 | 0.0000 | 0/0 (undefined) | 0 |
| missing_thigh_100 | sigmoid | 5738 | 0 | 0 | 0.0000 | 0/0 (undefined) | 0 |
| zero_back | raw | 0 | 0 | 5738 | 1.0000 | 5392/5738 (0.9397) | 4857 |
| zero_back | sigmoid | 0 | 0 | 5738 | 1.0000 | 5392/5738 (0.9397) | 4828 |
| gain_half | raw | 0 | 1223 | 4515 | 0.7869 | 819/4515 (0.1814) | 733 |
| gain_half | sigmoid | 0 | 1052 | 4686 | 0.8167 | 898/4686 (0.1916) | 723 |
| gain_double | raw | 0 | 866 | 4872 | 0.8491 | 510/4872 (0.1047) | 252 |
| gain_double | sigmoid | 0 | 361 | 5377 | 0.9371 | 768/5377 (0.1428) | 349 |
| bias_back_x_small | raw | 0 | 951 | 4787 | 0.8343 | 167/4787 (0.0349) | 42 |
| bias_back_x_small | sigmoid | 0 | 489 | 5249 | 0.9148 | 223/5249 (0.0425) | 21 |
| bias_back_x_large | raw | 0 | 588 | 5150 | 0.8975 | 4772/5150 (0.9266) | 4388 |
| bias_back_x_large | sigmoid | 0 | 673 | 5065 | 0.8827 | 4623/5065 (0.9127) | 4288 |
| invert_back_x | raw | 0 | 1380 | 4358 | 0.7595 | 3557/4358 (0.8162) | 3533 |
| invert_back_x | sigmoid | 0 | 4366 | 1372 | 0.2391 | 96/1372 (0.0700) | 21 |
| swap_sensors | raw | 0 | 1059 | 4679 | 0.8154 | 3652/4679 (0.7805) | 3510 |
| swap_sensors | sigmoid | 0 | 1055 | 4683 | 0.8161 | 3592/4683 (0.7670) | 3455 |

Newly wrong accepted counts clean accepted/correct windows that become accepted/wrong under the perturbation.
See report.json for participant/class counts, paired status transitions, shared-accepted error counts and cluster-bootstrap intervals.

Finite-value checks cannot detect every plausible sensor fault. High confidence can persist when features change.
These scenarios are not measured device failures or proof of clinical robustness. No model or threshold was tuned on these results.

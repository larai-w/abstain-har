# Exact-zero gate: exploratory results

Models and thresholds are fixed. Reference inputs are not independently adjudicated fault-free.

## Unperturbed input rejections

| Role | Rejected / windows | Standing, sitting, lying rejected / windows |
| --- | ---: | ---: |
| fit | 0/13263 | 0/7302 |
| calibration | 0/2961 | 0/1732 |
| threshold | 0/3569 | 0/2595 |
| test | 0/5738 | 0/4562 |

## Paired test scenarios

| Scenario | Variant | Input rejected (new gate) | Answers (finite → zero gate) | Errors (finite → zero gate) | New risk |
| --- | --- | ---: | ---: | ---: | ---: |
| clean | raw | 0 | 5452 → 5452 | 250 → 250 | 0.0459 |
| clean | sigmoid | 0 | 5417 → 5417 | 246 → 246 | 0.0454 |
| missing_thigh_10 | raw | 574 | 4901 → 4901 | 223 → 223 | 0.0455 |
| missing_thigh_10 | sigmoid | 574 | 4869 → 4869 | 220 → 220 | 0.0452 |
| missing_thigh_30 | raw | 1722 | 3813 → 3813 | 167 → 167 | 0.0438 |
| missing_thigh_30 | sigmoid | 1722 | 3793 → 3793 | 166 → 166 | 0.0438 |
| missing_thigh_100 | raw | 5738 | 0 → 0 | 0 → 0 | undefined |
| missing_thigh_100 | sigmoid | 5738 | 0 → 0 | 0 → 0 | undefined |
| zero_back | raw | 5738 | 5738 → 0 | 5392 → 0 | undefined |
| zero_back | sigmoid | 5738 | 5738 → 0 | 5392 → 0 | undefined |
| gain_half | raw | 0 | 4515 → 4515 | 819 → 819 | 0.1814 |
| gain_half | sigmoid | 0 | 4686 → 4686 | 898 → 898 | 0.1916 |
| gain_double | raw | 0 | 4872 → 4872 | 510 → 510 | 0.1047 |
| gain_double | sigmoid | 0 | 5377 → 5377 | 768 → 768 | 0.1428 |
| bias_back_x_small | raw | 0 | 4787 → 4787 | 167 → 167 | 0.0349 |
| bias_back_x_small | sigmoid | 0 | 5249 → 5249 | 223 → 223 | 0.0425 |
| bias_back_x_large | raw | 0 | 5150 → 5150 | 4772 → 4772 | 0.9266 |
| bias_back_x_large | sigmoid | 0 | 5065 → 5065 | 4623 → 4623 | 0.9127 |
| invert_back_x | raw | 0 | 4358 → 4358 | 3557 → 3557 | 0.8162 |
| invert_back_x | sigmoid | 0 | 1372 → 1372 | 96 → 96 | 0.0700 |
| swap_sensors | raw | 0 | 4679 → 4679 | 3652 → 3652 | 0.7805 |
| swap_sensors | sigmoid | 0 | 4683 → 4683 | 3592 → 3592 | 0.7670 |
| zero_thigh | raw | 5738 | 5657 → 0 | 1755 → 0 | undefined |
| zero_thigh | sigmoid | 5738 | 5738 → 0 | 1836 → 0 | undefined |
| zero_both | raw | 5738 | 5738 → 0 | 5392 → 0 | undefined |
| zero_both | sigmoid | 5738 | 5738 → 0 | 5392 → 0 | undefined |
| near_zero_back | raw | 0 | 5738 → 5738 | 5392 → 5392 | 0.9397 |
| near_zero_back | sigmoid | 0 | 5738 → 5738 | 5392 → 5392 | 0.9397 |
| constant_back_nonzero | raw | 0 | 4156 → 4156 | 3560 → 3560 | 0.8566 |
| constant_back_nonzero | sigmoid | 0 | 1577 → 1577 | 954 → 954 | 0.6049 |

Exact zero is deliberately narrow. Near-zero, nonzero flatline, gain, bias and placement faults can remain undetected.
No answers means no prediction utility and undefined risk. This is not a general fault detector or clinical safety evidence.

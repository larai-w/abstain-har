# Model card: HARTH activity classification with abstention

## Purpose and boundary

A reproducible research example of participant-separated activity classification, probability calibration and selective prediction. Intended users are researchers and engineers evaluating the recorded pipeline and its limitations. It is not validated for diagnosis, care decisions, fall prevention, real-time monitoring or unattended device operation. There are no production endpoints or measured user outcomes.

## Data and input

The pinned UCI HARTH distribution contains 22 people and 6,461,328 accelerometer samples at 50 Hz. Source attribution, CC BY 4.0 evidence, archive hash and header exceptions are in the [source manifest](../contracts/harth-v1.2-source.json) and [dataset notes](DATASETS.md). The model code's license does not replace the dataset license.

Thirty features describe each five-second, non-overlapping window: mean, population standard deviation, minimum, maximum and RMS for six axes across back and thigh sensors. Inputs use acceleration in g. There are 25,531 retained windows. Incomplete windows are discarded and timestamp gaps break windows; missingness is not imputed. Majority labels are used, including 3,754 mixed-label windows. Label agreement is audit metadata, never an input feature or quality filter.

Participants are assigned before modelling to fit (12), calibration (3), threshold selection (3) and test (4). The split is project-defined, not an official benchmark split. Neither participant identifiers nor labels enter the feature matrix. Sensitive participant characteristics are not evaluated; there is no demographic fairness claim.

## Models and selection

The most-frequent dummy, standardised logistic regression and Random Forest were compared using four-fold GroupKFold within fitting participants. Preprocessing is inside each fold. RF was selected by mean macro-F1, not by test scores: CV macro-F1 was 0.5954 for RF and 0.5808 for logistic regression. RF uses 200 trees, minimum leaf size 2, balanced-subsample weighting and seed 7.

Sigmoid calibration uses three different people and a frozen RF. A separate three-person role selects each confidence threshold. Both raw and sigmoid variants remain available; sigmoid is not declared the winner. See the [baseline](BASELINE.md) and [selective prediction](ABSTENTION.md) protocols.

## Recorded evaluation

| Variant | Test coverage | Error rate among answers |
| --- | ---: | ---: |
| Raw, no abstention | 100% | 7.41% |
| Raw, threshold 0.43 | 95.02% | 4.59% |
| Sigmoid, threshold 0.64 | 94.41% | 4.54% |

Baseline RF macro-F1 is 0.6407 and accuracy 0.9259. Logistic regression has higher test balanced accuracy. RF recall is only 0.3113 for shuffling and 0.3736 for ascending stairs. Class 140 has two test windows and no threshold-selection examples.

Calibration increased log loss (0.2486 to 0.2514) and top-label ECE (0.0388 to 0.0435); the small Brier reduction does not establish improved calibration. One calibration class has only one window and generates a recorded library warning. Three of four test people exceed 5% error among answers under each selected policy. The threshold role's empirical 5% target is not a future risk bound.

The baseline first used a held-out test split. Calibration, stress and gate extensions reuse those already viewed people and are explicitly exploratory. Participant bootstrap intervals from only four people are unstable. A new independent evaluation is required for stronger generalisation claims.

## Failure modes and input policy

A zeroed back sensor produces 93.97% errors with 100% coverage under both confidence policies. Confidence can remain high when features change. The optional `exact-zero-v1` gate rejects an entire zero-valued sensor block; it does not retrain or calibrate the model. On unperturbed data, no windows were rejected, including 16,191 standing/sitting/lying windows. These recordings were not independently adjudicated fault-free.

Near-zero back signals still yield 93.97% errors with full coverage. Nonzero flatlines, unit/gain changes, offsets and exchanged sensor columns can pass the gate. Zero standard deviation alone is not a fault criterion, since stationary inputs can be valid. Missing-input rejection, exact-zero rejection and model abstention are separate decisions. Rejected/abstained outputs never assert an activity label.

## Operation and reproducibility

The [local JSON contract](LOCAL_INFERENCE.md) limits rows and bytes, validates feature order, checks model hashes and dependency versions, records provenance and refuses output overwrites. Default input policy is `finite-v1`; `exact-zero-v1` requires explicit selection. Models use executable joblib serialization: only load trusted locally generated artifacts matching recorded evidence.

See [reproduction instructions](REPRODUCTION.md). Raw data, derived vectors and model binaries are excluded from Git. Test fixtures are synthetic. Passing CI establishes checks on code and fixtures, not clinical safety, cross-machine reproducibility or independent author proficiency. Training/validation code and documentation were developed with AI assistance; reviewers should inspect the linked evidence and decision boundaries rather than infer skill from code volume.

## Evidence index

- [Baseline results](../examples/baseline-v1/REPORT.md)
- [Calibration and abstention](../examples/selective-v1/REPORT.md)
- [Feature stress results](../examples/robustness-v1/REPORT.md)
- [Exact-zero gate results](../examples/zero-gate-v1/REPORT.md)
- [Five-minute technical walkthrough](WALKTHROUGH.md)

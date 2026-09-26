# Local inference: feature contract v1, response v2

A file-based research interface for the recorded HARTH models. It does not start a server, train a model, change thresholds, or deploy infrastructure. Both variants remain available; the caller explicitly selects `raw` or `sigmoid` without choosing a new winner on the test split.

## Request

Use strict JSON with exactly these top-level fields:

| Field | Contract |
| --- | --- |
| `contract` | Exactly `harth-summary-v1`. |
| `dataset_kind` | `synthetic` or `research`; caller-declared provenance, not independently authenticated. |
| `features` | All 30 names from `harth_input.FEATURES`, in that exact order. |
| `samples` | 1–1,000 rows, each with exactly `id` and `values`. |

Each ID is a unique opaque ASCII string of 1–80 letters, digits, underscores or hyphens. Do not use personal identifiers. Each vector contains exactly 30 JSON numbers or `null` for missing features. Booleans and numeric strings are not coerced. Duplicate JSON keys, NaN/Infinity literals, overflowed numbers, unknown fields and incorrect shapes reject the entire batch before model deserialization. Requests are limited to 2,000,000 bytes. Targets and participant metadata are not accepted as features.

Features represent 250 consecutive samples at 50 Hz (five seconds), acceleration in g, six channels in back-x/y/z then thigh-x/y/z order. Each channel has mean, population standard deviation, minimum, maximum and RMS, in that order. A matching column name does not prove that a caller actually used these units, placement or computation. The feature contract is distinct from the toolkit's synthetic care-summary contract.

## Response

`response_version: 2` returns the declared contract, dataset kind, selected `input_policy`, ordered `decisions`, status `counts`, and `provenance`. The provenance records model/reference/request/source hashes, model variant, fixed threshold and installed ML dependency versions. Confidence is the maximum class probability, not an assurance of correctness. The earlier v1 example is retained as a historical artifact; its source hashes identify that implementation.

| Status | Label | Confidence | Reason |
| --- | --- | --- | --- |
| `input_rejected` | `null` | `null` | `missing_feature`, `zero_back_sensor`, `zero_thigh_sensor` or `zero_both_sensors` |
| `model_abstained` | `null` | Maximum probability | `policy_abstention` |
| `accepted` | Activity class integer | Maximum probability | `threshold_met` |

A row with any missing value never reaches the classifier. Remaining rows use the recorded inclusive threshold (`confidence >= threshold`): raw 0.43, sigmoid 0.64. A null policy threshold abstains on every valid row. Diagnostic candidate labels for rejected or abstained rows are never emitted. Counts sum to the input row count. Successful processing may contain only rejections: exit code 0 means the batch was processed, not that a prediction was accepted.

`--input-policy finite-v1` is the default. Explicitly select `--input-policy exact-zero-v1` to also reject a window if all fifteen summaries of either three-axis sensor are exactly zero. Missing values take precedence. This new policy has a [fixed protocol](ZERO_GATE.md) and [separate evaluation](../examples/zero-gate-v1/REPORT.md). It does not reject a single zero axis or zero variance alone. The [v2 example response](../examples/zero-gate-v1/local-response.json) uses the same invented request with this policy enabled.

Malformed batches, missing or mismatched artifacts, dependency mismatch and existing output paths fail with exit code 2 and no new successful response. Output files are created exclusively; existing results are not overwritten. Diagnostics do not echo request contents. Other unexpected runtime faults may terminate with a different nonzero code; no fallback prediction is fabricated. This is a local batch tool, without an HTTP error protocol, concurrent artifact replacement protection or production availability guarantees.

## Reproduce

Prepare the pinned ML environment and local artifacts using [the selective protocol](ABSTENTION.md), then:

```bash
.venv/bin/python local_predict.py examples/local-inference-v1/request.json \
  --models artifacts/selective-v1 --variant sigmoid \
  --output artifacts/local-inference-v1/response.json
```

Use a fresh output path. The example contains invented constant and missing sensor summaries, no research feature rows and no ground-truth activity labels. The response uses the recorded research-trained model; it is a mechanics example, not an accuracy evaluation. Inspect the [recorded response](../examples/local-inference-v1/response.json).

Only deserialize locally generated joblib artifacts matching the checked-in selective report. Joblib can execute Python code. These hashes detect mismatched local artifacts; they do not authenticate a maliciously replaced model together with its reference. Model binaries are not distributed in this repository.

## Known limits and verification

The default finite-value gate cannot detect swapped sensors, incorrect units or plausible but faulty finite values. In particular, **all-zero inputs are not rejected by the default policy**. The opt-in exact-zero policy rejects that specific pattern, with zero observed rejections on 25,531 unperturbed windows, including 16,191 standing/sitting/lying windows. The recordings are not adjudicated fault-free and this is not a population false-positive bound. Near-zero and nonzero-flatline faults still pass and produce severe high-confidence errors. Further detectors require separate evaluation of missed faults and false rejection of valid stationary windows.

Synthetic regression checks cover raw-signal versus summary transformations, nested missingness, paired counts, undefined risk, rejected-row exclusion, output labels, input schema, model hashes, dependency/schema checks, provenance and overwrite prevention. They use temporary files and fake predictors; CI does not need research artifacts. A separate local CLI run exercises the recorded sigmoid artifact on the synthetic example. Neither check establishes clinical safety or generalisation to new people.

# Five-minute technical walkthrough

A five-minute reading path through the project's key decisions. Developed with AI assistance; design decisions, review and testing were mine.

## 0:00–1:00 — Question and data

Can a classifier recognise activity for unseen people, and what changes when it may decline to answer? Start with the [model card](MODEL_CARD.md). The 22-person HARTH archive becomes five-second windows with 30 acceleration summaries. Split by person to avoid evaluating highly related windows from the same person in both fitting and test sets. Keep identifiers, labels and label purity out of the model features.

## 1:00–2:00 — Selection without using test scores

Show `baseline.py`: fit-only GroupKFold selects RF by mean macro-F1. A dummy establishes context and logistic regression remains a credible comparison. RF has higher accuracy, but logistic regression has higher balanced accuracy on test. The choice of metric changes the story; inspect minority-class recall and support rather than quoting accuracy alone.

## 2:00–3:00 — Calibration is not automatically an improvement

Show [selective results](../examples/selective-v1/REPORT.md). Calibration has separate people and the base forest stays frozen. Thresholds come from a third role. Both raw and sigmoid are reported. Log loss and ECE get worse after calibration. A pooled error rate below 5% conceals three test participants above that target. The later experiments reuse the same test people and are exploratory.

## 3:00–4:00 — A concrete high-confidence failure

Show the [zero-back stress result](../examples/robustness-v1/REPORT.md): 100% coverage, 93.97% errors. Then show the [exact-zero gate](../examples/zero-gate-v1/REPORT.md). It catches the precise zero pattern without rejecting observed unperturbed windows, but misses near-zero and nonzero-flatline inputs. Rejecting everything produces undefined risk and no useful predictions, not perfect accuracy.

## 4:00–5:00 — Interface, provenance and remaining work

Show [the request](../examples/local-inference-v1/request.json) and [response v2](../examples/zero-gate-v1/local-response.json). Explain why a rejected or abstained row has a null label, why model deserialization requires trusted hash-matched artifacts, and why response status is different from process success. Walk through the [reproduction commands](REPRODUCTION.md). State the remaining boundaries: same-host reproduction is not independent validation; there is no deployment, external population evaluation or clinical claim.

## Follow-up review questions

1. What leaks if scaling is fitted before the group folds?
2. Why can a calibrated model's most probable class change?
3. Why do we count input rejection separately from model abstention?
4. What would count as evidence that the zero rule rejects valid stationary windows?
5. Which data could be used to tune a broader detector without calling the reused test independent?
6. What would have to change before exposing this interface as a service?

Useful live exercise: add a synthetic missing-feature case, explain its expected response before running it, and trace the model-call exclusion. Keep research data and previously recorded results unchanged.

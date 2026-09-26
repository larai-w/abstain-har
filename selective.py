#!/usr/bin/env python3
"""Calibrate the fixed HARTH forest and evaluate explicit abstention."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import warnings

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.frozen import FrozenEstimator
from threadpoolctl import threadpool_limits

from baseline import ROOT, load_inputs, metrics, models
from harth_input import LABELS, sha

GRID = [i / 100 for i in range(101)]
TARGET_RISK = 0.05
MIN_COVERAGE = 0.50
MIN_ACCEPTED = 100


def validate_probabilities(probabilities, y=None):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(LABELS):
        raise ValueError('expected N by 12 probabilities in declared class order')
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError('invalid probabilities')
    if y is not None:
        y = np.asarray(y)
        if y.shape != (len(p),) or not np.isin(y, LABELS).all():
            raise ValueError('labels do not match probabilities')
    return p


def decision_arrays(probabilities, threshold):
    p = validate_probabilities(probabilities)
    if threshold is not None and (isinstance(threshold, bool) or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('threshold must be null or finite in [0, 1]')
    confidence = p.max(axis=1)
    candidates = np.asarray(LABELS)[p.argmax(axis=1)]
    accepted = np.zeros(len(p), dtype=bool) if threshold is None else confidence >= threshold
    return candidates, confidence, accepted


def predict_with_abstention(probabilities, threshold):
    candidates, confidence, accepted = decision_arrays(probabilities, threshold)
    return [{'status': 'accepted' if keep else 'abstained', 'label': int(label) if keep else None,
             'confidence': float(score)} for label, score, keep in zip(candidates, confidence, accepted)]


def counts(total, accepted, errors):
    return {'total': int(total), 'accepted': int(accepted), 'errors': int(errors),
            'coverage': float(accepted / total) if total else None,
            'selective_risk': float(errors / accepted) if accepted else None}


def selective_metrics(y, probabilities, threshold):
    p = validate_probabilities(probabilities, y)
    predicted, _, accepted = decision_arrays(p, threshold)
    return counts(len(y), accepted.sum(), ((predicted != y) & accepted).sum())


def choose_threshold(y, probabilities):
    if not len(y):
        raise ValueError('threshold selection requires data')
    curve = [{'threshold': t, **selective_metrics(y, probabilities, t)} for t in GRID]
    feasible = [point for point in curve if point['accepted'] >= MIN_ACCEPTED
                and point['coverage'] >= MIN_COVERAGE and point['selective_risk'] <= TARGET_RISK]
    if not feasible:
        return {'status': 'no_feasible_threshold', 'threshold': None,
                'selection': selective_metrics(y, probabilities, None), 'curve': curve}
    chosen = sorted(feasible, key=lambda point: (-point['coverage'], point['threshold']))[0]
    return {'status': 'selected', 'threshold': chosen['threshold'], 'selection': chosen, 'curve': curve}


def probability_scores(y, probabilities):
    p = validate_probabilities(probabilities, y)
    if not len(y):
        raise ValueError('probability scoring requires data')
    indices = np.searchsorted(np.asarray(LABELS), y)
    truth = np.eye(len(LABELS))[indices]
    predicted, confidence, _ = decision_arrays(p, 0)
    correct = predicted == y
    assignments = np.minimum((confidence * 10).astype(int), 9)
    bins = []
    for i in range(10):
        mask = assignments == i
        bins.append({'lower': i / 10, 'upper': (i + 1) / 10, 'count': int(mask.sum()),
                     'mean_confidence': float(confidence[mask].mean()) if mask.any() else None,
                     'accuracy': float(correct[mask].mean()) if mask.any() else None})
    return {'log_loss': float(-np.log(np.clip(p[np.arange(len(y)), indices], np.finfo(float).eps, 1)).mean()),
            'multiclass_brier': float(np.mean(np.sum((p - truth) ** 2, axis=1))),
            'top_label_ece_10_bins': float(sum(b['count'] * abs(b['accuracy'] - b['mean_confidence']) for b in bins if b['count']) / len(y)),
            'reliability_bins': bins}


def fit_calibrator(forest, x, y):
    if list(forest.classes_) != list(LABELS) or set(np.asarray(y).tolist()) != set(LABELS):
        raise ValueError('all twelve classes are required for fitting and calibration')
    calibrated = CalibratedClassifierCV(FrozenEstimator(forest), method='sigmoid', ensemble=False)
    calibrated.fit(x, y)
    if list(calibrated.classes_) != list(LABELS):
        raise ValueError('calibration changed class ordering')
    return calibrated


def bootstrap(per_subject):
    values = np.array([[v['total'], v['accepted'], v['errors']] for v in per_subject.values()])
    rng = np.random.default_rng(7)
    coverages, risks = [], []
    for _ in range(1000):
        total, accepted, errors = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        coverages.append(float(accepted / total))
        if accepted:
            risks.append(float(errors / accepted))
    return {'method': 'participant cluster percentile bootstrap', 'participants': len(values),
            'replicates': 1000, 'seed': 7, 'valid_risk_replicates': len(risks),
            'coverage_interval_95': np.quantile(coverages, [0.025, 0.975]).tolist(),
            'risk_interval_95': np.quantile(risks, [0.025, 0.975]).tolist() if risks else None}


def evaluate(y, probabilities, subjects, policy):
    p = validate_probabilities(probabilities, y)
    t = policy['threshold']
    per_subject = {s: selective_metrics(y[subjects == s], p[subjects == s], t) for s in sorted(set(subjects.tolist()))}
    predicted, _, _ = decision_arrays(p, t)
    return {'probability_scores': probability_scores(y, p), 'classification_without_abstention': metrics(y, predicted),
            'without_abstention': selective_metrics(y, p, 0), 'selected': selective_metrics(y, p, t),
            'per_subject': per_subject,
            'per_true_class': {str(k): selective_metrics(y[y == k], p[y == k], t) for k in LABELS},
            'risk_coverage_curve': [{'threshold': point, **selective_metrics(y, p, point)} for point in [*GRID, None]],
            'participant_bootstrap': bootstrap(per_subject)}


def fit_and_select(data):
    masks = {role: data['roles'] == role for role in ('fit', 'calibration', 'threshold')}
    if not all(mask.any() for mask in masks.values()):
        raise ValueError('fitting, calibration and threshold data are all required')
    forest = models()['random_forest']
    forest.fit(data['X'][masks['fit']], data['y'][masks['fit']])
    sigmoid = fit_calibrator(forest, data['X'][masks['calibration']], data['y'][masks['calibration']])
    variants = {'raw': forest, 'sigmoid': sigmoid}
    policies = {name: choose_threshold(data['y'][masks['threshold']], model.predict_proba(data['X'][masks['threshold']]))
                for name, model in variants.items()}
    return variants, policies


def render(report):
    lines = ['# Calibration and abstention', '', '**Exploratory extension: the baseline test split was already viewed.**', '',
             '| Variant | Log loss | Brier (0–2) | ECE (10 bins) | Threshold | Test coverage | Accepted errors / answers | Test selective risk |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, value in report['test'].items():
        score, selected = value['probability_scores'], value['selected']
        risk = 'undefined' if selected['selective_risk'] is None else f'{selected["selective_risk"]:.4f}'
        lines.append(f'| {name} | {score["log_loss"]:.4f} | {score["multiclass_brier"]:.4f} | {score["top_label_ece_10_bins"]:.4f} | '
                     f'{report["policies"][name]["threshold"]} | {selected["coverage"]:.4f} | {selected["errors"]} / {selected["accepted"]} | {risk} |')
    lines += ['', 'Thresholds were selected on separate participants for maximum coverage with empirical risk ≤5%, coverage ≥50%, and ≥100 accepted windows.',
              'These are selection targets, not test or population guarantees. No test-based method selection was performed.', '',
              '## Participant operating points', '', '| Variant | Participant | Coverage | Accepted | Errors | Risk |', '| --- | --- | ---: | ---: | ---: | ---: |']
    for name, value in report['test'].items():
        for subject, entry in value['per_subject'].items():
            risk = 'undefined' if entry['selective_risk'] is None else f'{entry["selective_risk"]:.4f}'
            lines.append(f'| {name} | {subject} | {entry["coverage"]:.4f} | {entry["accepted"]} | {entry["errors"]} | {risk} |')
    lines += ['', '## Limits', '', '- Class 140 is absent from threshold selection. Inspect classwise counts before interpreting aggregate risk.',
              '- Some calibration classes have only a few examples. Sigmoid calibration can worsen results.',
              '- Four test participants yield unstable bootstrap intervals; no clinical or population safety guarantee follows.',
              '- Confidence is not correctness. Abstained predictions have no asserted label.',
              '- See report.json for full risk–coverage grids, reliability bins, class support, bootstrap intervals and provenance.', '']
    return '\n'.join(lines)


def run(directory: Path, output: Path):
    if output.exists():
        raise ValueError('output already exists; choose a new directory')
    data = load_inputs(directory)
    baseline_path = ROOT / 'examples/baseline-v1/report.json'
    baseline_report = json.loads(baseline_path.read_text())
    if (baseline_report['selection']['candidate'] != 'random_forest'
            or data['hashes'] != baseline_report['input_hashes']
            or sha(ROOT / 'baseline.py') != baseline_report['source_sha256']['baseline.py']
            or data['source_manifest_sha256'] != baseline_report['source_manifest_sha256']):
        raise ValueError('inputs or candidate differ from the recorded baseline')
    test = data['roles'] == 'test'
    if not test.any():
        raise ValueError('test data is required')
    output.mkdir(parents=True, exist_ok=False)
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('error', ConvergenceWarning)
        variants, policies = fit_and_select(data)
        (output / 'policy.json').write_text(json.dumps(policies, indent=2, allow_nan=False) + '\n')
        print('Calibration and threshold selection complete; policies saved before test prediction.', file=sys.stderr, flush=True)
        results = {name: evaluate(data['y'][test], model.predict_proba(data['X'][test]), data['subjects'][test], policies[name])
                   for name, model in variants.items()}
        for name, model in variants.items():
            joblib.dump(model, output / f'{name}.joblib')
    report = {'report_version': 1, 'dataset_kind': 'harth-research', 'evaluation_status': 'exploratory_previously_viewed_test',
              'input_hashes': data['hashes'], 'source_manifest_sha256': data['source_manifest_sha256'],
              'baseline_report_sha256': sha(baseline_path), 'base_model': 'random_forest',
              'base_model_changed': False, 'calibration': 'sigmoid with frozen base estimator',
              'selection_targets': {'empirical_risk': TARGET_RISK, 'minimum_coverage': MIN_COVERAGE, 'minimum_accepted': MIN_ACCEPTED},
              'policies': policies, 'test': results,
              'role_support': {role: {'participants': sorted(set(data['subjects'][data['roles'] == role].tolist())),
                                     'classes': {str(k): int(np.sum((data['roles'] == role) & (data['y'] == k))) for k in LABELS}}
                               for role in ('fit', 'calibration', 'threshold', 'test')},
              'source_sha256': {name: sha(ROOT / name) for name in ('selective.py', 'baseline.py', 'requirements-ml.txt', 'docs/ABSTENTION.md')},
              'model_sha256': {name: sha(output / f'{name}.joblib') for name in variants},
              'environment': {'python': sys.version, 'platform': platform.platform(), 'numeric_thread_limit': 1,
                              'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl')}},
              'git_commit_at_run': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
              'completed_at': datetime.now(timezone.utc).isoformat()}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'REPORT.md').write_text(render(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.directory, args.output)
    except (OSError, ValueError, KeyError, ConvergenceWarning) as error:
        print(f'Selective prediction error: {error}', file=sys.stderr)
        return 2
    print('Calibration and selective prediction experiment completed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

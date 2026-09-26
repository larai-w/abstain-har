#!/usr/bin/env python3
"""Fixed CPU baselines with fit-only grouped selection and held-out reporting."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import warnings

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from harth_input import FEATURES, LABELS, MANIFEST, sha

ROOT = Path(__file__).parent
HEADER = ['window_id', 'subject_id', 'role', 'label', 'label_agreement', *FEATURES]


def load_inputs(directory: Path, manifest_path: Path = MANIFEST) -> dict:
    paths = {name: directory / name for name in ('windows.csv', 'report.json')}
    hashes = {name: sha(path) for name, path in paths.items()}
    report = json.loads(paths['report.json'].read_text())
    manifest = json.loads(manifest_path.read_text())
    if (report.get('dataset_kind') != 'harth-research'
            or report.get('windows_sha256') != hashes['windows.csv']
            or report.get('archive_sha256') != manifest['archive_sha256']
            or report.get('source_manifest_sha256') != sha(manifest_path)
            or report.get('model_feature_columns') != FEATURES):
        raise ValueError('input provenance or feature contract mismatch')
    roles = manifest['split_plan']['roles']
    if set(roles) != {'fit', 'calibration', 'threshold', 'test'}:
        raise ValueError('expected four participant roles')
    ids = [s for group in roles.values() for s in group]
    if len(ids) != len(set(ids)):
        raise ValueError('participant roles overlap')
    assignments = {s: role for role, group in roles.items() for s in group}
    expected = {r['subject_id']: r for r in report['subjects']}
    if set(expected) != set(ids) or len(expected) != len(report['subjects']):
        raise ValueError('report subject coverage differs from manifest')
    values, labels, subjects, role_values = [], [], [], []
    seen = set()
    counts, class_counts = Counter(), Counter()
    previous = {}
    with paths['windows.csv'].open(newline='', encoding='utf-8') as source:
        reader = csv.reader(source, strict=True)
        if next(reader, None) != HEADER:
            raise ValueError('window CSV header must match explicit feature order')
        for row in reader:
            if len(row) != len(HEADER):
                raise ValueError('window CSV row width mismatch')
            window, subject, role = row[:3]
            if window in seen or assignments.get(subject) != role:
                raise ValueError('duplicate window or incorrect participant role')
            parts = window.split(':')
            if len(parts) != 2 or parts[0] != subject or not parts[1].isdigit():
                raise ValueError('window ID must preserve subject and source line')
            line = int(parts[1])
            if line < 2 or (subject in previous and line < previous[subject] + 250):
                raise ValueError('overlapping or unordered windows')
            previous[subject] = line
            label, agreement = int(row[3]), float(row[4])
            vector = [float(x) for x in row[5:]]
            if label not in LABELS or not math.isfinite(agreement) or not 0 < agreement <= 1:
                raise ValueError('invalid label or agreement metadata')
            if not all(math.isfinite(x) for x in vector):
                raise ValueError('non-finite model feature')
            seen.add(window); counts[subject] += 1; class_counts[subject, label] += 1
            values.append(vector); labels.append(label); subjects.append(subject); role_values.append(role)
    for subject, entry in expected.items():
        if (entry['role'] != assignments[subject] or entry['retained_windows'] != counts[subject]
                or entry['label_counts'] != {str(k): class_counts[subject, k] for k in LABELS}):
            raise ValueError('window counts or class support differ from input report')
    if not values or report['retained_windows'] != len(values):
        raise ValueError('empty or incomplete window data')
    if hashes != {name: sha(path) for name, path in paths.items()}:
        raise ValueError('inputs changed during loading')
    return {'X': np.asarray(values), 'y': np.asarray(labels), 'subjects': np.asarray(subjects),
            'roles': np.asarray(role_values), 'hashes': hashes,
            'source_manifest_sha256': sha(manifest_path), 'archive_sha256': report['archive_sha256']}


def models() -> dict:
    return {'dummy': DummyClassifier(strategy='most_frequent'),
            'logistic': make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, solver='lbfgs', class_weight='balanced', max_iter=2000, tol=1e-4)),
            'random_forest': RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                class_weight='balanced_subsample', random_state=7, n_jobs=1)}


def metrics_from_matrix(matrix) -> dict:
    matrix = np.asarray(matrix, dtype=np.int64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = np.diag(matrix)
    total = int(matrix.sum())
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(len(LABELS)), where=(support + predicted) != 0)
    recall = np.divide(correct, support, out=np.zeros(len(LABELS)), where=support != 0)
    return {'windows': total, 'accuracy': float(correct.sum() / total) if total else None,
            'macro_f1_12': float(f1.mean()) if total else None,
            'balanced_accuracy_present_classes': float(recall[support > 0].mean()) if total else None,
            'confusion_matrix': matrix.tolist(), 'label_order': list(LABELS),
            'per_class': {str(label): {'support': int(support[i]), 'predicted': int(predicted[i]),
                          'precision': float(correct[i] / predicted[i]) if predicted[i] else None,
                          'recall': float(recall[i]) if support[i] else None, 'f1': float(f1[i])}
                          for i, label in enumerate(LABELS)}}


def metrics(y, predicted) -> dict:
    return metrics_from_matrix(confusion_matrix(y, predicted, labels=LABELS))


def grouped_folds(subjects):
    if len(set(subjects)) < 4:
        raise ValueError('grouped baseline requires at least four fitting participants')
    return list(GroupKFold(n_splits=4).split(np.zeros(len(subjects)), groups=subjects))


def select_with_fit_only(data: dict) -> tuple[dict, str]:
    mask = data['roles'] == 'fit'
    x, y, subjects = data['X'][mask], data['y'][mask], data['subjects'][mask]
    folds = grouped_folds(subjects)
    results = {}
    for name, template in models().items():
        runs = []
        for train, valid in folds:
            model = clone(template)
            start = perf_counter(); model.fit(x[train], y[train]); elapsed = perf_counter() - start
            score = metrics(y[valid], model.predict(x[valid]))
            runs.append({'train_subjects': sorted(set(subjects[train].tolist())),
                         'validation_subjects': sorted(set(subjects[valid].tolist())),
                         'train_missing_classes': sorted(set(LABELS) - set(y[train].tolist())),
                         'fit_seconds': elapsed, 'metrics': score})
        results[name] = {'folds': runs, 'mean_macro_f1_12': float(np.mean([r['metrics']['macro_f1_12'] for r in runs]))}
        print(f'{name}: grouped CV complete', file=sys.stderr, flush=True)
    # The dummy remains a reference, not a candidate for later probability calibration.
    candidate = sorted(('logistic', 'random_forest'), key=lambda name: (-results[name]['mean_macro_f1_12'], name))[0]
    return results, candidate


def subject_bootstrap(per_subject: dict) -> dict:
    matrices = np.asarray([entry['confusion_matrix'] for entry in per_subject.values()])
    rng = np.random.default_rng(7)
    samples = {key: [] for key in ('accuracy', 'macro_f1_12', 'balanced_accuracy_present_classes')}
    for _ in range(1000):
        result = metrics_from_matrix(matrices[rng.integers(0, len(matrices), len(matrices))].sum(axis=0))
        for key in samples:
            samples[key].append(result[key])
    return {'method': 'participant cluster percentile bootstrap', 'replicates': 1000, 'seed': 7,
            'participants': len(matrices),
            'intervals_95': {key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in samples.items()},
            'limitation': 'Four test participants: intervals are unstable and do not establish population coverage.'}


def render(report: dict) -> str:
    lines = ['# HARTH baseline comparison', '', 'Research results on a fixed participant split; no clinical validation.', '',
             '| Model | Fit-only CV macro-F1 | Test macro-F1 (12 classes) | Test balanced accuracy | Test accuracy |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, result in report['models'].items():
        m = result['test']
        lines.append(f'| {name} | {report["selection"]["cv"][name]["mean_macro_f1_12"]:.4f} | '
                     f'{m["macro_f1_12"]:.4f} | {m["balanced_accuracy_present_classes"]:.4f} | {m["accuracy"]:.4f} |')
    lines += ['', f'Candidate chosen using fitting-participant CV only: **{report["selection"]["candidate"]}**.',
              'All three models use the same 30 features, fitting participants, and held-out test windows.', '',
              '## Test participant variation', '', '| Model | Participant | Windows | Accuracy | Macro-F1 (12 classes) |',
              '| --- | --- | ---: | ---: | ---: |']
    for name, result in report['models'].items():
        for subject, m in result['per_subject'].items():
            lines.append(f'| {name} | {subject} | {m["windows"]} | {m["accuracy"]:.4f} | {m["macro_f1_12"]:.4f} |')
    lines += ['', '## Interpretation boundaries', '',
              '- No calibration or abstention threshold has been fitted; every test window receives a prediction.',
              '- Fixed 12-class macro-F1 assigns zero to absent/unpredicted classes; balanced accuracy averages recall over classes with true support.',
              '- High accuracy can coexist with low rare-class recall. Inspect the confusion matrices and class support in report.json.',
              '- Participant bootstrap intervals use only four people and are unstable. Window count is not participant count.',
              '- Test results must not guide changes to this protocol. Further adaptation using these results is exploratory.',
              '- Runtime measurements describe this local CPU run, not a serving latency guarantee.', '']
    return '\n'.join(lines)


def run(directory: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('output already exists; choose a new directory')
    data = load_inputs(directory)
    fit = data['roles'] == 'fit'; test = data['roles'] == 'test'
    if not fit.any() or not test.any():
        raise ValueError('fit and test windows are required')
    output.mkdir(parents=True, exist_ok=False)
    # Warnings about incomplete optimisation invalidate the experiment, not silently its score.
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter('error', ConvergenceWarning)
        cv, candidate = select_with_fit_only(data)
        selection = {'criterion': 'highest mean four-fold fitting-participant macro_f1_12; alphabetical tie-break',
                     'candidate': candidate, 'test_used_for_selection': False, 'cv': cv}
        (output / 'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False) + '\n')
        results = {}
        for name, model in models().items():
            start = perf_counter(); model.fit(data['X'][fit], data['y'][fit]); fit_seconds = perf_counter() - start
            start = perf_counter(); predicted = model.predict(data['X'][test]); predict_seconds = perf_counter() - start
            target, subjects = data['y'][test], data['subjects'][test]
            per_subject = {s: metrics(target[subjects == s], predicted[subjects == s]) for s in sorted(set(subjects.tolist()))}
            joblib.dump(model, output / f'{name}.joblib')
            results[name] = {'test': metrics(target, predicted), 'per_subject': per_subject,
                             'participant_bootstrap': subject_bootstrap(per_subject),
                             'mean_participant_accuracy': float(np.mean([v['accuracy'] for v in per_subject.values()])),
                             'fit_seconds': fit_seconds, 'test_batch_predict_seconds': predict_seconds,
                             'model_sha256': sha(output / f'{name}.joblib')}
            print(f'{name}: held-out evaluation complete', file=sys.stderr, flush=True)
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True)
    report = {'report_version': 1, 'dataset_kind': 'harth-research', 'models_trained': True,
              'calibration_fitted': False, 'abstention_fitted': False,
              'input_hashes': data['hashes'], 'source_manifest_sha256': data['source_manifest_sha256'],
              'archive_sha256': data['archive_sha256'],
              'source_sha256': {name: sha(ROOT / name) for name in ('baseline.py', 'harth_input.py', 'requirements-ml.txt', 'docs/BASELINE.md')},
              'git_commit_at_run': git.stdout.strip(), 'feature_order': FEATURES,
              'selection': selection, 'models': results,
              'role_windows': {role: int(np.sum(data['roles'] == role)) for role in ('fit', 'calibration', 'threshold', 'test')},
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl')},
                              'numeric_thread_limit': 1},
              'completed_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'Frozen baseline comparison. No parameter search; calibration and threshold roles unused for fitting or scoring.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'REPORT.md').write_text(render(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.directory, args.output)
    except (OSError, ValueError, KeyError, csv.Error, ConvergenceWarning) as error:
        print(f'Baseline error: {error}', file=sys.stderr)
        return 2
    print('Baseline comparison completed; calibration and abstention remain unfitted.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

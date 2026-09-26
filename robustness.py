#!/usr/bin/env python3
"""Paired feature stress evaluation of frozen, locally generated HARTH models."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from baseline import ROOT, load_inputs, metrics
from harth_input import FEATURES, LABELS, sha
from selective import bootstrap, counts, decision_arrays, validate_probabilities

SCENARIOS = ('clean', 'missing_thigh_10', 'missing_thigh_30', 'missing_thigh_100',
             'zero_back', 'gain_half', 'gain_double', 'bias_back_x_small',
             'bias_back_x_large', 'invert_back_x', 'swap_sensors')
STATUSES = ('input_rejected', 'model_abstained', 'accepted')


def perturb(features, scenario):
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != 30 or not np.isfinite(x).all():
        raise ValueError('stress source must be a finite N by 30 array')
    out = x.copy()
    mask = np.zeros(len(x), dtype=bool)
    if scenario not in SCENARIOS:
        raise ValueError('unknown fixed scenario')
    if scenario.startswith('missing_thigh_'):
        fraction = int(scenario.rsplit('_', 1)[1]) / 100
        mask[np.random.default_rng(7).permutation(len(x))[:math.ceil(len(x) * fraction)]] = True
        out[mask, 15:] = np.nan
    elif scenario == 'zero_back':
        out[:, :15] = 0
    elif scenario.startswith('gain_'):
        out *= 0.5 if scenario == 'gain_half' else 2.0
    elif scenario.startswith('bias_back_x_'):
        bias = 0.25 if scenario.endswith('small') else 1.0
        squared_rms = x[:, 4] ** 2 + 2 * bias * x[:, 0] + bias ** 2
        if np.any(squared_rms < -1e-10):
            raise ValueError('inconsistent mean/RMS summaries')
        out[:, [0, 2, 3]] += bias
        out[:, 4] = np.sqrt(np.maximum(squared_rms, 0))
    elif scenario == 'invert_back_x':
        out[:, 0] = -x[:, 0]; out[:, 2] = -x[:, 3]; out[:, 3] = -x[:, 2]
    elif scenario == 'swap_sensors':
        out = np.concatenate([x[:, 15:], x[:, :15]], axis=1)
    changed = np.any(out != x, axis=1)
    metadata = {'scenario': scenario, 'changed_rows': int(changed.sum()),
                'changed_mask_sha256': hashlib.sha256(changed.astype('uint8').tobytes()).hexdigest(),
                'missing_rows': int(mask.sum()), 'seed': 7 if scenario.startswith('missing_') else None}
    return out, metadata


def infer(model, features, threshold):
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURES):
        raise ValueError('expected N by 30 features')
    valid = np.isfinite(x).all(axis=1)
    status = np.full(len(x), 'input_rejected', dtype='<U16')
    predicted = np.full(len(x), -1, dtype=int)
    confidence = np.full(len(x), np.nan)
    if valid.any():
        if list(model.classes_) != list(LABELS):
            raise ValueError('model class order mismatch')
        p = validate_probabilities(model.predict_proba(x[valid]))
        if len(p) != int(valid.sum()):
            raise ValueError('prediction row count mismatch')
        candidates, scores, accepted = decision_arrays(p, threshold)
        predicted[valid] = candidates; confidence[valid] = scores
        status[valid] = np.where(accepted, 'accepted', 'model_abstained')
    # Candidate labels are internal diagnostics, never asserted on rejected/abstained outputs.
    return {'status': status, 'predicted': predicted, 'confidence': confidence}


def summary(y, state):
    if state['status'].shape != (len(y),) or state['predicted'].shape != (len(y),):
        raise ValueError('decision rows do not match targets')
    if not np.isin(state['status'], STATUSES).all():
        raise ValueError('unknown decision status')
    accepted = state['status'] == 'accepted'
    valid = state['status'] != 'input_rejected'
    errors = state['predicted'] != y
    result = counts(len(y), accepted.sum(), (errors & accepted).sum())
    result.update({'input_rejected': int((~valid).sum()),
                   'model_abstained': int((state['status'] == 'model_abstained').sum()),
                   'valid_inputs': int(valid.sum()),
                   'coverage_among_valid': float(accepted.sum() / valid.sum()) if valid.any() else None,
                   'without_abstention': counts(valid.sum(), valid.sum(), (errors & valid).sum())})
    if result['input_rejected'] + result['model_abstained'] + result['accepted'] != result['total']:
        raise ValueError('decision counts do not partition the cohort')
    return result


def subset(state, mask):
    return {key: value[mask] for key, value in state.items()}


def compare(y, subjects, clean, stressed):
    result = summary(y, stressed)
    clean_result = summary(y, clean)
    result['per_subject'] = {s: summary(y[subjects == s], subset(stressed, subjects == s)) for s in sorted(set(subjects.tolist()))}
    result['per_true_class'] = {str(k): summary(y[y == k], subset(stressed, y == k)) for k in LABELS}
    result['participant_bootstrap'] = bootstrap(result['per_subject'])
    ca = clean['status'] == 'accepted'; sa = stressed['status'] == 'accepted'
    shared = ca & sa
    result['paired'] = {
        'status_transitions': {old: {new: int(((clean['status'] == old) & (stressed['status'] == new)).sum()) for new in STATUSES} for old in STATUSES},
        'clean_correct_accepted_to_stressed_wrong_accepted': int((ca & (clean['predicted'] == y) & sa & (stressed['predicted'] != y)).sum()),
        'shared_accepted_windows': int(shared.sum()),
        'clean_errors_on_shared': int(((clean['predicted'] != y) & shared).sum()),
        'stressed_errors_on_shared': int(((stressed['predicted'] != y) & shared).sum()),
        'coverage_delta': result['coverage'] - clean_result['coverage'],
        'risk_delta': result['selective_risk'] - clean_result['selective_risk'] if result['selective_risk'] is not None and clean_result['selective_risk'] is not None else None}
    return result


def load_frozen(model_directory, reference):
    # Validate every artifact before deserializing any of them.
    for name in ('raw', 'sigmoid'):
        if sha(model_directory / f'{name}.joblib') != reference['model_sha256'][name]:
            raise ValueError('model artifact hash mismatch')
    return {name: joblib.load(model_directory / f'{name}.joblib') for name in ('raw', 'sigmoid')}


def render(report):
    lines = ['# Frozen-model feature stress results', '', '**Exploratory, paired feature-level perturbations on the previously viewed test split.**', '',
             'Models and thresholds are unchanged. Missing inputs are rejected before prediction; this is not learned abstention.', '',
             '| Scenario | Variant | Input rejected | Model abstained | Answered | Coverage (all windows) | Errors / answers | Newly wrong accepted |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for scenario, case in report['scenarios'].items():
        for name, v in case['variants'].items():
            risk = 'undefined' if v['selective_risk'] is None else f'{v["selective_risk"]:.4f}'
            lines.append(f'| {scenario} | {name} | {v["input_rejected"]} | {v["model_abstained"]} | {v["accepted"]} | '
                         f'{v["coverage"]:.4f} | {v["errors"]}/{v["accepted"]} ({risk}) | {v["paired"]["clean_correct_accepted_to_stressed_wrong_accepted"]} |')
    lines += ['', 'Newly wrong accepted counts clean accepted/correct windows that become accepted/wrong under the perturbation.',
              'See report.json for participant/class counts, paired status transitions, shared-accepted error counts and cluster-bootstrap intervals.', '',
              'Finite-value checks cannot detect every plausible sensor fault. High confidence can persist when features change.',
              'These scenarios are not measured device failures or proof of clinical robustness. No model or threshold was tuned on these results.', '']
    return '\n'.join(lines)


def run(directory, model_directory, output):
    if output.exists():
        raise ValueError('output already exists; choose a new directory')
    data = load_inputs(directory)
    ref_path = ROOT / 'examples/selective-v1/report.json'
    reference = json.loads(ref_path.read_text())
    if data['hashes'] != reference['input_hashes'] or data['source_manifest_sha256'] != reference['source_manifest_sha256']:
        raise ValueError('input differs from frozen experiment')
    for name, expected in reference['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'frozen source changed: {name}')
    packages = {p: importlib.metadata.version(p) for p in reference['environment']['packages']}
    if packages != reference['environment']['packages']:
        raise ValueError('ML package versions differ from reference')
    variants = load_frozen(model_directory, reference)
    test = data['roles'] == 'test'
    x, y, subjects = data['X'][test], data['y'][test], data['subjects'][test]
    if not len(y):
        raise ValueError('test cohort is empty')
    policies = {name: reference['policies'][name]['threshold'] for name in variants}
    cases = {}
    with threadpool_limits(limits=1):
        clean = {name: infer(model, x, policies[name]) for name, model in variants.items()}
        for name, state in clean.items():
            observed = summary(y, state)
            if any(observed[k] != reference['test'][name]['selected'][k] for k in ('total', 'accepted', 'errors')):
                raise ValueError('clean operating point differs from reference')
            if metrics(y, state['predicted'])['confusion_matrix'] != reference['test'][name]['classification_without_abstention']['confusion_matrix']:
                raise ValueError('clean predictions differ from reference')
        for scenario in SCENARIOS:
            changed, meta = perturb(x, scenario)
            results = {name: compare(y, subjects, clean[name], infer(model, changed, policies[name])) for name, model in variants.items()}
            cases[scenario] = {'perturbation': meta, 'variants': results}
            print(f'{scenario}: complete', file=sys.stderr, flush=True)
    report = {'report_version': 1, 'dataset_kind': 'harth-research-with-feature-perturbations',
              'evaluation_status': 'exploratory_previously_viewed_test', 'models_refitted': False, 'thresholds_retuned': False,
              'reference_report_sha256': sha(ref_path), 'model_sha256': reference['model_sha256'],
              'input_hashes': data['hashes'], 'thresholds': policies, 'source_manifest_sha256': data['source_manifest_sha256'],
              'total_test_windows': len(y), 'scenarios': cases,
              'source_sha256': {name: sha(ROOT / name) for name in ('robustness.py', 'docs/ROBUSTNESS.md', 'baseline.py', 'selective.py')},
              'environment': {'python': sys.version, 'platform': platform.platform(), 'packages': packages, 'numeric_thread_limit': 1},
              'git_commit_at_run': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'REPORT.md').write_text(render(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.directory, args.models, args.output)
    except (ValueError, OSError, KeyError) as error:
        print(f'Robustness error: {error}', file=sys.stderr)
        return 2
    print('Frozen-model feature stress experiment completed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate a fixed exact-zero gate without retraining or policy selection."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from baseline import ROOT, load_inputs, metrics
from harth_input import LABELS, sha
from robustness import SCENARIOS, compare, infer, load_frozen, perturb, summary
from quality_gate import gated_infer, input_reasons

CASES = (*SCENARIOS, 'zero_thigh', 'zero_both', 'near_zero_back', 'constant_back_nonzero')


def change(x, case):
    if case in SCENARIOS:
        return perturb(x, case)[0]
    out = x.copy()
    if case == 'zero_thigh': out[:, 15:] = 0
    elif case == 'zero_both': out[:] = 0
    elif case == 'near_zero_back': out[:, :15] *= 1e-6
    elif case == 'constant_back_nonzero': out[:, :15] = [1, 0, 1, 1, 1] * 3
    else: raise ValueError('unknown scenario')
    return out


def rejection_counts(x):
    reasons = input_reasons(x, 'exact-zero-v1')
    n = len(x); rejected = int((reasons != '').sum())
    return {'total': n, 'rejected': rejected, 'fraction': rejected / n if n else None,
            'reasons': {reason: int((reasons == reason).sum()) for reason in
                        ('missing_feature', 'zero_back_sensor', 'zero_thigh_sensor', 'zero_both_sensors')}}


def render(report):
    lines = ['# Exact-zero gate: exploratory results', '',
             'Models and thresholds are fixed. Reference inputs are not independently adjudicated fault-free.', '',
             '## Unperturbed input rejections', '', '| Role | Rejected / windows | Standing, sitting, lying rejected / windows |',
             '| --- | ---: | ---: |']
    for role, v in report['unperturbed'].items():
        s = v['stationary_classes_6_7_8']
        lines.append(f'| {role} | {v["rejected"]}/{v["total"]} | {s["rejected"]}/{s["total"]} |')
    lines += ['', '## Paired test scenarios', '',
              '| Scenario | Variant | Input rejected (new gate) | Answers (finite → zero gate) | Errors (finite → zero gate) | New risk |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for case, item in report['scenarios'].items():
        for name, result in item['variants'].items():
            old, new = result['finite'], result['exact_zero']
            risk = 'undefined' if new['selective_risk'] is None else f'{new["selective_risk"]:.4f}'
            lines.append(f'| {case} | {name} | {new["input_rejected"]} | {old["accepted"]} → {new["accepted"]} | {old["errors"]} → {new["errors"]} | {risk} |')
    lines += ['', 'Exact zero is deliberately narrow. Near-zero, nonzero flatline, gain, bias and placement faults can remain undetected.',
              'No answers means no prediction utility and undefined risk. This is not a general fault detector or clinical safety evidence.', '']
    return '\n'.join(lines)


def run(directory, models, output):
    if output.exists(): raise ValueError('output exists')
    data = load_inputs(directory)
    ref_path = ROOT / 'examples/selective-v1/report.json'
    ref = json.loads(ref_path.read_text())
    if data['hashes'] != ref['input_hashes'] or data['source_manifest_sha256'] != ref['source_manifest_sha256']:
        raise ValueError('input provenance mismatch')
    for name, expected in ref['source_sha256'].items():
        if sha(ROOT / name) != expected: raise ValueError('frozen source mismatch')
    packages = {p: importlib.metadata.version(p) for p in ref['environment']['packages']}
    if packages != ref['environment']['packages']: raise ValueError('dependency mismatch')
    loaded = load_frozen(models, ref)
    unperturbed = {}
    for role in ('fit', 'calibration', 'threshold', 'test'):
        mask = data['roles'] == role
        x, y, people = data['X'][mask], data['y'][mask], data['subjects'][mask]
        unperturbed[role] = {**rejection_counts(x),
            'stationary_classes_6_7_8': rejection_counts(x[np.isin(y, [6, 7, 8])]),
            'per_class': {str(k): rejection_counts(x[y == k]) for k in LABELS},
            'per_subject': {s: rejection_counts(x[people == s]) for s in sorted(set(people.tolist()))}}
    test = data['roles'] == 'test'
    x, y, people = data['X'][test], data['y'][test], data['subjects'][test]
    cases = {}
    with threadpool_limits(limits=1):
        for name, model in loaded.items():
            clean = infer(model, x, ref['policies'][name]['threshold'])
            observed = summary(y, clean)
            if any(observed[k] != ref['test'][name]['selected'][k] for k in ('total', 'accepted', 'errors')):
                raise ValueError('clean counts differ')
            if metrics(y, clean['predicted'])['confusion_matrix'] != ref['test'][name]['classification_without_abstention']['confusion_matrix']:
                raise ValueError('clean predictions differ')
        for case in CASES:
            changed = change(x, case)
            affected = np.any(changed != x, axis=1)
            reasons = input_reasons(changed, 'exact-zero-v1')
            rejected_changed = int(((reasons != '') & affected).sum())
            count = int(affected.sum())
            variants = {}
            for name, model in loaded.items():
                threshold = ref['policies'][name]['threshold']
                finite = infer(model, changed, threshold)
                gated, _ = gated_infer(model, changed, threshold, 'exact-zero-v1')
                variants[name] = {'finite': summary(y, finite), 'exact_zero': compare(y, people, finite, gated)}
            cases[case] = {'changed_windows': count, 'rejected_changed_windows': rejected_changed,
                           'detection_fraction_on_changed': rejected_changed / count if count else None,
                           'variants': variants}
            print(f'{case}: complete', file=sys.stderr, flush=True)
    report = {'report_version': 1, 'evaluation_status': 'exploratory_previously_viewed_data',
              'rule': 'exact-zero-v1', 'models_refitted': False, 'thresholds_retuned': False,
              'reference_sha256': sha(ref_path), 'model_sha256': ref['model_sha256'],
              'thresholds': {k: v['threshold'] for k, v in ref['policies'].items()},
              'input_hashes': data['hashes'], 'source_manifest_sha256': data['source_manifest_sha256'],
              'source_sha256': {n: sha(ROOT / n) for n in ('quality_gate.py', 'evaluate_gate.py', 'docs/ZERO_GATE.md', 'robustness.py')},
              'packages': packages, 'python': sys.version, 'numeric_thread_limit': 1,
              'git_commit_at_run': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
              'unperturbed': unperturbed, 'scenarios': cases}
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
    try: run(args.directory, args.models, args.output)
    except (ValueError, OSError, KeyError) as error:
        print(f'Gate evaluation failed: {error}', file=sys.stderr)
        return 2
    print('Gate evaluation completed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

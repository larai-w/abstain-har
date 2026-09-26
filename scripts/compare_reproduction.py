#!/usr/bin/env python3
"""Compare regenerated aggregate evidence; never loosen artifact loading guards."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATOL = 1e-12
VOLATILE = {'completed_at', 'git_commit_at_run', 'fit_seconds', 'test_batch_predict_seconds'}
FILES = ('harth-input-v1/report.json', 'baseline-v1/report.json', 'selective-v1/report.json',
         'robustness-v1/report.json', 'zero-gate-v1/report.json', 'zero-gate-v1/local-response.json')


def compare(expected, observed):
    differences, excluded, artifact_hashes = [], [], []
    compared = 0
    max_error = 0.0

    def visit(a, b, path):
        nonlocal compared, max_error
        if type(a) is not type(b):
            differences.append(path + ': type mismatch')
        elif isinstance(a, dict):
            if a.keys() != b.keys(): differences.append(path + ': key mismatch')
            for key in sorted(a.keys() & b.keys()):
                location = path + '/' + key
                if key in VOLATILE or location in ('$/python', '$/environment/python', '$/environment/platform'):
                    excluded.append(location)
                elif key == 'model_sha256':
                    artifact_hashes.append({'path': location, 'equal': a[key] == b[key]})
                else:
                    visit(a[key], b[key], location)
        elif isinstance(a, list):
            if len(a) != len(b): differences.append(path + ': length mismatch')
            for i, (left, right) in enumerate(zip(a, b)):
                visit(left, right, path + '/' + str(i))
        else:
            compared += 1
            if isinstance(a, float):
                if not math.isfinite(a) or not math.isfinite(b):
                    differences.append(path + ': non-finite value')
                else:
                    delta = abs(a - b)
                    max_error = max(max_error, delta)
                    if delta > ATOL: differences.append(path + ': numeric mismatch')
            elif a != b:
                differences.append(path + ': value mismatch')
    visit(expected, observed, '$')
    return {'semantic_match': not differences, 'compared_leaves': compared,
            'absolute_tolerance': ATOL, 'relative_tolerance': 0,
            'max_absolute_numeric_difference': max_error,
            'difference_count': len(differences), 'differences_first_50': differences[:50],
            'excluded_paths': excluded, 'model_artifact_hashes': artifact_hashes,
            'model_artifacts_match': all(item['equal'] for item in artifact_hashes)}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(artifacts, output):
    if output.exists(): raise ValueError('output exists')
    results = {}
    for name in FILES:
        reference, regenerated = ROOT / 'examples' / name, artifacts / name
        result = compare(json.loads(reference.read_text()), json.loads(regenerated.read_text()))
        results[name] = {**result, 'reference_sha256': digest(reference), 'regenerated_sha256': digest(regenerated)}
    report = {'report_version': 1, 'all_semantic_matches': all(v['semantic_match'] for v in results.values()),
              'all_model_artifact_hashes_match': all(v['model_artifacts_match'] for v in results.values()),
              'comparisons': results, 'comparator_sha256': digest(Path(__file__))}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream: stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.artifacts, args.output)
    except (ValueError, OSError) as error:
        print(f'Comparison failed: {type(error).__name__}')
        return 2
    return 0 if result['all_semantic_matches'] and result['all_model_artifact_hashes_match'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

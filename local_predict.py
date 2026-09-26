#!/usr/bin/env python3
"""Versioned local batch inference with explicit rejection and abstention."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from baseline import ROOT
from harth_input import FEATURES, LABELS, sha
from quality_gate import POLICIES, gated_infer

CONTRACT = 'harth-summary-v1'
MAX_ROWS = 1000
MAX_BYTES = 2_000_000
REFERENCE = ROOT / 'examples/selective-v1/report.json'


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError('non-standard JSON number')


def read_request(path):
    with path.open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('request exceeds byte limit')
    request = json.loads(raw, object_pairs_hook=no_duplicates, parse_constant=invalid_constant)
    return request, hashlib.sha256(raw).hexdigest()


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {'contract', 'dataset_kind', 'features', 'samples'}:
        raise ValueError('unexpected request fields')
    if request['contract'] != CONTRACT or request['features'] != FEATURES:
        raise ValueError('feature contract or column order mismatch')
    if request['dataset_kind'] not in ('synthetic', 'research'):
        raise ValueError('dataset_kind must be synthetic or research')
    rows = request['samples']
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError('samples must contain 1 to 1000 rows')
    ids, values = [], []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'id', 'values'}:
            raise ValueError('unexpected sample fields')
        ident = row['id']
        if not isinstance(ident, str) or not ident or len(ident) > 80 or not ident.isascii() or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in ident):
            raise ValueError('id must be a short opaque ASCII identifier')
        if ident in ids:
            raise ValueError('duplicate sample id')
        vector = row['values']
        if not isinstance(vector, list) or len(vector) != len(FEATURES):
            raise ValueError('each sample requires 30 feature values')
        converted = []
        for value in vector:
            if value is None:
                converted.append(np.nan)
            elif type(value) not in (int, float):
                raise ValueError('feature values must be numbers or null, without coercion')
            else:
                try:
                    number = float(value)
                except OverflowError as error:
                    raise ValueError('feature number is out of range') from error
                if not math.isfinite(number):
                    raise ValueError('use null for missing input, not non-finite numbers')
                converted.append(number)
        ids.append(ident)
        values.append(converted)
    return ids, np.asarray(values, dtype=float)


def load_model(directory, variant):
    if variant not in ('raw', 'sigmoid'):
        raise ValueError('unknown model variant')
    reference = json.loads(REFERENCE.read_text())
    packages = {p: importlib.metadata.version(p) for p in reference['environment']['packages']}
    if packages != reference['environment']['packages']:
        raise ValueError('ML package versions differ from recorded model')
    model_path = directory / f'{variant}.joblib'
    expected = reference['model_sha256'][variant]
    if sha(model_path) != expected:
        raise ValueError('model artifact hash mismatch')
    model = joblib.load(model_path)
    if list(model.classes_) != list(LABELS) or model.n_features_in_ != len(FEATURES):
        raise ValueError('model schema mismatch')
    threshold = reference['policies'][variant]['threshold']
    if threshold is not None and (type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('invalid recorded threshold')
    provenance = {'variant': variant, 'model_sha256': expected,
                  'reference_sha256': sha(REFERENCE), 'threshold': threshold,
                  'packages': packages, 'feature_contract': CONTRACT}
    return model, threshold, provenance


def predict(request, model, threshold, input_policy='finite-v1'):
    ids, features = validate_request(request)
    with threadpool_limits(limits=1):
        state, reasons = gated_infer(model, features, threshold, input_policy)
    decisions = []
    for i, ident in enumerate(ids):
        status = str(state['status'][i])
        decisions.append({'id': ident, 'status': status,
                          'label': int(state['predicted'][i]) if status == 'accepted' else None,
                          'confidence': None if status == 'input_rejected' else float(state['confidence'][i]),
                          'reason': str(reasons[i]) if status == 'input_rejected' else
                                    {'accepted': 'threshold_met', 'model_abstained': 'policy_abstention'}[status]})
    return {'response_version': 2, 'contract': CONTRACT, 'dataset_kind': request['dataset_kind'],
            'input_policy': input_policy,
            'decisions': decisions,
            'counts': {status: sum(d['status'] == status for d in decisions)
                       for status in ('input_rejected', 'model_abstained', 'accepted')}}


def run(request_path, model_directory, variant, output, input_policy='finite-v1'):
    if output.exists():
        raise ValueError('output exists; choose a new path')
    request, request_hash = read_request(request_path)
    validate_request(request)  # Reject malformed batches before deserializing a model.
    model, threshold, provenance = load_model(model_directory, variant)
    result = predict(request, model, threshold, input_policy)
    result['provenance'] = {**provenance, 'request_sha256': request_hash,
                            'source_sha256': {name: sha(ROOT / name) for name in
                                              ('local_predict.py', 'quality_gate.py', 'robustness.py', 'selective.py', 'harth_input.py')}}
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(payload)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--variant', choices=('raw', 'sigmoid'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-policy', choices=POLICIES, default='finite-v1')
    args = parser.parse_args()
    try:
        run(args.request, args.models, args.variant, args.output, args.input_policy)
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as error:
        # No input payload or local identifiers are echoed in diagnostics.
        print(f'Local inference failed ({type(error).__name__}); check the request, model and output path.', file=sys.stderr)
        return 2
    print('Local inference completed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

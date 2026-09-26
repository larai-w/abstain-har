#!/usr/bin/env python3
"""Validate a synthetic care bundle and prepare subject-disjoint input batches."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / 'contracts/care-bundle-v1.json'
CONTRACT = json.loads(CONTRACT_PATH.read_text())


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def instant(value: object) -> datetime:
    if not isinstance(value, str) or 'T' not in value:
        raise ValueError('timestamps must be ISO 8601 strings with offsets')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            raise ValueError('timestamp offset required')
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        raise ValueError('invalid timestamp or UTC range') from None


def identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def check_row(row: dict) -> None:
    if not isinstance(row, dict) or not all(identifier(row.get(key)) for key in ('sample_id', 'subject_id')):
        raise ValueError('each row requires sample_id and subject_id')
    cutoff = instant(row.get('as_of'))
    audit = row.get('audit')
    if not isinstance(audit, dict) or not isinstance(audit.get('selected_versions'), list):
        raise ValueError('selected-version audit is required')
    versions = audit['selected_versions']
    ids, counts = set(), Counter()
    for event in versions:
        if not isinstance(event, dict) or not identifier(event.get('event_id')):
            raise ValueError('invalid selected event')
        event_id = event['event_id']
        if event_id in ids:
            raise ValueError('duplicate selected event_id')
        ids.add(event_id)
        if not instant(event.get('observed_at')) <= instant(event.get('received_at')) <= cutoff:
            raise ValueError('selected version is temporally unavailable')
        status = event.get('status')
        if not isinstance(status, str) or status not in ('observed', 'not_occurring', 'not_recorded'):
            raise ValueError('invalid audited status')
        counts[status] += 1
    lineage = audit.get('lineage')
    if not isinstance(lineage, list):
        raise ValueError('lineage must be an array')
    seen, selected = set(), set()
    for item in lineage:
        if not isinstance(item, dict):
            raise ValueError('invalid lineage')
        chain = item.get('chain')
        if not isinstance(chain, list) or not chain or not all(identifier(v) for v in chain):
            raise ValueError('invalid lineage chain')
        if len(set(chain)) != len(chain) or seen.intersection(chain):
            raise ValueError('lineage repeats a version')
        if item.get('root_event_id') != chain[0] or item.get('selected_event_id') != chain[-1]:
            raise ValueError('lineage endpoints do not match')
        seen.update(chain)
        selected.add(chain[-1])
    if selected != ids:
        raise ValueError('lineage does not cover selected versions')
    issues = row.get('quality_issues')
    if not isinstance(issues, list):
        raise ValueError('quality_issues must be an array')
    for issue in issues:
        if not isinstance(issue, dict) or not isinstance(issue.get('event_id'), str) or issue['event_id'] not in ids:
            raise ValueError('quality issue refers to an unknown selected event')
        if issue.get('rule') not in CONTRACT['quality_rules']:
            raise ValueError('unsupported quality rule')
    known = counts['observed'] + counts['not_occurring']
    expected_decision = 'quality_rejected' if issues else 'insufficient_data' if not known else 'eligible'
    if row.get('decision') != expected_decision:
        raise ValueError('decision is inconsistent with quality findings or available data')
    features = row.get('features')
    if expected_decision != 'eligible':
        if features is not None:
            raise ValueError('excluded samples must have null features')
        return
    if not isinstance(features, dict) or set(features) != set(CONTRACT['feature_names']):
        raise ValueError('feature schema mismatch')
    for key in ('known_count', 'unknown_count'):
        if type(features[key]) is not int or features[key] < 0:
            raise ValueError('feature counts must be non-negative integers')
    fraction = features['observed_fraction']
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('observed_fraction must be finite and between zero and one')
    if features['known_count'] != known or features['unknown_count'] != counts['not_recorded'] or abs(fraction - counts['observed'] / known) > 1e-12:
        raise ValueError('features do not match the audited statuses')


def prepare(bundle: dict, manifest: dict) -> dict:
    if not isinstance(bundle, dict) or type(bundle.get('contract_version')) is not int:
        raise ValueError('invalid contract version')
    for key in ('contract_version', 'dataset_kind', 'feature_set', 'feature_names'):
        if bundle.get(key) != CONTRACT[key]:
            raise ValueError(f'unsupported {key}')
    if bundle.get('contract_sha256') != hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest():
        raise ValueError('contract hash mismatch')
    rows = bundle.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('rows must be non-empty')
    seen, subjects = set(), set()
    for row in rows:
        check_row(row)
        if row['sample_id'] in seen:
            raise ValueError('duplicate sample_id')
        seen.add(row['sample_id'])
        subjects.add(row['subject_id'])
    if not isinstance(manifest, dict) or type(manifest.get('split_version')) is not int or manifest['split_version'] != 1:
        raise ValueError('unsupported split_version')
    groups = manifest.get('subjects')
    if not isinstance(groups, dict) or set(groups) != set(CONTRACT['split_roles']):
        raise ValueError('declare exactly fit, calibration, threshold and test subject groups')
    assignments = {}
    for role in CONTRACT['split_roles']:
        group = groups[role]
        if not isinstance(group, list) or not group or not all(identifier(s) for s in group):
            raise ValueError('each split needs non-empty subject identifiers')
        for subject in group:
            if subject in assignments:
                raise ValueError('subject overlap or duplicate in split manifest')
            assignments[subject] = role
    if set(assignments) != subjects:
        raise ValueError('split manifest must cover exactly the subjects in the bundle')
    batches = {role: [] for role in CONTRACT['split_roles']}
    excluded = []
    for row in sorted(rows, key=lambda r: r['sample_id']):
        role = assignments[row['subject_id']]
        if row['decision'] == 'eligible':
            batches[role].append({'sample_id': row['sample_id'], 'subject_id': row['subject_id'],
                                  'as_of': row['as_of'],
                                  'features': [row['features'][name] for name in CONTRACT['feature_names']]})
        else:
            excluded.append({'sample_id': row['sample_id'], 'subject_id': row['subject_id'],
                             'split': role, 'reason': row['decision']})
    return {'status': 'synthetic_contract_verified', 'feature_set': CONTRACT['feature_set'],
            'feature_names': CONTRACT['feature_names'], 'trained_model': False,
            'batches': batches, 'excluded': excluded,
            'summary': {'total': len(rows), 'eligible': sum(map(len, batches.values())),
                        'excluded': len(excluded), 'subjects': len(subjects),
                        'eligible_by_split': {role: len(batch) for role, batch in batches.items()}},
            'audit': {'bundle_sha256': digest(bundle), 'split_sha256': digest(manifest),
                      'contract_sha256': bundle['contract_sha256']}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new file')
    try:
        result = prepare(json.loads(args.bundle.read_text()), json.loads(args.splits.read_text()))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as output:
            output.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    except (OSError, ValueError) as error:
        print(f'Preparation error: {error}', file=sys.stderr)
        return 2
    print(f'Prepared {result["summary"]["eligible"]} synthetic rows; excluded {result["summary"]["excluded"]}. No model was trained.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

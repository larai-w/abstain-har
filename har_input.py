#!/usr/bin/env python3
"""Validate the UCI HAR release text layout and plan participant-disjoint roles."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import sys

ACTIVITIES = {1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
              4: 'SITTING', 5: 'STANDING', 6: 'LAYING'}
FEATURE_COUNT = 561
FILES = ['features.txt', 'activity_labels.txt'] + [f'{part}/{name}_{part}.txt'
        for part in ('train', 'test') for name in ('X', 'y', 'subject')]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def feature_definitions(path: Path) -> list[dict]:
    result = []
    for index, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] != str(index) or not fields[1].strip():
            raise ValueError(f'features.txt: expected position {index} and a name')
        result.append({'position': index, 'column_id': f'f{index:03d}', 'source_name': fields[1]})
    if len(result) != FEATURE_COUNT:
        raise ValueError('features.txt must declare exactly 561 ordered positions')
    # Source names in the real release are not unique; never key vectors by name.
    return result


def activity_definitions(path: Path) -> dict:
    result = {}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise ValueError('invalid activity_labels.txt row')
        label = int(fields[0])
        if label in result:
            raise ValueError('duplicate activity label')
        result[label] = fields[1]
    if result != ACTIVITIES:
        raise ValueError('activity_labels.txt differs from the six-class UCI HAR contract')
    return result


def read_partition(root: Path, part: str) -> list[dict]:
    rows = []
    paths = [root / part / f'{name}_{part}.txt' for name in ('X', 'y', 'subject')]
    with paths[0].open(encoding='utf-8-sig') as x, paths[1].open(encoding='utf-8-sig') as y, paths[2].open(encoding='utf-8-sig') as subjects:
        for index, lines in enumerate(zip_longest(x, y, subjects), 1):
            if any(line is None for line in lines):
                raise ValueError(f'{part}: feature/label/subject row counts differ at row {index}')
            values = lines[0].split()
            if len(values) != FEATURE_COUNT:
                raise ValueError(f'{part} row {index}: expected 561 feature values')
            features = [float(value) for value in values]
            if any(not math.isfinite(value) or abs(value) > 1 + 1e-6 for value in features):
                raise ValueError(f'{part} row {index}: features must be finite and within [-1,1] (tolerance 1e-6)')
            label_tokens, subject_tokens = lines[1].split(), lines[2].split()
            if len(label_tokens) != 1 or len(subject_tokens) != 1:
                raise ValueError(f'{part} row {index}: expected one label and one subject ID')
            label, subject = int(label_tokens[0]), int(subject_tokens[0])
            if label not in ACTIVITIES or not 1 <= subject <= 30:
                raise ValueError(f'{part} row {index}: label or subject ID outside the release vocabulary')
            rows.append({'row_id': f'{part}:{index}', 'subject_id': subject, 'label': label, 'features': features})
    if not rows:
        raise ValueError(f'{part}: empty partition')
    return rows


def load_dataset(root: Path) -> dict:
    # Detect changes during loading rather than attach hashes to different bytes.
    before = {name: sha(root / name) for name in FILES}
    features = feature_definitions(root / 'features.txt')
    activities = activity_definitions(root / 'activity_labels.txt')
    train, test = read_partition(root, 'train'), read_partition(root, 'test')
    train_subjects = {row['subject_id'] for row in train}
    test_subjects = {row['subject_id'] for row in test}
    if train_subjects & test_subjects:
        raise ValueError('official train/test subject overlap')
    if before != {name: sha(root / name) for name in FILES}:
        raise ValueError('input files changed during loading')
    return {'feature_definitions': features, 'activities': activities,
            'train': train, 'test': test, 'source_sha256': before}


def plan_splits(dataset: dict, seed: int, calibration_subjects: int, threshold_subjects: int) -> dict:
    if type(seed) is not int or type(calibration_subjects) is not int or type(threshold_subjects) is not int:
        raise ValueError('seed and subject counts must be integers')
    if calibration_subjects < 1 or threshold_subjects < 1:
        raise ValueError('calibration and threshold each need at least one subject')
    train_ids = sorted({row['subject_id'] for row in dataset['train']})
    test_ids = sorted({row['subject_id'] for row in dataset['test']})
    if set(train_ids) & set(test_ids):
        raise ValueError('official train/test subject overlap')
    if len(train_ids) - calibration_subjects - threshold_subjects < 2:
        raise ValueError('reserve at least two fit subjects for later grouped model selection')
    ordered = sorted(train_ids, key=lambda subject: hashlib.sha256(f'{seed}:{subject}'.encode()).hexdigest())
    c, t = calibration_subjects, threshold_subjects
    roles = {'fit': sorted(ordered[c + t:]), 'calibration': sorted(ordered[:c]),
             'threshold': sorted(ordered[c:c + t]), 'test': test_ids}
    result = {}
    for role, subjects in roles.items():
        partition = 'test' if role == 'test' else 'train'
        rows = [row for row in dataset[partition] if row['subject_id'] in subjects]
        counts = Counter(row['label'] for row in rows)
        result[role] = {'subjects': subjects, 'source_partition': partition,
                        'row_ids': [row['row_id'] for row in rows], 'rows': len(rows),
                        'label_counts': {str(label): counts[label] for label in ACTIVITIES},
                        'missing_classes': [label for label in ACTIVITIES if not counts[label]]}
    return {'algorithm': 'sha256-subject-order-v1', 'seed': seed,
            'selection_uses_labels': False, 'roles': result}


def build_report(root: Path, dataset_kind: str, seed: int = 7,
                 calibration_subjects: int = 3, threshold_subjects: int = 3) -> dict:
    if dataset_kind not in ('synthetic', 'uci-har-research'):
        raise ValueError('explicit dataset provenance is required')
    synthetic_marker = (root / 'SYNTHETIC.txt').is_file()
    if (dataset_kind == 'synthetic') != synthetic_marker:
        raise ValueError('dataset kind conflicts with SYNTHETIC.txt marker; do not relabel research recordings')
    dataset = load_dataset(root)
    features = dataset['feature_definitions']
    return {'input_contract_version': 1, 'dataset_kind': dataset_kind,
            'representation': 'uci_har_561_release_layout', 'models_trained': False,
            'source_sha256': dataset['source_sha256'], 'adapter_sha256': sha(Path(__file__)),
            'feature_definitions': features,
            'feature_positions': len(features), 'unique_source_names': len({f['source_name'] for f in features}),
            'partitions': {part: {'rows': len(dataset[part]),
                                 'subjects': sorted({row['subject_id'] for row in dataset[part]})}
                           for part in ('train', 'test')},
            'split_manifest': plan_splits(dataset, seed, calibration_subjects, threshold_subjects),
            'notes': ['Row IDs use the original partition and one-based line number.',
                      'No scaling, imputation, feature selection or model fitting was performed.',
                      'Provenance is declared by the caller; hashes do not authenticate dataset origin.',
                      'Class counts are diagnostics, not inputs to subject assignment.',
                      'The supplied text does not provide arrival timestamps; none were invented.']}


def render(report: dict) -> str:
    lines = ['# HAR input and split inspection', '', f"Dataset kind: **{report['dataset_kind']}**.",
             f"Feature positions: {report['feature_positions']}; unique source names: {report['unique_source_names']}.", '',
             '| Role | Source | Subjects | Rows | Missing classes |', '| --- | --- | --- | ---: | --- |']
    for role, value in report['split_manifest']['roles'].items():
        lines.append(f"| {role} | {value['source_partition']} | {', '.join(map(str, value['subjects']))} | "
                     f"{value['rows']} | {value['missing_classes'] or 'none'} |")
    lines += ['', 'The original test subjects remain exclusively in test. Internal roles use only original training subjects.',
              'This is an input-contract and split-planning result. No model performance has been measured.', '',
              'Features are addressed by position (`f001` through `f561`), retaining repeated source names as metadata.',
              'Missing classes are reported without changing the seed or silently moving subjects.', '',
              'See the accompanying report.json for row assignments, class counts, file hashes, and adapter hash.', '']
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--dataset-kind', choices=('synthetic', 'uci-har-research'), required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--calibration-subjects', type=int, default=3)
    parser.add_argument('--threshold-subjects', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new directory')
    try:
        report = build_report(args.directory, args.dataset_kind, args.seed,
                              args.calibration_subjects, args.threshold_subjects)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        (args.output / 'REPORT.md').write_text(render(report))
    except (OSError, ValueError) as error:
        print(f'HAR input error: {error}', file=sys.stderr)
        return 2
    print(f'Validated {report["partitions"]["train"]["rows"]} train and {report["partitions"]["test"]["rows"]} test rows; no model trained.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

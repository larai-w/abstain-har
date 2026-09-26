#!/usr/bin/env python3
"""Prepare local HARTH windows from the pinned UCI archive without fitting a model."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import zipfile

CHANNELS = ('back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z')
STATS = ('mean', 'std', 'min', 'max', 'rms')
FEATURES = [f'{channel}_{stat}' for channel in CHANNELS for stat in STATS]
LABELS = (1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140)
WINDOW = 250
MANIFEST = Path(__file__).parent / 'contracts/harth-v1.2-source.json'


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def describe_window(samples: list) -> list[float]:
    result = []
    for axis in range(len(CHANNELS)):
        values = [sample[0][axis] for sample in samples]
        mean = math.fsum(values) / WINDOW
        result.extend((mean, math.sqrt(math.fsum((x - mean) ** 2 for x in values) / WINDOW),
                       min(values), max(values), math.sqrt(math.fsum(x * x for x in values) / WINDOW)))
    if not all(math.isfinite(x) for x in result):
        raise ValueError('non-finite derived features')
    return result


def inspect_subject(stream, header: list[str], subject: str, role: str, writer) -> dict:
    reader = csv.reader(stream, strict=True)
    if next(reader, None) != header:
        raise ValueError(f'{subject}: header differs from pinned manifest')
    indices = [header.index(name) for name in CHANNELS]
    ti, li = header.index('timestamp'), header.index('label')
    counts = Counter()
    label_counts = Counter()
    samples = []
    previous = None
    window_start = None
    mixed = 0
    purity_sum = 0.0
    for line, row in enumerate(reader, 2):
        counts['rows'] += 1
        if len(row) != len(header):
            raise ValueError(f'{subject} line {line}: unexpected column count')
        try:
            timestamp = datetime.fromisoformat(row[ti])
            label = int(row[li])
        except ValueError as error:
            raise ValueError(f'{subject} line {line}: invalid timestamp or label') from error
        if timestamp.tzinfo is not None or label not in LABELS:
            raise ValueError(f'{subject} line {line}: unexpected timezone or label')
        if previous is not None:
            delta = (timestamp - previous).total_seconds()
            if delta <= 0 or delta > 0.030:
                counts['nonpositive_time_breaks' if delta <= 0 else 'gap_breaks'] += 1
                counts['incomplete_samples'] += len(samples)
                samples = []
        previous = timestamp
        try:
            values = [float(row[i]) for i in indices]
            valid = all(math.isfinite(v) for v in values)
        except ValueError:
            values, valid = [], False
        if not valid:
            counts['invalid_sensor_rows'] += 1
        if not samples:
            window_start = line
        samples.append((values, label, valid))
        if len(samples) < WINDOW:
            continue
        counts['complete_windows'] += 1
        if not all(sample[2] for sample in samples):
            counts['rejected_windows'] += 1
        else:
            votes = Counter(sample[1] for sample in samples)
            target = min(votes, key=lambda k: (-votes[k], k))
            purity = votes[target] / WINDOW
            mixed += len(votes) > 1
            purity_sum += purity
            vector = describe_window(samples)
            writer.writerow([f'{subject}:{window_start}', subject, role, target, purity, *vector])
            counts['retained_windows'] += 1
            label_counts[target] += 1
        samples = []
    counts['incomplete_samples'] += len(samples)
    if not counts['rows']:
        raise ValueError(f'{subject}: empty recording')
    retained = counts['retained_windows']
    return {'subject_id': subject, 'role': role,
            **{key: counts[key] for key in ('rows', 'complete_windows', 'retained_windows',
               'rejected_windows', 'invalid_sensor_rows', 'incomplete_samples',
               'gap_breaks', 'nonpositive_time_breaks')},
            'retained_nominal_seconds': retained * WINDOW / 50,
            'mixed_label_windows': mixed,
            'mean_label_agreement': purity_sum / retained if retained else None,
            'label_counts': {str(k): label_counts[k] for k in LABELS}}


def prepare(archive: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError('output already exists; choose a new directory')
    manifest = json.loads(MANIFEST.read_text())
    original_hash = sha(archive)
    if original_hash != manifest['archive_sha256'] or archive.stat().st_size != manifest['archive_bytes']:
        raise ValueError('archive differs from pinned HARTH distribution')
    assignments = {subject: role for role, ids in manifest['split_plan']['roles'].items() for subject in ids}
    subjects = [Path(member['member']).stem for member in manifest['members']]
    flat = [s for ids in manifest['split_plan']['roles'].values() for s in ids]
    if len(flat) != len(set(flat)) or set(flat) != set(subjects):
        raise ValueError('manifest subject assignment is incomplete or overlapping')
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    with tempfile.TemporaryDirectory(prefix='.harth-', dir=output.parent) as temp:
        staging = Path(temp)
        with zipfile.ZipFile(archive) as z, (staging / 'windows.csv').open('w', newline='', encoding='utf-8') as target:
            files = [i.filename for i in z.infolist() if not i.is_dir()]
            if len(files) != len(set(files)) or set(files) != {m['member'] for m in manifest['members']}:
                raise ValueError('archive member inventory differs from manifest')
            writer = csv.writer(target)
            writer.writerow(['window_id', 'subject_id', 'role', 'label', 'label_agreement', *FEATURES])
            for member in manifest['members']:
                subject = Path(member['member']).stem
                with z.open(member['member']) as raw, io.TextIOWrapper(raw, encoding='utf-8-sig', newline='') as stream:
                    report = inspect_subject(stream, member['header'], subject, assignments[subject], writer)
                reports.append(report)
                print(f'{subject}: {report["retained_windows"]} windows retained', file=sys.stderr, flush=True)
        if sha(archive) != original_hash:
            raise ValueError('archive changed during processing')
        roles = {}
        for role, ids in manifest['split_plan']['roles'].items():
            group = [r for r in reports if r['role'] == role]
            classes = {str(k): sum(r['label_counts'][str(k)] for r in group) for k in LABELS}
            roles[role] = {'subjects': ids, 'retained_windows': sum(r['retained_windows'] for r in group),
                           'label_counts': classes, 'missing_classes': [int(k) for k, v in classes.items() if not v]}
        report = {'report_version': 1, 'dataset_kind': 'harth-research', 'models_trained': False,
                  'archive_sha256': original_hash, 'source_manifest_sha256': sha(MANIFEST),
                  'adapter_sha256': sha(Path(__file__)), 'windows_sha256': sha(staging / 'windows.csv'),
                  'window_samples': WINDOW, 'nominal_hz': 50, 'overlap_samples': 0,
                  'gap_threshold_seconds': 0.030, 'feature_order': FEATURES,
                  'model_feature_columns': FEATURES,
                  'excluded_from_model_features': ['window_id', 'subject_id', 'role', 'label', 'label_agreement'],
                  'subjects': reports, 'roles': roles,
                  'total_rows': sum(r['rows'] for r in reports),
                  'retained_windows': sum(r['retained_windows'] for r in reports),
                  'notes': ['No model, scaler, imputer, calibration or threshold was fitted.',
                            'Label agreement is evaluation metadata, never a feature or quality gate.',
                            'Durations are nominal sample durations, not wall-clock elapsed times.',
                            'All roles were inspected for input diagnostics; these are not model results.']}
        (staging / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        lines = ['# HARTH input preparation', '', 'Research recordings; no model trained.', '',
                 f'Input rows: {report["total_rows"]}. Retained windows: {report["retained_windows"]}.', '',
                 '| Role | Participants | Retained windows | Missing classes |', '| --- | ---: | ---: | --- |']
        for role, result in roles.items():
            lines.append(f'| {role} | {len(result["subjects"])} | {result["retained_windows"]} | {result["missing_classes"] or "none"} |')
        lines += ['', '250 samples per window, no overlap, 30 explicitly named features.',
                  'See report.json for subject-level retention, timestamp breaks, mixed labels and hashes.',
                  'windows.csv is a local derived dataset and must remain outside version control.', '']
        (staging / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
        output.mkdir(exist_ok=False)
        for name in ('windows.csv', 'report.json', 'REPORT.md'):
            (staging / name).replace(output / name)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        report = prepare(args.archive, args.output)
    except (OSError, ValueError, csv.Error, zipfile.BadZipFile, OverflowError) as error:
        print(f'HARTH input error: {error}', file=sys.stderr)
        return 2
    print(f'Prepared {report["retained_windows"]} windows; no model trained.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

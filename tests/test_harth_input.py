"""Synthetic, temporary-file regression coverage; never reads research recordings."""
import csv
from datetime import datetime, timedelta
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import harth_input as harth

HEADER = ['timestamp', *harth.CHANNELS, 'label']


def rows(count=250):
    start = datetime(2020, 1, 1)
    return [[(start + timedelta(milliseconds=20 * i)).isoformat(),
             *[str(i % 2 * 2 + axis) for axis in range(6)], '1'] for i in range(count)]


def as_csv(header, records):
    output = io.StringIO(newline='')
    csv.writer(output).writerows([header, *records])
    output.seek(0)
    return output


def inspect(records, header=HEADER):
    output = io.StringIO(newline='')
    report = harth.inspect_subject(as_csv(header, records), header, 'S999', 'fit', csv.writer(output))
    output.seek(0)
    return report, list(csv.reader(output))


class HarthInputTests(unittest.TestCase):
    def test_feature_values_have_independent_closed_form_expectations(self):
        report, windows = inspect(rows(500))
        self.assertEqual((report['rows'], report['retained_windows']), (500, 2))
        self.assertEqual([row[0] for row in windows], ['S999:2', 'S999:252'])
        self.assertEqual(len(windows[0]), 35)
        for axis in range(6):
            expected = [axis + 1, 1, axis, axis + 2, math.sqrt((axis**2 + (axis + 2)**2) / 2)]
            for actual, value in zip(windows[0][5 + axis * 5:10 + axis * 5], expected):
                self.assertAlmostEqual(float(actual), value)

    def test_named_and_unnamed_export_indices_never_become_features(self):
        baseline = inspect(rows())[1]
        for name, position in [('index', 1), ('', 0)]:
            with self.subTest(name=name):
                header = HEADER.copy(); header.insert(position, name)
                records = rows()
                for row in records:
                    row.insert(position, '987654321')
                self.assertEqual(inspect(records, header)[1], baseline)

    def test_mixed_label_windows_survive_and_ties_choose_smallest_code(self):
        records = rows()
        for i, row in enumerate(records):
            row[-1] = '7' if i < 125 else '1'
        report, windows = inspect(records)
        self.assertEqual(report['mixed_label_windows'], 1)
        self.assertEqual(windows[0][3:5], ['1', '0.5'])

    def test_invalid_sensor_values_reject_whole_window_without_realigning(self):
        for value in ('NaN', 'inf', '-inf', 'not-a-number'):
            with self.subTest(value=value):
                records = rows(500); records[12][1] = value
                report, windows = inspect(records)
                self.assertEqual(report['invalid_sensor_rows'], 1)
                self.assertEqual(report['rejected_windows'], 1)
                self.assertEqual([row[0] for row in windows], ['S999:252'])

    def test_incomplete_tail_and_gap_accounting(self):
        records = rows(510)
        for row in records[100:]:
            row[0] = (datetime.fromisoformat(row[0]) + timedelta(seconds=1)).isoformat()
        report, windows = inspect(records)
        self.assertEqual(report['gap_breaks'], 1)
        self.assertEqual(report['incomplete_samples'], 260)
        self.assertEqual([row[0] for row in windows], ['S999:102'])
        self.assertEqual(report['rows'], report['complete_windows'] * 250 + report['incomplete_samples'])

    def test_nonpositive_time_breaks_and_inclusive_gap_boundary(self):
        for milliseconds, breaks in [(30, 0), (31, 1), (0, 1), (-1, 1)]:
            with self.subTest(milliseconds=milliseconds):
                records = rows(2)
                records[1][0] = (datetime.fromisoformat(records[0][0]) + timedelta(milliseconds=milliseconds)).isoformat()
                report, windows = inspect(records)
                self.assertEqual(report['gap_breaks'] + report['nonpositive_time_breaks'], breaks)
                self.assertEqual(report['incomplete_samples'], 2)
                self.assertEqual(windows, [])

    def test_bad_row_width_timestamp_label_and_timezone_fail(self):
        bad_rows = [rows(1)[0][:-1], ['bad', *rows(1)[0][1:]],
                    [*rows(1)[0][:-1], '999'], [*rows(1)[0][:-1], '1.5'],
                    ['2020-01-01T00:00:00+00:00', *rows(1)[0][1:]]]
        for row in bad_rows:
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    inspect([row])

    def test_empty_input_and_changed_header_fail(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            inspect([])
        with self.assertRaisesRegex(ValueError, 'header'):
            harth.inspect_subject(as_csv(HEADER + ['extra'], rows()), HEADER, 'S999', 'fit', csv.writer(io.StringIO()))

    def synthetic_archive(self, root):
        archive = root / 'synthetic.zip'
        members = []
        roles = {}
        with zipfile.ZipFile(archive, 'w') as z:
            for index, role in enumerate(('fit', 'calibration', 'threshold', 'test')):
                subject = f'S90{index}'
                member = f'harth/{subject}.csv'
                payload = as_csv(HEADER, rows()).getvalue().encode()
                z.writestr(member, payload)
                members.append({'member': member, 'header': HEADER})
                roles[role] = [subject]
        manifest = {'archive_sha256': harth.sha(archive), 'archive_bytes': archive.stat().st_size,
                    'members': members, 'split_plan': {'roles': roles}}
        path = root / 'synthetic-manifest.json'
        path.write_text(json.dumps(manifest))
        return archive, path, manifest

    def test_archive_preparation_role_isolation_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive, manifest, _ = self.synthetic_archive(root)
            with patch.object(harth, 'MANIFEST', manifest):
                report = harth.prepare(archive, root / 'out')
                self.assertEqual(report['retained_windows'], 4)
                self.assertEqual(report['windows_sha256'], harth.sha(root / 'out/windows.csv'))
                self.assertFalse(report['models_trained'])
                for role, value in report['roles'].items():
                    self.assertEqual(value['retained_windows'], 1)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    harth.prepare(archive, root / 'out')

    def test_archive_mismatch_does_not_leave_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive, manifest, _ = self.synthetic_archive(root)
            with archive.open('ab') as f:
                f.write(b'changed')
            with patch.object(harth, 'MANIFEST', manifest), self.assertRaisesRegex(ValueError, 'archive differs'):
                harth.prepare(archive, root / 'out')
            self.assertFalse((root / 'out').exists())

    def test_overlapping_subject_roles_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive, manifest, data = self.synthetic_archive(root)
            data['split_plan']['roles']['test'] = data['split_plan']['roles']['fit']
            manifest.write_text(json.dumps(data))
            with patch.object(harth, 'MANIFEST', manifest), self.assertRaisesRegex(ValueError, 'overlapping'):
                harth.prepare(archive, root / 'out')
            self.assertFalse((root / 'out').exists())

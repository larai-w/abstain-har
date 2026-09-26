from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import har_input
from har_input import build_report, load_dataset, plan_splits
from scripts.make_synthetic_har import generate

ROOT = Path(__file__).resolve().parents[1]


class HARInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'synthetic'
        generate(self.root)

    def replace_first(self, filename, value):
        p = self.root / filename
        lines = p.read_text().splitlines()
        lines[0] = value
        p.write_text('\n'.join(lines) + '\n')

    def test_three_files_align_without_losing_repeated_feature_names(self):
        data = load_dataset(self.root)
        self.assertEqual(len(data['train']), 36)
        self.assertEqual(len(data['test']), 12)
        self.assertEqual(data['train'][0]['row_id'], 'train:1')
        self.assertEqual(data['train'][0]['subject_id'], 1)
        self.assertEqual(data['train'][0]['label'], 1)
        self.assertEqual(len(data['train'][0]['features']), 561)
        self.assertEqual(data['feature_definitions'][-1]['source_name'], data['feature_definitions'][-2]['source_name'])
        self.assertNotEqual(data['feature_definitions'][-1]['column_id'], data['feature_definitions'][-2]['column_id'])
        self.assertEqual(len({f['column_id'] for f in data['feature_definitions']}), 561)

    def test_split_roles_are_subject_disjoint_and_test_is_untouched(self):
        data = load_dataset(self.root)
        split = plan_splits(data, 7, 1, 1)
        roles = split['roles']
        self.assertEqual(roles['test']['subjects'], [7, 8])
        self.assertEqual(roles['test']['row_ids'], [f'test:{i}' for i in range(1, 13)])
        self.assertEqual([len(roles[r]['subjects']) for r in ('fit','calibration','threshold','test')], [4,1,1,2])
        seen = set()
        all_rows = []
        for role in roles.values():
            self.assertFalse(seen.intersection(role['subjects']))
            seen.update(role['subjects'])
            all_rows.extend(role['row_ids'])
        self.assertEqual(len(all_rows), 48)
        self.assertEqual(len(set(all_rows)), 48)

    def test_assignment_does_not_use_labels_or_feature_values(self):
        data = load_dataset(self.root)
        original = plan_splits(data, 7, 1, 1)
        modified = deepcopy(data)
        for row in modified['train'] + modified['test']:
            row['label'] = 1
            row['features'] = [0.0] * 561
        changed = plan_splits(modified, 7, 1, 1)
        for role in original['roles']:
            self.assertEqual(original['roles'][role]['subjects'], changed['roles'][role]['subjects'])
            self.assertEqual(original['roles'][role]['row_ids'], changed['roles'][role]['row_ids'])
            self.assertEqual(changed['roles'][role]['missing_classes'], [2,3,4,5,6])
        self.assertFalse(changed['selection_uses_labels'])

    def test_feature_dimension_and_order_are_checked(self):
        self.replace_first('train/X_train.txt', ' '.join(['0'] * 560))
        with self.assertRaisesRegex(ValueError, '561'):
            load_dataset(self.root)
        self.replace_first('train/X_train.txt', ' '.join(['0'] * 561))
        self.replace_first('features.txt', '2 wrong-position')
        with self.assertRaisesRegex(ValueError, 'position 1'):
            load_dataset(self.root)

    def test_nonfinite_and_out_of_range_features_are_rejected(self):
        for token in ('nan', 'inf', '-inf', '1.1', 'not-a-number'):
            with self.subTest(token=token):
                self.replace_first('train/X_train.txt', ' '.join([token] + ['0'] * 560))
                with self.assertRaises(ValueError):
                    load_dataset(self.root)

    def test_row_count_mismatch_fails_instead_of_truncating(self):
        path = self.root / 'train/y_train.txt'
        path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')
        with self.assertRaisesRegex(ValueError, 'row counts differ'):
            load_dataset(self.root)

    def test_subject_overlap_is_rejected(self):
        self.replace_first('test/subject_test.txt', '1')
        with self.assertRaisesRegex(ValueError, 'subject overlap'):
            load_dataset(self.root)

    def test_invalid_label_subject_and_blank_row(self):
        for filename, value in [('train/y_train.txt','7'), ('train/y_train.txt','1 2'),
                                ('train/subject_train.txt','31'), ('train/subject_train.txt','0'),
                                ('train/X_train.txt','')]:
            with self.subTest(filename=filename, value=value):
                p = self.root / filename
                before = p.read_text()
                self.replace_first(filename, value)
                with self.assertRaises(ValueError):
                    load_dataset(self.root)
                p.write_text(before)

    def test_activity_vocabulary_is_not_silently_remapped(self):
        self.replace_first('activity_labels.txt', '1 RUNNING')
        with self.assertRaisesRegex(ValueError, 'six-class'):
            load_dataset(self.root)

    def test_invalid_reservations_fail_instead_of_reusing_subjects(self):
        data = load_dataset(self.root)
        for calibration, threshold in ((0,1),(1,0),(3,3),(4,1),(True,1)):
            with self.subTest(calibration=calibration, threshold=threshold), self.assertRaises(ValueError):
                plan_splits(data, 7, calibration, threshold)

    def test_seed_is_reproducible_and_test_subjects_never_change(self):
        data = load_dataset(self.root)
        first = plan_splits(data, 7, 1, 1)
        self.assertEqual(first, plan_splits(data, 7, 1, 1))
        different = plan_splits(data, 19, 1, 1)
        self.assertEqual(first['roles']['test'], different['roles']['test'])
        self.assertNotEqual(first['roles']['fit']['subjects'], different['roles']['fit']['subjects'])

    def test_changes_during_loading_are_detected(self):
        original = har_input.read_partition
        def mutate(root, part):
            result = original(root, part)
            if part == 'test':
                path = root / 'features.txt'
                path.write_text(path.read_text() + '\n')
            return result
        with patch('har_input.read_partition', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'changed during loading'):
                load_dataset(self.root)

    def test_synthetic_provenance_cannot_be_accidentally_relabelled(self):
        with self.assertRaises(ValueError):
            build_report(self.root, 'uci-har-research', 7, 1, 1)
        (self.root / 'SYNTHETIC.txt').unlink()
        with self.assertRaises(ValueError):
            build_report(self.root, 'synthetic', 7, 1, 1)

    def test_checked_in_report_is_reproducible(self):
        expected = json.loads((ROOT / 'examples/har-input-v1/report.json').read_text())
        self.assertEqual(build_report(self.root, 'synthetic', 7, 1, 1), expected)

    def test_cli_is_offline_and_preserves_existing_output(self):
        output = Path(self.temp.name) / 'report'
        command = [sys.executable, str(ROOT / 'har_input.py'), str(self.root), '--dataset-kind', 'synthetic', '--calibration-subjects', '1', '--threshold-subjects', '1', '--output', str(output)]
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        saved = (output / 'report.json').read_bytes()
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(saved, (output / 'report.json').read_bytes())


if __name__ == '__main__':
    unittest.main()

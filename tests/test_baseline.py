"""Temporary synthetic baselines: no research data download or access."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score

import baseline as b


class BaselineTests(unittest.TestCase):
    def dataset(self, root):
        roles = {'fit': ['S901', 'S902', 'S903', 'S904'], 'calibration': ['S905'],
                 'threshold': ['S906'], 'test': ['S907', 'S908']}
        manifest = {'archive_sha256': 'synthetic-test-only', 'split_plan': {'roles': roles}}
        mp = root / 'manifest.json'; mp.write_text(json.dumps(manifest))
        subject_reports = []
        with (root / 'windows.csv').open('w', newline='') as f:
            writer = csv.writer(f); writer.writerow(b.HEADER)
            for role, subjects in roles.items():
                for s in subjects:
                    for index, label in enumerate(b.LABELS):
                        writer.writerow([f'{s}:{2 + index * 250}', s, role, label, 0.5,
                                         *[float(index + axis % 3) for axis in range(30)]])
                    subject_reports.append({'subject_id': s, 'role': role, 'retained_windows': 12,
                                            'label_counts': {str(k): 1 for k in b.LABELS}})
        report = {'dataset_kind': 'harth-research', 'windows_sha256': b.sha(root / 'windows.csv'),
                  'archive_sha256': manifest['archive_sha256'], 'source_manifest_sha256': b.sha(mp),
                  'model_feature_columns': b.FEATURES, 'subjects': subject_reports, 'retained_windows': 96}
        (root / 'report.json').write_text(json.dumps(report))
        return mp

    def rewrite_csv(self, root, mutate):
        path = root / 'windows.csv'
        with path.open(newline='') as f:
            rows = list(csv.reader(f))
        mutate(rows)
        with path.open('w', newline='') as f:
            csv.writer(f).writerows(rows)
        rp = root / 'report.json'; report = json.loads(rp.read_text())
        report['windows_sha256'] = b.sha(path); rp.write_text(json.dumps(report))

    def test_only_thirty_explicit_features_enter_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mp = self.dataset(root)
            data = b.load_inputs(root, mp)
            self.assertEqual(data['X'].shape, (96, 30))
            np.testing.assert_equal(data['X'][0], [float(i % 3) for i in range(30)])
            before = data['X'].copy()
            self.rewrite_csv(root, lambda rows: [row.__setitem__(4, '1.0') for row in rows[1:]])
            np.testing.assert_equal(b.load_inputs(root, mp)['X'], before)

    def test_hash_role_nan_duplicate_and_extra_column_rejection(self):
        changes = [lambda r: r[1].__setitem__(2, 'test'),
                   lambda r: r[1].__setitem__(5, 'NaN'),
                   lambda r: r[2].__setitem__(0, r[1][0]),
                   lambda r: r[0].append('leaky_feature'),
                   lambda r: r[1].__setitem__(0, 'S901:3'),  # overlaps the next window
                   lambda r: r[1].__setitem__(3, '999')]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); mp = self.dataset(root)
                self.rewrite_csv(root, change)
                with self.assertRaises(ValueError):
                    b.load_inputs(root, mp)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mp = self.dataset(root)
            with (root / 'windows.csv').open('a') as f:
                f.write('tampered\n')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                b.load_inputs(root, mp)

    def test_report_class_support_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mp = self.dataset(root)
            self.rewrite_csv(root, lambda r: r[1].__setitem__(3, '2'))
            with self.assertRaisesRegex(ValueError, 'support'):
                b.load_inputs(root, mp)

    def test_grouped_folds_keep_each_person_out_of_training(self):
        subjects = np.repeat(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], 3)
        visits = []
        for train, valid in b.grouped_folds(subjects):
            self.assertFalse(set(subjects[train]) & set(subjects[valid]))
            visits.extend(valid.tolist())
        self.assertEqual(sorted(visits), list(range(len(subjects))))
        with self.assertRaises(ValueError):
            b.grouped_folds(np.array(['a', 'b', 'c']))

    def test_scaler_is_fit_on_training_rows_only(self):
        x = np.vstack([np.zeros(30), np.ones(30), np.ones(30) * 2, np.ones(30) * 3])
        model = b.models()['logistic']; model.fit(x, [1, 2, 1, 2])
        np.testing.assert_allclose(model.named_steps['standardscaler'].mean_, 1.5)
        model.predict(np.ones((2, 30)) * 10000)
        np.testing.assert_allclose(model.named_steps['standardscaler'].mean_, 1.5)

    def test_selection_cannot_use_nonfit_values_or_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mp = self.dataset(root); data = b.load_inputs(root, mp)
            templates = {name: DummyClassifier(strategy='most_frequent') for name in ('dummy', 'logistic', 'random_forest')}
            with patch.object(b, 'models', return_value=templates):
                first, candidate = b.select_with_fit_only(data)
                data['X'][data['roles'] != 'fit'] = 99999
                data['y'][data['roles'] != 'fit'] = 140
                second, next_candidate = b.select_with_fit_only(data)
            self.assertEqual(candidate, next_candidate)
            self.assertEqual(candidate, 'logistic')  # declared tie-break
            for name in first:
                self.assertEqual(first[name]['mean_macro_f1_12'], second[name]['mean_macro_f1_12'])
                for before, after in zip(first[name]['folds'], second[name]['folds']):
                    self.assertEqual(before['metrics'], after['metrics'])

    def test_metric_definitions_match_independent_library_reference(self):
        y = [1, 1, 2, 2]; pred = [1, 2, 2, 2]
        result = b.metrics(y, pred)
        self.assertEqual(result['accuracy'], 0.75)
        self.assertAlmostEqual(result['macro_f1_12'], f1_score(y, pred, labels=b.LABELS, average='macro', zero_division=0))
        self.assertAlmostEqual(result['balanced_accuracy_present_classes'], balanced_accuracy_score(y, pred))
        self.assertIsNone(result['per_class']['140']['recall'])
        self.assertIsNone(result['per_class']['140']['precision'])
        self.assertEqual(result['per_class']['140']['f1'], 0)
        self.assertIsNone(b.metrics([], [])['accuracy'])

    def test_bootstrap_resamples_people_and_repeats_deterministically(self):
        per_subject = {'one': b.metrics([1, 1], [1, 1]), 'two': b.metrics([2], [1])}
        first = b.subject_bootstrap(per_subject)
        self.assertEqual(first, b.subject_bootstrap(per_subject))
        self.assertEqual(first['participants'], 2)
        self.assertEqual(first['intervals_95']['accuracy'], [0.0, 1.0])

    def test_small_end_to_end_run_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mp = self.dataset(root); data = b.load_inputs(root, mp)
            with patch.object(b, 'load_inputs', return_value=data):
                report = b.run(root, root / 'out')
                self.assertTrue(report['models_trained'])
                self.assertFalse(report['calibration_fitted'])
                self.assertEqual(report['role_windows']['fit'], 48)
                self.assertEqual(set(report['models']), {'dummy', 'logistic', 'random_forest'})
                self.assertTrue((root / 'out/selection.json').exists())
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    b.run(root, root / 'out')

"""Synthetic calibration/abstention checks; no research files are read."""
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

import selective as s


def probabilities(confidences):
    p = np.zeros((len(confidences), 12))
    p[:, 0] = confidences
    p[:, 1] = 1 - np.asarray(confidences)
    return p


def synthetic_data():
    xs, ys, roles, subjects = [], [], [], []
    for role, repeats in [('fit', 8), ('calibration', 8), ('threshold', 16), ('test', 4)]:
        for repeat in range(repeats):
            for i, label in enumerate(s.LABELS):
                xs.append([i, i % 3, repeat * 0.001])
                ys.append(label); roles.append(role); subjects.append(role)
    return {'X': np.asarray(xs), 'y': np.asarray(ys), 'roles': np.asarray(roles), 'subjects': np.asarray(subjects)}


class SelectiveTests(unittest.TestCase):
    def test_inclusive_boundary_and_abstained_label_is_null(self):
        p = probabilities([0.8, 0.79, 1.0])
        decisions = s.predict_with_abstention(p, 0.8)
        self.assertEqual([v['status'] for v in decisions], ['accepted', 'abstained', 'accepted'])
        self.assertIsNone(decisions[1]['label'])
        self.assertTrue(all(v['label'] is None for v in s.predict_with_abstention(p, None)))
        self.assertEqual(s.selective_metrics(np.array([1, 1, 1]), p, 1.0)['accepted'], 1)

    def test_zero_accepted_or_zero_support_has_undefined_risk(self):
        value = s.selective_metrics(np.array([1, 2]), probabilities([0.8, 0.8]), None)
        self.assertEqual((value['accepted'], value['coverage']), (0, 0.0))
        self.assertIsNone(value['selective_risk'])
        empty = s.selective_metrics(np.array([], dtype=int), np.empty((0, 12)), 0.5)
        self.assertIsNone(empty['coverage'])
        self.assertIsNone(empty['selective_risk'])

    def test_selection_uses_maximum_feasible_coverage_and_lowest_threshold(self):
        y = np.array([1] * 200 + [2] * 200)
        result = s.choose_threshold(y, probabilities([0.9] * 200 + [0.6] * 200))
        self.assertEqual(result['status'], 'selected')
        self.assertEqual(result['threshold'], 0.61)
        self.assertEqual(result['selection']['coverage'], 0.5)
        self.assertEqual(result['selection']['selective_risk'], 0)

    def test_infeasible_selection_never_relaxes_constraints(self):
        for y, p in [(np.full(200, 2), probabilities([1.0] * 200)),
                     (np.full(99, 1), probabilities([1.0] * 99))]:
            result = s.choose_threshold(y, p)
            self.assertEqual(result['status'], 'no_feasible_threshold')
            self.assertIsNone(result['threshold'])
            self.assertIsNone(result['selection']['selective_risk'])
        with self.assertRaises(ValueError):
            s.choose_threshold([], np.empty((0, 12)))

    def test_invalid_probabilities_labels_and_thresholds_fail(self):
        for p in [np.zeros((2, 12)), np.ones((2, 2)), np.full((1, 12), np.nan),
                  np.array([[1.1, -0.1] + [0] * 10])]:
            with self.subTest(p=p), self.assertRaises(ValueError):
                s.validate_probabilities(p)
        for y in [[999], [1, 2]]:
            with self.subTest(y=y), self.assertRaises(ValueError):
                s.validate_probabilities(probabilities([0.8]), y)
        for threshold in [-1, 1.01, np.nan, True]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                s.decision_arrays(probabilities([0.8]), threshold)

    def test_probability_scores_match_closed_form_and_library(self):
        y = np.array([1, 2]); p = probabilities([0.8, 0.8])
        scores = s.probability_scores(y, p)
        self.assertAlmostEqual(scores['log_loss'], log_loss(y, p, labels=s.LABELS))
        self.assertAlmostEqual(scores['multiclass_brier'], 0.68)
        self.assertAlmostEqual(scores['top_label_ece_10_bins'], 0.3)
        self.assertEqual(scores['reliability_bins'][8]['count'], 2)
        self.assertIsNone(scores['reliability_bins'][0]['accuracy'])
        perfect = s.probability_scores(np.asarray(s.LABELS), np.eye(12))
        self.assertEqual(perfect['reliability_bins'][9]['count'], 12)
        self.assertEqual(perfect['multiclass_brier'], 0.0)

    def test_frozen_calibration_never_refits_forest(self):
        data = synthetic_data(); fit = data['roles'] == 'fit'; cal = data['roles'] == 'calibration'
        forest = RandomForestClassifier(n_estimators=8, random_state=7).fit(data['X'][fit], data['y'][fit])
        before = forest.predict_proba(data['X'][cal])
        with patch.object(forest, 'fit', side_effect=AssertionError('base estimator refit')):
            calibrated = s.fit_calibrator(forest, data['X'][cal], data['y'][cal])
        np.testing.assert_array_equal(before, forest.predict_proba(data['X'][cal]))
        s.validate_probabilities(calibrated.predict_proba(data['X'][cal]))
        missing = cal & (data['y'] != 140)
        with self.assertRaisesRegex(ValueError, 'twelve'):
            s.fit_calibrator(forest, data['X'][missing], data['y'][missing])

    def test_test_values_and_labels_cannot_change_calibration_or_policy(self):
        data = synthetic_data()
        factory = lambda: {'random_forest': RandomForestClassifier(n_estimators=8, random_state=7)}
        with patch.object(s, 'models', side_effect=factory):
            first_models, first = s.fit_and_select(data)
            probe = data['X'][data['roles'] == 'threshold'].copy()
            before = first_models['sigmoid'].predict_proba(probe)
            data['X'][data['roles'] == 'test'] = 999999
            data['y'][data['roles'] == 'test'] = 140
            next_models, second = s.fit_and_select(data)
        self.assertEqual(first, second)
        # Independent numeric fits may differ at machine epsilon; policies must still match exactly.
        np.testing.assert_allclose(before, next_models['sigmoid'].predict_proba(probe), rtol=0, atol=1e-12)

    def test_threshold_labels_cannot_change_calibrator(self):
        data = synthetic_data()
        factory = lambda: {'random_forest': RandomForestClassifier(n_estimators=8, random_state=7)}
        with patch.object(s, 'models', side_effect=factory):
            models, _ = s.fit_and_select(data)
            before = models['sigmoid'].predict_proba(data['X'])
            data['y'][data['roles'] == 'threshold'] = 140
            after_models, _ = s.fit_and_select(data)
        np.testing.assert_allclose(before, after_models['sigmoid'].predict_proba(data['X']), rtol=0, atol=1e-12)

    def test_bootstrap_retains_undefined_risk_information(self):
        none = {'a': s.counts(10, 0, 0), 'b': s.counts(10, 0, 0)}
        result = s.bootstrap(none)
        self.assertEqual(result['valid_risk_replicates'], 0)
        self.assertIsNone(result['risk_interval_95'])
        mixed = {'a': s.counts(10, 0, 0), 'b': s.counts(10, 10, 10)}
        result = s.bootstrap(mixed)
        self.assertEqual(result, s.bootstrap(mixed))
        self.assertTrue(0 < result['valid_risk_replicates'] < 1000)
        self.assertEqual(result['risk_interval_95'], [1.0, 1.0])

    def test_evaluation_counts_match_per_person_and_per_class(self):
        y = np.array([1, 2, 1, 2]); p = probabilities([0.9, 0.6, 0.8, 0.9])
        result = s.evaluate(y, p, np.array(['a', 'a', 'b', 'b']), {'threshold': 0.8})
        self.assertEqual(result['selected']['accepted'], 3)
        self.assertEqual(result['selected']['errors'], 1)
        for field in ('total', 'accepted', 'errors'):
            self.assertEqual(sum(v[field] for v in result['per_subject'].values()), result['selected'][field])
            self.assertEqual(sum(v[field] for v in result['per_true_class'].values()), result['selected'][field])
        self.assertIsNone(result['risk_coverage_curve'][-1]['selective_risk'])

"""Synthetic fixed-model stress checks; no research artifacts are accessed."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import robustness as r


class Predictor:
    classes_ = np.asarray(r.LABELS)
    n_features_in_ = 30

    def __init__(self):
        self.calls = []

    def predict_proba(self, x):
        if not np.isfinite(x).all():
            raise AssertionError('invalid rows reached model')
        self.calls.append(x.copy())
        p = np.zeros((len(x), 12))
        p[:, 0] = np.where(x[:, 0] > 0, 0.9, 0.6)
        p[:, 1] = 1 - p[:, 0]
        return p

    def fit(self, *args):
        raise AssertionError('inference must not fit')


def summaries(raw):
    # Independent raw-signal statistics oracle for the algebraic transformations.
    return np.stack([raw.mean(axis=1), raw.std(axis=1), raw.min(axis=1),
                     raw.max(axis=1), np.sqrt((raw ** 2).mean(axis=1))], axis=2).reshape(-1, 30)


class RobustnessTests(unittest.TestCase):
    def test_transformations_match_raw_signal_oracle_without_mutation(self):
        raw = np.random.default_rng(3).normal(size=(5, 250, 6))
        x = summaries(raw); original = x.copy()
        for scenario in r.SCENARIOS:
            if scenario.startswith('missing_'):
                continue
            changed = raw.copy()
            if scenario == 'zero_back': changed[:, :, :3] = 0
            elif scenario == 'gain_half': changed *= 0.5
            elif scenario == 'gain_double': changed *= 2
            elif scenario == 'bias_back_x_small': changed[:, :, 0] += 0.25
            elif scenario == 'bias_back_x_large': changed[:, :, 0] += 1
            elif scenario == 'invert_back_x': changed[:, :, 0] *= -1
            elif scenario == 'swap_sensors': changed = changed[:, :, [3, 4, 5, 0, 1, 2]]
            with self.subTest(scenario=scenario):
                observed, _ = r.perturb(x, scenario)
                np.testing.assert_allclose(observed, summaries(changed), rtol=0, atol=1e-12)
                np.testing.assert_array_equal(x, original)
                self.assertFalse(np.shares_memory(x, observed))

    def test_missing_masks_are_nested_reproducible_and_ceil_sized(self):
        x = np.ones((11, 30)); masks = []
        for percent, expected in [(10, 2), (30, 4), (100, 11)]:
            scenario = f'missing_thigh_{percent}'
            changed, meta = r.perturb(x, scenario)
            again, again_meta = r.perturb(x, scenario)
            np.testing.assert_array_equal(changed, again)
            self.assertEqual(meta, again_meta)
            self.assertEqual(meta['missing_rows'], expected)
            np.testing.assert_array_equal(changed[:, :15], x[:, :15])
            masks.append(np.isnan(changed).any(axis=1))
        self.assertTrue(np.all(~masks[0] | masks[1]))
        self.assertTrue(masks[2].all())

    def test_invalid_perturbations_fail(self):
        for x, case in [(np.ones((1, 29)), 'clean'), (np.full((1, 30), np.nan), 'clean'),
                        (np.ones((1, 30)), 'unknown')]:
            with self.assertRaises(ValueError): r.perturb(x, case)
        bad = np.zeros((1, 30)); bad[:, 0] = -10
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            r.perturb(bad, 'bias_back_x_small')

    def test_invalid_rows_never_reach_predictor(self):
        x = np.ones((4, 30)); x[1, 0] = 0; x[2, 2] = np.nan; x[3, 3] = np.inf
        model = Predictor(); state = r.infer(model, x, 0.8)
        self.assertEqual(state['status'].tolist(), ['accepted', 'model_abstained', 'input_rejected', 'input_rejected'])
        self.assertEqual(model.calls[0].shape, (2, 30))
        value = r.summary(np.ones(4), state)
        self.assertEqual((value['accepted'], value['model_abstained'], value['input_rejected']), (1, 1, 2))
        self.assertEqual(value['coverage'], 0.25)
        self.assertEqual(value['coverage_among_valid'], 0.5)

    def test_all_rejected_and_empty_class_risk_is_undefined(self):
        model = Predictor(); state = r.infer(model, np.full((2, 30), np.nan), 0.8)
        self.assertEqual(model.calls, [])
        result = r.compare(np.array([1, 1]), np.array(['a', 'b']), state, state)
        self.assertEqual(result['coverage'], 0)
        self.assertIsNone(result['selective_risk'])
        self.assertIsNone(result['per_true_class']['2']['coverage'])
        self.assertIsNone(result['participant_bootstrap']['risk_interval_95'])
        self.assertEqual(result['participant_bootstrap']['valid_risk_replicates'], 0)

    def test_schema_and_model_output_guards(self):
        with self.assertRaises(ValueError): r.infer(Predictor(), np.ones((1, 29)), 0.8)
        model = Predictor(); model.classes_ = np.array([1])
        with self.assertRaisesRegex(ValueError, 'class order'): r.infer(model, np.ones((1, 30)), 0.8)
        for output in [np.zeros((1, 12)), np.ones((1, 2)), np.tile([1] + [0] * 11, (2, 1))]:
            model = Predictor()
            with patch.object(model, 'predict_proba', return_value=output), self.assertRaises(ValueError):
                r.infer(model, np.ones((1, 30)), 0.8)

    def test_paired_transitions_use_fixed_shared_denominator(self):
        def state(status, predicted):
            return {'status': np.array(status), 'predicted': np.array(predicted), 'confidence': np.ones(4)}
        clean = state(['accepted', 'accepted', 'model_abstained', 'accepted'], [1, 2, 1, 1])
        stressed = state(['accepted', 'accepted', 'accepted', 'input_rejected'], [2, 1, 2, -1])
        result = r.compare(np.ones(4, dtype=int), np.array(['a', 'a', 'b', 'b']), clean, stressed)
        pair = result['paired']
        self.assertEqual(pair['clean_correct_accepted_to_stressed_wrong_accepted'], 1)
        self.assertEqual(pair['shared_accepted_windows'], 2)
        self.assertEqual((pair['clean_errors_on_shared'], pair['stressed_errors_on_shared']), (1, 1))
        self.assertEqual(sum(sum(row.values()) for row in pair['status_transitions'].values()), 4)
        for field in ('total', 'accepted', 'errors'):
            self.assertEqual(sum(v[field] for v in result['per_subject'].values()), result[field])
            self.assertEqual(sum(v[field] for v in result['per_true_class'].values()), result[field])

    def test_hashes_checked_before_any_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('raw', 'sigmoid'): (root / f'{name}.joblib').write_bytes(b'not a model')
            ref = {'model_sha256': {'raw': r.sha(root / 'raw.joblib'), 'sigmoid': 'wrong'}}
            with patch.object(r.joblib, 'load') as loader, self.assertRaisesRegex(ValueError, 'hash mismatch'):
                r.load_frozen(root, ref)
            loader.assert_not_called()

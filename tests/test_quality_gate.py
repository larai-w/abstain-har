"""Synthetic input-gate checks, without research data or model files."""
import unittest
import numpy as np
from quality_gate import gated_infer, input_reasons
from evaluate_gate import change, rejection_counts
from local_predict import predict
from test_local_predict import request
from test_robustness import Predictor


class QualityGateTests(unittest.TestCase):
    def test_sensor_reasons_and_missing_precedence(self):
        x = np.ones((5, 30))
        x[0, :15] = 0; x[1, 15:] = 0; x[2] = 0
        x[3, :15] = 0; x[3, 29] = np.nan
        self.assertEqual(input_reasons(x, 'exact-zero-v1').tolist(),
                         ['zero_back_sensor', 'zero_thigh_sensor', 'zero_both_sensors', 'missing_feature', ''])
        self.assertEqual(input_reasons(x, 'finite-v1').tolist(), ['', '', '', 'missing_feature', ''])

    def test_stationary_gravity_and_single_zero_axis_are_kept(self):
        # Two constant gravity vectors: zero variance alone must not reject them.
        x = np.array([[0, 0, 0, 0, 0] * 2 + [1, 0, 1, 1, 1]] * 2).reshape(1, 30)
        self.assertEqual(input_reasons(x, 'exact-zero-v1').tolist(), [''])
        self.assertEqual(input_reasons(np.ones((1, 30)) * 1e-9, 'exact-zero-v1').tolist(), [''])
        self.assertEqual(input_reasons(np.array([[1, 0, 1, 1, 1] * 6]), 'exact-zero-v1').tolist(), [''])

    def test_rejected_inputs_never_reach_model_and_input_is_unchanged(self):
        x = np.ones((3, 30)); x[0, :15] = 0; x[1, 15:] = 0
        before = x.copy(); model = Predictor()
        state, _ = gated_infer(model, x, 0.8, 'exact-zero-v1')
        self.assertEqual(state['status'].tolist(), ['input_rejected', 'input_rejected', 'accepted'])
        self.assertEqual(model.calls[0].shape, (1, 30))
        np.testing.assert_array_equal(x, before)
        model = Predictor()
        gated_infer(model, np.zeros((2, 30)), 0.8, 'exact-zero-v1')
        self.assertEqual(model.calls, [])

    def test_local_response_is_versioned_and_suppresses_rejected_labels(self):
        item = request()
        old = predict(item, Predictor(), 0.8)
        new = predict(item, Predictor(), 0.8, 'exact-zero-v1')
        self.assertEqual(old['input_policy'], 'finite-v1')
        self.assertEqual(new['response_version'], 2)
        self.assertEqual(new['counts'], {'input_rejected': 2, 'model_abstained': 0, 'accepted': 1})
        zero = new['decisions'][1]
        self.assertEqual(zero['reason'], 'zero_both_sensors')
        self.assertIsNone(zero['label']); self.assertIsNone(zero['confidence'])

    def test_invalid_policy_or_shape_fails(self):
        with self.assertRaises(ValueError): input_reasons(np.ones((1, 30)), 'unknown')
        with self.assertRaises(ValueError): input_reasons(np.ones((1, 29)), 'exact-zero-v1')

    def test_evaluation_cases_and_rejection_counts(self):
        x = np.array([[1, 0, 1, 1, 1] * 6], dtype=float)
        for case in ('zero_back', 'zero_thigh', 'zero_both'):
            with self.subTest(case=case):
                self.assertEqual(rejection_counts(change(x, case))['rejected'], 1)
        for case in ('near_zero_back', 'constant_back_nonzero', 'clean'):
            with self.subTest(case=case):
                self.assertEqual(rejection_counts(change(x, case))['rejected'], 0)
        self.assertIsNone(rejection_counts(np.empty((0, 30)))['fraction'])
        with self.assertRaises(ValueError): change(x, 'unknown')

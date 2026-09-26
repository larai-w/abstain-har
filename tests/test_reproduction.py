"""Synthetic evidence-comparison checks."""
import unittest
from scripts.compare_reproduction import compare


class ReproductionTests(unittest.TestCase):
    def test_discrete_changes_and_structure_never_use_tolerance(self):
        for other in [{'n': 2}, {'n': 1.0}, {'n': 1, 'extra': 0}, {'n': [1]}]:
            self.assertFalse(compare({'n': 1}, other)['semantic_match'])

    def test_float_tolerance_is_fixed_absolute_and_rejects_nonfinite(self):
        self.assertTrue(compare(0.5, 0.5 + 1e-13)['semantic_match'])
        self.assertFalse(compare(0.5, 0.5 + 1e-9)['semantic_match'])
        self.assertFalse(compare(float('nan'), float('nan'))['semantic_match'])
        self.assertFalse(compare([1], [1, 2])['semantic_match'])

    def test_artifact_mismatch_is_separate_and_visible(self):
        value = compare({'model_sha256': 'a', 'n': 2}, {'model_sha256': 'b', 'n': 2})
        self.assertTrue(value['semantic_match'])
        self.assertFalse(value['model_artifacts_match'])
        self.assertEqual(value['model_artifact_hashes'], [{'path': '$/model_sha256', 'equal': False}])

    def test_only_declared_metadata_is_excluded(self):
        before = {'completed_at': 'a', 'fit_seconds': 1.0, 'environment': {'platform': 'a', 'packages': {'numpy': '1'}}}
        after = {'completed_at': 'b', 'fit_seconds': 2.0, 'environment': {'platform': 'b', 'packages': {'numpy': '1'}}}
        result = compare(before, after)
        self.assertTrue(result['semantic_match'])
        self.assertEqual(len(result['excluded_paths']), 3)
        after['environment']['packages']['numpy'] = '2'
        self.assertFalse(compare(before, after)['semantic_match'])

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from prepare_inputs import prepare

ROOT = Path(__file__).resolve().parents[1]


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.bundle = json.loads((ROOT / 'examples/integration/bundle.json').read_text())
        self.splits = json.loads((ROOT / 'examples/integration/splits.json').read_text())

    def eligible(self):
        return next(r for r in self.bundle['rows'] if r['decision'] == 'eligible')

    def test_valid_bundle_separates_gates_from_model_abstention(self):
        result = prepare(self.bundle, self.splits)
        self.assertEqual(result['summary']['total'], 6)
        self.assertEqual(result['summary']['eligible'], 4)
        self.assertEqual(result['summary']['excluded'], 2)
        self.assertEqual(result['summary']['eligible_by_split'], dict.fromkeys(['fit','calibration','threshold','test'], 1))
        self.assertFalse(result['trained_model'])
        self.assertEqual({r['reason'] for r in result['excluded']}, {'quality_rejected', 'insufficient_data'})
        self.assertEqual(result['batches']['test'][0]['features'], [.5, 2, 1])
        self.assertEqual(result, json.loads((ROOT / 'examples/integration/prepared.json').read_text()))

    def test_subject_overlap_is_rejected(self):
        self.splits['subjects']['test'].append('synthetic-fit')
        with self.assertRaisesRegex(ValueError, 'subject overlap'):
            prepare(self.bundle, self.splits)

    def test_missing_extra_and_duplicate_subject_assignments(self):
        for mutation in ('missing', 'extra', 'duplicate'):
            splits = deepcopy(self.splits)
            if mutation == 'missing': splits['subjects']['fit'] = []
            if mutation == 'extra': splits['subjects']['fit'].append('unknown')
            if mutation == 'duplicate': splits['subjects']['fit'].append('synthetic-fit')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                prepare(self.bundle, splits)

    def test_future_arrivals_cannot_be_passed_as_current_features(self):
        self.eligible()['audit']['selected_versions'][0]['received_at'] = '2099-01-01T00:00:00Z'
        with self.assertRaisesRegex(ValueError, 'temporally unavailable'):
            prepare(self.bundle, self.splits)

    def test_feature_values_must_match_audited_counts(self):
        for value in (0.125, float('nan'), float('inf'), True):
            bundle = deepcopy(self.bundle)
            next(r for r in bundle['rows'] if r['decision'] == 'eligible')['features']['observed_fraction'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare(bundle, self.splits)

    def test_rejected_samples_cannot_smuggle_features(self):
        row = next(r for r in self.bundle['rows'] if r['decision'] == 'quality_rejected')
        row['features'] = {'observed_fraction': 1, 'known_count': 1, 'unknown_count': 1}
        with self.assertRaisesRegex(ValueError, 'null features'):
            prepare(self.bundle, self.splits)

    def test_quality_flag_cannot_be_relabelled_as_eligible(self):
        next(r for r in self.bundle['rows'] if r['decision'] == 'quality_rejected')['decision'] = 'eligible'
        with self.assertRaisesRegex(ValueError, 'decision is inconsistent'):
            prepare(self.bundle, self.splits)

    def test_feature_names_versions_and_dataset_kind_are_explicit(self):
        for key, value in [('contract_version', True), ('contract_version', 2), ('dataset_kind', 'research'), ('feature_set', 'uci-har'), ('feature_names', ['other']), ('contract_sha256', 'wrong')]:
            bundle = deepcopy(self.bundle)
            bundle[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                prepare(bundle, self.splits)

    def test_lineage_and_sample_ids_are_checked(self):
        self.bundle['rows'].append(deepcopy(self.bundle['rows'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate sample_id'):
            prepare(self.bundle, self.splits)
        self.bundle['rows'].pop()
        self.eligible()['audit']['lineage'][0]['chain'].append('unknown')
        with self.assertRaisesRegex(ValueError, 'lineage endpoints'):
            prepare(self.bundle, self.splits)

    def test_missing_audit_or_nonstring_status_is_rejected(self):
        self.eligible()['audit']['selected_versions'][0]['status'] = []
        with self.assertRaisesRegex(ValueError, 'invalid audited status'):
            prepare(self.bundle, self.splits)

    def test_input_is_unchanged_and_batches_are_order_independent(self):
        before = deepcopy((self.bundle, self.splits))
        expected = prepare(self.bundle, self.splits)
        self.assertEqual((self.bundle, self.splits), before)
        self.bundle['rows'].reverse()
        actual = prepare(self.bundle, self.splits)
        self.assertEqual(actual['batches'], expected['batches'])
        self.assertEqual(actual['excluded'], expected['excluded'])
        # Archive order is still traceable in its audit hash.
        self.assertNotEqual(actual['audit']['bundle_sha256'], expected['audit']['bundle_sha256'])

    def test_empty_eligible_split_is_visible_not_a_training_claim(self):
        row = self.eligible()
        row['audit']['selected_versions'] = []
        row['audit']['lineage'] = []
        row.update(decision='insufficient_data', features=None)
        result = prepare(self.bundle, self.splits)
        self.assertIn(0, result['summary']['eligible_by_split'].values())
        self.assertFalse(result['trained_model'])

    def test_cli_creates_output_once_and_rejects_leaky_split(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'prepared.json'
            command = [sys.executable, str(ROOT / 'prepare_inputs.py'), str(ROOT / 'examples/integration/bundle.json'), '--splits', str(ROOT / 'examples/integration/splits.json'), '--output', str(out)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            old = out.read_bytes()
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            self.assertEqual(old, out.read_bytes())
            split = Path(directory) / 'split.json'
            self.splits['subjects']['test'].append('synthetic-fit')
            split.write_text(json.dumps(self.splits))
            new = Path(directory) / 'rejected.json'
            result = subprocess.run(command[:3] + ['--splits', str(split), '--output', str(new)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(new.exists())
            self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()

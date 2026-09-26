"""Local inference contract checks using temporary synthetic data only."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import local_predict as p
from test_robustness import Predictor


def request():
    return {'contract': p.CONTRACT, 'dataset_kind': 'synthetic', 'features': list(p.FEATURES),
            'samples': [{'id': 'accepted', 'values': [1] * 30},
                        {'id': 'abstained', 'values': [0] * 30},
                        {'id': 'missing', 'values': [None] + [1] * 29}]}


class LocalPredictTests(unittest.TestCase):
    def test_three_outcomes_and_null_labels(self):
        value = p.predict(request(), Predictor(), 0.8)
        self.assertEqual(value['counts'], {'accepted': 1, 'model_abstained': 1, 'input_rejected': 1})
        first, second, third = value['decisions']
        self.assertEqual(first['label'], 1)
        self.assertIsNone(second['label']); self.assertIsNone(third['label'])
        self.assertEqual(second['confidence'], 0.6)
        self.assertIsNone(third['confidence'])
        json.dumps(value, allow_nan=False)

    def test_null_policy_abstains_on_all_valid_rows(self):
        value = p.predict(request(), Predictor(), None)
        self.assertEqual(value['counts'], {'accepted': 0, 'model_abstained': 2, 'input_rejected': 1})
        self.assertTrue(all(d['label'] is None for d in value['decisions']))

    def test_wrong_contract_order_and_unknown_fields_fail(self):
        for field, value in [('contract', 'v2'), ('dataset_kind', 'clinical'), ('features', list(reversed(p.FEATURES))), ('extra', 1)]:
            item = request(); item[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): p.validate_request(item)

    def test_batch_and_row_structure_fail(self):
        bad_rows = [[], [{}] * 1001, None, [{'id': 'x', 'values': [1] * 29}],
                    [{'id': 'x', 'values': [1] * 30, 'label': 1}],
                    [{'id': 'x', 'values': [1] * 30}] * 2,
                    [{'id': '../private', 'values': [1] * 30}]]
        for rows in bad_rows:
            item = request(); item['samples'] = rows
            with self.subTest(rows=str(rows)[:80]), self.assertRaises(ValueError): p.validate_request(item)

    def test_no_numeric_coercion_or_nonfinite_json_values(self):
        for number in [True, '1.0', float('nan'), float('inf'), 10 ** 400, {}, []]:
            item = request(); item['samples'][0]['values'][0] = number
            with self.subTest(value=str(number)[:30]), self.assertRaises(ValueError): p.validate_request(item)

    def test_duplicate_json_keys_and_nonstandard_constants_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'request.json'
            for raw in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":-Infinity}']:
                source.write_text(raw)
                with self.assertRaises(ValueError): p.read_request(source)
            source.write_bytes(b' ' * (p.MAX_BYTES + 1))
            with self.assertRaisesRegex(ValueError, 'byte limit'): p.read_request(source)

    def test_malformed_batch_never_loads_model_or_writes_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'request.json'; output = root / 'output.json'
            source.write_text('{}')
            with patch.object(p, 'load_model') as loader, self.assertRaises(ValueError):
                p.run(source, root, 'raw', output)
            loader.assert_not_called(); self.assertFalse(output.exists())

    def test_output_provenance_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'request.json'; output = root / 'out' / 'response.json'
            source.write_text(json.dumps(request()))
            with patch.object(p, 'load_model', return_value=(Predictor(), 0.8, {'variant': 'raw'})):
                value = p.run(source, root, 'raw', output)
                self.assertEqual(json.loads(output.read_text()), value)
                self.assertEqual(value['provenance']['request_sha256'], p.sha(source))
                before = output.read_bytes()
                with self.assertRaises(ValueError): p.run(source, root, 'raw', output)
                self.assertEqual(output.read_bytes(), before)

    def test_model_hash_and_package_mismatch_prevent_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ref_path = root / 'ref.json'
            (root / 'raw.joblib').write_bytes(b'not a model')
            reference = {'environment': {'packages': {'numpy': 'expected'}},
                         'model_sha256': {'raw': 'wrong'}, 'policies': {'raw': {'threshold': 0.8}}}
            ref_path.write_text(json.dumps(reference))
            for version in ['different', 'expected']:
                with patch.object(p, 'REFERENCE', ref_path), patch.object(p.importlib.metadata, 'version', return_value=version), patch.object(p.joblib, 'load') as loader:
                    with self.assertRaises(ValueError): p.load_model(root, 'raw')
                    loader.assert_not_called()

    def test_model_schema_and_policy_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / 'raw.joblib'; path.write_bytes(b'fixture')
            ref_path = root / 'ref.json'
            ref = {'environment': {'packages': {}}, 'model_sha256': {'raw': p.sha(path)},
                   'policies': {'raw': {'threshold': 0.8}}}
            for field, value in [('n_features_in_', 29), ('classes_', [1])]:
                model = Predictor(); setattr(model, field, value)
                ref_path.write_text(json.dumps(ref))
                with patch.object(p, 'REFERENCE', ref_path), patch.object(p.joblib, 'load', return_value=model), self.assertRaises(ValueError):
                    p.load_model(root, 'raw')
            ref['policies']['raw']['threshold'] = True
            ref_path.write_text(json.dumps(ref))
            with patch.object(p, 'REFERENCE', ref_path), patch.object(p.joblib, 'load', return_value=Predictor()), self.assertRaises(ValueError):
                p.load_model(root, 'raw')

    def test_cli_failure_is_nonzero_and_does_not_echo_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'request.json'; source.write_text('{"secret_marker": true}')
            run = subprocess.run([sys.executable, str(p.ROOT / 'local_predict.py'), str(source), '--models', str(root), '--variant', 'raw', '--output', str(root / 'out.json')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertNotIn('secret_marker', run.stderr)
            self.assertFalse((root / 'out.json').exists())

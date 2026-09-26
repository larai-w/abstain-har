#!/usr/bin/env python3
"""Run both checked-out repositories against their fixed synthetic integration example."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--toolkit', type=Path, required=True)
    args = parser.parse_args()
    toolkit = args.toolkit.resolve()
    if (toolkit / 'contracts/care-bundle-v1.json').read_bytes() != (ROOT / 'contracts/care-bundle-v1.json').read_bytes():
        raise ValueError('Producer and consumer contract copies differ')
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory) / 'bundle.json'
        prepared = Path(directory) / 'prepared.json'
        subprocess.run([sys.executable, str(toolkit / 'care_export.py'), str(toolkit / 'fixtures/integration-request.json'), '--output', str(bundle)], check=True)
        subprocess.run([sys.executable, str(ROOT / 'prepare_inputs.py'), str(bundle), '--splits', str(ROOT / 'examples/integration/splits.json'), '--output', str(prepared)], check=True)
        actual = json.loads(prepared.read_text())
        expected = json.loads((ROOT / 'examples/integration/prepared.json').read_text())
        if actual != expected:
            raise ValueError('Fresh cross-repository output differs from the checked-in example')
    print('Fresh producer -> consumer integration passed: four eligible, two excluded, no training.')


if __name__ == '__main__':
    main()

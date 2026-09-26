#!/usr/bin/env python3
"""Render aggregate selective-prediction evidence; never reads sensor rows."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new directory')
    report = json.loads(args.report.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for name, color in [('raw', '#0072B2'), ('sigmoid', '#D55E00')]:
        data = report['test'][name]
        bins = [b for b in data['probability_scores']['reliability_bins'] if b['count']]
        axes[0].plot([b['mean_confidence'] for b in bins], [b['accuracy'] for b in bins], 'o-', color=color, label=name)
        points = [p for p in data['risk_coverage_curve'] if p['selective_risk'] is not None]
        axes[1].plot([p['coverage'] for p in points], [p['selective_risk'] for p in points], color=color, label=name)
        chosen = data['selected']
        if chosen['selective_risk'] is not None:
            axes[1].scatter([chosen['coverage']], [chosen['selective_risk']], s=70, color=color, edgecolors='black', zorder=4)
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='confidence = accuracy')
    axes[0].set(xlabel='Mean top-label confidence (10 fixed bins)', ylabel='Accuracy within bin',
                title='Reliability on the reused test split', xlim=(0, 1), ylim=(0, 1))
    axes[1].axhline(report['selection_targets']['empirical_risk'], ls='--', color='gray', label='selection target (not guarantee)')
    axes[1].set(xlabel='Coverage (fraction answered)', ylabel='Selective risk (errors / answers)',
                title='Risk–coverage; dots mark selected policies', xlim=(0, 1))
    axes[1].set_ylim(bottom=0)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle('Exploratory HARTH extension • 4 previously evaluated test participants', fontsize=11)
    args.output.mkdir(parents=True, exist_ok=False)
    fig.savefig(args.output / 'calibration-risk-coverage.png', dpi=180)
    plt.close(fig)
    metadata = {'report_sha256': hashlib.sha256(args.report.read_bytes()).hexdigest(),
                'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'matplotlib_version': importlib.metadata.version('matplotlib'),
                'source': 'report.json aggregate bins and grid only; no fitted parameters changed'}
    (args.output / 'plot-source.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()

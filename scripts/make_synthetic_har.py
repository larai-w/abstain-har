#!/usr/bin/env python3
"""Generate fake numbers in the release text layout, never research recordings."""
import argparse
from pathlib import Path

# Intentionally maintained independently from the input parser.
ACTIVITIES = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']


def generate(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    (root / 'SYNTHETIC.txt').write_text('Generated contract fixture. No research recordings or model results.\n')
    (root / 'features.txt').write_text(''.join(f'{i} ' + (f'synthetic_feature_{i}' if i < 559 else 'synthetic_repeated_name') + '\n' for i in range(1, 562)))
    (root / 'activity_labels.txt').write_text(''.join(f'{i} {name}\n' for i, name in enumerate(ACTIVITIES, 1)))
    for part, subjects in [('train', range(1, 7)), ('test', range(7, 9))]:
        directory = root / part
        directory.mkdir()
        xs, ys, ids = [], [], []
        for subject in subjects:
            for label in range(1, 7):
                xs.append(' '.join(f'{((subject * 23 + label * 7 + feature) % 201 - 100) / 100:.2f}' for feature in range(561)))
                ys.append(str(label))
                ids.append(str(subject))
        for name, lines in [('X', xs), ('y', ys), ('subject', ids)]:
            (directory / f'{name}_{part}.txt').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists; choose a new directory')
    generate(args.output)
    print(f'Generated 48 synthetic rows, not UCI research data: {args.output}')


if __name__ == '__main__':
    main()

"""Compare reproduced raw scientific CSVs with the versioned baseline."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated', type=Path, required=True)
    parser.add_argument('--exact', action='store_true', help='Require byte identity instead of numerical agreement.')
    args = parser.parse_args()
    baselines = sorted(p for p in (ROOT / 'results').glob('*.csv') if not p.name.endswith('_summary.csv'))
    if len(baselines) != 21:
        raise RuntimeError(f'Expected 21 raw scientific CSVs; found {len(baselines)}.')
    for reference in baselines:
        generated = args.generated / reference.name
        if not generated.is_file():
            raise FileNotFoundError(generated)
        if args.exact:
            if reference.read_bytes() != generated.read_bytes():
                raise AssertionError(f'Byte mismatch: {reference.name}')
        else:
            pd.testing.assert_frame_equal(pd.read_csv(reference), pd.read_csv(generated),
                                          check_exact=False, rtol=1e-9, atol=1e-12)
        print(f'PASS {reference.name}')
    print(f'{len(baselines)} raw files verified; runtime metadata intentionally excluded.')

if __name__ == '__main__':
    main()

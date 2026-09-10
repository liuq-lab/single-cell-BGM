#!/usr/bin/env python3
from pathlib import Path
import anndata as ad

root = Path('data/splits')
files = {
    'train': root / 'pbmc68k_train_raw.h5ad',
    'validation': root / 'pbmc68k_validation_raw.h5ad',
    'test': root / 'pbmc68k_test_raw.h5ad',
}
counts = {}
for name, path in files.items():
    if not path.exists():
        raise FileNotFoundError(path)
    x = ad.read_h5ad(path, backed='r')
    counts[name] = int(x.n_obs)
    x.file.close()

total = sum(counts.values())
print('split counts')
for name in ('train', 'validation', 'test'):
    print(f'{name:10s}: {counts[name]:6d}  ({counts[name]/total:.4%})')
print(f'total     : {total:6d}')

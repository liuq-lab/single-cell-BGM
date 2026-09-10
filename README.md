# Single-cell BayesGM

This branch adds the updated unconditional BayesGM workflow used for the
Bouchet AENB experiments. It uses a Negative-Binomial RNA autoencoder and a
compact, label-free BayesGM training pipeline.

The implementation is isolated under [`unconditional/`](unconditional/).

## Repository layout

```text
.
└── unconditional/          # Updated label-free NB-AE + BayesGM workflow
```

Raw datasets, H5AD files, model weights, generated arrays, checkpoints, logs,
and cluster-specific local configurations are intentionally excluded from Git.

## Quick start

Use Python 3.9. Install the dependencies for the updated workflow:

```bash
python -m pip install -r unconditional/requirements.txt
```

For unconditional generation:

```bash
cd unconditional
python preprocess.py --help
python autoencoder.py --help
python train.py --bgm-z-dim 16
python evaluate.py
```

See [`unconditional/README.md`](unconditional/README.md) for the input layout,
full commands, and artifact contract before starting a long training run.

## Reproducibility contract

- Gene filtering and all learned preprocessing are fit on TRAIN only.
- VALIDATION is used for model/policy selection.
- TEST is reserved for the frozen final evaluation.
- Generated count data are normalized to 10,000 counts per cell and transformed
  with `log1p` exactly once before the reported expression-space metrics.
- Saved configuration records the split identities, random seeds, and
  model/artifact settings.

The active code was organized from the verified Bouchet experiment snapshots on
2026-09-08. The unconditional core is the NB-AE implementation used by the AENB
experiments; the experiment-level benchmarking harness is intentionally omitted.

## References

- [BayesGM](https://github.com/liuq-lab/bayesgm)
- [scDiffusion](https://github.com/EperLuo/scDiffusion)
- [scDiffusion-X](https://github.com/EperLuo/scDiffusion-X)

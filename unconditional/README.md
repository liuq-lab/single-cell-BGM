# Unconditional BayesGM with an NB RNA autoencoder

This is the active label-free workflow organized from the Bouchet AENB
experiment. BayesGM models a 128-dimensional latent representation learned by a
Negative-Binomial (NB) RNA autoencoder, then decodes generated latent cells back
to sampled UMI counts.

No cluster label, classifier, label embedding, or class-balanced sampling is
used in this workflow.

## Model contract

```text
raw UMI counts
  -> log1p
  -> NB autoencoder (genes -> 512 -> 300 -> 128)
  -> unconditional BayesGM
  -> frozen NB decoder + TRAIN-derived library size
  -> sampled counts
  -> normalize_total(1e4) + log1p
```

The compact training recipe uses BayesGM `z=16`. The base trainer defaults to
`z=32` for backward compatibility, so pass `--bgm-z-dim 16` explicitly when
using the recipe below.

## Input layout

The core runner expects one fixed split directory with these canonical names:

```text
data/splits/
├── pbmc68k_train_raw.h5ad
├── pbmc68k_validation_raw.h5ad
└── pbmc68k_test_raw.h5ad
```

For a non-PBMC dataset, create an isolated split view using the same three
filenames. Do not mix datasets in one directory. `train.py` opens only TRAIN and
VALIDATION; `evaluate.py` is the separate final-test entry point.

## Install

Python 3.9 is recommended.

```bash
python -m pip install -r requirements.txt
```

On an HPC system, reuse a tested TensorFlow/PyTorch environment when possible;
do not blindly replace its CUDA packages.

## Run

From this directory:

```bash
# PBMC68k only: build the fixed raw-count split from 10x input.
python preprocess.py --help
python preprocess.py

# Train and select the NB autoencoder on TRAIN/VALIDATION.
python autoencoder.py \
  --split-dir data/splits \
  --output-dir results/autoencoder \
  --latent-dim 128

# Optional frozen-AE validation diagnostic.
python evaluate_ae.py \
  --split-dir data/splits \
  --ae-artifact-dir results/autoencoder

# Train BayesGM. z=16 matches the recommended model dimension.
python train.py \
  --split-dir data/splits \
  --ae-artifact-dir results/autoencoder \
  --output-dir results/bayesgm_z16 \
  --bgm-z-dim 16

# Open TEST only after the model and evaluation choices are frozen.
python evaluate.py \
  --split-dir data/splits \
  --result-dir results/bayesgm_z16
```

Run `python <script> --help` before a long job; all model, seed, decoder, and
selection settings are recorded in the output configuration.

## Library-size model

The included trainer fits one global Normal distribution to TRAIN log-library
sizes and samples one library size per generated cell. VALIDATION and TEST
library sizes are not used. Experiment-level screening and materialization
scripts are intentionally omitted from this compact source tree.

## Outputs excluded from Git

Autoencoder weights, BayesGM weights, H5AD files, NPY/NPZ arrays, logs, and
generated matrices belong under `results/` or another artifact store. Only
source code, dependency metadata, and documentation should be committed.

The NB observation model follows scDiffusion-X commit
`d60d928b635ab7a52ace030a641efd02137c193f`; the mathematical alignment is
documented in the `autoencoder.py` module header.

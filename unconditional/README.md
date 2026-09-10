# Unconditional BayesGM

This directory contains the compact label-free single-cell BayesGM workflow.
It uses a Negative-Binomial RNA autoencoder to encode raw counts, trains
BayesGM in the 128-dimensional AE latent space, and decodes generated cells
back to gene expression.

## Files

| File | Purpose |
|---|---|
| `preprocess.py` | Create fixed TRAIN, VALIDATION, and TEST raw-count splits. |
| `check_data.py` | Report split sizes and proportions. |
| `autoencoder.py` | Train the 128-dimensional Negative-Binomial RNA autoencoder. |
| `evaluate_ae.py` | Optionally evaluate the frozen autoencoder on VALIDATION. |
| `train.py` | Train and select unconditional BayesGM using TRAIN and VALIDATION. |
| `evaluate.py` | Evaluate frozen generated output on the final TEST split. |

## Expected data

The default workflow uses:

```text
data/splits/
├── pbmc68k_train_raw.h5ad
├── pbmc68k_validation_raw.h5ad
└── pbmc68k_test_raw.h5ad
```

The three files must contain raw non-negative UMI counts, identical gene names
and order, and disjoint cell identities. `train.py` opens only TRAIN and
VALIDATION. TEST is opened separately by `evaluate.py` after model selection.

## Run order

Run commands from this directory. Use Python 3.9 and inspect all options with
`python <script> --help` before starting a long job.

```bash
# 1. Build fixed raw-count splits.
python preprocess.py --input data/raw/hg19 --output-dir data/splits

# 2. Check the split sizes.
python check_data.py

# 3. Train the Negative-Binomial autoencoder.
python autoencoder.py \
  --split-dir data/splits \
  --output-dir results/autoencoder \
  --latent-dim 128

# 4. Optional validation-only AE diagnostic.
python evaluate_ae.py \
  --split-dir data/splits \
  --ae-artifact-dir results/autoencoder

# 5. Train unconditional BayesGM.
python train.py \
  --split-dir data/splits \
  --ae-artifact-dir results/autoencoder \
  --output-dir results/bayesgm_z16 \
  --bgm-z-dim 16

# 6. Evaluate the frozen model on TEST.
python evaluate.py \
  --split-dir data/splits \
  --result-dir results/bayesgm_z16
```

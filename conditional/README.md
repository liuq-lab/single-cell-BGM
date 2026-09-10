# Conditional BayesGM

This directory contains the compact conditional single-cell BayesGM workflow.
Conditions are supplied through an existing `AnnData.obs` column; training does
not infer clusters or create labels automatically.

## Files

| File | Purpose |
|---|---|
| `preprocess.py` | Create fixed TRAIN, VALIDATION, and TEST raw-count splits. |
| `prepare_official_pbmc68k_labels.py` | Attach the pinned official PBMC68k cell-type annotations. |
| `select_markers.py` | Optionally select TRAIN-only marker genes for evaluation. |
| `autoencoder.py` | Train the 128-dimensional Negative-Binomial RNA autoencoder. |
| `train.py` | Train and select the conditional BayesGM using TRAIN and VALIDATION. |
| `generate.py` | Generate labeled cells from a frozen conditional model. |
| `evaluate.py` | Evaluate condition fidelity, real/generated AUC, macro-MMD, and optional marker fidelity. |

## Expected data

The default PBMC68k workflow uses:

```text
data/splits/
├── pbmc68k_train_raw.h5ad
├── pbmc68k_validation_raw.h5ad
├── pbmc68k_test_raw.h5ad
├── pbmc68k_train_raw_official_celltype.h5ad
├── pbmc68k_validation_raw_official_celltype.h5ad
└── pbmc68k_test_raw_official_celltype.h5ad
```

All splits must have the same gene order, disjoint cell identities, and a
non-missing `celltype` column in the labeled files. TEST is not used during
training or model selection.

## Run order

Run commands from this directory. Inspect all available options with
`python <script> --help` before starting a long job.

```bash
# 1. Build fixed raw-count splits.
python preprocess.py --input data/raw/hg19 --output-dir data/splits

# 2. Add the official PBMC68k labels.
python prepare_official_pbmc68k_labels.py --split-dir data/splits

# 3. Train the Negative-Binomial autoencoder.
python autoencoder.py \
  --split-dir data/splits \
  --output-dir results/autoencoder \
  --latent-dim 128

# 4. Train conditional BayesGM.
python train.py \
  --split-dir data/splits \
  --ae-artifact-dir results/autoencoder \
  --output-dir results/conditional_bayesgm \
  --bgm-z-dim 32 \
  --label-embed-dim 16 \
  --label-injection early_concat \
  --condition-dx \
  --library-mode label-lognormal-shrunk

# 5. Generate cells for every trained condition.
python generate.py \
  --model-dir results/conditional_bayesgm \
  --labels all \
  --n-per-label 1000 \
  --output results/conditional_bayesgm/generated.h5ad

# 6. Evaluate on VALIDATION while developing.
python evaluate.py \
  --real-train data/splits/pbmc68k_train_raw_official_celltype.h5ad \
  --real-eval data/splits/pbmc68k_validation_raw_official_celltype.h5ad \
  --generated results/conditional_bayesgm/generated.h5ad \
  --output-dir results/conditional_evaluation
```

Use the frozen TEST split only after the model, seeds, generation policy, and
evaluation settings have been finalized.

# Single-cell BayesGM

This repository contains compact unconditional and conditional BayesGM
workflows for single-cell RNA-seq generation. Both workflows use a
Negative-Binomial RNA autoencoder and keep TRAIN, VALIDATION, and TEST roles
separate.

## Repository layout

```text
single-cell-BGM/
├── unconditional/
│   ├── preprocess.py
│   ├── check_data.py
│   ├── autoencoder.py
│   ├── evaluate_ae.py
│   ├── train.py
│   └── evaluate.py
└── conditional/
    ├── preprocess.py
    ├── prepare_official_pbmc68k_labels.py
    ├── select_markers.py
    ├── autoencoder.py
    ├── train.py
    ├── generate.py
    └── evaluate.py
```

## Shared data contract

- Input matrices contain raw, finite, non-negative UMI counts.
- TRAIN, VALIDATION, and TEST cell identities are disjoint.
- Gene names and gene order are identical across splits.
- Preprocessing and learned distributions use TRAIN only.
- VALIDATION is used for model selection.
- TEST is opened only after the model and evaluation settings are frozen.

## Data split

The full dataset is first divided into 80% development data and 20% final test
data. Ten percent of the development data is used for validation. The resulting
proportions are approximately:

| Split | Fraction | Use |
|---|---:|---|
| Train | 72% | All autoencoder and BayesGM gradient updates |
| Validation | 8% | Autoencoder, Step 1, and Step 2 checkpoint selection |
| Test | 20% | One final evaluation after all settings are frozen |

Gene filtering is learned from the training cells only. The validation and test
sets are never used for gradient updates.

## Workflow

### Unconditional

1. `preprocess.py` creates the fixed TRAIN, VALIDATION, and TEST splits.
2. `check_data.py` checks the split sizes and proportions.
3. `autoencoder.py` trains the Negative-Binomial autoencoder on TRAIN and
   selects its checkpoint with VALIDATION.
4. `train.py` trains BayesGM in the frozen autoencoder latent space and uses
   VALIDATION for checkpoint selection.
5. `evaluate.py` evaluates the frozen generated output on TEST.

`evaluate_ae.py` is an optional validation-only autoencoder diagnostic.

### Conditional

1. `preprocess.py` creates the fixed raw-count splits.
2. `prepare_official_pbmc68k_labels.py` attaches the condition labels.
3. `autoencoder.py` trains the Negative-Binomial autoencoder.
4. `train.py` trains the label-conditioned BayesGM using TRAIN and VALIDATION.
5. `generate.py` generates cells for requested labels from the frozen model.
6. `evaluate.py` measures condition fidelity, real/generated discrimination,
   distribution similarity, and optional marker fidelity.

`select_markers.py` is optional. See each workflow README for full commands and
expected input/output paths.

## References

- [BayesGM](https://github.com/liuq-lab/bayesgm)
- [scDiffusion](https://github.com/EperLuo/scDiffusion)
- [scDiffusion-X](https://github.com/EperLuo/scDiffusion-X)

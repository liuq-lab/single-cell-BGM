# BayesGM for PBMC68k

This repository trains an unconditional BayesGM model on PBMC68k single-cell
RNA-seq data. A deterministic scDiffusion-style autoencoder maps full-gene
expression to a 128-dimensional latent space. BayesGM is then trained in that
space and the generated cells are decoded back to full-gene expression.

## Data

Download the **Fresh 68k PBMCs, Donor A** dataset from 10x Genomics:

https://www.10xgenomics.com/cn/datasets/fresh-68-k-pbm-cs-donor-a-1-standard-1-1-0

Place the extracted 10x files here:

```text
data/raw/hg19/
├── barcodes.tsv
├── genes.tsv
└── matrix.mtx
```

Compressed `.gz` versions are also supported.

## Files

```text
.
├── preprocess.py
├── autoencoder.py
├── train.py
├── evaluate.py
├── evaluate.ipynb
├── requirements.txt
└── README.md
```

- `preprocess.py` creates the train, validation, and test splits.
- `autoencoder.py` trains the scDiffusion-style deterministic autoencoder.
- `train.py` trains BayesGM Step 1 and Step 2.
- `evaluate.py` contains final-test metrics and plotting functions.
- `evaluate.ipynb` audits the split, evaluates the frozen model, and plots the results.

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

## Model

### Autoencoder

- Input: full-gene counts normalized to 10,000 counts per cell and transformed with `log1p`
- Encoder: `n_genes -> 1024 -> 1024 -> 1024 -> 128`
- Hidden layers: BatchNorm and PReLU
- Latent representation: per-cell L2 normalization
- Decoder: symmetric architecture
- Loss: mean squared error
- Optimizer: AdamW, learning rate `5e-4`, weight decay `1e-2`

The upstream scDiffusion repository calls this network a VAE, but the model used
here is deterministic: it has no `mu`, `logvar`, reparameterization, or KL loss.

### BayesGM Step 1

- Input dimension: 128
- BayesGM latent dimension: 32
- Generator and encoder: five hidden layers of 512 units
- Data discriminator: `[256, 128, 64, 16]`
- Latent discriminator: `[128, 64, 32, 8]`
- Learning rate: `1e-4`
- Batch size: 512
- Iterations: 100,000
- Selection metric: validation PCA20 iLISI

### BayesGM Step 2

- Epochs: 500
- Batch size: 512
- Generator learning rate: `5e-7`
- Variance learning rate: `1e-8`
- Initial variance: `0.30`
- Variance range: `[0.15, 0.60]`
- Objective: Gaussian NLL, latent MMD, Step 1 anchor, and variance regularization
- Selection metric: validation PCA20 iLISI

The final test set is not loaded by `train.py`.


## Run

All commands are executed from the repository root.

### 1. Preprocess the data

```bash
python preprocess.py
```

This writes the split files to `data/splits/`.

### 2. Train the autoencoder

```bash
python autoencoder.py
```

This writes the selected autoencoder and latent representations to
`results/autoencoder/`.

### 3. Train BayesGM

```bash
python train.py
```

This writes the selected checkpoints, histories, and frozen generated cells to
`results/bayesgm/`.


### 4. Evaluate

```bash
jupyter lab evaluate.ipynb
```

The notebook performs the protocol audit, reads the final test set, computes
the metrics, and creates two simple plots.

## Evaluation

All metrics start from decoded full-gene log-normalized expression, not from
the BayesGM latent space.

- PCA20 iLISI: real and generated cells are combined, projected with joint PCA,
  and evaluated on a 10-nearest-neighbor graph using the first 20 PCs.
- PCA50 MMD: computed from the first 50 PCs of the same joint PCA.
- Additional checks: zero fraction, mean cell sum, and gene-mean correlation.

## References

- [BayesGM](https://github.com/liuq-lab/bayesgm)
- [scDiffusion](https://github.com/EperLuo/scDiffusion)

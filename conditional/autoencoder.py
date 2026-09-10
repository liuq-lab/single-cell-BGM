#!/usr/bin/env python3
"""Negative-Binomial RNA autoencoder used by the PBMC68k BayesGM pipeline.

This module reimplements only the single-RNA autoencoder observation model used
by scDiffusion-X, without requiring PyTorch Lightning, Hydra, or scvi-tools.

Alignment to scDiffusion-X (official commit d60d928):
- encoder input: log1p(raw UMI counts)
- MLP encoder dims: n_genes -> 512 -> 300 -> 100
- MLP decoder dims: 100 -> 300 -> 512 -> n_genes
- hidden normalization: BatchNorm1d
- hidden activation: ELU
- RNA likelihood: Negative Binomial
- mean: library_size * softmax(decoder(z))
- inverse dispersion: one learnable parameter per gene, theta = exp(log_theta)

The implementation intentionally keeps the same mathematical model while using
plain PyTorch so it can be dropped into the existing BayesGM workflow.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from scipy import sparse


OFFICIAL_COMMIT = "d60d928b635ab7a52ace030a641efd02137c193f"
OFFICIAL_REPO = "https://github.com/EperLuo/scDiffusion-X"


def sha256_lines(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


class MLP(nn.Module):
    """scDiffusion-X style MLP: Norm+ELU on hidden layers, linear final layer."""

    def __init__(
        self,
        dims: Sequence[int],
        norm: bool = True,
        norm_type: str = "batchnorm",
        dropout: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [int(v) for v in dims]
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output")
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if norm:
                if norm_type == "batchnorm":
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                elif norm_type == "layernorm":
                    layers.append(nn.LayerNorm(dims[i + 1]))
                else:
                    raise ValueError(f"Unknown norm_type={norm_type!r}")
            layers.append(nn.ELU())
            if dropout:
                layers.append(nn.Dropout(float(dropout_p)))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNAEncoderDecoder(nn.Module):
    """Single-RNA Negative-Binomial autoencoder.

    The count likelihood and decoder parameterization follow the RNA autoencoder
    used by scDiffusion-X. The latent width is configurable so the default 128-D
    setting can match the previous PBMC68k BayesGM pipeline.
    """

    def __init__(
        self,
        n_genes: int,
        hidden_dims: Sequence[int] = (512, 300),
        latent_dim: int = 100,
        norm_type: str = "batchnorm",
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.hidden_dims = tuple(int(v) for v in hidden_dims)
        self.latent_dim = int(latent_dim)
        encoder_dims = [self.n_genes, *self.hidden_dims, self.latent_dim]
        decoder_dims = list(reversed(encoder_dims))
        self.encoder = MLP(
            encoder_dims,
            norm=True,
            norm_type=norm_type,
            dropout=False,
            dropout_p=0.0,
        )
        self.decoder = MLP(
            decoder_dims,
            norm=True,
            norm_type=norm_type,
            dropout=False,
            dropout_p=0.0,
        )
        # Official scDiffusion-X initializes this unconstrained parameter randomly.
        self.log_theta = nn.Parameter(torch.randn(self.n_genes), requires_grad=True)

    def encode(self, raw_counts: torch.Tensor) -> torch.Tensor:
        if raw_counts.ndim != 2 or raw_counts.shape[1] != self.n_genes:
            raise ValueError(f"Expected (N,{self.n_genes}) raw counts, got {tuple(raw_counts.shape)}")
        if torch.any(raw_counts < 0):
            raise ValueError("RNA counts must be non-negative")
        return self.encoder(torch.log1p(raw_counts))

    def decode_mu(self, latent: torch.Tensor, library_size: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected (N,{self.latent_dim}) latent, got {tuple(latent.shape)}")
        if library_size.ndim == 1:
            library_size = library_size[:, None]
        if library_size.ndim != 2 or library_size.shape[1] != 1:
            raise ValueError(f"Expected library_size shape (N,) or (N,1), got {tuple(library_size.shape)}")
        logits = self.decoder(latent)
        rho = torch.softmax(logits, dim=1)
        return rho * library_size

    def forward(self, raw_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        library_size = raw_counts.sum(dim=1, keepdim=True)
        if torch.any(library_size <= 0):
            raise ValueError("Every RNA cell must have positive total counts")
        latent = self.encode(raw_counts)
        mu = self.decode_mu(latent, library_size)
        theta = torch.exp(self.log_theta)
        return latent, mu, theta


def negative_binomial_log_prob(
    counts: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """NB log probability with mean/inverse-dispersion parameterization.

    Var[X] = mu + mu^2 / theta.
    """
    counts = counts.to(mu.dtype)
    theta = theta.to(mu.dtype)
    mu = torch.clamp(mu, min=eps)
    theta = torch.clamp(theta, min=eps)
    log_theta_mu = torch.log(theta + mu)
    return (
        torch.lgamma(counts + theta)
        - torch.lgamma(theta)
        - torch.lgamma(counts + 1.0)
        + theta * (torch.log(theta) - log_theta_mu)
        + counts * (torch.log(mu) - log_theta_mu)
    )


def negative_binomial_nll(
    counts: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Official objective shape: -log_prob(X).sum(genes).mean(cells)."""
    return -negative_binomial_log_prob(counts, mu, theta).sum(dim=1).mean()


@torch.no_grad()
def sample_negative_binomial(
    mu: torch.Tensor,
    theta: torch.Tensor,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample NB counts using PyTorch's total_count/logits parameterization."""
    if seed is not None:
        # This intentionally resets the generator for deterministic checkpoint comparisons.
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    mu = torch.clamp(mu, min=1e-8)
    theta = torch.clamp(theta, min=1e-8)
    logits = torch.log(mu) - torch.log(theta)
    distribution = torch.distributions.NegativeBinomial(
        total_count=theta,
        logits=logits,
    )
    return distribution.sample()


def dense_rows(matrix, indices: np.ndarray) -> np.ndarray:
    values = matrix[indices]
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def row_sums(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        result = np.asarray(matrix.sum(axis=1)).reshape(-1)
    else:
        result = np.asarray(matrix, dtype=np.float64).sum(axis=1)
    return np.asarray(result, dtype=np.float32)


@torch.no_grad()
def encode_matrix(
    model: RNAEncoderDecoder,
    matrix,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    model = model.to(device).eval()
    chunks: list[np.ndarray] = []
    for start in range(0, matrix.shape[0], batch_size):
        stop = min(start + batch_size, matrix.shape[0])
        idx = np.arange(start, stop)
        batch = torch.as_tensor(dense_rows(matrix, idx), dtype=torch.float32, device=device)
        chunks.append(model.encode(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def decode_mu_matrix(
    model: RNAEncoderDecoder,
    latent: np.ndarray,
    library_size: np.ndarray,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    model = model.to(device).eval()
    latent = np.asarray(latent, dtype=np.float32)
    library_size = np.asarray(library_size, dtype=np.float32).reshape(-1)
    if len(latent) != len(library_size):
        raise ValueError("latent and library_size lengths differ")
    chunks: list[np.ndarray] = []
    for start in range(0, len(latent), batch_size):
        stop = min(start + batch_size, len(latent))
        z = torch.as_tensor(latent[start:stop], dtype=torch.float32, device=device)
        s = torch.as_tensor(library_size[start:stop], dtype=torch.float32, device=device)
        mu = model.decode_mu(z, s)
        chunks.append(mu.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def decode_sample_counts(
    model: RNAEncoderDecoder,
    latent: np.ndarray,
    library_size: np.ndarray,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> np.ndarray:
    """Decode latent + library size and sample NB counts in deterministic batches."""
    model = model.to(device).eval()
    latent = np.asarray(latent, dtype=np.float32)
    library_size = np.asarray(library_size, dtype=np.float32).reshape(-1)
    if len(latent) != len(library_size):
        raise ValueError("latent and library_size lengths differ")
    theta = torch.exp(model.log_theta).to(device)
    chunks: list[np.ndarray] = []
    for batch_index, start in enumerate(range(0, len(latent), batch_size)):
        stop = min(start + batch_size, len(latent))
        z = torch.as_tensor(latent[start:stop], dtype=torch.float32, device=device)
        s = torch.as_tensor(library_size[start:stop], dtype=torch.float32, device=device)
        mu = model.decode_mu(z, s)
        counts = sample_negative_binomial(mu, theta, seed=int(seed) + batch_index)
        chunks.append(counts.cpu().numpy().astype(np.float32))
    result = np.concatenate(chunks, axis=0)
    # Entirely zero cells are essentially impossible at PBMC library sizes, but guard anyway.
    zero_cells = result.sum(axis=1) <= 0
    if np.any(zero_cells):
        mu = decode_mu_matrix(model, latent[zero_cells], library_size[zero_cells], batch_size, device)
        result[zero_cells] = mu
    return result


def normalize_total_log1p_dense(
    matrix: np.ndarray,
    target_sum: float = 1e4,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if np.any(matrix < 0) or not np.isfinite(matrix).all():
        raise ValueError("matrix must be finite and non-negative")
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Every cell must have positive total expression")
    normalized = matrix * np.float32(target_sum) / totals
    return np.log1p(normalized).astype(np.float32, copy=False)


@torch.no_grad()
def decode_to_log_expression(
    model: RNAEncoderDecoder,
    latent: np.ndarray,
    library_size: np.ndarray,
    mode: str = "sample",
    target_sum: float = 1e4,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    seed: int = 42,
) -> np.ndarray:
    if mode == "sample":
        decoded = decode_sample_counts(
            model, latent, library_size, batch_size=batch_size, device=device, seed=seed
        )
    elif mode == "mean":
        decoded = decode_mu_matrix(
            model, latent, library_size, batch_size=batch_size, device=device
        )
    else:
        raise ValueError("mode must be 'sample' or 'mean'")
    return normalize_total_log1p_dense(decoded, target_sum=target_sum)


@dataclass
class AEBundle:
    model: RNAEncoderDecoder
    metadata: dict
    train_latent: np.ndarray
    validation_latent: np.ndarray
    train_library_size: np.ndarray
    validation_library_size: np.ndarray


def load_ae_bundle(
    directory: Path | str,
    device: str | torch.device = "cpu",
) -> AEBundle:
    directory = Path(directory).expanduser().resolve()
    with (directory / "metadata.json").open() as handle:
        metadata = json.load(handle)
    model = RNAEncoderDecoder(
        n_genes=int(metadata["n_genes"]),
        hidden_dims=tuple(metadata.get("hidden_dims", [512, 300])),
        latent_dim=int(metadata.get("latent_dim", 100)),
        norm_type=str(metadata.get("norm_type", "batchnorm")),
    )
    state = torch.load(directory / "model.pt", map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(device).eval()
    return AEBundle(
        model=model,
        metadata=metadata,
        train_latent=np.load(directory / "train_latent.npy").astype(np.float32),
        validation_latent=np.load(directory / "validation_latent.npy").astype(np.float32),
        train_library_size=np.load(directory / "train_library_size.npy").astype(np.float32),
        validation_library_size=np.load(directory / "validation_library_size.npy").astype(np.float32),
    )

import argparse
import gc
import os
import random

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/pbmc68k_ae_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pbmc68k_ae_matplotlib")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--train-file", default="pbmc68k_train_raw.h5ad")
    p.add_argument("--validation-file", default="pbmc68k_validation_raw.h5ad")
    p.add_argument("--output-dir", type=Path, default=Path("results/autoencoder"))
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--hidden-dims", default="512,300")
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--norm-type", choices=("batchnorm", "layernorm"), default="batchnorm")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--encode-batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_raw(adata: ad.AnnData, name: str) -> None:
    adata.var_names_make_unique()
    values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X).ravel()
    if values.size and (not np.isfinite(values).all() or float(values.min()) < 0):
        raise ValueError(f"{name} must contain finite non-negative raw counts")
    sums = row_sums(adata.X)
    if np.any(sums <= 0):
        raise ValueError(f"{name} contains zero-library cells")


def dense_batch(matrix, indices: np.ndarray) -> np.ndarray:
    x = matrix[indices]
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def validation_nll(
    model: RNAEncoderDecoder,
    matrix,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_cells = 0
    with torch.no_grad():
        for start in range(0, matrix.shape[0], batch_size):
            stop = min(start + batch_size, matrix.shape[0])
            idx = np.arange(start, stop)
            counts = torch.as_tensor(dense_batch(matrix, idx), device=device)
            _, mu, theta = model(counts)
            loss = negative_binomial_nll(counts, mu, theta)
            total_loss += float(loss.item()) * len(idx)
            total_cells += len(idx)
    return total_loss / max(total_cells, 1)


def main() -> None:
    import anndata as ad
    import pandas as pd
    from tqdm import tqdm

    args = parse_args()
    seed_all(args.seed)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is non-empty; refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    split = args.split_dir.expanduser().resolve()
    train_path = split / args.train_file
    val_path = split / args.validation_file
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing train/validation files under {split}")

    print("Loading TRAIN raw counts:", train_path)
    train = ad.read_h5ad(train_path)
    print("Loading VALIDATION raw counts:", val_path)
    validation = ad.read_h5ad(val_path)
    validate_raw(train, "train")
    validate_raw(validation, "validation")
    if set(train.obs_names) & set(validation.obs_names):
        raise RuntimeError("Train/validation barcode overlap detected")
    if not np.array_equal(train.var_names.astype(str), validation.var_names.astype(str)):
        raise RuntimeError("Train/validation gene order mismatch")

    hidden_dims = tuple(int(v) for v in args.hidden_dims.split(",") if v.strip())
    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))

    model = RNAEncoderDecoder(
        n_genes=train.n_vars,
        hidden_dims=hidden_dims,
        latent_dim=args.latent_dim,
        norm_type=args.norm_type,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print(model)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("TRAIN cells:", train.n_obs, "VALIDATION cells:", validation.n_obs)
    print("Genes:", train.n_vars)

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_epoch = -1
    best_path = output / "model.pt"
    rng = np.random.default_rng(args.seed + 100)

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        indices = np.arange(train.n_obs)
        rng.shuffle(indices)
        # Official loader uses drop_last=True.
        usable = (len(indices) // args.batch_size) * args.batch_size
        indices = indices[:usable]
        running = 0.0
        seen = 0
        progress = tqdm(
            range(0, usable, args.batch_size),
            desc=f"AE epoch {epoch}/{args.max_epochs}",
        )
        for start in progress:
            batch_idx = indices[start : start + args.batch_size]
            counts = torch.as_tensor(dense_batch(train.X, batch_idx), device=device)
            optimizer.zero_grad(set_to_none=True)
            _, mu, theta = model(counts)
            loss = negative_binomial_nll(counts, mu, theta)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite AE loss at epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            running += float(loss.item()) * len(batch_idx)
            seen += len(batch_idx)
            progress.set_postfix(nll=f"{float(loss.item()):.2f}")

        train_nll = running / max(seen, 1)
        should_eval = epoch % args.eval_every == 0 or epoch == args.max_epochs
        if should_eval:
            val_nll = validation_nll(model, validation.X, args.batch_size, device)
            row = {
                "epoch": epoch,
                "train_nll_per_cell": train_nll,
                "train_nll_per_gene": train_nll / train.n_vars,
                "validation_nll_per_cell": val_nll,
                "validation_nll_per_gene": val_nll / train.n_vars,
                "theta_q01": float(torch.quantile(torch.exp(model.log_theta.detach()).cpu(), 0.01)),
                "theta_q50": float(torch.quantile(torch.exp(model.log_theta.detach()).cpu(), 0.50)),
                "theta_q99": float(torch.quantile(torch.exp(model.log_theta.detach()).cpu(), 0.99)),
            }
            history.append(row)
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
            print(
                f"AE epoch={epoch} train_NLL/gene={row['train_nll_per_gene']:.6f} "
                f"val_NLL/gene={row['validation_nll_per_gene']:.6f}"
            )
            if val_nll < best_val:
                best_val = val_nll
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
                print(f"  new best AE checkpoint: epoch={epoch}")

        if epoch % args.checkpoint_every == 0 or epoch == args.max_epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                output / f"checkpoint_epoch_{epoch:04d}.pt",
            )

    if best_epoch < 0:
        raise RuntimeError("No validation checkpoint was selected")

    print("Reloading best AE:", best_path)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    print("Encoding TRAIN and VALIDATION with selected AE ...")
    train_latent = encode_matrix(model, train.X, args.encode_batch_size, device)
    val_latent = encode_matrix(model, validation.X, args.encode_batch_size, device)
    train_library = row_sums(train.X)
    val_library = row_sums(validation.X)

    np.save(output / "train_latent.npy", train_latent)
    np.save(output / "validation_latent.npy", val_latent)
    np.save(output / "train_library_size.npy", train_library)
    np.save(output / "validation_library_size.npy", val_library)
    np.save(output / "genes.npy", train.var_names.astype(str).to_numpy())

    metadata = {
        "model": "single-RNA Negative-Binomial autoencoder",
        "official_repo": OFFICIAL_REPO,
        "official_reference_commit": OFFICIAL_COMMIT,
        "n_genes": int(train.n_vars),
        "latent_dim": int(args.latent_dim),
        "hidden_dims": list(hidden_dims),
        "norm_type": args.norm_type,
        "activation": "ELU",
        "encoder_input": "log1p(raw_counts)",
        "decoder_mean": "library_size * softmax(decoder(latent))",
        "likelihood": "NegativeBinomial(mean=mu, inverse_dispersion=exp(log_theta))",
        "theta": "gene-specific",
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "gradient_clip": float(args.gradient_clip),
        "max_epochs": int(args.max_epochs),
        "best_epoch": int(best_epoch),
        "best_validation_nll_per_cell": float(best_val),
        "best_validation_nll_per_gene": float(best_val / train.n_vars),
        "train_n_obs": int(train.n_obs),
        "validation_n_obs": int(validation.n_obs),
        "gene_order_sha256": sha256_lines(train.var_names.astype(str)),
        "train_obs_sha256": sha256_lines(train.obs_names.astype(str)),
        "validation_obs_sha256": sha256_lines(validation.obs_names.astype(str)),
        "checkpoint": "model.pt",
        "final_test_loaded_during_training": False,
    }
    with (output / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print("Saved AE artifacts to:", output)
    print("Selected epoch:", best_epoch)
    print("Validation NLL/gene:", best_val / train.n_vars)
    gc.collect()


if __name__ == "__main__":
    main()

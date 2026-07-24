"""scDiffusion-style PyTorch autoencoder used by this experiment.

The network follows EperLuo/scDiffusion ``VAE/VAE_model.py``:

* encoder: 17,678 genes -> 1024 -> 1024 -> 1024 -> 128;
* BatchNorm1d + PReLU after each hidden linear layer;
* L2-normalized encoder output;
* symmetric decoder;
* MSE reconstruction loss and AdamW;
* ReLU is applied only when decoding generated latent vectors.

It is called ``VAE`` upstream, but its latent code is deterministic and there
is no KL term, so statistically it is an autoencoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from scipy import sparse
from torch import nn
from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: Sequence[int] = (1024, 1024, 1024),
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if residual and len(set(hidden_dim)) != 1:
            raise ValueError("Residual encoder requires equal hidden widths")
        self.residual = residual
        self.network = nn.ModuleList()
        for index, width in enumerate(hidden_dim):
            previous = n_genes if index == 0 else hidden_dim[index - 1]
            rate = input_dropout if index == 0 else dropout
            self.network.append(
                nn.Sequential(
                    nn.Dropout(p=rate),
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width),
                    nn.PReLU(),
                )
            )
        self.network.append(nn.Linear(hidden_dim[-1], latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.network):
            if self.residual and 0 < index < len(self.network) - 1:
                x = layer(x) + x
            else:
                x = layer(x)
        return F.normalize(x, p=2, dim=1)


class Decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: Sequence[int] = (1024, 1024, 1024),
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        if residual and len(set(hidden_dim)) != 1:
            raise ValueError("Residual decoder requires equal hidden widths")
        self.residual = residual
        self.network = nn.ModuleList()
        for index, width in enumerate(hidden_dim):
            previous = latent_dim if index == 0 else hidden_dim[index - 1]
            layers: list[nn.Module] = []
            if index > 0:
                layers.append(nn.Dropout(p=dropout))
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width),
                    nn.PReLU(),
                ]
            )
            self.network.append(nn.Sequential(*layers))
        self.network.append(nn.Linear(hidden_dim[-1], n_genes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.network):
            if self.residual and 0 < index < len(self.network) - 1:
                x = layer(x) + x
            else:
                x = layer(x)
        return x


class ScDiffusionAutoencoder(nn.Module):
    """Deterministic model matching scDiffusion's upstream ``VAE`` class."""

    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 128,
        hidden_dim: Sequence[int] = (1024, 1024, 1024),
    ) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = tuple(int(value) for value in hidden_dim)
        # These are the actual values hard-coded by scDiffusion's VAE class.
        self.encoder = Encoder(
            self.n_genes,
            self.latent_dim,
            self.hidden_dim,
            dropout=0.0,
            input_dropout=0.0,
            residual=False,
        )
        self.decoder = Decoder(
            self.n_genes,
            self.latent_dim,
            tuple(reversed(self.hidden_dim)),
            dropout=0.0,
            residual=False,
        )

    def forward(
        self,
        genes: torch.Tensor,
        return_latent: bool = False,
        return_decoded: bool = False,
    ) -> torch.Tensor:
        if return_decoded:
            return torch.relu(self.decoder(genes))
        latent = self.encoder(genes)
        if return_latent:
            return latent
        return self.decoder(latent)


def load_autoencoder(
    checkpoint: Path | str,
    n_genes: int,
    latent_dim: int = 128,
    hidden_dim: Sequence[int] = (1024, 1024, 1024),
    device: str | torch.device = "cpu",
) -> ScDiffusionAutoencoder:
    model = ScDiffusionAutoencoder(n_genes, latent_dim, hidden_dim)
    state = torch.load(Path(checkpoint), map_location=device)
    # Our files and the official scDiffusion checkpoints are direct state dicts.
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def encode_array(
    model: ScDiffusionAutoencoder,
    matrix,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    model.to(device).eval()
    chunks: list[np.ndarray] = []
    for start in range(0, matrix.shape[0], batch_size):
        values = matrix[start : start + batch_size]
        if hasattr(values, "toarray"):
            values = values.toarray()
        batch = torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)
        chunks.append(model(batch, return_latent=True).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def decode_array(
    model: ScDiffusionAutoencoder,
    latent: np.ndarray,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    model.to(device).eval()
    latent = np.asarray(latent, dtype=np.float32)
    latent = latent / np.maximum(np.linalg.norm(latent, axis=1, keepdims=True), 1e-8)
    chunks: list[np.ndarray] = []
    for start in range(0, len(latent), batch_size):
        batch = torch.as_tensor(latent[start : start + batch_size], device=device)
        chunks.append(model(batch, return_decoded=True).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the scDiffusion-style autoencoder"
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/autoencoder"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--min-steps", type=int, default=150_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


def _hash_lines(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def _normalize_h5ad(path: Path, target_sum: float):
    data = sc.read_h5ad(path)
    data.var_names_make_unique()
    sc.pp.normalize_total(data, target_sum=target_sum)
    sc.pp.log1p(data)
    if sparse.issparse(data.X):
        data.X = data.X.tocsr().astype(np.float32)
    else:
        data.X = np.asarray(data.X, dtype=np.float32)
    return data


@torch.no_grad()
def _validation_mse(model, matrix, batch_size: int, device: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, matrix.shape[0], batch_size):
        values = matrix[start : start + batch_size]
        if sparse.issparse(values):
            values = values.toarray()
        batch = torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)
        prediction = model(batch)
        total += float(torch.sum((prediction - batch) ** 2).cpu())
        count += batch.numel()
    return total / max(count, 1)


def train_shared_autoencoder(args: argparse.Namespace) -> None:
    """Fit the shared AE on TRAIN; use VALIDATION only for checkpoint selection."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "autoencoder.py training requires a CUDA-enabled PyTorch installation."
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    split = args.split_dir.expanduser()
    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    train_path = split / "pbmc68k_train_raw.h5ad"
    validation_path = split / "pbmc68k_validation_raw.h5ad"
    print("Loading TRAIN:", train_path)
    train = _normalize_h5ad(train_path, args.target_sum)
    print("Loading VALIDATION:", validation_path)
    validation = _normalize_h5ad(validation_path, args.target_sum)
    if set(train.obs_names) & set(validation.obs_names):
        raise RuntimeError("Train/validation overlap")
    if not np.array_equal(train.var_names, validation.var_names):
        raise RuntimeError("Train/validation gene order differs")

    device = "cuda"
    model = ScDiffusionAutoencoder(train.n_vars, args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("AE parameters:", sum(parameter.numel() for parameter in model.parameters()))

    rng = np.random.default_rng(args.seed)
    best_mse = float("inf")
    best_step = -1
    stale = 0
    history: list[dict[str, float]] = []
    best_path = output / "model.pt"

    progress = tqdm(range(1, args.max_steps + 1), desc="scDiffusion AE")
    for step in progress:
        indices = rng.integers(0, train.n_obs, size=args.batch_size)
        values = train.X[indices]
        if sparse.issparse(values):
            values = values.toarray()
        batch = torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)
        model.train()
        prediction = model(batch)
        loss = torch.mean((prediction - batch) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        progress.set_postfix(loss=f"{loss_value:.6f}")

        if step % args.checkpoint_every == 0 or step == args.max_steps:
            torch.save(model.state_dict(), output / f"checkpoint_{step}.pt")
        if step != 1 and step % args.eval_every != 0 and step != args.max_steps:
            continue

        value = _validation_mse(model, validation.X, args.batch_size, device)
        history.append(
            {"step": step, "train_batch_mse": loss_value, "validation_mse": value}
        )
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(f"AE step={step} validation_mse={value:.8f}")
        if value < best_mse - 1e-8:
            best_mse = value
            best_step = step
            stale = 0
            torch.save(model.state_dict(), best_path)
        else:
            stale += 1
        if step >= args.min_steps and stale >= args.patience:
            print("AE early stopping after minimum steps.")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    print(f"Selected AE step={best_step} validation_mse={best_mse:.8f}")
    print("Encoding TRAIN and VALIDATION with selected AE ...")
    train_latent = encode_array(model, train.X, 512, device)
    validation_latent = encode_array(model, validation.X, 512, device)
    np.save(output / "train_latent.npy", train_latent)
    np.save(output / "validation_latent.npy", validation_latent)
    metadata = {
        "framework": "pytorch",
        "upstream": "EperLuo/scDiffusion/VAE/VAE_model.py",
        "checkpoint": best_path.name,
        "best_step": best_step,
        "best_validation_mse": best_mse,
        "n_genes": train.n_vars,
        "latent_dim": args.latent_dim,
        "hidden_dim": [1024, 1024, 1024],
        "train_n_obs": train.n_obs,
        "validation_n_obs": validation.n_obs,
        "gene_order_sha256": _hash_lines(train.var_names.astype(str)),
        "train_obs_sha256": _hash_lines(train.obs_names.astype(str)),
        "validation_obs_sha256": _hash_lines(validation.obs_names.astype(str)),
    }
    with (output / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print("Saved AE artifacts to:", output)


if __name__ == "__main__":
    train_shared_autoencoder(parse_args())

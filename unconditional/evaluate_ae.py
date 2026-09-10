#!/usr/bin/env python3
"""Validation-only gate for the Negative-Binomial RNA autoencoder.

This script never opens the final-test split. It compares real validation cells
with two reconstructions from the frozen AE:

1. decoder mean (mu) -> normalize_total(1e4) -> log1p
2. Negative-Binomial sampled counts -> normalize_total(1e4) -> log1p

The sampled reconstruction is the relevant generative diagnostic because the
NB observation model, not its dense mean alone, is what can reproduce zeros.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/pbmc68k_ae_eval_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pbmc68k_ae_eval_matplotlib")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scib
from scipy import sparse
from sklearn.decomposition import PCA

from autoencoder import (
    decode_to_log_expression,
    load_ae_bundle,
    normalize_total_log1p_dense,
    row_sums,
    sha256_lines,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--validation-file", default="pbmc68k_validation_raw.h5ad")
    p.add_argument("--ae-artifact-dir", type=Path, default=Path("results/autoencoder"))
    p.add_argument("--output-dir", type=Path, default=Path("results/autoencoder_evaluation"))
    p.add_argument("--n-eval", type=int, default=2000)
    p.add_argument("--mmd-n-eval", type=int, default=500)
    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--decode-batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-matrices", action="store_true")
    return p.parse_args()


def dense_float32(x) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def rbf_mmd_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    total = np.concatenate([x, y], axis=0)
    distances = np.maximum(
        np.sum(total * total, axis=1)[:, None]
        + np.sum(total * total, axis=1)[None, :]
        - 2.0 * total @ total.T,
        0.0,
    )
    n_total = len(total)
    bandwidth = float(distances.sum()) / max(n_total * n_total - n_total, 1)
    bandwidth = max(bandwidth / (2.0 ** (5 // 2)), 1e-8)
    kernels = np.zeros_like(distances)
    for index in range(5):
        kernels += np.exp(-distances / (bandwidth * (2.0 ** index)))
    n = len(x)
    return float(
        kernels[:n, :n].mean()
        + kernels[n:, n:].mean()
        - kernels[:n, n:].mean()
        - kernels[n:, :n].mean()
    )


def distribution_metrics(
    real: np.ndarray,
    generated: np.ndarray,
    n_eval: int,
    mmd_n_eval: int,
    seed: int,
) -> dict[str, float]:
    """Match the existing unconditional BayesGM joint-PCA evaluation protocol."""
    n = min(int(n_eval), len(real), len(generated))
    rng = np.random.default_rng(seed)
    real_idx = rng.choice(len(real), n, replace=False)
    generated_idx = rng.choice(len(generated), n, replace=False)
    real_eval = np.asarray(real[real_idx], dtype=np.float32)
    generated_eval = np.asarray(generated[generated_idx], dtype=np.float32)
    joint = np.concatenate([real_eval, generated_eval], axis=0)
    n_components = min(50, joint.shape[0] - 1, joint.shape[1])
    pcs = PCA(n_components=n_components, random_state=seed).fit_transform(joint)

    graph = ad.AnnData(X=np.zeros((2 * n, 1), dtype=np.float32))
    graph.obsm["X_metric_pca20"] = pcs[:, : min(20, n_components)].astype(np.float32)
    graph.obs["batch"] = pd.Categorical(["real"] * n + ["generated"] * n)
    sc.pp.neighbors(graph, use_rep="X_metric_pca20", n_neighbors=10, random_state=seed)
    ilisi = float(scib.me.ilisi_graph(graph, batch_key="batch", type_="knn"))

    m = min(int(mmd_n_eval), n)
    mmd = rbf_mmd_np(
        pcs[:m, : min(50, n_components)],
        pcs[n : n + m, : min(50, n_components)],
    )
    real_mean = real_eval.mean(axis=0)
    generated_mean = generated_eval.mean(axis=0)
    return {
        "ilisi_pca20": ilisi,
        "mmd_pca50": mmd,
        "real_zero_frac": float(np.mean(real_eval <= 0)),
        "generated_zero_frac": float(np.mean(generated_eval <= 0)),
        "real_cell_sum_mean": float(real_eval.sum(axis=1).mean()),
        "generated_cell_sum_mean": float(generated_eval.sum(axis=1).mean()),
        "gene_mean_corr": float(np.corrcoef(real_mean, generated_mean)[0, 1]),
        "n_eval": int(n),
        "mmd_n_eval": int(m),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    validation_path = args.split_dir.expanduser().resolve() / args.validation_file
    if not validation_path.exists():
        raise FileNotFoundError(validation_path)

    validation = sc.read_h5ad(validation_path)
    validation.var_names_make_unique()
    values = validation.X.data if sparse.issparse(validation.X) else np.asarray(validation.X).ravel()
    if values.size and (not np.isfinite(values).all() or float(values.min()) < 0):
        raise ValueError("Validation matrix must contain raw non-negative counts")

    device = args.device
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            device = "cpu"
    bundle = load_ae_bundle(args.ae_artifact_dir, device=device)
    metadata = bundle.metadata

    expected = {
        "n_genes": int(validation.n_vars),
        "latent_dim": int(bundle.model.latent_dim),
        "validation_n_obs": int(validation.n_obs),
        "gene_order_sha256": sha256_lines(validation.var_names.astype(str)),
        "validation_obs_sha256": sha256_lines(validation.obs_names.astype(str)),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"AE/validation mismatch for {key}: stored={metadata.get(key)!r}, current={value!r}"
            )

    n = min(args.n_eval, validation.n_obs)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(validation.n_obs, n, replace=False)
    raw = dense_float32(validation.X[indices])
    real = normalize_total_log1p_dense(raw, args.target_sum)
    latent = bundle.validation_latent[indices]
    library = bundle.validation_library_size[indices]
    # Audit saved library sizes against the actual raw matrix.
    if not np.allclose(library, row_sums(validation.X)[indices], rtol=0, atol=1e-3):
        raise RuntimeError("Saved validation library sizes do not match raw validation counts")

    mean_recon = decode_to_log_expression(
        bundle.model,
        latent,
        library,
        mode="mean",
        target_sum=args.target_sum,
        batch_size=args.decode_batch_size,
        device=device,
        seed=args.seed,
    )

    rows: list[dict[str, object]] = []
    mean_metrics = distribution_metrics(real, mean_recon, n, args.mmd_n_eval, args.seed + 10)
    rows.append({"reconstruction": "NB_mean", "replicate": 0, **mean_metrics})

    sampled_matrices: list[np.ndarray] = []
    for replicate in range(args.replicates):
        sampled = decode_to_log_expression(
            bundle.model,
            latent,
            library,
            mode="sample",
            target_sum=args.target_sum,
            batch_size=args.decode_batch_size,
            device=device,
            seed=args.seed + 1000 * (replicate + 1),
        )
        sampled_matrices.append(sampled)
        metrics = distribution_metrics(
            real,
            sampled,
            n,
            args.mmd_n_eval,
            args.seed + 100 + replicate,
        )
        rows.append({"reconstruction": "NB_sample", "replicate": replicate + 1, **metrics})
        print(
            f"sample replicate={replicate + 1} "
            f"iLISI={metrics['ilisi_pca20']:.6f} "
            f"MMD50={metrics['mmd_pca50']:.6f} "
            f"zero={metrics['generated_zero_frac']:.6f} "
            f"corr={metrics['gene_mean_corr']:.6f}"
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)

    sampled_frame = frame[frame["reconstruction"] == "NB_sample"]
    summary = {
        "protocol": "validation-only AE gate; final test not loaded",
        "n_eval": int(n),
        "target_sum": float(args.target_sum),
        "real_zero_frac": float(np.mean(real <= 0)),
        "mean_reconstruction": {
            key: float(mean_metrics[key])
            for key in (
                "ilisi_pca20",
                "mmd_pca50",
                "generated_zero_frac",
                "generated_cell_sum_mean",
                "gene_mean_corr",
            )
        },
        "sampled_reconstruction_mean": {
            key: float(sampled_frame[key].mean())
            for key in (
                "ilisi_pca20",
                "mmd_pca50",
                "generated_zero_frac",
                "generated_cell_sum_mean",
                "gene_mean_corr",
            )
        },
        "sampled_reconstruction_sd": {
            key: float(sampled_frame[key].std(ddof=0))
            for key in (
                "ilisi_pca20",
                "mmd_pca50",
                "generated_zero_frac",
                "generated_cell_sum_mean",
                "gene_mean_corr",
            )
        },
        "ae_metadata": metadata,
        "final_test_loaded": False,
    }
    with (output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    if args.save_matrices:
        np.save(output / "validation_real_log_expression.npy", real)
        np.save(output / "validation_nb_mean_reconstruction.npy", mean_recon)
        for index, matrix in enumerate(sampled_matrices, 1):
            np.save(output / f"validation_nb_sample_reconstruction_rep{index}.npy", matrix)

    print("\n===== AUTOENCODER VALIDATION =====")
    print(f"Real zero fraction             : {summary['real_zero_frac']:.6f}")
    print(f"NB mean recon iLISI            : {mean_metrics['ilisi_pca20']:.6f}")
    print(f"NB mean recon zero fraction    : {mean_metrics['generated_zero_frac']:.6f}")
    print(
        "NB sampled recon iLISI mean±sd: "
        f"{summary['sampled_reconstruction_mean']['ilisi_pca20']:.6f} ± "
        f"{summary['sampled_reconstruction_sd']['ilisi_pca20']:.6f}"
    )
    print(
        "NB sampled recon zero mean±sd : "
        f"{summary['sampled_reconstruction_mean']['generated_zero_frac']:.6f} ± "
        f"{summary['sampled_reconstruction_sd']['generated_zero_frac']:.6f}"
    )
    print("Saved:", output / "metrics.csv")
    print("Saved:", output / "summary.json")


if __name__ == "__main__":
    main()

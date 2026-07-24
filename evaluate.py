#!/usr/bin/env python3
"""Final-test evaluation helpers for the leakage-safe PBMC68k experiment.

Import this module from ``evaluate.ipynb``. It never trains or selects a model.
All quantitative metrics start from full-gene log-normalized expression.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scib
from scipy import sparse
from sklearn.decomposition import PCA


def sha256_lines(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def dense_float32(x) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def normalize_log1p(adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
    result = adata.copy()
    sc.pp.normalize_total(result, target_sum=target_sum)
    sc.pp.log1p(result)
    return result


def align_test_to_genes(test: ad.AnnData, gene_names: np.ndarray) -> ad.AnnData:
    gene_names = np.asarray(gene_names).astype(str)
    test.var_names_make_unique()
    missing = np.setdiff1d(gene_names, test.var_names)
    if len(missing):
        raise ValueError(f"Final test is missing {len(missing)} generated genes; first={missing[:5]}")
    return test[:, gene_names].copy()


def load_final_test_and_generated(
    split_dir: Path,
    result_dir: Path,
    target_sum: float = 1e4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ad.AnnData]:
    split_dir = Path(split_dir).expanduser()
    result_dir = Path(result_dir).expanduser()
    test_path = split_dir / "pbmc68k_test_raw.h5ad"
    generated_path = result_dir / "generated.npy"
    genes_path = result_dir / "genes.npy"
    for path in (test_path, generated_path, genes_path):
        if not path.exists():
            raise FileNotFoundError(path)

    genes = np.load(genes_path, allow_pickle=False).astype(str)
    test_raw = align_test_to_genes(sc.read_h5ad(test_path), genes)
    test_log = normalize_log1p(test_raw, target_sum=target_sum)
    real = dense_float32(test_log.X)
    generated = np.load(generated_path, allow_pickle=False).astype(np.float32)
    if generated.ndim != 2 or generated.shape[1] != len(genes):
        raise ValueError(
            f"Generated shape {generated.shape} is incompatible with {len(genes)} genes"
        )
    if not np.isfinite(generated).all() or float(generated.min()) < 0:
        raise ValueError("Generated expression must be finite and non-negative")
    return real, generated, genes, test_raw


def rbf_mmd(x: np.ndarray, y: np.ndarray) -> float:
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


def compute_quantitative_metrics(
    real: np.ndarray,
    generated: np.ndarray,
    n_eval: int = 2_000,
    mmd_n_eval: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Compute the same joint-PCA iLISI/MMD used during validation."""
    n = min(n_eval, len(real), len(generated))
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

    m = min(mmd_n_eval, n)
    mmd = rbf_mmd(pcs[:m, : min(50, n_components)], pcs[n : n + m, : min(50, n_components)])
    real_mean = real_eval.mean(axis=0)
    generated_mean = generated_eval.mean(axis=0)
    return {
        "test_ilisi_pca20": ilisi,
        "test_mmd_pca50": mmd,
        "test_real_zero_frac": float(np.mean(real_eval <= 0)),
        "generated_zero_frac": float(np.mean(generated_eval <= 0)),
        "test_real_cell_sum_mean": float(real_eval.sum(axis=1).mean()),
        "generated_cell_sum_mean": float(generated_eval.sum(axis=1).mean()),
        "test_gene_mean_corr": float(np.corrcoef(real_mean, generated_mean)[0, 1]),
        "test_cells_used": int(n),
        "mmd_cells_used": int(m),
        "evaluation_seed": int(seed),
    }


def save_metrics(metrics: Mapping[str, float], result_dir: Path) -> None:
    result_dir = Path(result_dir)
    pd.Series(metrics, name="value").to_csv(result_dir / "metrics.csv")
    with (result_dir / "metrics.json").open("w") as handle:
        json.dump(dict(metrics), handle, indent=2)


def audit_protocol(split_dir: Path, result_dir: Path) -> dict:
    split_dir = Path(split_dir).expanduser()
    result_dir = Path(result_dir).expanduser()
    manifest_path = split_dir / "split_manifest.json"
    config_path = result_dir / "config.json"
    if not manifest_path.exists() or not config_path.exists():
        raise FileNotFoundError("split_manifest.json and config.json are required")
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    with config_path.open() as handle:
        config = json.load(handle)

    if config.get("final_test_loaded_during_training") is not False:
        raise RuntimeError("Training config does not certify that final test was untouched")
    if manifest["gene_order_sha256"] != config["gene_order_sha256"]:
        raise RuntimeError("Training and split gene hashes differ")
    split_table = pd.read_csv(split_dir / "cell_split.csv")
    groups = {
        name: set(split_table.loc[split_table["split"] == name, "barcode"].astype(str))
        for name in ("train", "validation", "test")
    }
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise RuntimeError("Barcode overlap detected")
    return {"manifest": manifest, "run_config": config}


def prepare_embedding(
    real: np.ndarray,
    generated: np.ndarray,
    n_plot: int = 2_000,
    seed: int = 42,
) -> dict:
    """Fit PCA and UMAP on real test cells, then transform generated cells."""
    import umap

    rng = np.random.default_rng(seed)
    real_idx = rng.choice(len(real), min(n_plot, len(real)), replace=False)
    generated_idx = rng.choice(
        len(generated), min(n_plot, len(generated)), replace=False
    )
    real_plot = np.asarray(real[real_idx], dtype=np.float32)
    generated_plot = np.asarray(generated[generated_idx], dtype=np.float32)

    pca = PCA(n_components=min(50, len(real_plot) - 1, real_plot.shape[1]), random_state=seed)
    real_pca = pca.fit_transform(real_plot)
    generated_pca = pca.transform(generated_plot)

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.3,
        metric="euclidean",
        random_state=seed,
        transform_seed=seed,
    )
    real_umap = reducer.fit_transform(real_pca[:, :20])
    generated_umap = reducer.transform(generated_pca[:, :20])
    return {
        "real_indices": real_idx,
        "generated_indices": generated_idx,
        "real_pca": real_pca,
        "generated_pca": generated_pca,
        "real_umap": real_umap,
        "generated_umap": generated_umap,
    }


def plot_umap(embedding: dict, plot_dir: Path) -> Path:
    """Plot real cells in gray and generated cells in orange."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    real_xy = embedding["real_umap"]
    generated_xy = embedding["generated_umap"]
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        real_xy[:, 0],
        real_xy[:, 1],
        s=8,
        color="#C7CDD3",
        alpha=0.45,
        linewidths=0,
        rasterized=True,
        label="Real test",
    )
    axis.scatter(
        generated_xy[:, 0],
        generated_xy[:, 1],
        s=10,
        color="#E87722",
        alpha=0.80,
        linewidths=0,
        rasterized=True,
        label="BayesGM",
    )
    axis.set_title("BayesGM")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(frameon=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    path = plot_dir / "umap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path

#!/usr/bin/env python3
"""Standalone evaluation for conditional scRNA-seq generation.

The evaluation separates four questions:

1. Condition fidelity: an independent KNN classifier is fitted only on REAL
   TRAIN cells, checked on held-out REAL EVAL cells, and then applied to all
   generated cells.
2. Within-condition realism: for each requested condition, a 5-NN classifier
   tries to distinguish REAL EVAL from generated cells. ROC AUC should approach
   0.5. An optional cscGAN-style random-forest AUC is also available.
3. Distribution similarity: per-condition multiscale RBF MMD is computed in a
   PCA space fitted only on REAL TRAIN cells.
4. Marker fidelity: optional cell-type-specific marker distributions are compared
   using KL, symmetric KL, JS, Wasserstein distance, zero-rate error, and mean
   expression error.

No local project module is imported, so this script can be run from any path.
Generated ``obs[label_key]`` is interpreted as the REQUESTED condition and is
never used as an independent prediction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--real-train", type=Path, required=True)
    parser.add_argument(
        "--real-eval",
        type=Path,
        required=True,
        help="Validation during development; final test only after all settings are frozen.",
    )
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--label-key", default="celltype")
    parser.add_argument(
        "--generated-label-key",
        default=None,
        help=(
            "Explicit generated-label column override. If omitted, the generated "
            "file must contain exactly obs[label_key]; no fallback is attempted."
        ),
    )
    parser.add_argument("--target-sum", type=float, default=1e4)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--classifier-max-train", type=int, default=30_000)
    parser.add_argument("--label-knn-neighbors", type=int, default=15)
    parser.add_argument("--auc-max-per-origin", type=int, default=1000)
    parser.add_argument("--auc-repeats", type=int, default=20)
    parser.add_argument("--auc-min-per-origin", type=int, default=20)
    parser.add_argument("--mmd-max-per-origin", type=int, default=1000)
    parser.add_argument("--skip-random-forest", action="store_true")
    parser.add_argument("--rf-estimators", type=int, default=1000)
    parser.add_argument("--rf-max-depth", type=int, default=5)
    parser.add_argument("--rf-folds", type=int, default=5)
    parser.add_argument(
        "--markers",
        type=Path,
        default=None,
        help="CSV from select_markers.py or a manual CSV with celltype,gene columns.",
    )
    parser.add_argument("--marker-celltype-column", default="celltype")
    parser.add_argument("--marker-gene-column", default="gene")
    parser.add_argument("--marker-positive-bins", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/conditional_evaluation"),
    )
    return parser.parse_args()


def natural_order(values: Iterable[str]) -> list[str]:
    def key(value: str):
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted(set(map(str, values)), key=key)


def check_finite_nonnegative(data: ad.AnnData, name: str) -> None:
    values = data.X.data if sparse.issparse(data.X) else np.asarray(data.X).ravel()
    if values.size and (not np.isfinite(values).all() or float(values.min()) < 0):
        raise ValueError(f"{name} expression must be finite and non-negative")


def load_real(path: Path, target_sum: float) -> ad.AnnData:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    data = sc.read_h5ad(path)
    data.obs_names = data.obs_names.astype(str)
    data.var_names = data.var_names.astype(str)
    data.var_names_make_unique()
    check_finite_nonnegative(data, str(path))
    sc.pp.normalize_total(data, target_sum=float(target_sum))
    sc.pp.log1p(data)
    if sparse.issparse(data.X):
        data.X = data.X.tocsr().astype(np.float32)
    else:
        data.X = np.asarray(data.X, dtype=np.float32)
    return data


def load_generated(path: Path) -> ad.AnnData:
    """Load generated expression already stored as normalize_total+log1p values.

    Conditional NB generation writes normalized/log1p expression to ``X`` and
    retains sampled UMI counts separately.  Normalizing ``X`` again here would
    distort the generated distribution, so this loader intentionally performs
    no expression transform.
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    data = sc.read_h5ad(path)
    data.obs_names = data.obs_names.astype(str)
    data.var_names = data.var_names.astype(str)
    data.var_names_make_unique()
    check_finite_nonnegative(data, str(path))
    if sparse.issparse(data.X):
        data.X = data.X.tocsr().astype(np.float32)
    else:
        data.X = np.asarray(data.X, dtype=np.float32)
    return data


def align_genes(
    train: ad.AnnData, evaluation: ad.AnnData, generated: ad.AnnData
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    if not train.var_names.equals(evaluation.var_names):
        missing = train.var_names.difference(evaluation.var_names)
        extra = evaluation.var_names.difference(train.var_names)
        if len(missing) or len(extra):
            raise ValueError(
                f"REAL TRAIN/EVAL genes differ: missing={len(missing)}, extra={len(extra)}"
            )
        evaluation = evaluation[:, train.var_names].copy()

    missing_generated = train.var_names.difference(generated.var_names)
    if len(missing_generated):
        raise ValueError(
            f"Generated data is missing {len(missing_generated)} genes; "
            f"examples={missing_generated[:10].tolist()}"
        )
    if not generated.var_names.equals(train.var_names):
        generated = generated[:, train.var_names].copy()
    return train, evaluation, generated


def choose_generated_label_key(
    data: ad.AnnData, requested: str, explicit: str | None
) -> str:
    """Resolve the generated label column without guessing or silent fallback."""
    key = requested if explicit is None else explicit
    if key not in data.obs.columns:
        qualifier = "requested label key" if explicit is None else "explicit override"
        raise KeyError(
            f"Generated data are missing obs[{key!r}] ({qualifier}). "
            f"Available={data.obs.columns.tolist()}"
        )
    return key


def validate_labels(
    train: ad.AnnData,
    evaluation: ad.AnnData,
    generated: ad.AnnData,
    label_key: str,
    generated_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    for name, data, key in (
        ("real train", train, label_key),
        ("real eval", evaluation, label_key),
        ("generated", generated, generated_key),
    ):
        if key not in data.obs.columns:
            raise KeyError(f"{name} has no obs[{key!r}]")
        if data.obs[key].isna().any():
            raise ValueError(f"{name}.obs[{key!r}] contains missing labels")

    train_y = train.obs[label_key].astype(str).to_numpy()
    eval_y = evaluation.obs[label_key].astype(str).to_numpy()
    gen_y = generated.obs[generated_key].astype(str).to_numpy()
    train_classes = set(train_y)
    unseen_eval = set(eval_y) - train_classes
    unseen_gen = set(gen_y) - train_classes
    if unseen_eval:
        raise ValueError(f"REAL EVAL contains classes absent from TRAIN: {sorted(unseen_eval)}")
    if unseen_gen:
        raise ValueError(f"Generated requested classes absent from TRAIN: {sorted(unseen_gen)}")
    shared = natural_order(set(eval_y).intersection(set(gen_y)))
    if not shared:
        raise RuntimeError("REAL EVAL and generated data share no condition labels")
    return train_y, eval_y, gen_y, shared


def stratified_sample_indices(
    labels: np.ndarray, maximum: int, rng: np.random.Generator
) -> np.ndarray:
    if maximum <= 0 or len(labels) <= maximum:
        return np.arange(len(labels), dtype=np.int64)
    classes = natural_order(np.unique(labels).tolist())
    per_class = max(1, int(math.ceil(maximum / len(classes))))
    parts: list[np.ndarray] = []
    for label in classes:
        candidates = np.flatnonzero(labels == label)
        parts.append(
            rng.choice(candidates, size=min(per_class, len(candidates)), replace=False)
        )
    result = np.concatenate(parts)
    if len(result) > maximum:
        result = rng.choice(result, size=maximum, replace=False)
    rng.shuffle(result)
    return result.astype(np.int64)


def dense_rows_columns(matrix, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    values = matrix[rows][:, columns]
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def dense_columns(matrix, columns: np.ndarray) -> np.ndarray:
    values = matrix[:, columns]
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def fit_real_train_reference(
    train: ad.AnnData,
    train_y: np.ndarray,
    n_hvg: int,
    n_components: int,
    max_train: int,
    seed: int,
) -> tuple[np.ndarray, StandardScaler, PCA, np.ndarray, np.ndarray]:
    n_hvg = min(int(n_hvg), train.n_vars)
    sc.pp.highly_variable_genes(
        train,
        n_top_genes=n_hvg,
        flavor="seurat",
        subset=False,
    )
    hvg_mask = train.var["highly_variable"].to_numpy(dtype=bool)
    hvg_indices = np.flatnonzero(hvg_mask)
    if len(hvg_indices) < 2:
        raise RuntimeError("Fewer than two HVGs were selected")

    rng = np.random.default_rng(seed)
    rows = stratified_sample_indices(train_y, int(max_train), rng)
    x_train = dense_rows_columns(train.X, rows, hvg_indices)

    scaler = StandardScaler(copy=False)
    x_train = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    n_components = min(int(n_components), x_train.shape[0] - 1, x_train.shape[1])
    if n_components < 2:
        raise RuntimeError("Not enough training cells/features for PCA")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    train_pca = pca.fit_transform(x_train).astype(np.float32)
    return hvg_indices, scaler, pca, rows, train_pca


def transform_reference_space(
    data: ad.AnnData,
    hvg_indices: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    values = dense_columns(data.X, hvg_indices)
    values = scaler.transform(values).astype(np.float32, copy=False)
    return pca.transform(values).astype(np.float32)


def save_confusion(
    truth: np.ndarray,
    prediction: np.ndarray,
    classes: Sequence[str],
    path: Path,
) -> None:
    matrix = confusion_matrix(truth, prediction, labels=list(classes))
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(path)


def condition_fidelity(
    train_pca: np.ndarray,
    sampled_train_y: np.ndarray,
    eval_pca: np.ndarray,
    eval_y: np.ndarray,
    gen_pca: np.ndarray,
    gen_y: np.ndarray,
    classes: Sequence[str],
    neighbors: int,
    output: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    k = min(int(neighbors), len(train_pca))
    classifier = KNeighborsClassifier(n_neighbors=max(1, k), weights="distance", n_jobs=-1)
    classifier.fit(train_pca, sampled_train_y)
    eval_pred = classifier.predict(eval_pca)
    gen_pred = classifier.predict(gen_pca)

    save_confusion(eval_y, eval_pred, classes, output / "condition_confusion_real_eval.csv")
    save_confusion(gen_y, gen_pred, classes, output / "condition_confusion_generated.csv")

    rows: list[dict[str, object]] = []
    for label in classes:
        eval_mask = eval_y == label
        gen_mask = gen_y == label
        rows.append(
            {
                "celltype": label,
                "n_real_eval": int(eval_mask.sum()),
                "n_generated": int(gen_mask.sum()),
                "real_eval_recall": (
                    float(np.mean(eval_pred[eval_mask] == label)) if eval_mask.any() else math.nan
                ),
                "generated_requested_recall": (
                    float(np.mean(gen_pred[gen_mask] == label)) if gen_mask.any() else math.nan
                ),
            }
        )
    per_class = pd.DataFrame(rows)
    summary = {
        "real_eval_accuracy": float(accuracy_score(eval_y, eval_pred)),
        "real_eval_balanced_accuracy": float(balanced_accuracy_score(eval_y, eval_pred)),
        "real_eval_macro_f1": float(f1_score(eval_y, eval_pred, average="macro", zero_division=0)),
        "generated_requested_accuracy": float(accuracy_score(gen_y, gen_pred)),
        "generated_requested_balanced_accuracy": float(
            balanced_accuracy_score(gen_y, gen_pred)
        ),
        "generated_requested_macro_f1": float(
            f1_score(gen_y, gen_pred, average="macro", zero_division=0)
        ),
    }
    return per_class, summary


def rbf_mmd(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    total = np.concatenate([x, y], axis=0)
    squared = np.sum(total * total, axis=1)
    distances = np.maximum(
        squared[:, None] + squared[None, :] - 2.0 * total @ total.T,
        0.0,
    )
    n_total = len(total)
    bandwidth = float(distances.sum()) / max(n_total * n_total - n_total, 1)
    bandwidth = max(bandwidth / (2.0 ** (5 // 2)), 1e-8)
    kernel = np.zeros_like(distances)
    for index in range(5):
        kernel += np.exp(-distances / (bandwidth * (2.0 ** index)))
    n_x = len(x)
    return float(
        kernel[:n_x, :n_x].mean()
        + kernel[n_x:, n_x:].mean()
        - kernel[:n_x, n_x:].mean()
        - kernel[n_x:, :n_x].mean()
    )


def sample_equal_indices(
    labels: np.ndarray,
    label: str,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = np.flatnonzero(labels == label)
    if count > len(candidates):
        raise ValueError("Requested sample exceeds candidates")
    return rng.choice(candidates, size=count, replace=False)


def knn_real_generated_auc(
    eval_pca: np.ndarray,
    gen_pca: np.ndarray,
    eval_y: np.ndarray,
    gen_y: np.ndarray,
    classes: Sequence[str],
    max_per_origin: int,
    repeats: int,
    min_per_origin: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for class_index, label in enumerate(classes):
        n = min(
            int(max_per_origin),
            int(np.sum(eval_y == label)),
            int(np.sum(gen_y == label)),
        )
        if n < int(min_per_origin):
            summary.append(
                {
                    "celltype": label,
                    "n_each": n,
                    "knn_auc_mean": math.nan,
                    "knn_auc_sd": math.nan,
                    "mean_abs_auc_minus_0.5": math.nan,
                    "eligible": False,
                }
            )
            continue
        aucs: list[float] = []
        for repeat in range(int(repeats)):
            rng = np.random.default_rng(seed + 10_000 * class_index + repeat)
            real_idx = sample_equal_indices(eval_y, label, n, rng)
            gen_idx = sample_equal_indices(gen_y, label, n, rng)
            values = np.concatenate([eval_pca[real_idx], gen_pca[gen_idx]], axis=0)
            # real=1 and generated=0, following a real-vs-generated discriminator.
            target = np.concatenate(
                [np.ones(n, dtype=np.int32), np.zeros(n, dtype=np.int32)]
            )
            train_x, test_x, train_t, test_t = train_test_split(
                values,
                target,
                test_size=0.30,
                stratify=target,
                random_state=seed + 1000 * class_index + repeat,
            )
            k = min(5, len(train_x))
            model = KNeighborsClassifier(n_neighbors=max(1, k), n_jobs=-1)
            model.fit(train_x, train_t)
            auc = float(roc_auc_score(test_t, model.predict_proba(test_x)[:, 1]))
            aucs.append(auc)
            detail.append(
                {
                    "celltype": label,
                    "repeat": repeat,
                    "n_each": n,
                    "knn_auc": auc,
                    "abs_auc_minus_0.5": abs(auc - 0.5),
                }
            )
        summary.append(
            {
                "celltype": label,
                "n_each": n,
                "knn_auc_mean": float(np.mean(aucs)),
                "knn_auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                "mean_abs_auc_minus_0.5": float(np.mean(np.abs(np.asarray(aucs) - 0.5))),
                "eligible": True,
            }
        )
    return pd.DataFrame(summary), pd.DataFrame(detail)


def rf_real_generated_auc(
    eval_pca: np.ndarray,
    gen_pca: np.ndarray,
    eval_y: np.ndarray,
    gen_y: np.ndarray,
    classes: Sequence[str],
    max_per_origin: int,
    min_per_origin: int,
    estimators: int,
    max_depth: int,
    folds: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for class_index, label in enumerate(classes):
        n = min(
            int(max_per_origin),
            int(np.sum(eval_y == label)),
            int(np.sum(gen_y == label)),
        )
        if n < max(int(min_per_origin), int(folds)):
            rows.append(
                {
                    "celltype": label,
                    "n_each": n,
                    "rf_auc_mean": math.nan,
                    "rf_auc_sd": math.nan,
                    "mean_abs_auc_minus_0.5": math.nan,
                    "eligible": False,
                }
            )
            continue
        rng = np.random.default_rng(seed + 20_000 + class_index)
        real_idx = sample_equal_indices(eval_y, label, n, rng)
        gen_idx = sample_equal_indices(gen_y, label, n, rng)
        values = np.concatenate([eval_pca[real_idx], gen_pca[gen_idx]], axis=0)
        target = np.concatenate(
            [np.ones(n, dtype=np.int32), np.zeros(n, dtype=np.int32)]
        )
        splitter = StratifiedKFold(
            n_splits=int(folds), shuffle=True, random_state=seed + class_index
        )
        aucs: list[float] = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(values, target)):
            model = RandomForestClassifier(
                n_estimators=int(estimators),
                max_depth=int(max_depth),
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed + 100 * class_index + fold,
            )
            model.fit(values[train_idx], target[train_idx])
            aucs.append(
                float(
                    roc_auc_score(
                        target[test_idx], model.predict_proba(values[test_idx])[:, 1]
                    )
                )
            )
        rows.append(
            {
                "celltype": label,
                "n_each": n,
                "rf_auc_mean": float(np.mean(aucs)),
                "rf_auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                "mean_abs_auc_minus_0.5": float(np.mean(np.abs(np.asarray(aucs) - 0.5))),
                "eligible": True,
            }
        )
    return pd.DataFrame(rows)


def mmd_by_celltype(
    eval_pca: np.ndarray,
    gen_pca: np.ndarray,
    eval_y: np.ndarray,
    gen_y: np.ndarray,
    classes: Sequence[str],
    maximum: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, label in enumerate(classes):
        n = min(int(maximum), int(np.sum(eval_y == label)), int(np.sum(gen_y == label)))
        if n < 2:
            rows.append({"celltype": label, "n_each": n, "mmd_pca": math.nan})
            continue
        rng = np.random.default_rng(seed + 30_000 + index)
        real_idx = sample_equal_indices(eval_y, label, n, rng)
        gen_idx = sample_equal_indices(gen_y, label, n, rng)
        rows.append(
            {
                "celltype": label,
                "n_each": n,
                "mmd_pca": rbf_mmd(eval_pca[real_idx], gen_pca[gen_idx]),
            }
        )
    return pd.DataFrame(rows)


def extract_gene_values(data: ad.AnnData, rows: np.ndarray, gene_index: int) -> np.ndarray:
    values = data.X[rows, gene_index]
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def reference_bin_edges(values: np.ndarray, positive_bins: int) -> np.ndarray:
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if len(positive) == 0:
        return np.asarray([0.0, 1.0], dtype=float)
    quantiles = np.linspace(0.0, 1.0, int(positive_bins) + 1)
    edges = np.unique(np.quantile(positive, quantiles))
    if len(edges) < 2:
        value = max(float(edges[0]), 1e-6)
        edges = np.asarray([0.0, value], dtype=float)
    edges[0] = max(0.0, float(edges[0]))
    return edges


def histogram_probabilities(values: np.ndarray, positive_edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    positive_bins = len(positive_edges) - 1
    counts = np.zeros(positive_bins + 1, dtype=np.float64)  # first bin is exact zero
    counts[0] = np.sum(values <= 0)
    positive = values[values > 0]
    if len(positive):
        ids = np.searchsorted(positive_edges[1:-1], positive, side="right")
        ids = np.clip(ids, 0, positive_bins - 1)
        counts[1:] = np.bincount(ids, minlength=positive_bins)
    epsilon = 1e-8
    probabilities = counts + epsilon
    return probabilities / probabilities.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def marker_metrics(
    train: ad.AnnData,
    evaluation: ad.AnnData,
    generated: ad.AnnData,
    train_y: np.ndarray,
    eval_y: np.ndarray,
    gen_y: np.ndarray,
    markers_path: Path,
    celltype_column: str,
    gene_column: str,
    positive_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    markers = pd.read_csv(markers_path.expanduser().resolve())
    if gene_column not in markers.columns and "names" in markers.columns:
        markers = markers.rename(columns={"names": gene_column})
    required = {celltype_column, gene_column}
    missing = required - set(markers.columns)
    if missing:
        raise KeyError(f"Marker CSV is missing columns: {sorted(missing)}")

    gene_to_index = {str(gene): index for index, gene in enumerate(train.var_names)}
    rows: list[dict[str, object]] = []
    for _, marker in markers.iterrows():
        label = str(marker[celltype_column])
        gene = str(marker[gene_column])
        if gene not in gene_to_index:
            rows.append(
                {
                    "celltype": label,
                    "gene": gene,
                    "available": False,
                    "reason": "gene_absent_from_model_output",
                }
            )
            continue
        train_rows = np.flatnonzero(train_y == label)
        eval_rows = np.flatnonzero(eval_y == label)
        gen_rows = np.flatnonzero(gen_y == label)
        if len(train_rows) == 0 or len(eval_rows) == 0 or len(gen_rows) == 0:
            rows.append(
                {
                    "celltype": label,
                    "gene": gene,
                    "available": False,
                    "reason": "celltype_missing_from_train_eval_or_generated",
                }
            )
            continue
        gene_index = gene_to_index[gene]
        reference = extract_gene_values(train, train_rows, gene_index)
        real = extract_gene_values(evaluation, eval_rows, gene_index)
        fake = extract_gene_values(generated, gen_rows, gene_index)
        edges = reference_bin_edges(reference, positive_bins)
        p = histogram_probabilities(real, edges)
        q = histogram_probabilities(fake, edges)
        midpoint = 0.5 * (p + q)
        kl_real_gen = kl_divergence(p, q)
        kl_gen_real = kl_divergence(q, p)
        js = 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)
        rows.append(
            {
                "celltype": label,
                "gene": gene,
                "available": True,
                "reason": "",
                "n_real_eval": len(real),
                "n_generated": len(fake),
                "kl_real_to_generated": kl_real_gen,
                "kl_generated_to_real": kl_gen_real,
                "symmetric_kl": 0.5 * (kl_real_gen + kl_gen_real),
                "js_divergence": float(js),
                "wasserstein": float(wasserstein_distance(real, fake)),
                "real_zero_rate": float(np.mean(real <= 0)),
                "generated_zero_rate": float(np.mean(fake <= 0)),
                "absolute_zero_rate_error": float(
                    abs(np.mean(real <= 0) - np.mean(fake <= 0))
                ),
                "real_mean": float(np.mean(real)),
                "generated_mean": float(np.mean(fake)),
                "absolute_mean_error": float(abs(np.mean(real) - np.mean(fake))),
                "positive_histogram_bins": int(len(edges) - 1),
            }
        )

    detail = pd.DataFrame(rows)
    eligible = detail.loc[detail.get("available", False) == True].copy()  # noqa: E712
    metric_columns = [
        "kl_real_to_generated",
        "symmetric_kl",
        "js_divergence",
        "wasserstein",
        "absolute_zero_rate_error",
        "absolute_mean_error",
    ]
    if len(eligible):
        by_celltype = (
            eligible.groupby("celltype", as_index=False)[metric_columns]
            .mean()
            .merge(
                eligible.groupby("celltype").size().rename("n_markers").reset_index(),
                on="celltype",
            )
        )
        macro = {
            f"marker_macro_{column}": float(by_celltype[column].mean())
            for column in metric_columns
        }
        macro["marker_celltypes_evaluated"] = int(len(by_celltype))
        macro["marker_genes_evaluated"] = int(len(eligible))
    else:
        by_celltype = pd.DataFrame(columns=["celltype", "n_markers", *metric_columns])
        macro = {f"marker_macro_{column}": math.nan for column in metric_columns}
        macro["marker_celltypes_evaluated"] = 0
        macro["marker_genes_evaluated"] = 0
    macro["marker_genes_unavailable"] = int(len(detail) - len(eligible))
    return detail, by_celltype, macro


def finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("Loading and normalizing REAL TRAIN ...")
    train = load_real(args.real_train, args.target_sum)
    print("Loading and normalizing REAL EVAL ...")
    evaluation = load_real(args.real_eval, args.target_sum)
    print("Loading GENERATED ...")
    generated = load_generated(args.generated)
    train, evaluation, generated = align_genes(train, evaluation, generated)

    generated_key = choose_generated_label_key(
        generated, args.label_key, args.generated_label_key
    )
    train_y, eval_y, gen_y, classes = validate_labels(
        train, evaluation, generated, args.label_key, generated_key
    )
    print("Shared conditions:", classes)

    hvg_indices, scaler, pca, train_rows, train_pca = fit_real_train_reference(
        train,
        train_y,
        args.n_hvg,
        args.pca_components,
        args.classifier_max_train,
        args.seed,
    )
    np.save(output / "reference_hvg_indices.npy", hvg_indices)
    pd.Series(train.var_names[hvg_indices]).to_csv(
        output / "reference_hvg_genes.txt", index=False, header=False
    )
    eval_pca = transform_reference_space(evaluation, hvg_indices, scaler, pca)
    gen_pca = transform_reference_space(generated, hvg_indices, scaler, pca)
    np.save(output / "real_eval_pca.npy", eval_pca)
    np.save(output / "generated_pca.npy", gen_pca)

    fidelity_table, fidelity_summary = condition_fidelity(
        train_pca,
        train_y[train_rows],
        eval_pca,
        eval_y,
        gen_pca,
        gen_y,
        classes,
        args.label_knn_neighbors,
        output,
    )
    fidelity_table.to_csv(output / "condition_fidelity_by_celltype.csv", index=False)

    knn_auc, knn_auc_repeats = knn_real_generated_auc(
        eval_pca,
        gen_pca,
        eval_y,
        gen_y,
        classes,
        args.auc_max_per_origin,
        args.auc_repeats,
        args.auc_min_per_origin,
        args.seed,
    )
    knn_auc.to_csv(output / "knn_real_vs_generated_auc_by_celltype.csv", index=False)
    knn_auc_repeats.to_csv(output / "knn_real_vs_generated_auc_repeats.csv", index=False)

    mmd = mmd_by_celltype(
        eval_pca,
        gen_pca,
        eval_y,
        gen_y,
        classes,
        args.mmd_max_per_origin,
        args.seed,
    )
    mmd.to_csv(output / "mmd_by_celltype.csv", index=False)

    rf = None
    if not args.skip_random_forest:
        rf = rf_real_generated_auc(
            eval_pca,
            gen_pca,
            eval_y,
            gen_y,
            classes,
            args.auc_max_per_origin,
            args.auc_min_per_origin,
            args.rf_estimators,
            args.rf_max_depth,
            args.rf_folds,
            args.seed,
        )
        rf.to_csv(output / "rf_real_vs_generated_auc_by_celltype.csv", index=False)

    marker_summary: dict[str, float] = {}
    if args.markers is not None:
        detail, by_celltype, marker_summary = marker_metrics(
            train,
            evaluation,
            generated,
            train_y,
            eval_y,
            gen_y,
            args.markers,
            args.marker_celltype_column,
            args.marker_gene_column,
            args.marker_positive_bins,
        )
        detail.to_csv(output / "marker_metrics_by_gene.csv", index=False)
        by_celltype.to_csv(output / "marker_metrics_by_celltype.csv", index=False)

    summary: dict[str, object] = {
        "real_train": str(args.real_train.expanduser().resolve()),
        "real_eval": str(args.real_eval.expanduser().resolve()),
        "generated": str(args.generated.expanduser().resolve()),
        "label_key_real": args.label_key,
        "label_key_generated": generated_key,
        "target_sum": float(args.target_sum),
        "generated_space": "normalize_total_log1p",
        "conditions": classes,
        "n_real_train": int(train.n_obs),
        "n_real_eval": int(evaluation.n_obs),
        "n_generated": int(generated.n_obs),
        "n_genes": int(train.n_vars),
        "n_hvg": int(len(hvg_indices)),
        "pca_components": int(pca.n_components_),
        **fidelity_summary,
        "knn_auc_macro_mean": finite_mean(knn_auc["knn_auc_mean"]),
        "knn_auc_macro_mean_abs_distance_from_0.5": finite_mean(
            knn_auc["mean_abs_auc_minus_0.5"]
        ),
        "mmd_macro": finite_mean(mmd["mmd_pca"]),
        **marker_summary,
    }
    if rf is not None:
        summary["rf_auc_macro_mean"] = finite_mean(rf["rf_auc_mean"])
        summary["rf_auc_macro_mean_abs_distance_from_0.5"] = finite_mean(
            rf["mean_abs_auc_minus_0.5"]
        )

    with (output / "evaluation_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    pd.DataFrame([summary]).to_csv(output / "evaluation_summary.csv", index=False)

    print("\nEvaluation summary:")
    for key, value in summary.items():
        if isinstance(value, (int, float, str)):
            print(f"  {key}: {value}")
    print(f"\nSaved all results under: {output}")


if __name__ == "__main__":
    main()

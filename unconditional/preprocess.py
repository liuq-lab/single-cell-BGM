#!/usr/bin/env python3
"""Create a leakage-safe PBMC68k train/validation/final-test split.

Protocol
--------
1. Load raw 10x counts.
2. Remove cells with fewer than ``min_genes`` detected genes.
3. Randomly reserve 20% of cells as the untouched final test set.
4. Reserve 10% of the remaining 80% as validation.
5. Determine retained genes using the *actual training cells only*.

The resulting proportions are approximately 72% train, 8% validation and
20% final test.  All three H5AD files contain raw integer counts and exactly
the same genes in exactly the same order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmread
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw/hg19"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--outer-test-frac", type=float, default=0.20)
    parser.add_argument("--inner-validation-frac", type=float, default=0.10)
    parser.add_argument("--min-genes", type=int, default=10)
    parser.add_argument("--min-cells-train", type=int, default=3)
    return parser.parse_args()


def _find_one(directory: Path, names: list[str]) -> Path:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of {names} exists in {directory}")


def read_raw_10x(path: Path) -> ad.AnnData:
    path = path.expanduser()
    if path.suffix == ".h5ad":
        result = sc.read_h5ad(path)
        result.var_names_make_unique()
        return result

    if not path.is_dir():
        raise FileNotFoundError(path)
    if not (path / "matrix.mtx").exists() and not (path / "matrix.mtx.gz").exists():
        nested = path / "hg19"
        if nested.is_dir():
            path = nested

    matrix_path = _find_one(path, ["matrix.mtx", "matrix.mtx.gz"])
    gene_path = _find_one(path, ["genes.tsv", "genes.tsv.gz", "features.tsv", "features.tsv.gz"])
    barcode_path = _find_one(path, ["barcodes.tsv", "barcodes.tsv.gz"])

    matrix = mmread(matrix_path).tocsr().T
    genes = pd.read_csv(gene_path, sep="\t", header=None)
    barcodes = pd.read_csv(barcode_path, sep="\t", header=None)
    if matrix.shape != (len(barcodes), len(genes)):
        raise ValueError(
            f"10x dimensions disagree: matrix={matrix.shape}, "
            f"barcodes={len(barcodes)}, genes={len(genes)}"
        )

    gene_symbols = genes.iloc[:, 1] if genes.shape[1] >= 2 else genes.iloc[:, 0]
    var = pd.DataFrame(index=gene_symbols.astype(str).to_numpy())
    var["gene_ids"] = genes.iloc[:, 0].astype(str).to_numpy()
    if genes.shape[1] >= 3:
        var["feature_types"] = genes.iloc[:, 2].astype(str).to_numpy()

    result = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=barcodes.iloc[:, 0].astype(str).to_numpy()),
        var=var,
    )
    result.var_names_make_unique()
    return result


def sha256_lines(values: np.ndarray) -> str:
    payload = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_raw_counts(adata: ad.AnnData, name: str) -> None:
    x = adata.X
    values = x.data if sparse.issparse(x) else np.asarray(x).ravel()
    if values.size == 0:
        raise ValueError(f"{name} contains no non-zero counts")
    if not np.isfinite(values).all() or float(values.min()) < 0:
        raise ValueError(f"{name} is not a finite non-negative count matrix")
    sample = values[: min(values.size, 1_000_000)]
    if not np.allclose(sample, np.rint(sample), atol=1e-6):
        raise ValueError(f"{name} does not appear to contain raw integer counts")


def write_barcodes(path: Path, adata: ad.AnnData) -> None:
    pd.DataFrame({"barcode": adata.obs_names.astype(str)}).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    if not 0 < args.outer_test_frac < 1:
        raise ValueError("outer-test-frac must be between 0 and 1")
    if not 0 < args.inner_validation_frac < 1:
        raise ValueError("inner-validation-frac must be between 0 and 1")

    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)

    print("Loading raw 10x PBMC68k counts ...")
    full = read_raw_10x(args.input)
    assert_raw_counts(full, "input")
    full.obs_names_make_unique()

    detected = np.asarray((full.X > 0).sum(axis=1)).ravel()
    full.obs["n_genes"] = detected.astype(np.int32)
    full = full[detected >= args.min_genes].copy()

    # Sorting first makes the split independent of file row order.
    full = full[np.argsort(full.obs_names.astype(str))].copy()
    all_indices = np.arange(full.n_obs)
    train_pool_idx, test_idx = train_test_split(
        all_indices,
        test_size=args.outer_test_frac,
        random_state=args.seed,
        shuffle=True,
    )
    train_idx, validation_idx = train_test_split(
        train_pool_idx,
        test_size=args.inner_validation_frac,
        random_state=args.seed,
        shuffle=True,
    )

    # Gene inclusion is learned from actual training cells only.
    train_counts = full.X[train_idx]
    train_gene_ncells = np.asarray((train_counts > 0).sum(axis=0)).ravel()
    keep_genes = train_gene_ncells >= args.min_cells_train
    if int(keep_genes.sum()) == 0:
        raise RuntimeError("Training-only gene filter removed every gene")

    full = full[:, keep_genes].copy()
    full.var["n_cells_train"] = train_gene_ncells[keep_genes].astype(np.int32)
    train = full[train_idx].copy()
    validation = full[validation_idx].copy()
    test = full[test_idx].copy()

    for name, split in (("train", train), ("validation", validation), ("test", test)):
        split.obs["split"] = name
        assert_raw_counts(split, name)

    sets = [set(x.obs_names) for x in (train, validation, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("Cell leakage detected between splits")
    if not (
        np.array_equal(train.var_names, validation.var_names)
        and np.array_equal(train.var_names, test.var_names)
    ):
        raise RuntimeError("Gene order differs between splits")

    paths = {
        "train": output / "pbmc68k_train_raw.h5ad",
        "validation": output / "pbmc68k_validation_raw.h5ad",
        "test": output / "pbmc68k_test_raw.h5ad",
    }
    train.write_h5ad(paths["train"], compression="gzip")
    validation.write_h5ad(paths["validation"], compression="gzip")
    test.write_h5ad(paths["test"], compression="gzip")

    write_barcodes(output / "train_barcodes.csv", train)
    write_barcodes(output / "validation_barcodes.csv", validation)
    write_barcodes(output / "test_barcodes.csv", test)

    split_table = pd.concat(
        [
            pd.DataFrame({"barcode": x.obs_names.astype(str), "split": name})
            for name, x in (("train", train), ("validation", validation), ("test", test))
        ],
        ignore_index=True,
    )
    split_table.to_csv(output / "cell_split.csv", index=False)

    manifest = {
        "protocol": "outer 80/20; inner validation is 10% of the 80% train pool",
        "seed": args.seed,
        "outer_test_frac": args.outer_test_frac,
        "inner_validation_frac": args.inner_validation_frac,
        "effective_train_frac": (1 - args.outer_test_frac) * (1 - args.inner_validation_frac),
        "effective_validation_frac": (1 - args.outer_test_frac) * args.inner_validation_frac,
        "effective_test_frac": args.outer_test_frac,
        "min_genes_per_cell": args.min_genes,
        "min_cells_per_gene_on_train_only": args.min_cells_train,
        "n_cells": {
            "train": train.n_obs,
            "validation": validation.n_obs,
            "test": test.n_obs,
            "total": train.n_obs + validation.n_obs + test.n_obs,
        },
        "n_genes": train.n_vars,
        "gene_order_sha256": sha256_lines(train.var_names.to_numpy()),
        "barcode_sha256": {
            "train": sha256_lines(train.obs_names.to_numpy()),
            "validation": sha256_lines(validation.obs_names.to_numpy()),
            "test": sha256_lines(test.obs_names.to_numpy()),
        },
        "files": {key: value.name for key, value in paths.items()},
    }
    with (output / "split_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"Saved leakage-safe split to: {output}")


if __name__ == "__main__":
    main()

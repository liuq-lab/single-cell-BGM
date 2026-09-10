#!/usr/bin/env python3
"""Attach the official PBMC68k cell-type annotations to fixed raw splits.

scDiffusion's PBMC68k loader uses the 10x Genomics
``68k_pbmc_barcodes_annotation.tsv`` file and its ``celltype`` column.  This
script pins that exact file by SHA-256, joins annotations by the complete 10x
barcode (never by row position), and adds the labels to the existing
train/validation/test H5AD files without changing expression values or order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


OFFICIAL_ANNOTATION_URL = (
    "https://raw.githubusercontent.com/10XGenomics/single-cell-3prime-paper/"
    "master/pbmc68k_analysis/68k_pbmc_barcodes_annotation.tsv"
)
OFFICIAL_ANNOTATION_SHA256 = (
    "98dc85c047b56cb209ece200b7bebeb2ad53634ff3d5f42143987f59510624b3"
)
OFFICIAL_ANNOTATION_ROWS = 68_579
OFFICIAL_CELLTYPES = (
    "CD14+ Monocyte",
    "CD19+ B",
    "CD34+",
    "CD4+ T Helper2",
    "CD4+/CD25 T Reg",
    "CD4+/CD45RA+/CD25- Naive T",
    "CD4+/CD45RO+ Memory",
    "CD56+ NK",
    "CD8+ Cytotoxic T",
    "CD8+/CD45RA+ Naive Cytotoxic",
    "Dendritic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--annotation-tsv",
        type=Path,
        default=None,
        help=(
            "Local copy of the pinned 10x TSV. If omitted, the exact official "
            "file is downloaded to SPLIT_DIR."
        ),
    )
    parser.add_argument("--label-key", default="celltype")
    parser.add_argument("--output-suffix", default="_official_celltype")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Defaults to SPLIT_DIR/official_celltype_manifest.json.",
    )
    parser.add_argument("--download-timeout", type=float, default=60.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing labeled outputs and manifest.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def canonical_sparse(matrix):
    result = matrix.tocsr(copy=True)
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    return result


def matrix_equal(left, right, row_batch_size: int = 1024) -> bool:
    if left.shape != right.shape:
        return False
    if sparse.issparse(left) and sparse.issparse(right):
        left_csr = canonical_sparse(left)
        right_csr = canonical_sparse(right)
        return bool(
            np.array_equal(left_csr.indptr, right_csr.indptr)
            and np.array_equal(left_csr.indices, right_csr.indices)
            and np.array_equal(left_csr.data, right_csr.data)
        )
    for start in range(0, left.shape[0], int(row_batch_size)):
        stop = min(start + int(row_batch_size), left.shape[0])
        left_chunk = left[start:stop]
        right_chunk = right[start:stop]
        if sparse.issparse(left_chunk):
            left_chunk = left_chunk.toarray()
        if sparse.issparse(right_chunk):
            right_chunk = right_chunk.toarray()
        if not np.array_equal(np.asarray(left_chunk), np.asarray(right_chunk)):
            return False
    return True


def download_official_annotation(destination: Path, timeout: float) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        observed = file_sha256(destination)
        if observed != OFFICIAL_ANNOTATION_SHA256:
            raise RuntimeError(
                f"Existing annotation has SHA-256 {observed}, expected "
                f"{OFFICIAL_ANNOTATION_SHA256}: {destination}"
            )
        return destination

    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        request = urllib.request.Request(
            OFFICIAL_ANNOTATION_URL,
            headers={"User-Agent": "PBMC68k-BayesGM-official-label-preparation/1.0"},
        )
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            with temporary.open("wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
        observed = file_sha256(temporary)
        if observed != OFFICIAL_ANNOTATION_SHA256:
            raise RuntimeError(
                f"Downloaded annotation has SHA-256 {observed}, expected "
                f"{OFFICIAL_ANNOTATION_SHA256}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_official_annotations(path: Path) -> pd.Series:
    observed_sha256 = file_sha256(path)
    if observed_sha256 != OFFICIAL_ANNOTATION_SHA256:
        raise RuntimeError(
            f"Annotation SHA-256 mismatch: observed={observed_sha256}, "
            f"expected={OFFICIAL_ANNOTATION_SHA256}, path={path}"
        )
    frame = pd.read_csv(path, sep="\t", dtype={"barcodes": str, "celltype": str})
    required = {"barcodes", "celltype"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Official TSV must contain columns {sorted(required)}; "
            f"found={frame.columns.tolist()}"
        )
    if len(frame) != OFFICIAL_ANNOTATION_ROWS:
        raise ValueError(
            f"Official TSV row count is {len(frame)}, expected {OFFICIAL_ANNOTATION_ROWS}"
        )
    if frame["barcodes"].isna().any() or frame["celltype"].isna().any():
        raise ValueError("Official TSV contains missing barcodes or celltype labels")
    if frame["barcodes"].duplicated().any():
        examples = frame.loc[frame["barcodes"].duplicated(), "barcodes"].head().tolist()
        raise ValueError(f"Official TSV contains duplicate barcodes, e.g. {examples}")
    if (frame["barcodes"].str.strip() != frame["barcodes"]).any():
        raise ValueError("Official TSV contains barcodes with surrounding whitespace")
    if (frame["celltype"].str.strip() == "").any():
        raise ValueError("Official TSV contains blank celltype labels")

    observed_labels = tuple(sorted(frame["celltype"].unique().tolist()))
    if observed_labels != OFFICIAL_CELLTYPES:
        raise ValueError(
            "Official TSV label vocabulary differs from the pinned 11 classes: "
            f"observed={observed_labels}, expected={OFFICIAL_CELLTYPES}"
        )
    return frame.set_index("barcodes", verify_integrity=True)["celltype"]


def write_verified_labeled_split(
    raw: ad.AnnData,
    labels: pd.Series,
    label_key: str,
    output_path: Path,
    overwrite: bool,
) -> ad.AnnData:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing labeled split: {output_path}. "
            "Use --overwrite only when replacement is intentional."
        )
    if label_key in raw.obs.columns:
        raise ValueError(
            f"Raw split already contains obs[{label_key!r}]; refusing to replace it"
        )
    raw.obs_names = raw.obs_names.astype(str)
    raw.var_names = raw.var_names.astype(str)
    if not raw.obs_names.is_unique:
        raise ValueError("Raw split contains duplicate barcodes")

    joined = labels.reindex(raw.obs_names)
    if joined.isna().any():
        missing = raw.obs_names[joined.isna().to_numpy()].tolist()
        raise KeyError(
            f"{len(missing)} split barcodes have no official annotation; "
            f"examples={missing[:10]}"
        )
    unknown = sorted(set(joined.astype(str)) - set(OFFICIAL_CELLTYPES))
    if unknown:
        raise ValueError(f"Joined labels contain unexpected values: {unknown}")

    labeled = raw.copy()
    labeled.obs[label_key] = pd.Categorical(
        joined.to_numpy(dtype=str),
        categories=list(OFFICIAL_CELLTYPES),
        ordered=True,
    )
    if not raw.obs_names.equals(labeled.obs_names):
        raise RuntimeError("Adding labels changed cell order before writing")
    if not raw.var_names.equals(labeled.var_names):
        raise RuntimeError("Adding labels changed gene order before writing")
    if not matrix_equal(raw.X, labeled.X):
        raise RuntimeError("Adding labels changed expression values before writing")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}.h5ad")
    try:
        labeled.write_h5ad(temporary, compression="gzip")
        verified = sc.read_h5ad(temporary)
        verified.obs_names = verified.obs_names.astype(str)
        verified.var_names = verified.var_names.astype(str)
        if verified.shape != raw.shape:
            raise RuntimeError("Written labeled split changed matrix shape")
        if not raw.obs_names.equals(verified.obs_names):
            raise RuntimeError("Written labeled split changed cell order")
        if not raw.var_names.equals(verified.var_names):
            raise RuntimeError("Written labeled split changed gene order")
        if not matrix_equal(raw.X, verified.X):
            raise RuntimeError("Written labeled split changed expression values")
        if label_key not in verified.obs:
            raise RuntimeError(f"Written labeled split lost obs[{label_key!r}]")
        written_labels = verified.obs[label_key]
        if not isinstance(written_labels.dtype, pd.CategoricalDtype):
            raise RuntimeError("Written official label column is not categorical")
        if tuple(map(str, written_labels.cat.categories)) != OFFICIAL_CELLTYPES:
            raise RuntimeError("Written official label category order changed")
        if not bool(written_labels.cat.ordered):
            raise RuntimeError("Written official label categories are not ordered")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return labeled


def main() -> None:
    args = parse_args()
    if args.label_key != "celltype":
        raise ValueError(
            "The scDiffusion PBMC68k annotation key is fixed to 'celltype'; "
            "do not rename it for this benchmark"
        )
    if args.output_suffix != "_official_celltype":
        raise ValueError(
            "The official benchmark output suffix is fixed to "
            "'_official_celltype'"
        )
    split_dir = args.split_dir.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else split_dir / "official_celltype_manifest.json"
    )
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {manifest_path}. "
            "Use --overwrite only when replacement is intentional."
        )
    for split in ("train", "validation", "test"):
        raw_path = split_dir / f"pbmc68k_{split}_raw.h5ad"
        output_path = split_dir / f"pbmc68k_{split}_raw{args.output_suffix}.h5ad"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing labeled split: {output_path}. "
                "Use --overwrite only when replacement is intentional."
            )

    if args.annotation_tsv is None:
        annotation_path = download_official_annotation(
            split_dir / "68k_pbmc_barcodes_annotation.tsv",
            timeout=args.download_timeout,
        )
    else:
        annotation_path = args.annotation_tsv.expanduser().resolve()
        if not annotation_path.exists():
            raise FileNotFoundError(annotation_path)
    annotations = load_official_annotations(annotation_path)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "method": "exact full-barcode join; never row-position matching",
        "label_key": args.label_key,
        "output_suffix": args.output_suffix,
        "official_annotation": {
            "url": OFFICIAL_ANNOTATION_URL,
            "sha256": OFFICIAL_ANNOTATION_SHA256,
            "local_path": str(annotation_path),
            "n_rows": int(len(annotations)),
        },
        "categories": list(OFFICIAL_CELLTYPES),
        "category_order": "alphabetical",
        "annotation_counts": {
            str(label): int(count)
            for label, count in annotations.value_counts().sort_index().items()
        },
        "splits": {},
    }

    observed_barcodes: set[str] = set()
    gene_order: tuple[str, ...] | None = None
    for split in ("train", "validation", "test"):
        raw_path = split_dir / f"pbmc68k_{split}_raw.h5ad"
        output_path = split_dir / f"pbmc68k_{split}_raw{args.output_suffix}.h5ad"
        raw = sc.read_h5ad(raw_path)
        raw.obs_names = raw.obs_names.astype(str)
        raw.var_names = raw.var_names.astype(str)
        current_barcodes = set(raw.obs_names)
        overlap = observed_barcodes.intersection(current_barcodes)
        if overlap:
            raise RuntimeError(
                f"Cell leakage between fixed splits; examples={sorted(overlap)[:10]}"
            )
        observed_barcodes.update(current_barcodes)
        current_gene_order = tuple(raw.var_names)
        if gene_order is None:
            gene_order = current_gene_order
        elif current_gene_order != gene_order:
            raise RuntimeError(f"Gene order differs in the {split} raw split")

        labeled = write_verified_labeled_split(
            raw=raw,
            labels=annotations,
            label_key=args.label_key,
            output_path=output_path,
            overwrite=args.overwrite,
        )
        counts = labeled.obs[args.label_key].value_counts(sort=False)
        if tuple(map(str, counts.index)) != OFFICIAL_CELLTYPES:
            raise RuntimeError(f"{split} label category order is not the pinned order")
        if (counts <= 0).any():
            missing_classes = counts.index[counts <= 0].astype(str).tolist()
            raise RuntimeError(
                f"{split} does not contain every official class: {missing_classes}"
            )
        manifest["splits"][split] = {
            "raw_file": str(raw_path),
            "labeled_file": str(output_path),
            "n_obs": int(raw.n_obs),
            "n_vars": int(raw.n_vars),
            "barcode_sha256": sha256_lines(raw.obs_names),
            "gene_order_sha256": sha256_lines(raw.var_names),
            "label_counts": {
                str(label): int(count) for label, count in counts.items()
            },
            "shape_preserved": True,
            "cell_order_preserved": True,
            "gene_order_preserved": True,
            "expression_preserved": True,
        }
        print(f"{split}: {dict(counts.items())}")
        print(f"Saved: {output_path}")

    annotation_barcodes = set(annotations.index.astype(str))
    if observed_barcodes != annotation_barcodes:
        raise RuntimeError(
            "The fixed train/validation/test union is not exactly the official "
            "PBMC68k barcode set: "
            f"missing={len(annotation_barcodes - observed_barcodes)}, "
            f"unexpected={len(observed_barcodes - annotation_barcodes)}"
        )
    manifest["split_union_n_obs"] = len(observed_barcodes)
    manifest["annotation_barcodes_not_in_filtered_splits"] = int(
        len(annotation_barcodes - observed_barcodes)
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_name(
        manifest_path.name + f".tmp.{os.getpid()}"
    )
    try:
        with temporary_manifest.open("w") as handle:
            json.dump(manifest, handle, indent=2)
        os.replace(temporary_manifest, manifest_path)
    finally:
        if temporary_manifest.exists():
            temporary_manifest.unlink()
    print(f"Official-label manifest: {manifest_path}")


if __name__ == "__main__":
    main()

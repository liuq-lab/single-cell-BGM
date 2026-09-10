#!/usr/bin/env python3
"""Select reproducible official-cell-type marker genes from REAL TRAIN cells.

Two modes are provided:

* ``all-celltypes``: rank every cell type against the rest and retain ``top_n``
  positive markers per cell type.  This is the recommended complete conditional
  benchmark.
* ``largest-two``: identify the two largest annotated cell types and retain five
  positive markers from each, matching the marker-count convention of the
  historical cscGAN comparison without replacing the official annotations.

The final validation/test sets are never used for marker selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--real-train", type=Path, required=True)
    parser.add_argument("--label-key", default="celltype")
    parser.add_argument(
        "--mode",
        choices=("all-celltypes", "largest-two"),
        default="all-celltypes",
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--target-sum",
        type=float,
        default=20_000,
        help="Library size before log1p for marker ranking.",
    )
    parser.add_argument(
        "--method",
        default="t-test_overestim_var",
        choices=("t-test_overestim_var", "t-test", "wilcoxon", "logreg"),
    )
    parser.add_argument(
        "--rank-n-genes",
        type=int,
        default=100,
        help="Number of ranked genes requested from Scanpy before top-N filtering.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Keep only genes with a positive score by default.",
    )
    parser.add_argument(
        "--max-adjusted-p",
        type=float,
        default=None,
        help="Optional adjusted-p cutoff. Omitted for paper-aligned ranking.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/markers/celltype_markers.csv"),
    )
    return parser.parse_args()


def natural_order(values: list[str]) -> list[str]:
    def key(value: str):
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted(set(map(str, values)), key=key)


def check_raw_nonnegative(data: sc.AnnData) -> None:
    values = data.X.data if sparse.issparse(data.X) else np.asarray(data.X).ravel()
    if values.size and (not np.isfinite(values).all() or float(values.min()) < 0):
        raise ValueError("Input expression must be finite and non-negative")


def main() -> None:
    args = parse_args()
    if args.top_n <= 0 or args.rank_n_genes <= 0:
        raise ValueError("--top-n and --rank-n-genes must be positive")

    path = args.real_train.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    data = sc.read_h5ad(path)
    data.var_names = data.var_names.astype(str)
    data.obs_names = data.obs_names.astype(str)
    data.var_names_make_unique()
    check_raw_nonnegative(data)

    if args.label_key not in data.obs.columns:
        raise KeyError(
            f"{path} has no obs[{args.label_key!r}]; "
            f"available={data.obs.columns.tolist()}"
        )
    if data.obs[args.label_key].isna().any():
        raise ValueError(f"obs[{args.label_key!r}] contains missing labels")

    data.obs[args.label_key] = data.obs[args.label_key].astype(str).astype("category")
    counts = data.obs[args.label_key].astype(str).value_counts()

    if args.mode == "largest-two":
        if len(counts) < 2:
            raise RuntimeError("largest-two mode requires at least two cell types")
        groups = counts.sort_values(ascending=False).index[:2].astype(str).tolist()
        top_n = 5
    else:
        groups = natural_order(counts.index.astype(str).tolist())
        top_n = int(args.top_n)

    print("Cell-type counts:")
    print(counts.sort_index())
    print("Groups selected for marker ranking:", groups)

    sc.pp.normalize_total(data, target_sum=float(args.target_sum))
    sc.pp.log1p(data)

    n_rank = min(max(int(args.rank_n_genes), top_n), data.n_vars)
    sc.tl.rank_genes_groups(
        data,
        groupby=args.label_key,
        groups=groups,
        reference="rest",
        method=args.method,
        n_genes=n_rank,
        use_raw=False,
        rankby_abs=False,
    )

    tables: list[pd.DataFrame] = []
    for group in groups:
        table = sc.get.rank_genes_groups_df(data, group=group).copy()
        table = table.rename(columns={"names": "gene", "scores": "score"})
        if "score" not in table:
            raise RuntimeError("Scanpy result does not contain a score column")
        table = table.loc[np.isfinite(table["score"]) & (table["score"] > args.min_score)]
        if args.max_adjusted_p is not None and "pvals_adj" in table:
            table = table.loc[table["pvals_adj"] <= args.max_adjusted_p]
        table = table.sort_values("score", ascending=False).head(top_n).copy()
        if len(table) < top_n:
            raise RuntimeError(
                f"Celltype {group!r} has only {len(table)} eligible markers; "
                f"requested {top_n}"
            )
        table.insert(0, "rank", np.arange(1, len(table) + 1, dtype=int))
        table.insert(0, "celltype", str(group))
        keep = [
            column
            for column in (
                "celltype",
                "rank",
                "gene",
                "score",
                "logfoldchanges",
                "pvals",
                "pvals_adj",
            )
            if column in table.columns
        ]
        tables.append(table[keep])

    markers = pd.concat(tables, ignore_index=True)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    markers.to_csv(output, index=False)

    metadata = {
        "real_train": str(path),
        "label_key": args.label_key,
        "mode": args.mode,
        "groups": groups,
        "top_n_per_group": top_n,
        "target_sum": float(args.target_sum),
        "rank_method": args.method,
        "reference": "rest",
        "rankby_abs": False,
        "source_split": "real_train_only",
        "output": str(output),
    }
    with output.with_suffix(".metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nSelected markers:")
    print(markers.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()

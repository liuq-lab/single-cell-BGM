#!/usr/bin/env python3
"""Final-test evaluation for frozen unconditional BayesGM outputs.

generated.npy must already be in normalize_total(1e4)+log1p space.
Marker KLD uses TRAIN only to define bins, then compares TEST vs generated.
"""

from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/pbmc68k_bgm_eval_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pbmc68k_bgm_eval_matplotlib")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scib
from scipy import sparse
from sklearn.decomposition import PCA


MARKERS = [
    "LST1", "S100A8", "S100A9",
    "MS4A1", "CD79A", "CD74",
    "IL2RA", "FOXP3", "CTLA4",
    "CCR7", "TCF7", "LEF1",
    "IL7R", "LTB", "S100A4",
    "GNLY", "NKG7", "KLRD1",
    "CCL5", "CTSW", "GZMK",
    "CD8B", "FCER1A", "HLA-DRA",
]
MARKER_BINS = 20
KLD_EPS = 1e-8


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--result-dir", type=Path, default=Path("results/bayesgm"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--target-sum", type=float, default=1e4)
    p.add_argument("--n-eval", type=int, default=2000)
    p.add_argument("--mmd-n-eval", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-plot", type=int, default=2000)
    return p.parse_args()


def sha256_lines(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def dense_float32(x):
    return np.asarray(x.toarray() if sparse.issparse(x) else x, dtype=np.float32)


def sample_indices(n_real, n_gen, n_eval, seed):
    n = min(n_eval, n_real, n_gen)
    rng = np.random.default_rng(seed)
    return n, rng.choice(n_real, n, replace=False), rng.choice(n_gen, n, replace=False)


def rbf_mmd_np(x, y):
    x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
    total = np.concatenate([x, y])
    ss = np.sum(total * total, axis=1)
    d = np.maximum(ss[:, None] + ss[None, :] - 2 * total @ total.T, 0)
    nt = len(total)
    bw = max((float(d.sum()) / max(nt * nt - nt, 1)) / (2 ** (5 // 2)), 1e-8)
    kernels = sum(np.exp(-d / (bw * 2**i)) for i in range(5))
    n = len(x)
    return float(kernels[:n, :n].mean() + kernels[n:, n:].mean()
                 - kernels[:n, n:].mean() - kernels[n:, :n].mean())


def compute_metrics(real, generated, n_eval, mmd_n, seed):
    n, ri, gi = sample_indices(len(real), len(generated), n_eval, seed)
    r = np.asarray(real[ri], np.float32)
    g = np.asarray(generated[gi], np.float32)

    joint = np.concatenate([r, g])
    npc = min(50, len(joint) - 1, joint.shape[1])
    pcs = PCA(n_components=npc, random_state=seed).fit_transform(joint)

    graph = ad.AnnData(X=np.zeros((2 * n, 1), np.float32))
    graph.obsm["X_metric_pca20"] = pcs[:, :min(20, npc)].astype(np.float32)
    graph.obs["batch"] = pd.Categorical(["real"] * n + ["generated"] * n)
    sc.pp.neighbors(graph, use_rep="X_metric_pca20", n_neighbors=10, random_state=seed)
    ilisi = float(scib.me.ilisi_graph(graph, batch_key="batch", type_="knn"))

    m = min(mmd_n, n)
    mmd = rbf_mmd_np(pcs[:m, :min(50, npc)], pcs[n:n + m, :min(50, npc)])

    rz, gz = float(np.mean(r <= 0)), float(np.mean(g <= 0))
    return {
        "test_ilisi_pca20": ilisi,
        "test_mmd_pca50": mmd,
        "test_real_zero_frac": rz,
        "test_generated_zero_frac": gz,
        "test_zero_frac_gap": abs(rz - gz),
        "test_real_cell_sum_mean": float(r.sum(1).mean()),
        "test_generated_cell_sum_mean": float(g.sum(1).mean()),
        "test_gene_mean_corr": float(np.corrcoef(r.mean(0), g.mean(0))[0, 1]),
        "test_cells_used": int(n),
        "mmd_cells_used": int(m),
        "evaluation_seed": int(seed),
    }


def normalized_markers_from_raw(adata, cols, target_sum):
    x = adata.X
    if sparse.issparse(x):
        x = x.tocsr()
        lib = np.asarray(x.sum(1)).ravel()
        sub = x[:, cols].toarray()
    else:
        x = np.asarray(x)
        lib, sub = x.sum(1), x[:, cols].copy()

    lib, sub = np.asarray(lib, float), np.asarray(sub, float)
    if np.any(lib <= 0):
        raise ValueError("Real data contain zero-library cells")
    return np.log1p(sub * (target_sum / lib)[:, None]).astype(np.float32)


def train_thresholds(x, n_bins=MARKER_BINS):
    pos = np.asarray(x, float)
    pos = pos[pos > 0]
    if len(pos) == 0:
        return np.array([], float)
    edges = np.unique(np.quantile(pos, np.linspace(0, 1, n_bins + 1)))
    return edges[1:-1] if len(edges) > 2 else np.array([], float)


def marker_distribution(x, cuts):
    x = np.asarray(x, float)
    if not np.isfinite(x).all() or x.min() < -1e-6:
        raise ValueError("Invalid marker expression")
    x = np.maximum(x, 0)
    zero = x <= 0
    pos = x[~zero]

    counts = np.zeros(2 + len(cuts), float)
    counts[0] = zero.sum()
    if len(pos):
        ids = np.searchsorted(cuts, pos, side="right")
        counts[1:] = np.bincount(ids, minlength=1 + len(cuts))

    counts += KLD_EPS
    return counts / counts.sum()


def compute_marker_kld(train_marker, test_marker, gen_marker, n_eval, seed):
    n, ri, gi = sample_indices(len(test_marker), len(gen_marker), n_eval, seed)
    r, g = test_marker[ri], gen_marker[gi]
    rows = []

    for j, gene in enumerate(MARKERS):
        cuts = train_thresholds(train_marker[:, j])
        p = marker_distribution(r[:, j], cuts)
        q = marker_distribution(g[:, j], cuts)
        kld = float(np.sum(p * np.log(p / q)))

        rz = float(np.mean(r[:, j] <= 0))
        gz = float(np.mean(g[:, j] <= 0))
        rows.append({
            "gene": gene,
            "train_positive_frac": float(np.mean(train_marker[:, j] > 0)),
            "real_zero_frac": rz,
            "generated_zero_frac": gz,
            "zero_frac_gap": abs(rz - gz),
            "positive_bins_effective": int(len(cuts) + 1),
            "kld_real_to_generated": kld,
        })

    table = pd.DataFrame(rows)
    k = table["kld_real_to_generated"].to_numpy(float)
    summary = {
        "test_marker_kld_mean": float(k.mean()),
        "test_marker_kld_median": float(np.median(k)),
        "test_marker_kld_min": float(k.min()),
        "test_marker_kld_max": float(k.max()),
        "test_marker_kld_n_genes": int(len(k)),
        "test_marker_kld_cells_used": int(n),
        "test_marker_kld_positive_bins_requested": MARKER_BINS,
        "test_marker_kld_epsilon": KLD_EPS,
    }
    return table, summary


def audit(split, result, train, test, genes):
    config_path = result / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = json.loads(config_path.read_text())
    if config.get("final_test_loaded_during_training") is not False:
        raise RuntimeError("Training config does not certify final test was untouched")

    gene_hash = sha256_lines(genes)
    if config.get("gene_order_sha256") != gene_hash:
        raise RuntimeError("Training config gene hash differs from genes.npy")
    if not np.array_equal(train.var_names.astype(str), genes):
        raise RuntimeError("TRAIN gene order differs from genes.npy")
    if not np.array_equal(test.var_names.astype(str), genes):
        raise RuntimeError("TEST gene order differs from genes.npy")

    out = {
        "config_final_test_loaded_during_training": False,
        "gene_order_sha256": gene_hash,
        "marker_panel_sha256": sha256_lines(MARKERS),
        "marker_genes": MARKERS,
        "marker_kld_direction": "KL(real_test || generated)",
        "marker_kld_bin_source": "real TRAIN only",
        "marker_kld_zero_bin": "exact zero",
        "marker_kld_positive_bins": MARKER_BINS,
        "marker_kld_epsilon": KLD_EPS,
    }

    cell_split = split / "cell_split.csv"
    if cell_split.exists():
        t = pd.read_csv(cell_split)
        if {"split", "barcode"}.issubset(t.columns):
            groups = {s: set(t.loc[t["split"] == s, "barcode"].astype(str))
                      for s in ("train", "validation", "test")}
            overlaps = {
                "train_validation": len(groups["train"] & groups["validation"]),
                "train_test": len(groups["train"] & groups["test"]),
                "validation_test": len(groups["validation"] & groups["test"]),
            }
            if any(overlaps.values()):
                raise RuntimeError(f"Split barcode overlap detected: {overlaps}")
            out["barcode_overlap"] = overlaps
    return out


def save_umap(real, generated, output, n_plot, seed):
    import umap

    rng = np.random.default_rng(seed)
    nr, ng = min(n_plot, len(real)), min(n_plot, len(generated))
    r = np.asarray(real[rng.choice(len(real), nr, replace=False)], np.float32)
    g = np.asarray(generated[rng.choice(len(generated), ng, replace=False)], np.float32)
    joint = np.concatenate([r, g])

    npc = min(50, len(joint) - 1, joint.shape[1])
    pcs = PCA(n_components=npc, random_state=seed).fit_transform(joint)
    xy = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=seed,
                   transform_seed=seed).fit_transform(pcs[:, :20])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xy[:nr, 0], xy[:nr, 1], s=8, alpha=.35, label="Real test")
    ax.scatter(xy[nr:, 0], xy[nr:, 1], s=8, alpha=.55, label="Generated")
    ax.set(xticks=[], yticks=[], title="PBMC68k final test: real vs generated")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "umap_real_vs_generated.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    split = args.split_dir.expanduser().resolve()
    result = args.result_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve() if args.output_dir else result / "final_test_evaluation"
    output.mkdir(parents=True, exist_ok=True)

    train_path = split / "pbmc68k_train_raw.h5ad"
    test_path = split / "pbmc68k_test_raw.h5ad"
    generated_path = result / "generated.npy"
    genes_path = result / "genes.npy"

    for p in (train_path, test_path, generated_path, genes_path):
        if not p.exists():
            raise FileNotFoundError(p)

    print("Opening TRAIN for marker-KLD bins:", train_path)
    train = sc.read_h5ad(train_path)
    train.var_names_make_unique()

    print("FINAL TEST IS NOW BEING OPENED:", test_path)
    test = sc.read_h5ad(test_path)
    test.var_names_make_unique()

    genes = np.load(genes_path, allow_pickle=False).astype(str)
    audit_result = audit(split, result, train, test, genes)

    # Real TEST: raw -> normalize_total(1e4) -> log1p
    test_log = test.copy()
    sc.pp.normalize_total(test_log, target_sum=args.target_sum)
    sc.pp.log1p(test_log)
    real = dense_float32(test_log.X)

    # Generated is ALREADY normalized/log. Do not transform again.
    generated = np.load(generated_path, allow_pickle=False).astype(np.float32)
    if generated.ndim != 2 or generated.shape[1] != len(genes):
        raise ValueError(f"Generated shape {generated.shape} incompatible with {len(genes)} genes")
    if not np.isfinite(generated).all() or generated.min() < 0:
        raise ValueError("Generated expression must be finite and non-negative")

    result_metrics = compute_metrics(real, generated, args.n_eval, args.mmd_n_eval, args.seed)

    gene_to_idx = {g: i for i, g in enumerate(genes)}
    missing = [g for g in MARKERS if g not in gene_to_idx]
    if missing:
        raise RuntimeError(f"Markers missing from model gene space: {missing}")

    marker_idx = np.array([gene_to_idx[g] for g in MARKERS], int)
    train_marker = normalized_markers_from_raw(train, marker_idx, args.target_sum)
    test_marker = np.asarray(real[:, marker_idx], np.float32)
    gen_marker = np.asarray(generated[:, marker_idx], np.float32)

    marker_table, marker_summary = compute_marker_kld(
        train_marker, test_marker, gen_marker, args.n_eval, args.seed
    )
    result_metrics.update(marker_summary)

    marker_table.to_csv(output / "final_test_marker_kld.csv", index=False)
    pd.Series(result_metrics, name="value").to_csv(output / "final_test_metrics.csv")
    (output / "final_test_metrics.json").write_text(json.dumps(result_metrics, indent=2) + "\n")
    (output / "protocol_audit.json").write_text(json.dumps(audit_result, indent=2) + "\n")
    save_umap(real, generated, output, args.n_plot, args.seed)

    print("\n===== FINAL TEST =====")
    for k, v in result_metrics.items():
        print(f"{k}: {v}")

    print("\n===== MARKER KLD =====")
    print(marker_table[["gene", "real_zero_frac", "generated_zero_frac",
                        "kld_real_to_generated"]].to_string(index=False))
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
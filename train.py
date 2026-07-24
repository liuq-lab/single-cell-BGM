#!/usr/bin/env python3
"""Train BayesGM from a frozen scDiffusion-style autoencoder without test leakage.

This script reads *only* ``pbmc68k_train_raw.h5ad`` and
``pbmc68k_validation_raw.h5ad``.  The final-test H5AD is deliberately not
opened here.  Model and epoch selection use validation PCA20 iLISI.

Pipeline
--------
raw counts -> normalize_total(1e4) -> log1p -> frozen PyTorch 128-D AE
-> train-only latent standardization -> BayesGM (z_dim=32)
-> Step 1 EGM warm start -> Step 2 learnable variance + latent MMD + anchor
-> frozen final generated full-gene log-expression matrix.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/pbmc68k_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pbmc68k_matplotlib")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scib
from scipy import sparse
from sklearn.decomposition import PCA
from tqdm import tqdm

import torch
import tensorflow as tf
from bayesgm.models.bgm import BGM
from bayesgm.models.networks import BaseVariationalNet

from autoencoder import ScDiffusionAutoencoder, decode_array, load_autoencoder


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bayesgm"),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-sum", type=float, default=1e4)

    # Frozen PyTorch autoencoder artifacts produced by autoencoder.py.
    parser.add_argument(
        "--ae-artifact-dir",
        type=Path,
        default=Path("results/autoencoder"),
    )
    parser.add_argument("--ae-latent-dim", type=int, default=128)

    # BayesGM Step 1.
    parser.add_argument("--bgm-z-dim", type=int, default=32)
    parser.add_argument("--g-units", type=parse_int_list, default=[512, 512, 512, 512, 512])
    parser.add_argument("--e-units", type=parse_int_list, default=[512, 512, 512, 512, 512])
    parser.add_argument("--dx-units", type=parse_int_list, default=[256, 128, 64, 16])
    parser.add_argument("--dz-units", type=parse_int_list, default=[128, 64, 32, 8])
    parser.add_argument("--step1-lr", type=float, default=1e-4)
    parser.add_argument("--step1-iterations", type=int, default=100_000)
    parser.add_argument("--step1-batch-size", type=int, default=512)
    parser.add_argument("--step1-eval-every", type=int, default=10_000)
    parser.add_argument("--step1-save-iters", type=parse_int_list, default=[20_000, 30_000, 50_000, 100_000])
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--generator-variance-eps", type=float, default=1e-3)

    # Distribution-preserving Step 2, matching the best-performing branch.
    parser.add_argument("--step2-epochs", type=int, default=500)
    parser.add_argument("--step2-batch-size", type=int, default=512)
    parser.add_argument("--step2-eval-every", type=int, default=5)
    parser.add_argument("--step2-save-every", type=int, default=50)
    parser.add_argument("--generator-lr", type=float, default=5e-7)
    parser.add_argument("--variance-lr", type=float, default=1e-8)
    parser.add_argument("--variance-init", type=float, default=0.30)
    parser.add_argument("--variance-min", type=float, default=0.15)
    parser.add_argument("--variance-max", type=float, default=0.60)
    parser.add_argument("--variance-log-prior-weight", type=float, default=5.0)
    parser.add_argument("--mmd-weight", type=float, default=50.0)
    parser.add_argument("--anchor-weight", type=float, default=0.10)
    parser.add_argument("--mmd-samples", type=int, default=256)
    parser.add_argument("--gradient-clip", type=float, default=10.0)

    # Validation and frozen generation.
    parser.add_argument("--n-eval", type=int, default=2_000)
    parser.add_argument("--mmd-n-eval", type=int, default=500)
    parser.add_argument("--validation-generation-seed", type=int, default=42)
    parser.add_argument("--final-generation-seed", type=int, default=2027)
    parser.add_argument("--final-n-generate", type=int, default=2_000)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def configure_tensorflow() -> None:
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass


def install_generator_variance_epsilon(epsilon: float) -> None:
    """Use the same positive variance floor in both BayesGM stages."""
    if not hasattr(BaseVariationalNet, "_pbmc_original_call"):
        BaseVariationalNet._pbmc_original_call = BaseVariationalNet.call
    original = BaseVariationalNet._pbmc_original_call

    def patched(self, inputs, eps=epsilon, training=True):
        return original(self, inputs, eps=epsilon, training=training)

    BaseVariationalNet.call = patched


def sha256_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def dense_float32(x) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def get_rows(x, indices: np.ndarray) -> np.ndarray:
    return dense_float32(x[indices])


def assert_disjoint(train: ad.AnnData, validation: ad.AnnData) -> None:
    overlap = set(train.obs_names) & set(validation.obs_names)
    if overlap:
        raise RuntimeError(f"Train/validation overlap: {len(overlap)} cells")
    if not np.array_equal(train.var_names, validation.var_names):
        raise RuntimeError("Train and validation genes/order differ")


def load_log_expression(path: Path, target_sum: float) -> ad.AnnData:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    data = sc.read_h5ad(path)
    data.var_names_make_unique()
    values = data.X.data if sparse.issparse(data.X) else np.asarray(data.X).ravel()
    if values.size and (float(values.min()) < 0 or not np.isfinite(values).all()):
        raise ValueError(f"{path} is not a raw non-negative matrix")
    sc.pp.normalize_total(data, target_sum=target_sum)
    sc.pp.log1p(data)
    if sparse.issparse(data.X):
        data.X = data.X.tocsr().astype(np.float32)
    else:
        data.X = np.asarray(data.X, dtype=np.float32)
    return data


def load_ae_artifacts(
    train: ad.AnnData,
    validation: ad.AnnData,
    args: argparse.Namespace,
) -> tuple[ScDiffusionAutoencoder, np.ndarray, np.ndarray]:
    directory = args.ae_artifact_dir.expanduser()
    metadata_path = directory / "metadata.json"
    checkpoint = directory / "model.pt"
    train_latent_path = directory / "train_latent.npy"
    validation_latent_path = directory / "validation_latent.npy"
    for path in [metadata_path, checkpoint, train_latent_path, validation_latent_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing AE artifact: {path}. Run autoencoder.py first."
            )
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    expected = {
        "n_genes": train.n_vars,
        "latent_dim": args.ae_latent_dim,
        "train_n_obs": train.n_obs,
        "validation_n_obs": validation.n_obs,
        "gene_order_sha256": sha256_lines(train.var_names.astype(str)),
        "train_obs_sha256": sha256_lines(train.obs_names.astype(str)),
        "validation_obs_sha256": sha256_lines(validation.obs_names.astype(str)),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"AE artifact mismatch for {key}: stored={metadata.get(key)!r}, current={value!r}"
            )
    train_latent = np.load(train_latent_path, allow_pickle=False).astype(np.float32)
    validation_latent = np.load(validation_latent_path, allow_pickle=False).astype(np.float32)
    if train_latent.shape != (train.n_obs, args.ae_latent_dim):
        raise RuntimeError(f"Unexpected train latent shape: {train_latent.shape}")
    if validation_latent.shape != (validation.n_obs, args.ae_latent_dim):
        raise RuntimeError(f"Unexpected validation latent shape: {validation_latent.shape}")
    model = load_autoencoder(
        checkpoint,
        n_genes=train.n_vars,
        latent_dim=args.ae_latent_dim,
        device="cpu",
    )
    print("Loaded frozen scDiffusion-style autoencoder:", checkpoint)
    print("Selected AE step:", metadata["best_step"])
    print("Selected AE validation MSE:", metadata["best_validation_mse"])
    return model, train_latent, validation_latent


def select_fixed_rows(matrix, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.choice(matrix.shape[0], size=min(n, matrix.shape[0]), replace=False)
    return get_rows(matrix, indices)


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
    mmd = rbf_mmd_np(pcs[:m, : min(50, n_components)], pcs[n : n + m, : min(50, n_components)])
    real_mean = real_eval.mean(axis=0)
    gen_mean = generated_eval.mean(axis=0)
    corr = float(np.corrcoef(real_mean, gen_mean)[0, 1])
    return {
        "ilisi_pca20": ilisi,
        "mmd_pca50": mmd,
        "real_zero_frac": float(np.mean(real_eval <= 0)),
        "generated_zero_frac": float(np.mean(generated_eval <= 0)),
        "real_cell_sum_mean": float(real_eval.sum(axis=1).mean()),
        "generated_cell_sum_mean": float(generated_eval.sum(axis=1).mean()),
        "gene_mean_corr": corr,
    }


def build_bgm(args: argparse.Namespace, output: Path) -> BGM:
    params = {
        "x_dim": args.ae_latent_dim,
        "z_dim": args.bgm_z_dim,
        "dataset": "pbmc68k_ae_latent",
        "output_dir": str(output),
        "use_bnn": False,
        "g_units": list(args.g_units),
        "e_units": list(args.e_units),
        "dz_units": list(args.dz_units),
        "dx_units": list(args.dx_units),
        "lr": args.step1_lr,
        "lr_theta": args.generator_lr,
        "lr_z": 1e-7,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "g_d_freq": 1,
        "kl_weight": 5e-5,
        "save_model": False,
        "save_res": False,
    }
    model = BGM(params=params, random_seed=args.seed)
    # Explicitly build every network so count/save/load calls are reliable.
    model.g_net(tf.zeros((1, args.bgm_z_dim), dtype=tf.float32), training=False)
    model.e_net(tf.zeros((1, args.ae_latent_dim), dtype=tf.float32), training=False)
    model.dz_net(tf.zeros((1, args.bgm_z_dim), dtype=tf.float32), training=False)
    model.dx_net(tf.zeros((1, args.ae_latent_dim), dtype=tf.float32), training=False)
    return model


def generate_latent_mean(model: BGM, prior_z: np.ndarray) -> np.ndarray:
    mean, _ = model.g_net(tf.convert_to_tensor(prior_z, dtype=tf.float32), training=False)
    return mean.numpy().astype(np.float32)


def decode_bgm_latent(
    standardized: np.ndarray,
    latent_mean: np.ndarray,
    latent_sd: np.ndarray,
    ae: ScDiffusionAutoencoder,
) -> np.ndarray:
    ae_latent = standardized * latent_sd[None, :] + latent_mean[None, :]
    return decode_array(ae, ae_latent, batch_size=512, device="cpu")


def evaluate_generator(
    model: BGM,
    prior_z: np.ndarray,
    train_real: np.ndarray,
    validation_real: np.ndarray,
    latent_mean: np.ndarray,
    latent_sd: np.ndarray,
    ae: ScDiffusionAutoencoder,
    args: argparse.Namespace,
) -> tuple[dict[str, float], np.ndarray]:
    generated_standardized = generate_latent_mean(model, prior_z)
    generated = decode_bgm_latent(generated_standardized, latent_mean, latent_sd, ae)
    train_metrics = distribution_metrics(
        train_real, generated, args.n_eval, args.mmd_n_eval, args.seed + 101
    )
    validation_metrics = distribution_metrics(
        validation_real, generated, args.n_eval, args.mmd_n_eval, args.seed + 202
    )
    result = {}
    result.update({f"train_{key}": value for key, value in train_metrics.items()})
    result.update({f"validation_{key}": value for key, value in validation_metrics.items()})
    mean_var = model.g_net(tf.convert_to_tensor(prior_z[: min(512, len(prior_z))]), training=False)[1]
    variance = mean_var.numpy().ravel()
    result.update(
        {
            "bgm_var_q01": float(np.quantile(variance, 0.01)),
            "bgm_var_q50": float(np.quantile(variance, 0.50)),
            "bgm_var_q99": float(np.quantile(variance, 0.99)),
        }
    )
    return result, generated


def save_bgm_weights(model: BGM, generator: Path, encoder: Path | None = None) -> None:
    model.g_net.save_weights(str(generator))
    if encoder is not None:
        model.e_net.save_weights(str(encoder))


def run_step1(
    model: BGM,
    train_latent: np.ndarray,
    train_real: np.ndarray,
    validation_real: np.ndarray,
    latent_mean: np.ndarray,
    latent_sd: np.ndarray,
    ae: ScDiffusionAutoencoder,
    prior_z_eval: np.ndarray,
    args: argparse.Namespace,
    output: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed + 300)
    history: list[dict[str, float]] = []
    best_ilisi = -math.inf
    best_generator = output / "step1_generator.weights.h5"
    best_encoder = output / "step1_encoder.weights.h5"
    checkpoint_dir = output / "step1_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print("\nBayesGM Step 1 EGM warm start ...")
    for iteration in range(1, args.step1_iterations + 1):
        for _ in range(model.params["g_d_freq"]):
            idx = rng.integers(0, len(train_latent), size=args.step1_batch_size)
            batch_x = tf.convert_to_tensor(train_latent[idx], dtype=tf.float32)
            batch_z = tf.convert_to_tensor(
                rng.standard_normal((args.step1_batch_size, args.bgm_z_dim)).astype(np.float32)
            )
            dz_loss, dx_loss, d_loss = model.train_disc_step(batch_z, batch_x)

        idx = rng.integers(0, len(train_latent), size=args.step1_batch_size)
        batch_x = tf.convert_to_tensor(train_latent[idx], dtype=tf.float32)
        batch_z = tf.convert_to_tensor(
            rng.standard_normal((args.step1_batch_size, args.bgm_z_dim)).astype(np.float32)
        )
        losses = model.train_gen_step(batch_z, batch_x)

        should_eval = iteration % args.step1_eval_every == 0 or iteration == args.step1_iterations
        if not should_eval:
            continue
        g_adv, e_adv, l2_z, l2_x, var_loss, ge_loss = [float(x.numpy()) for x in losses]
        metrics, generated = evaluate_generator(
            model,
            prior_z_eval,
            train_real,
            validation_real,
            latent_mean,
            latent_sd,
            ae,
            args,
        )
        row = {
            "iteration": iteration,
            "g_loss_adv": g_adv,
            "e_loss_adv": e_adv,
            "l2_loss_z": l2_z,
            "l2_loss_x": l2_x,
            "variance_loss": var_loss,
            "g_e_loss": ge_loss,
            "dz_loss": float(dz_loss.numpy()),
            "dx_loss": float(dx_loss.numpy()),
            "d_loss": float(d_loss.numpy()),
            **metrics,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output / "step1_history.csv", index=False)
        print(
            f"Step1 iter={iteration} | val_iLISI={metrics['validation_ilisi_pca20']:.4f} "
            f"| val_MMD50={metrics['validation_mmd_pca50']:.4f} "
            f"| train_iLISI={metrics['train_ilisi_pca20']:.4f}"
        )

        if iteration in set(args.step1_save_iters):
            directory = checkpoint_dir / f"iter_{iteration:06d}"
            directory.mkdir(exist_ok=True)
            save_bgm_weights(
                model,
                directory / "generator.weights.h5",
                directory / "encoder.weights.h5",
            )
            np.save(directory / "validation_generated.npy", generated)

        score = metrics["validation_ilisi_pca20"]
        if score > best_ilisi:
            best_ilisi = score
            save_bgm_weights(model, best_generator, best_encoder)
            np.save(output / "step1_validation_generated.npy", generated)
            with (output / "step1.json").open("w") as handle:
                json.dump(
                    {"iteration": iteration, "validation_ilisi_pca20": score},
                    handle,
                    indent=2,
                )
            print("  new validation-selected Step 1 checkpoint")

    model.g_net.load_weights(str(best_generator))
    model.e_net.load_weights(str(best_encoder))
    return pd.DataFrame(history)


def inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


def initialize_variance_head(model: BGM, initial_variance: float, epsilon: float) -> None:
    target = max(initial_variance - epsilon, 1e-8)
    model.g_net.var_layer.kernel.assign(tf.zeros_like(model.g_net.var_layer.kernel))
    model.g_net.var_layer.bias.assign(
        tf.fill(tf.shape(model.g_net.var_layer.bias), tf.cast(inverse_softplus(target), tf.float32))
    )


def pairwise_sq_dist_tf(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.maximum(
        tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        + tf.transpose(tf.reduce_sum(tf.square(y), axis=1, keepdims=True))
        - 2.0 * tf.matmul(x, y, transpose_b=True),
        0.0,
    )


def adaptive_mmd_tf(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    total = tf.concat([x, y], axis=0)
    total_dist = pairwise_sq_dist_tf(total, total)
    flat = tf.reshape(total_dist, [-1])
    positive = tf.boolean_mask(flat, flat > 0)
    positive = tf.sort(positive)
    median = tf.cond(
        tf.size(positive) > 0,
        lambda: positive[tf.size(positive) // 2],
        lambda: tf.constant(1.0, dtype=tf.float32),
    )
    median = tf.stop_gradient(tf.maximum(median, 1e-6))
    xx = pairwise_sq_dist_tf(x, x)
    yy = pairwise_sq_dist_tf(y, y)
    xy = pairwise_sq_dist_tf(x, y)
    result = tf.constant(0.0, tf.float32)
    for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0):
        bandwidth = median * multiplier
        result += (
            tf.reduce_mean(tf.exp(-xx / (2.0 * bandwidth)))
            + tf.reduce_mean(tf.exp(-yy / (2.0 * bandwidth)))
            - 2.0 * tf.reduce_mean(tf.exp(-xy / (2.0 * bandwidth)))
        )
    return result


def apply_gradients_clipped(
    optimizer: tf.keras.optimizers.Optimizer,
    gradients: Iterable[tf.Tensor | None],
    variables: Sequence[tf.Variable],
    clip_norm: float,
) -> None:
    pairs = [(g, v) for g, v in zip(gradients, variables) if g is not None]
    if not pairs:
        return
    grad_values, var_values = zip(*pairs)
    clipped, _ = tf.clip_by_global_norm(list(grad_values), clip_norm)
    optimizer.apply_gradients(zip(clipped, var_values))


def make_step2_update(
    model: BGM,
    args: argparse.Namespace,
    anchor_variables: list[tf.Tensor],
):
    # Freeze Step-1 BatchNorm statistics; dense weights remain trainable.
    model.g_net.norm_layer.trainable = False
    variance_variables = list(model.g_net.var_layer.trainable_variables)
    variance_ids = {id(v) for v in variance_variables}
    mean_variables = [v for v in model.g_net.trainable_variables if id(v) not in variance_ids]
    anchor_by_name = {v.name: value for v, value in zip(mean_variables, anchor_variables)}
    mean_optimizer = tf.keras.optimizers.Adam(args.generator_lr, beta_1=0.9, beta_2=0.99)
    variance_optimizer = tf.keras.optimizers.Adam(args.variance_lr, beta_1=0.9, beta_2=0.99)

    @tf.function(reduce_retracing=True)
    def update(batch_z: tf.Tensor, batch_x: tf.Tensor, prior_z: tf.Tensor):
        with tf.GradientTape(persistent=True) as tape:
            mu, variance = model.g_net(batch_z, training=False)
            variance_safe = tf.clip_by_value(variance, args.variance_min, args.variance_max)
            nll = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(batch_x - mu) / (2.0 * variance_safe)
                    + 0.5 * tf.math.log(variance_safe),
                    axis=1,
                )
            )
            mse = tf.reduce_mean(tf.square(batch_x - mu))

            prior_mu, _ = model.g_net(prior_z, training=False)
            mmd_count = tf.minimum(tf.shape(prior_mu)[0], tf.shape(batch_x)[0])
            latent_mmd = adaptive_mmd_tf(prior_mu[:mmd_count], batch_x[:mmd_count])

            anchor_terms = [
                tf.reduce_mean(tf.square(v - anchor_by_name[v.name]))
                for v in mean_variables
                if v.name in anchor_by_name
            ]
            anchor = tf.add_n(anchor_terms) / max(len(anchor_terms), 1)
            log_prior = tf.reduce_mean(
                tf.square(tf.math.log(variance_safe) - math.log(args.variance_init))
            )
            bounds = tf.reduce_mean(
                tf.square(tf.nn.relu(args.variance_min - variance))
                + tf.square(tf.nn.relu(variance - args.variance_max))
            )
            variance_regularizer = args.variance_log_prior_weight * log_prior + 10.0 * bounds
            total = (
                nll
                + args.mmd_weight * latent_mmd
                + args.anchor_weight * anchor
                + variance_regularizer
            )

        mean_gradients = tape.gradient(total, mean_variables)
        variance_gradients = tape.gradient(total, variance_variables)
        del tape
        apply_gradients_clipped(mean_optimizer, mean_gradients, mean_variables, args.gradient_clip)
        apply_gradients_clipped(
            variance_optimizer, variance_gradients, variance_variables, args.gradient_clip
        )
        return total, nll, mse, latent_mmd, anchor, variance_regularizer

    return update, mean_variables


def run_step2(
    model: BGM,
    train_latent: np.ndarray,
    train_real: np.ndarray,
    validation_real: np.ndarray,
    latent_mean: np.ndarray,
    latent_sd: np.ndarray,
    ae: ScDiffusionAutoencoder,
    prior_z_eval: np.ndarray,
    args: argparse.Namespace,
    output: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed + 400)
    data_z = model.e_net(tf.convert_to_tensor(train_latent), training=False).numpy().astype(np.float32)
    initialize_variance_head(model, args.variance_init, args.generator_variance_eps)

    # Snapshot mean-network parameters after the selected Step 1 checkpoint.
    model.g_net.norm_layer.trainable = False
    var_ids = {id(v) for v in model.g_net.var_layer.trainable_variables}
    mean_vars_before = [v for v in model.g_net.trainable_variables if id(v) not in var_ids]
    anchor = [tf.identity(v) for v in mean_vars_before]
    update, _ = make_step2_update(model, args, anchor)

    history: list[dict[str, float]] = []
    best_score = -math.inf
    best_generator = output / "generator.weights.h5"
    best_data_z = output / "data_z.npy"
    checkpoint_dir = output / "step2_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    baseline_metrics, baseline_generated = evaluate_generator(
        model,
        prior_z_eval,
        train_real,
        validation_real,
        latent_mean,
        latent_sd,
        ae,
        args,
    )
    baseline = {"epoch": -1, **baseline_metrics}
    history.append(baseline)
    best_score = baseline_metrics["validation_ilisi_pca20"]
    model.g_net.save_weights(str(best_generator))
    np.save(best_data_z, data_z)
    np.save(output / "validation_generated.npy", baseline_generated)
    with (output / "step2.json").open("w") as handle:
        json.dump(
            {"epoch": -1, "validation_ilisi_pca20": best_score},
            handle,
            indent=2,
        )

    print("\nBayesGM distribution-preserving Step 2 ...")
    print(
        f"baseline validation iLISI={best_score:.4f}; variance init={args.variance_init}; "
        f"generator lr={args.generator_lr}; variance lr={args.variance_lr}"
    )
    last_losses = [math.nan] * 6
    for epoch in range(0, args.step2_epochs + 1):
        permutation = rng.permutation(len(train_latent))
        batches = len(train_latent) // args.step2_batch_size
        progress = tqdm(range(batches), desc=f"Step 2 epoch {epoch}/{args.step2_epochs}")
        for batch_number in progress:
            indices = permutation[
                batch_number * args.step2_batch_size : (batch_number + 1) * args.step2_batch_size
            ]
            batch_z = tf.convert_to_tensor(data_z[indices], dtype=tf.float32)
            batch_x = tf.convert_to_tensor(train_latent[indices], dtype=tf.float32)
            prior_count = min(args.mmd_samples, args.step2_batch_size)
            prior_z = tf.convert_to_tensor(
                rng.standard_normal((prior_count, args.bgm_z_dim)).astype(np.float32)
            )
            values = update(batch_z, batch_x, prior_z)
            last_losses = [float(value.numpy()) for value in values]
            progress.set_postfix(
                total=f"{last_losses[0]:.3f}",
                mse=f"{last_losses[2]:.5f}",
                mmd=f"{last_losses[3]:.4f}",
            )

        should_eval = epoch % args.step2_eval_every == 0 or epoch == args.step2_epochs
        if not should_eval:
            continue
        metrics, generated = evaluate_generator(
            model,
            prior_z_eval,
            train_real,
            validation_real,
            latent_mean,
            latent_sd,
            ae,
            args,
        )
        row = {
            "epoch": epoch,
            "loss_total_last": last_losses[0],
            "loss_nll_last": last_losses[1],
            "loss_mse_last": last_losses[2],
            "latent_mmd_last": last_losses[3],
            "anchor_last": last_losses[4],
            "variance_regularizer_last": last_losses[5],
            **metrics,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output / "step2_history.csv", index=False)
        score = metrics["validation_ilisi_pca20"]
        print(
            f"Step2 epoch={epoch} | val_iLISI={score:.4f} "
            f"| val_MMD50={metrics['validation_mmd_pca50']:.4f} "
            f"| var={metrics['bgm_var_q01']:.4f}/{metrics['bgm_var_q50']:.4f}/"
            f"{metrics['bgm_var_q99']:.4f}"
        )

        if epoch % args.step2_save_every == 0:
            directory = checkpoint_dir / f"epoch_{epoch:04d}"
            directory.mkdir(exist_ok=True)
            model.g_net.save_weights(str(directory / "generator.weights.h5"))
            np.save(directory / "validation_generated.npy", generated)

        if score > best_score:
            best_score = score
            model.g_net.save_weights(str(best_generator))
            np.save(best_data_z, data_z)
            np.save(output / "validation_generated.npy", generated)
            with (output / "step2.json").open("w") as handle:
                json.dump(
                    {"epoch": epoch, "validation_ilisi_pca20": score},
                    handle,
                    indent=2,
                )
            print("  new validation-selected Step 2 checkpoint")

    model.g_net.load_weights(str(best_generator))
    return pd.DataFrame(history)


def write_run_config(
    args: argparse.Namespace,
    output: Path,
    train: ad.AnnData,
    validation: ad.AnnData,
) -> None:
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    config.update(
        {
            "protocol": "outer 80/20; validation is 10% of outer train pool",
            "model_selection": "validation_ilisi_pca20 only",
            "final_test_loaded_during_training": False,
            "ae_type": "deterministic scDiffusion-style AE; not variational",
            "n_train": train.n_obs,
            "n_validation": validation.n_obs,
            "n_genes": train.n_vars,
            "gene_order_sha256": sha256_lines(train.var_names.astype(str)),
            "train_barcodes_sha256": sha256_lines(train.obs_names.astype(str)),
            "validation_barcodes_sha256": sha256_lines(validation.obs_names.astype(str)),
        }
    )
    with (output / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)


def main() -> None:
    args = parse_args()
    if not (args.variance_min < args.variance_init < args.variance_max):
        raise ValueError("variance-min < variance-init < variance-max is required")
    configure_tensorflow()
    seed_everything(args.seed)
    install_generator_variance_epsilon(args.generator_variance_eps)
    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    split = args.split_dir.expanduser()
    train_path = split / "pbmc68k_train_raw.h5ad"
    validation_path = split / "pbmc68k_validation_raw.h5ad"

    print("Loading TRAIN for gradients:", train_path)
    train = load_log_expression(train_path, args.target_sum)
    print("Loading VALIDATION for checkpoint selection:", validation_path)
    validation = load_log_expression(validation_path, args.target_sum)
    assert_disjoint(train, validation)
    write_run_config(args, output, train, validation)
    # Force a fixed-width Unicode array so eval can load with allow_pickle=False.
    np.save(output / "genes.npy", np.asarray(train.var_names.astype(str), dtype="U"))

    ae, train_encoded, validation_encoded = load_ae_artifacts(train, validation, args)
    latent_mean = train_encoded.mean(axis=0).astype(np.float32)
    latent_sd = train_encoded.std(axis=0).astype(np.float32)
    latent_sd = np.maximum(latent_sd, 1e-4)
    train_latent = ((train_encoded - latent_mean) / latent_sd).astype(np.float32)
    validation_latent = ((validation_encoded - latent_mean) / latent_sd).astype(np.float32)
    np.savez(
        output / "latent_scale.npz",
        mean=latent_mean,
        sd=latent_sd,
        train_latent_mean=train_latent.mean(axis=0),
        train_latent_sd=train_latent.std(axis=0),
    )
    # Kept only for auditing; validation latent is never used for gradients.
    np.save(output / "validation_latent.npy", validation_latent)

    # Fixed expression subsets make every checkpoint directly comparable.
    train_real = select_fixed_rows(train.X, args.n_eval, args.seed + 11)
    validation_real = select_fixed_rows(validation.X, args.n_eval, args.seed + 12)
    np.save(output / "validation_real.npy", validation_real)

    gc.collect()

    model = build_bgm(args, output)
    print("BayesGM x_dim:", args.ae_latent_dim)
    print("BayesGM z_dim:", args.bgm_z_dim)
    print("Generator parameters:", model.g_net.count_params())
    print("Encoder parameters:", model.e_net.count_params())

    eval_rng = np.random.default_rng(args.validation_generation_seed)
    prior_z_eval = eval_rng.standard_normal((args.n_eval, args.bgm_z_dim)).astype(np.float32)
    np.save(output / "validation_prior_z.npy", prior_z_eval)

    run_step1(
        model,
        train_latent,
        train_real,
        validation_real,
        latent_mean,
        latent_sd,
        ae,
        prior_z_eval,
        args,
        output,
    )
    run_step2(
        model,
        train_latent,
        train_real,
        validation_real,
        latent_mean,
        latent_sd,
        ae,
        prior_z_eval,
        args,
        output,
    )

    # Freeze checkpoint, then make one separately seeded unconditional sample.
    final_rng = np.random.default_rng(args.final_generation_seed)
    final_prior_z = final_rng.standard_normal(
        (args.final_n_generate, args.bgm_z_dim)
    ).astype(np.float32)
    final_standardized = generate_latent_mean(model, final_prior_z)
    final_generated = decode_bgm_latent(final_standardized, latent_mean, latent_sd, ae)
    np.save(output / "generated.npy", final_generated)
    np.save(output / "prior_z.npy", final_prior_z)

    print("\nTraining complete. FINAL TEST HAS NOT BEEN LOADED OR INSPECTED.")
    print("Validation-selected generator:", output / "generator.weights.h5")
    print("Frozen generated expression:", output / "generated.npy")
    print("Run evaluate.ipynb only after all model choices are frozen.")


if __name__ == "__main__":
    main()

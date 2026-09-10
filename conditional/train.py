#!/usr/bin/env python3
"""Train conditional BayesGM on an arbitrary categorical observation label.

The BayesGM operates in the latent space of the frozen Negative-Binomial RNA
autoencoder implemented in ``autoencoder.py``.  Real train/validation H5ADs
must contain raw UMI counts plus the requested ``obs[label_key]`` labels.  The
PBMC68k filenames and ``obs['celltype']`` remain the command-line defaults, but
the label vocabulary can be derived deterministically from any training split
or pinned explicitly with ``--label-vocabulary-json``.  The final test split is
deliberately never opened here.

The default conditional generator is

    y_emb = Embedding(y)
    (mu_x, var_x) = G(concat([z, y_emb]))

The additive-per-block experiment instead uses a 32-dimensional embedding,
adds it to z, and adds an independently projected copy before every ResMLP
block.  Both variants keep the heteroscedastic diagonal Gaussian likelihood
used by BayesGM.
The encoder stays unconditional.  During Step 2, every cell-specific latent
variable z_i is optimized jointly with its observed class label y_i.

Training phases
---------------
1. Conditional EGM warm start.
2. Alternating generator / per-cell latent MAP updates with
   conditional MMD, Step-1 anchor, and learnable-variance regularization.
3. Validation checkpoint selection by the shared per-class PCA20/kNN10 iLISI.
   Nearest-centroid label fidelity remains a reported diagnostic only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/conditional_bgm_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/conditional_bgm_matplotlib")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scib
import tensorflow as tf
import torch
from bayesgm.models.bgm import BGM
from scipy import sparse
from sklearn.decomposition import PCA
from tqdm import tqdm

from autoencoder import (
    AEBundle,
    decode_mu_matrix,
    decode_sample_counts,
    load_ae_bundle,
    normalize_total_log1p_dense,
    row_sums,
)

SELECTION_METRIC = "validation_macro_ilisi_pca20"


def configure_tensorflow() -> None:
    """Enable TensorFlow GPU memory growth when a GPU is visible."""
    for device in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            # TensorFlow may already have initialized the device.
            pass


def dense_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def get_rows(matrix, indices: np.ndarray) -> np.ndarray:
    return dense_float32(matrix[indices])


def assert_disjoint(train: ad.AnnData, validation: ad.AnnData) -> None:
    overlap = set(train.obs_names) & set(validation.obs_names)
    if overlap:
        raise RuntimeError(f"Train/validation overlap: {len(overlap)} cells")
    if not np.array_equal(train.var_names, validation.var_names):
        raise RuntimeError("Train and validation genes/order differ")


def load_raw_counts(path: Path) -> ad.AnnData:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    data = sc.read_h5ad(path)
    if not data.var_names.is_unique:
        raise ValueError(f"{path} contains duplicate gene names")
    if not data.obs_names.is_unique:
        raise ValueError(f"{path} contains duplicate cell barcodes")
    values = data.X.data if sparse.issparse(data.X) else np.asarray(data.X).ravel()
    if values.size:
        if float(values.min()) < 0 or not np.isfinite(values).all():
            raise ValueError(f"{path} must contain finite non-negative raw counts")
        if not np.allclose(values, np.rint(values), atol=1e-5, rtol=0.0):
            raise ValueError(f"{path} is not raw integer UMI counts")
    if np.any(row_sums(data.X) <= 0):
        raise ValueError(f"{path} contains zero-library cells")
    return data


def normalize_log_expression_inplace(data: ad.AnnData, target_sum: float) -> None:
    """Convert a raw-count AnnData copy to normalize_total + log1p once."""
    sc.pp.normalize_total(data, target_sum=target_sum)
    sc.pp.log1p(data)
    if sparse.issparse(data.X):
        data.X = data.X.tocsr().astype(np.float32)
    else:
        data.X = np.asarray(data.X, dtype=np.float32)


def load_ae_artifacts(
    train: ad.AnnData,
    validation: ad.AnnData,
    args: argparse.Namespace,
) -> AEBundle:
    directory = args.ae_artifact_dir.expanduser()
    required = [
        directory / "metadata.json",
        directory / "model.pt",
        directory / "train_latent.npy",
        directory / "validation_latent.npy",
        directory / "train_library_size.npy",
        directory / "validation_library_size.npy",
        directory / "genes.npy",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Negative-Binomial AE artifact: {path}. Run "
                "autoencoder.py first or set --ae-artifact-dir correctly."
            )

    bundle = load_ae_bundle(directory, device=args.decode_device)
    metadata = bundle.metadata

    expected = {
        "n_genes": int(train.n_vars),
        "latent_dim": int(args.ae_latent_dim),
        "train_n_obs": int(train.n_obs),
        "validation_n_obs": int(validation.n_obs),
        "gene_order_sha256": sha256_lines(train.var_names.astype(str)),
        "train_obs_sha256": sha256_lines(train.obs_names.astype(str)),
        "validation_obs_sha256": sha256_lines(validation.obs_names.astype(str)),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"AE artifact mismatch for {key}: "
                f"stored={metadata.get(key)!r}, current={value!r}"
            )

    likelihood = str(metadata.get("likelihood", ""))
    if "negativebinomial" not in likelihood.replace(" ", "").lower():
        raise RuntimeError(
            "AE metadata is not a Negative-Binomial model: "
            f"likelihood={likelihood!r}"
        )
    if metadata.get("final_test_loaded_during_training") is not False:
        raise RuntimeError(
            "AE metadata must certify final_test_loaded_during_training=false"
        )
    if str(metadata.get("encoder_input", "")) != "log1p(raw_counts)":
        raise RuntimeError("AE metadata encoder_input is not log1p(raw_counts)")
    decoder_mean = str(metadata.get("decoder_mean", ""))
    if "library_size" not in decoder_mean or "softmax" not in decoder_mean:
        raise RuntimeError("AE metadata decoder_mean is not library_size*softmax")

    train_latent = np.asarray(bundle.train_latent, dtype=np.float32)
    validation_latent = np.asarray(bundle.validation_latent, dtype=np.float32)
    expected_train_shape = (int(train.n_obs), int(args.ae_latent_dim))
    expected_validation_shape = (int(validation.n_obs), int(args.ae_latent_dim))
    if train_latent.shape != expected_train_shape:
        raise RuntimeError(
            f"Unexpected train latent shape: {train_latent.shape}; "
            f"expected {expected_train_shape}"
        )
    if validation_latent.shape != expected_validation_shape:
        raise RuntimeError(
            f"Unexpected validation latent shape: {validation_latent.shape}; "
            f"expected {expected_validation_shape}"
        )
    if not np.isfinite(train_latent).all() or not np.isfinite(validation_latent).all():
        raise RuntimeError("AE latent artifacts contain NaN or Inf")

    # Existing AENB runs saved pandas Index values, which are commonly object
    # dtype in NumPy.  The file is a locally produced, hash-checked AE artifact.
    genes = np.load(directory / "genes.npy", allow_pickle=True).astype(str)
    if not np.array_equal(genes, train.var_names.astype(str).to_numpy()):
        raise RuntimeError("AE genes.npy does not match the current gene order")

    expected_libraries = (row_sums(train.X), row_sums(validation.X))
    stored_libraries = (
        np.asarray(bundle.train_library_size, dtype=np.float32).reshape(-1),
        np.asarray(bundle.validation_library_size, dtype=np.float32).reshape(-1),
    )
    for name, stored, current in zip(
        ("train", "validation"), stored_libraries, expected_libraries
    ):
        if stored.shape != current.shape or not np.all(np.isfinite(stored)):
            raise RuntimeError(f"Invalid {name} library-size artifact")
        if np.any(stored <= 0) or not np.allclose(stored, current, atol=1e-3, rtol=0):
            raise RuntimeError(
                f"{name} library-size artifact does not match raw H5AD row sums"
            )

    print("Loaded frozen Negative-Binomial autoencoder:", directory / "model.pt")
    print("Selected AE epoch:", metadata.get("best_epoch", "unknown"))
    print("Selected AE validation NLL/gene:", metadata.get("best_validation_nll_per_gene", "unknown"))
    return bundle


def rbf_mmd_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    total = np.concatenate([x, y], axis=0)
    squared_norm = np.sum(total * total, axis=1)
    distances = np.maximum(
        squared_norm[:, None] + squared_norm[None, :] - 2.0 * total @ total.T,
        0.0,
    )
    n_total = len(total)
    bandwidth = float(distances.sum()) / max(n_total * n_total - n_total, 1)
    bandwidth = max(bandwidth / (2.0 ** (5 // 2)), 1e-8)
    kernels = np.zeros_like(distances)
    for index in range(5):
        kernels += np.exp(-distances / (bandwidth * (2.0 ** index)))
    n_x = len(x)
    return float(
        kernels[:n_x, :n_x].mean()
        + kernels[n_x:, n_x:].mean()
        - kernels[:n_x, n_x:].mean()
        - kernels[n_x:, :n_x].mean()
    )


def distribution_metrics(
    real: np.ndarray,
    generated: np.ndarray,
    n_eval: int,
    mmd_n_eval: int,
    seed: int,
    ilisi_knn_neighbors: int = 10,
) -> dict[str, float]:
    """Evaluate one condition using the shared final graph-iLISI protocol.

    To keep scores comparable across conditions and checkpoints, iLISI is
    reported only when the joint real/generated sample contains at least
    ``ilisi_knn_neighbors + 1`` cells. Small conditions still receive MMD and
    descriptive metrics instead of making the whole evaluation invalid. The
    eligible-condition calculation uses the frozen graph-iLISI protocol:
    joint PCA50, first 20 PCs, Scanpy kNN graph, then scIB ``type_="knn"``.
    """
    n = min(int(n_eval), len(real), len(generated))
    if n < 2:
        raise ValueError("At least two real and generated cells are required")

    rng = np.random.default_rng(seed)
    real_idx = rng.choice(len(real), n, replace=False)
    generated_idx = rng.choice(len(generated), n, replace=False)
    real_eval = np.asarray(real[real_idx], dtype=np.float32)
    generated_eval = np.asarray(generated[generated_idx], dtype=np.float32)
    joint = np.concatenate([real_eval, generated_eval], axis=0)
    n_components = min(50, joint.shape[0] - 1, joint.shape[1])
    pcs = PCA(n_components=n_components, random_state=seed).fit_transform(joint)

    ilisi_knn_neighbors = int(ilisi_knn_neighbors)
    if ilisi_knn_neighbors <= 0:
        raise ValueError("ilisi_knn_neighbors must be positive")
    ilisi_eligible = joint.shape[0] - 1 >= ilisi_knn_neighbors
    ilisi = math.nan
    if ilisi_eligible:
        graph = ad.AnnData(X=np.zeros((2 * n, 1), dtype=np.float32))
        representation = "X_metric_pca"
        graph.obsm[representation] = pcs[:, : min(20, n_components)].astype(
            np.float32
        )
        graph.obs["origin"] = pd.Categorical(
            ["real"] * n + ["generated"] * n,
            categories=["real", "generated"],
        )
        sc.pp.neighbors(
            graph,
            use_rep=representation,
            n_neighbors=ilisi_knn_neighbors,
            random_state=seed,
        )
        ilisi = float(
            scib.me.ilisi_graph(
                graph,
                batch_key="origin",
                type_="knn",
            )
        )

    m = min(int(mmd_n_eval), n)
    mmd = rbf_mmd_np(
        pcs[:m, : min(50, n_components)],
        pcs[n : n + m, : min(50, n_components)],
    )
    real_mean = real_eval.mean(axis=0)
    generated_mean = generated_eval.mean(axis=0)
    if np.std(real_mean) == 0 or np.std(generated_mean) == 0:
        correlation = math.nan
    else:
        correlation = float(np.corrcoef(real_mean, generated_mean)[0, 1])
    return {
        "eval_n_per_origin": float(n),
        "ilisi_knn_neighbors": float(ilisi_knn_neighbors),
        "ilisi_eligible": float(ilisi_eligible),
        "ilisi_pca20": ilisi,
        "mmd_pca50": mmd,
        "real_zero_frac": float(np.mean(real_eval <= 0)),
        "generated_zero_frac": float(np.mean(generated_eval <= 0)),
        "real_cell_sum_mean": float(real_eval.sum(axis=1).mean()),
        "generated_cell_sum_mean": float(generated_eval.sum(axis=1).mean()),
        "gene_mean_corr": correlation,
    }


def decode_bgm_latent(
    standardized: np.ndarray,
    latent_mean: np.ndarray,
    latent_sd: np.ndarray,
    ae: AEBundle,
    library_size: np.ndarray,
    args: argparse.Namespace,
    decoder_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-standardize and decode with the frozen NB observation model.

    Unlike the former MSE AE adapter, NB latents are never L2-normalized.  The
    returned expression is normalized/log1p exactly once after count decoding.
    """
    ae_latent = standardized * latent_sd[None, :] + latent_mean[None, :]
    if args.decoder_mode == "sample":
        decoded = decode_sample_counts(
            ae.model,
            ae_latent,
            library_size,
            batch_size=args.decode_batch_size,
            device=args.decode_device,
            seed=decoder_seed,
        )
    elif args.decoder_mode == "mean":
        decoded = decode_mu_matrix(
            ae.model,
            ae_latent,
            library_size,
            batch_size=args.decode_batch_size,
            device=args.decode_device,
        )
    else:  # guarded by argparse; retained for programmatic calls.
        raise ValueError(f"Unknown decoder mode: {args.decoder_mode}")
    expression = normalize_total_log1p_dense(decoded, target_sum=args.target_sum)
    return expression, np.asarray(decoded, dtype=np.float32)


def sample_library_sizes(
    train_library_size: np.ndarray,
    train_labels: np.ndarray,
    requested_labels: np.ndarray,
    mode: str,
    seed: int,
    shrinkage_strength: float = 200.0,
) -> np.ndarray:
    """Draw library sizes using TRAIN-only information.

    ``label-lognormal-shrunk`` fits one Normal distribution to log library
    sizes per official cell type.  The class mean is kept class-specific, while
    the class variance is shrunk toward the global TRAIN log-library variance:

        w_c = n_c / (n_c + shrinkage_strength)
        var_c* = w_c * var_c + (1 - w_c) * var_global

    This keeps rare official cell types stable without leaking validation/test
    information into the decoder-side size-factor model.
    """
    libraries = np.asarray(train_library_size, dtype=np.float32).reshape(-1)
    train_labels = np.asarray(train_labels, dtype=np.int32).reshape(-1)
    requested_labels = np.asarray(requested_labels, dtype=np.int32).reshape(-1)
    if len(libraries) != len(train_labels):
        raise ValueError("train library sizes and labels have different lengths")
    if np.any(libraries <= 0) or not np.isfinite(libraries).all():
        raise ValueError("training library sizes must be finite and positive")
    if not np.isfinite(shrinkage_strength) or float(shrinkage_strength) < 0:
        raise ValueError("library shrinkage strength must be finite and non-negative")

    rng = np.random.default_rng(seed)
    result = np.empty(len(requested_labels), dtype=np.float32)
    if mode == "global-empirical":
        result[:] = rng.choice(libraries, size=len(result), replace=True)
    elif mode == "label-empirical":
        for class_id in np.unique(requested_labels):
            target = np.flatnonzero(requested_labels == class_id)
            pool = libraries[train_labels == class_id]
            if len(pool) == 0:
                raise ValueError(f"No training library sizes for class {class_id}")
            result[target] = rng.choice(pool, size=len(target), replace=True)
    elif mode == "label-lognormal-shrunk":
        log_libraries = np.log(libraries.astype(np.float64))
        global_var = float(np.var(log_libraries, ddof=1)) if len(log_libraries) > 1 else 0.0
        global_var = max(global_var, 1e-8)
        tau = float(shrinkage_strength)
        for class_id in np.unique(requested_labels):
            target = np.flatnonzero(requested_labels == class_id)
            local = log_libraries[train_labels == class_id]
            if len(local) == 0:
                raise ValueError(f"No training library sizes for class {class_id}")
            local_mean = float(np.mean(local))
            local_var = float(np.var(local, ddof=1)) if len(local) > 1 else global_var
            local_var = max(local_var, 1e-8)
            weight = len(local) / (len(local) + tau) if tau > 0 else 1.0
            shrunk_var = weight * local_var + (1.0 - weight) * global_var
            sampled_log = rng.normal(
                loc=local_mean,
                scale=np.sqrt(max(shrunk_var, 1e-8)),
                size=len(target),
            )
            result[target] = np.exp(sampled_log).astype(np.float32)
    else:
        raise ValueError(
            "library mode must be global-empirical, label-empirical, "
            "or label-lognormal-shrunk"
        )
    if np.any(result <= 0) or not np.isfinite(result).all():
        raise FloatingPointError("Sampled library sizes are invalid")
    return result


def inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-12)
    return float(np.log(np.expm1(value)))


def initialize_variance_head(model, initial_variance: float, epsilon: float) -> None:
    target = max(float(initial_variance) - float(epsilon), 1e-8)
    if not model.g_net.var_layer.built:
        raise RuntimeError("Generator variance head must be built before initialization")
    model.g_net.var_layer.kernel.assign(
        tf.zeros_like(model.g_net.var_layer.kernel)
    )
    model.g_net.var_layer.bias.assign(
        tf.fill(
            tf.shape(model.g_net.var_layer.bias),
            tf.cast(inverse_softplus(target), tf.float32),
        )
    )


def pairwise_sq_dist_tf(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.maximum(
        tf.reduce_sum(tf.square(x), axis=1, keepdims=True)
        + tf.transpose(tf.reduce_sum(tf.square(y), axis=1, keepdims=True))
        - 2.0 * tf.matmul(x, y, transpose_b=True),
        0.0,
    )


def adaptive_mmd_tf(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
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
    result = tf.constant(0.0, dtype=tf.float32)
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
    gradients: Iterable[tf.Tensor | tf.IndexedSlices | None],
    variables: Sequence[tf.Variable],
    clip_norm: float,
) -> None:
    pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, variables)
        if gradient is not None
    ]
    if not pairs:
        return
    gradient_values, variable_values = zip(*pairs)
    clipped, _ = tf.clip_by_global_norm(list(gradient_values), float(clip_norm))
    optimizer.apply_gradients(zip(clipped, variable_values))


class ConditionalVariationalGenerator(tf.keras.Model):
    """Dense conditional generator with a learnable diagonal variance head."""

    def __init__(
        self,
        z_dim: int,
        output_dim: int,
        num_classes: int,
        label_embed_dim: int,
        hidden_units: Sequence[int],
        variance_epsilon: float = 1e-3,
        name: str = "conditional_g_net",
    ) -> None:
        super().__init__(name=name)
        if z_dim <= 0 or output_dim <= 0:
            raise ValueError("z_dim and output_dim must be positive")
        if num_classes <= 1:
            raise ValueError("Conditional generation requires at least two labels")
        if label_embed_dim <= 0:
            raise ValueError("label_embed_dim must be positive")

        self.input_dim = int(z_dim)
        self.output_dim = int(output_dim)
        self.num_classes = int(num_classes)
        self.label_embed_dim = int(label_embed_dim)
        self.variance_epsilon = float(variance_epsilon)

        self.label_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_classes,
            output_dim=self.label_embed_dim,
            embeddings_initializer="glorot_uniform",
            name="label_embedding",
        )
        # Kept public because the Step-2 code freezes its moving statistics.
        self.norm_layer = tf.keras.layers.BatchNormalization(name="input_batchnorm")
        self.all_layers = [
            tf.keras.layers.Dense(int(width), activation=None, name=f"hidden_{index + 1}")
            for index, width in enumerate(hidden_units)
        ]
        self.mean_layer = tf.keras.layers.Dense(self.output_dim, name="mean_output")
        # Kept public because the variance head uses its own optimizer in Step 2.
        self.var_layer = tf.keras.layers.Dense(self.output_dim, name="variance_output")

    def _normalize_inputs(
        self,
        z: tf.Tensor | tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor],
        labels: tf.Tensor | None,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if labels is None:
            if not isinstance(z, (tuple, list)) or len(z) != 2:
                raise ValueError("Call the conditional generator as G(z, labels)")
            z, labels = z
        z = tf.cast(z, tf.float32)
        labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
        tf.debugging.assert_equal(
            tf.shape(z)[0], tf.shape(labels)[0], message="z/label batch sizes differ"
        )
        tf.debugging.assert_greater_equal(labels, 0)
        tf.debugging.assert_less(labels, self.num_classes)
        return z, labels

    def call(
        self,
        z: tf.Tensor | tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor],
        labels: tf.Tensor | None = None,
        eps: float | tf.Tensor | None = None,
        training: bool = True,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        z, labels = self._normalize_inputs(z, labels)
        label_features = self.label_embedding(labels)
        hidden = tf.concat([z, label_features], axis=-1)
        hidden = self.norm_layer(hidden, training=training)
        for layer in self.all_layers:
            hidden = tf.nn.leaky_relu(layer(hidden), alpha=0.2)
        mean = self.mean_layer(hidden)
        epsilon = self.variance_epsilon if eps is None else eps
        variance = tf.nn.softplus(self.var_layer(hidden)) + tf.cast(epsilon, mean.dtype)
        return mean, variance

    @staticmethod
    def reparameterize(mean: tf.Tensor, variance: tf.Tensor) -> tf.Tensor:
        noise = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
        return mean + noise * tf.sqrt(variance)


class RMSNorm(tf.keras.layers.Layer):
    """RMS normalization without a learned bias."""

    def __init__(self, epsilon: float = 1e-6, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def build(self, input_shape: tf.TensorShape) -> None:
        self.scale = self.add_weight(
            name="scale",
            shape=(int(input_shape[-1]),),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        inverse_rms = tf.math.rsqrt(
            tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.epsilon
        )
        return inputs * inverse_rms * self.scale


@dataclass(frozen=True)
class ResidualConfig:
    width: int = 256
    generator_blocks: int = 4
    encoder_blocks: int = 3
    expansion: int = 4
    dropout: float = 0.0
    normalization: str = "layernorm"


def _make_norm(kind: str, name: str) -> tf.keras.layers.Layer:
    if kind == "layernorm":
        return tf.keras.layers.LayerNormalization(epsilon=1e-5, name=name)
    if kind == "rmsnorm":
        return RMSNorm(epsilon=1e-6, name=name)
    raise ValueError(f"Unknown normalization: {kind}")


class PreNormResidualMLPBlock(tf.keras.layers.Layer):
    """h <- h + W2(Dropout(SiLU(W1(Norm(h)))))."""

    def __init__(self, config: ResidualConfig, name: str) -> None:
        super().__init__(name=name)
        expanded = int(config.width * config.expansion)
        self.norm = _make_norm(config.normalization, "pre_norm")
        self.fc1 = tf.keras.layers.Dense(
            expanded, kernel_initializer="he_normal", name="expand"
        )
        self.dropout1 = tf.keras.layers.Dropout(config.dropout, name="dropout_1")
        self.fc2 = tf.keras.layers.Dense(
            config.width, kernel_initializer="he_normal", name="contract"
        )
        self.dropout2 = tf.keras.layers.Dropout(config.dropout, name="dropout_2")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden = self.norm(inputs)
        hidden = tf.nn.silu(self.fc1(hidden))
        hidden = self.dropout1(hidden, training=training)
        hidden = self.fc2(hidden)
        hidden = self.dropout2(hidden, training=training)
        return inputs + hidden


class ConditionalResidualVariationalGenerator(tf.keras.Model):
    """Conditional ResMLP generator preserving the BayesGM mean/variance API."""

    def __init__(
        self,
        z_dim: int,
        output_dim: int,
        num_classes: int,
        label_embed_dim: int,
        variance_epsilon: float,
        config: ResidualConfig,
        label_injection: str = "early_concat",
    ) -> None:
        super().__init__(name="conditional_residual_g_net")
        self.input_dim = int(z_dim)
        self.output_dim = int(output_dim)
        self.num_classes = int(num_classes)
        self.label_embed_dim = int(label_embed_dim)
        self.variance_epsilon = float(variance_epsilon)
        self.config = config
        self.label_injection = str(label_injection)
        per_block_modes = {"additive_per_block", "concat_per_block"}
        if self.label_injection not in {"early_concat", *per_block_modes}:
            raise ValueError(f"Unknown label injection mode: {self.label_injection}")
        if (
            self.label_injection == "additive_per_block"
            and self.label_embed_dim != self.input_dim
        ):
            raise ValueError(
                "additive_per_block requires label_embed_dim == z_dim; "
                f"got {self.label_embed_dim} != {self.input_dim}"
            )

        self.label_embedding = tf.keras.layers.Embedding(
            self.num_classes,
            self.label_embed_dim,
            embeddings_initializer="glorot_uniform",
            name="label_embedding",
        )
        self.input_projection = tf.keras.layers.Dense(
            config.width,
            kernel_initializer="he_normal",
            name="input_projection",
        )
        self.blocks = [
            PreNormResidualMLPBlock(config, name=f"generator_block_{index + 1}")
            for index in range(config.generator_blocks)
        ]
        self.block_label_projections = (
            [
                tf.keras.layers.Dense(
                    config.width,
                    use_bias=False,
                    kernel_initializer="glorot_uniform",
                    name=f"block_label_projection_{index + 1}",
                )
                for index in range(config.generator_blocks)
            ]
            if self.label_injection in per_block_modes
            else []
        )
        self.final_norm = _make_norm(config.normalization, "final_norm")
        self.mean_layer = tf.keras.layers.Dense(output_dim, name="mean_output")
        self.var_layer = tf.keras.layers.Dense(output_dim, name="variance_output")
        # Compatibility handle used by the Step-2 code.  The actual LayerNorm /
        # RMSNorm layers remain trainable.
        self.norm_layer = tf.keras.layers.Activation(
            "linear", name="batchnorm_compatibility_handle"
        )

    def call(
        self,
        z: tf.Tensor | tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor],
        labels: tf.Tensor | None = None,
        eps: float | tf.Tensor | None = None,
        training: bool = True,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if labels is None:
            if not isinstance(z, (tuple, list)) or len(z) != 2:
                raise ValueError("Call the conditional generator as G(z, labels)")
            z, labels = z
        z = tf.cast(z, tf.float32)
        labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
        tf.debugging.assert_equal(tf.shape(z)[0], tf.shape(labels)[0])
        tf.debugging.assert_greater_equal(labels, 0)
        tf.debugging.assert_less(labels, self.num_classes)

        label_features = self.label_embedding(labels)
        if self.label_injection in {"additive_per_block", "concat_per_block"}:
            generator_input = (
                z + label_features
                if self.label_injection == "additive_per_block"
                else tf.concat([z, label_features], axis=-1)
            )
            hidden = self.input_projection(generator_input)
            for block, label_projection in zip(
                self.blocks, self.block_label_projections
            ):
                hidden = block(
                    hidden + label_projection(label_features),
                    training=training,
                )
        else:
            hidden = self.input_projection(tf.concat([z, label_features], axis=-1))
            for block in self.blocks:
                hidden = block(hidden, training=training)
        hidden = self.final_norm(hidden)
        mean = self.mean_layer(hidden)
        epsilon = self.variance_epsilon if eps is None else eps
        variance = tf.nn.softplus(self.var_layer(hidden)) + tf.cast(epsilon, mean.dtype)
        return mean, variance

    @staticmethod
    def reparameterize(mean: tf.Tensor, variance: tf.Tensor) -> tf.Tensor:
        noise = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
        return mean + noise * tf.sqrt(variance)


class ResidualEncoder(tf.keras.Model):
    """ResMLP encoder used by the optional residual architecture."""

    def __init__(self, input_dim: int, output_dim: int, config: ResidualConfig) -> None:
        super().__init__(name="residual_e_net")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_projection = tf.keras.layers.Dense(
            config.width, kernel_initializer="he_normal", name="input_projection"
        )
        self.blocks = [
            PreNormResidualMLPBlock(config, name=f"encoder_block_{index + 1}")
            for index in range(config.encoder_blocks)
        ]
        self.final_norm = _make_norm(config.normalization, "final_norm")
        self.output_layer = tf.keras.layers.Dense(output_dim, name="latent_output")

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        hidden = self.input_projection(tf.cast(inputs, tf.float32))
        for block in self.blocks:
            hidden = block(hidden, training=training)
        return self.output_layer(self.final_norm(hidden))


class ConditionalProjectionDiscriminator(tf.keras.Model):
    """Projection discriminator D(x, y), used optionally during Step 1.

    The requested conditional generator works without this class.  Enabling it
    makes the adversarial warm start label-aware and reduces the chance that G
    ignores the label embedding.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_units: Sequence[int],
    ) -> None:
        super().__init__(name="conditional_dx_net")
        if not hidden_units:
            raise ValueError("Conditional discriminator needs at least one hidden layer")
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.hidden_layers = [
            tf.keras.layers.Dense(int(width), activation=None, name=f"hidden_{index + 1}")
            for index, width in enumerate(hidden_units)
        ]
        final_width = int(hidden_units[-1])
        self.unconditional_head = tf.keras.layers.Dense(1, name="unconditional_head")
        self.label_embedding = tf.keras.layers.Embedding(
            self.num_classes,
            final_width,
            embeddings_initializer="glorot_uniform",
            name="projection_embedding",
        )
        self.scale = tf.math.rsqrt(tf.cast(final_width, tf.float32))

    def call(
        self,
        inputs: tf.Tensor,
        labels: tf.Tensor,
        training: bool = True,
    ) -> tf.Tensor:
        del training
        hidden = tf.cast(inputs, tf.float32)
        labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
        for layer in self.hidden_layers:
            hidden = tf.nn.leaky_relu(layer(hidden), alpha=0.2)
        label_features = self.label_embedding(labels)
        projection = tf.reduce_sum(hidden * label_features, axis=1, keepdims=True)
        return self.unconditional_head(hidden) + self.scale * projection


def _getattr(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_conditional_bgm(
    args: Any,
    output: Path,
    num_classes: int,
) -> BGM:
    """Build BGM and replace only the networks that need conditioning."""

    params = {
        "x_dim": int(args.ae_latent_dim),
        "z_dim": int(args.bgm_z_dim),
        "dataset": "conditional_scrna_ae_latent",
        "output_dir": str(output),
        "use_bnn": False,
        "g_units": list(args.g_units),
        "e_units": list(args.e_units),
        "dz_units": list(args.dz_units),
        "dx_units": list(args.dx_units),
        "lr": float(args.step1_lr),
        "lr_theta": float(args.generator_lr),
        "lr_z": float(args.latent_lr),
        "gamma": float(args.gamma),
        "alpha": float(args.alpha),
        "g_d_freq": 1,
        "kl_weight": 5e-5,
        "save_model": False,
        "save_res": False,
    }
    model = BGM(params=params, random_seed=int(args.seed))

    variant = str(_getattr(args, "network_variant", "mlp"))
    if variant == "resmlp":
        residual = ResidualConfig(
            width=int(_getattr(args, "res_width", 256)),
            generator_blocks=int(_getattr(args, "res_generator_blocks", 4)),
            encoder_blocks=int(_getattr(args, "res_encoder_blocks", 3)),
            expansion=int(_getattr(args, "res_expansion", 4)),
            dropout=float(_getattr(args, "res_dropout", 0.0)),
            normalization=str(_getattr(args, "res_normalization", "layernorm")),
        )
        model.g_net = ConditionalResidualVariationalGenerator(
            z_dim=int(args.bgm_z_dim),
            output_dim=int(args.ae_latent_dim),
            num_classes=int(num_classes),
            label_embed_dim=int(args.label_embed_dim),
            variance_epsilon=float(args.generator_variance_eps),
            config=residual,
            label_injection=str(_getattr(args, "label_injection", "early_concat")),
        )
        model.e_net = ResidualEncoder(
            input_dim=int(args.ae_latent_dim),
            output_dim=int(args.bgm_z_dim),
            config=residual,
        )
    elif variant == "mlp":
        model.g_net = ConditionalVariationalGenerator(
            z_dim=int(args.bgm_z_dim),
            output_dim=int(args.ae_latent_dim),
            num_classes=int(num_classes),
            label_embed_dim=int(args.label_embed_dim),
            hidden_units=list(args.g_units),
            variance_epsilon=float(args.generator_variance_eps),
        )
    else:
        raise ValueError(f"Unknown network variant: {variant}")

    condition_dx = bool(_getattr(args, "condition_dx", True))
    if condition_dx:
        model.dx_net = ConditionalProjectionDiscriminator(
            input_dim=int(args.ae_latent_dim),
            num_classes=int(num_classes),
            hidden_units=list(args.dx_units),
        )

    dummy_x = tf.zeros((2, int(args.ae_latent_dim)), dtype=tf.float32)
    dummy_z = tf.zeros((2, int(args.bgm_z_dim)), dtype=tf.float32)
    dummy_y = tf.zeros((2,), dtype=tf.int32)
    mean, variance = model.g_net(dummy_z, dummy_y, training=False)
    encoded = model.e_net(dummy_x, training=False)
    model.dz_net(dummy_z, training=False)
    if condition_dx:
        model.dx_net(dummy_x, dummy_y, training=False)
    else:
        model.dx_net(dummy_x, training=False)

    if tuple(mean.shape) != (2, int(args.ae_latent_dim)):
        raise RuntimeError(f"Unexpected conditional generator mean shape: {mean.shape}")
    if tuple(variance.shape) != (2, int(args.ae_latent_dim)):
        raise RuntimeError(f"Unexpected conditional generator variance shape: {variance.shape}")
    if tuple(encoded.shape) != (2, int(args.bgm_z_dim)):
        raise RuntimeError(f"Unexpected encoder shape: {encoded.shape}")
    tf.debugging.assert_all_finite(mean, "Generator mean is non-finite")
    tf.debugging.assert_positive(variance, "Generator variance must be positive")

    # BGM constructed its checkpoint before the conditional networks were
    # installed.  Rebind it so future checkpoint use cannot silently reference
    # the discarded unconditional generator.
    model.ckpt = tf.train.Checkpoint(
        g_net=model.g_net,
        e_net=model.e_net,
        dz_net=model.dz_net,
        dx_net=model.dx_net,
        g_pre_optimizer=model.g_pre_optimizer,
        d_pre_optimizer=model.d_pre_optimizer,
        g_optimizer=model.g_optimizer,
        posterior_optimizer=model.posterior_optimizer,
    )
    model.ckpt_manager = tf.train.CheckpointManager(
        model.ckpt, model.checkpoint_path, max_to_keep=100
    )
    model.num_classes = int(num_classes)
    model.label_embed_dim = int(args.label_embed_dim)
    model.label_injection = str(_getattr(args, "label_injection", "early_concat"))
    model.condition_dx = condition_dx
    model.network_variant = variant
    return model



def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Conditional single-cell BayesGM with G(z, label).",
    )

    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--train-file", default="pbmc68k_train_raw_official_celltype.h5ad"
    )
    parser.add_argument(
        "--validation-file",
        default="pbmc68k_validation_raw_official_celltype.h5ad",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/conditional_bayesgm"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow training into a non-empty output directory.",
    )
    parser.add_argument(
        "--dataset-name",
        default="PBMC68k",
        help="Dataset identifier recorded in config.json for multi-dataset runs.",
    )
    parser.add_argument("--label-key", default="celltype")
    parser.add_argument(
        "--label-vocabulary-json",
        type=Path,
        default=None,
        help=(
            "Optional strict label order as a JSON string array (or an object "
            "containing id_to_label). Without it, labels are sorted from TRAIN; "
            "VALIDATION labels absent from TRAIN are always rejected."
        ),
    )
    parser.add_argument("--min-train-cells-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-sum", type=float, default=1e4)

    parser.add_argument("--ae-artifact-dir", type=Path, default=Path("results/autoencoder"))
    parser.add_argument("--ae-latent-dim", type=int, default=128)
    parser.add_argument("--decoder-mode", choices=("sample", "mean"), default="sample")
    parser.add_argument("--decode-device", default="cpu")
    parser.add_argument("--decode-batch-size", type=int, default=256)
    parser.add_argument(
        "--library-mode",
        choices=("global-empirical", "label-empirical", "label-lognormal-shrunk"),
        default="global-empirical",
        help=(
            "Sample library sizes from all TRAIN cells, within the requested "
            "condition label, or from a TRAIN-only per-condition log-normal "
            "whose variance is shrunk toward the global TRAIN variance."
        ),
    )
    parser.add_argument(
        "--library-shrinkage-strength",
        type=float,
        default=200.0,
        help=(
            "Prior strength tau for label-lognormal-shrunk variance pooling: "
            "w_c=n_c/(n_c+tau). The condition log-library mean is not shrunk."
        ),
    )

    parser.add_argument("--bgm-z-dim", type=int, default=32)
    parser.add_argument("--label-embed-dim", type=int, default=16)
    parser.add_argument("--network-variant", choices=("mlp", "resmlp"), default="resmlp")
    parser.add_argument(
        "--label-injection",
        choices=("early_concat", "additive_per_block", "concat_per_block"),
        default="early_concat",
        help=(
            "Generator label injection. early_concat uses concat[z, e_y] once; "
            "additive_per_block requires dim(e_y)=dim(z), uses z+e_y at the "
            "input, and adds a learned projection of e_y before every ResMLP block; "
            "concat_per_block concatenates z and e_y at the input and also injects "
            "a learned projection before every block."
        ),
    )
    parser.add_argument(
        "--condition-dx",
        action="store_true",
        help=(
            "Also condition the Step-1 data discriminator with a projection label "
            "embedding. Leave unset for the exact requested G-only architecture."
        ),
    )
    parser.add_argument("--g-units", type=parse_int_list, default=[512, 512, 512, 512, 512])
    parser.add_argument("--e-units", type=parse_int_list, default=[512, 512, 512, 512, 512])
    parser.add_argument("--dx-units", type=parse_int_list, default=[256, 128, 64, 16])
    parser.add_argument("--dz-units", type=parse_int_list, default=[128, 64, 32, 8])

    parser.add_argument("--res-width", type=int, default=256)
    parser.add_argument("--res-generator-blocks", type=int, default=4)
    parser.add_argument("--res-encoder-blocks", type=int, default=3)
    parser.add_argument("--res-expansion", type=int, default=4)
    parser.add_argument("--res-dropout", type=float, default=0.0)
    parser.add_argument(
        "--res-normalization",
        choices=("layernorm", "rmsnorm"),
        default="layernorm",
    )

    parser.add_argument("--step1-lr", type=float, default=1e-4)
    parser.add_argument("--step1-iterations", type=int, default=100_000)
    parser.add_argument("--step1-batch-size", type=int, default=512)
    parser.add_argument("--step1-eval-every", type=int, default=10_000)
    parser.add_argument(
        "--step1-save-iters",
        type=parse_int_list,
        default=[20_000, 30_000, 50_000, 100_000],
    )
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1e-3)

    parser.add_argument("--step2-epochs", type=int, default=500)
    parser.add_argument("--step2-batch-size", type=int, default=512)
    parser.add_argument("--step2-eval-every", type=int, default=5)
    parser.add_argument("--step2-save-every", type=int, default=50)
    parser.add_argument("--generator-lr", type=float, default=5e-7)
    parser.add_argument("--variance-lr", type=float, default=1e-8)
    parser.add_argument("--latent-lr", type=float, default=1e-7)
    parser.add_argument("--latent-prior-weight", type=float, default=1.0)
    parser.add_argument("--generator-variance-eps", type=float, default=1e-3)
    parser.add_argument("--variance-init", type=float, default=0.30)
    parser.add_argument("--variance-min", type=float, default=0.15)
    parser.add_argument("--variance-max", type=float, default=0.60)
    parser.add_argument("--variance-log-prior-weight", type=float, default=5.0)
    parser.add_argument("--variance-bounds-weight", type=float, default=10.0)
    parser.add_argument("--mmd-weight", type=float, default=50.0)
    parser.add_argument("--anchor-weight", type=float, default=0.10)
    parser.add_argument(
        "--mmd-samples",
        type=int,
        default=256,
        help="Approximate total number of prior samples used by conditional MMD.",
    )
    parser.add_argument("--gradient-clip", type=float, default=10.0)

    balance = parser.add_mutually_exclusive_group()
    balance.add_argument(
        "--balanced-batches",
        dest="batch_sampling",
        action="store_const",
        const="balanced",
        help="Use approximately equal class probabilities in every training batch.",
    )
    balance.add_argument(
        "--sqrt-balanced-batches",
        dest="batch_sampling",
        action="store_const",
        const="sqrt",
        help=(
            "Sample classes with probability proportional to sqrt(n_class), "
            "then sample cells within class."
        ),
    )
    balance.add_argument(
        "--natural-batches",
        dest="batch_sampling",
        action="store_const",
        const="natural",
        help="Use the observed class proportions.",
    )
    parser.set_defaults(batch_sampling="natural")

    parser.add_argument("--n-eval-per-class", type=int, default=200)
    parser.add_argument("--mmd-n-eval", type=int, default=200)
    parser.add_argument(
        "--ilisi-knn-neighbors",
        type=int,
        default=10,
        help=(
            "Scanpy kNN graph size for validation iLISI; this is the same "
            "parameter and protocol used by the final evaluator."
        ),
    )
    parser.add_argument("--validation-generation-seed", type=int, default=42)
    parser.add_argument("--validation-library-seed", type=int, default=43)
    parser.add_argument("--validation-decoder-seed", type=int, default=44)
    parser.add_argument("--final-generation-seed", type=int, default=2027)
    parser.add_argument("--final-library-seed", type=int, default=2028)
    parser.add_argument("--final-decoder-seed", type=int, default=2029)
    parser.add_argument("--final-n-per-class", type=int, default=1_000)
    parser.add_argument(
        "--sample-observation-noise",
        action="store_true",
        help=(
            "For the final frozen sample, draw from N(mu, variance). Validation "
            "selection always uses mu so checkpoints are directly comparable."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not str(args.dataset_name).strip():
        raise ValueError("--dataset-name must not be empty")
    if not str(args.label_key).strip():
        raise ValueError("--label-key must not be empty")
    if not np.isfinite(args.target_sum) or args.target_sum <= 0:
        raise ValueError("--target-sum must be finite and positive")
    if not np.isfinite(args.library_shrinkage_strength) or args.library_shrinkage_strength < 0:
        raise ValueError("--library-shrinkage-strength must be finite and non-negative")
    positive_ints = {
        "ae_latent_dim": args.ae_latent_dim,
        "bgm_z_dim": args.bgm_z_dim,
        "label_embed_dim": args.label_embed_dim,
        "decode_batch_size": args.decode_batch_size,
        "step1_batch_size": args.step1_batch_size,
        "step2_batch_size": args.step2_batch_size,
        "n_eval_per_class": args.n_eval_per_class,
        "ilisi_knn_neighbors": args.ilisi_knn_neighbors,
        "final_n_per_class": args.final_n_per_class,
    }
    for name, value in positive_ints.items():
        if int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not (0.0 < args.variance_min < args.variance_init < args.variance_max):
        raise ValueError("Require 0 < variance-min < variance-init < variance-max")
    if args.generator_variance_eps <= 0:
        raise ValueError("generator-variance-eps must be positive")
    if args.mmd_samples <= 0:
        raise ValueError("mmd-samples must be positive")
    if args.step1_eval_every <= 0 or args.step2_eval_every <= 0:
        raise ValueError("evaluation intervals must be positive")
    if args.res_width <= 0 or args.res_expansion <= 0:
        raise ValueError("ResMLP width/expansion must be positive")
    if args.res_generator_blocks <= 0 or args.res_encoder_blocks <= 0:
        raise ValueError("ResMLP block counts must be positive")
    if not 0.0 <= args.res_dropout < 1.0:
        raise ValueError("res-dropout must be in [0, 1)")
    if args.label_injection in {"additive_per_block", "concat_per_block"}:
        if args.network_variant != "resmlp":
            raise ValueError("per-block label injection requires --network-variant resmlp")
    if args.label_injection == "additive_per_block":
        if args.label_embed_dim != args.bgm_z_dim:
            raise ValueError(
                "additive_per_block requires --label-embed-dim == --bgm-z-dim"
            )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def sha256_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_label_vocabulary_json(path: Path) -> list[str]:
    """Load a strict, ordered label vocabulary from JSON.

    A plain JSON string array is preferred.  ``{"id_to_label": [...]}`` is
    also accepted so a previous run's ``label_map.json`` can be reused.
    """
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping):
        if "id_to_label" not in payload:
            raise ValueError(
                f"{path} must be a JSON string array or contain 'id_to_label'"
            )
        payload = payload["id_to_label"]
    if not isinstance(payload, list) or not all(
        isinstance(item, str) for item in payload
    ):
        raise ValueError(f"{path} label vocabulary must be a JSON string array")
    names = list(payload)
    if len(names) < 2:
        raise ValueError("Label vocabulary must contain at least two classes")
    if any(not name for name in names):
        raise ValueError("Label vocabulary must not contain empty strings")
    if len(set(names)) != len(names):
        raise ValueError("Label vocabulary contains duplicate class names")
    return names


def resolve_label_vocabulary(
    train_names: Sequence[str],
    label_vocabulary: Sequence[str] | None = None,
) -> list[str]:
    """Return a deterministic training vocabulary or validate a strict one."""
    observed = sorted(set(map(str, train_names)))
    if len(observed) < 2:
        raise ValueError("Conditional generation requires at least two observed classes")
    if label_vocabulary is None:
        return observed

    names = list(map(str, label_vocabulary))
    if len(names) < 2 or any(not name for name in names):
        raise ValueError("Label vocabulary must contain at least two non-empty classes")
    if len(set(names)) != len(names):
        raise ValueError("Label vocabulary contains duplicate class names")
    missing_from_train = sorted(set(names) - set(observed))
    unexpected_in_train = sorted(set(observed) - set(names))
    if missing_from_train or unexpected_in_train:
        raise ValueError(
            "Training labels do not exactly match --label-vocabulary-json: "
            f"missing_from_train={missing_from_train}, "
            f"unexpected_in_train={unexpected_in_train}"
        )
    return names


def encode_labels(
    train: ad.AnnData,
    validation: ad.AnnData,
    key: str,
    min_train_cells: int,
    label_vocabulary: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if key not in train.obs:
        raise KeyError(f"Training AnnData does not contain obs[{key!r}]")
    if key not in validation.obs:
        raise KeyError(f"Validation AnnData does not contain obs[{key!r}]")
    if train.obs[key].isna().any() or validation.obs[key].isna().any():
        raise ValueError(f"obs[{key!r}] contains missing labels")

    train_series = train.obs[key]
    train_names = train_series.astype(str).to_numpy()
    names = resolve_label_vocabulary(train_names, label_vocabulary)
    name_to_id = {name: index for index, name in enumerate(names)}
    validation_names = validation.obs[key].astype(str).to_numpy()
    unseen = sorted(set(validation_names) - set(name_to_id))
    if unseen:
        raise ValueError(f"Validation contains labels absent from training: {unseen}")

    train_y = np.asarray([name_to_id[value] for value in train_names], dtype=np.int32)
    validation_y = np.asarray([name_to_id[value] for value in validation_names], dtype=np.int32)
    train_counts = np.bincount(train_y, minlength=len(names))
    validation_counts = np.bincount(validation_y, minlength=len(names))

    too_small = [
        names[index]
        for index, count in enumerate(train_counts)
        if int(count) < int(min_train_cells)
    ]
    if too_small:
        raise ValueError(
            "Classes below --min-train-cells-per-class: " + ", ".join(too_small)
        )
    missing_validation = [
        names[index] for index, count in enumerate(validation_counts) if int(count) == 0
    ]
    if missing_validation:
        warnings.warn(
            "No validation cells for classes: " + ", ".join(missing_validation),
            RuntimeWarning,
        )

    metadata: dict[str, object] = {
        "label_key": key,
        "label_vocabulary_source": (
            "explicit_json" if label_vocabulary is not None else "training_sorted_unique"
        ),
        "num_classes": len(names),
        "id_to_label": names,
        "label_to_id": name_to_id,
        "train_counts": {names[i]: int(train_counts[i]) for i in range(len(names))},
        "validation_counts": {
            names[i]: int(validation_counts[i]) for i in range(len(names))
        },
    }
    return train_y, validation_y, metadata


class BatchIndexSampler:
    """Natural, sqrt-balanced, or fully balanced mini-batch sampler."""

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        seed: int,
        mode: str,
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.int32)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.mode = str(mode)
        self.num_classes = int(self.labels.max()) + 1
        self.by_class = [np.flatnonzero(self.labels == index) for index in range(self.num_classes)]
        self.class_counts = np.asarray([len(x) for x in self.by_class], dtype=np.float64)
        if np.any(self.class_counts <= 0):
            raise ValueError("Every class must contain at least one training cell")
        if self.mode not in {"natural", "sqrt", "balanced"}:
            raise ValueError(f"Unknown batch-sampling mode: {self.mode!r}")
        if self.mode == "balanced":
            weights = np.ones(self.num_classes, dtype=np.float64)
        elif self.mode == "sqrt":
            weights = np.sqrt(self.class_counts)
        else:
            weights = self.class_counts.copy()
        self.class_probabilities = weights / weights.sum()

    def sample(self) -> np.ndarray:
        if self.mode == "natural":
            replace = self.batch_size > len(self.labels)
            return self.rng.choice(len(self.labels), self.batch_size, replace=replace)

        class_draws = self.rng.multinomial(self.batch_size, self.class_probabilities)
        indices: list[np.ndarray] = []
        for class_id, count in enumerate(class_draws):
            if count <= 0:
                continue
            candidates = self.by_class[class_id]
            chosen = self.rng.choice(candidates, int(count), replace=count > len(candidates))
            indices.append(np.asarray(chosen, dtype=np.int64))
        combined = np.concatenate(indices)
        self.rng.shuffle(combined)
        return combined


def epoch_batches(
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    balanced_sampler: BatchIndexSampler | None,
) -> Iterable[np.ndarray]:
    n_batches = max(len(labels) // batch_size, 1)
    if balanced_sampler is not None:
        for _ in range(n_batches):
            yield balanced_sampler.sample()
        return

    permutation = rng.permutation(len(labels))
    for batch_number in range(n_batches):
        start = batch_number * batch_size
        stop = start + batch_size
        selected = permutation[start:stop]
        if len(selected) < batch_size:
            extra = rng.choice(len(labels), batch_size - len(selected), replace=True)
            selected = np.concatenate([selected, extra])
        yield selected


def call_dx(
    model,
    data_x: tf.Tensor,
    labels: tf.Tensor,
    training: bool,
) -> tf.Tensor:
    if model.condition_dx:
        return model.dx_net(data_x, labels, training=training)
    return model.dx_net(data_x, training=training)


def apply_unclipped_gradients(
    optimizer: tf.keras.optimizers.Optimizer,
    gradients: Sequence[tf.Tensor | None],
    variables: Sequence[tf.Variable],
) -> None:
    pairs = [(gradient, variable) for gradient, variable in zip(gradients, variables) if gradient is not None]
    if pairs:
        optimizer.apply_gradients(pairs)


def make_step1_updates(model, args: argparse.Namespace):
    """Create label-aware EGM discriminator and generator update functions."""

    @tf.function(reduce_retracing=True)
    def update_discriminators(
        prior_z: tf.Tensor,
        real_x: tf.Tensor,
        labels: tf.Tensor,
    ):
        prior_z = tf.cast(prior_z, tf.float32)
        real_x = tf.cast(real_x, tf.float32)
        labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
        batch_size = tf.shape(real_x)[0]
        epsilon_z = tf.random.uniform((batch_size, 1), 0.0, 1.0)
        epsilon_x = tf.random.uniform((batch_size, 1), 0.0, 1.0)

        with tf.GradientTape(persistent=True) as discriminator_tape:
            encoded_z = model.e_net(real_x, training=True)

            with tf.GradientTape() as gpz_tape:
                mixed_z = prior_z * epsilon_z + encoded_z * (1.0 - epsilon_z)
                gpz_tape.watch(mixed_z)
                mixed_z_score = model.dz_net(mixed_z, training=True)

            generated_mean, generated_variance = model.g_net(
                prior_z, labels, training=True
            )
            generated_x = model.g_net.reparameterize(
                generated_mean, generated_variance
            )

            with tf.GradientTape() as gpx_tape:
                mixed_x = real_x * epsilon_x + generated_x * (1.0 - epsilon_x)
                gpx_tape.watch(mixed_x)
                mixed_x_score = call_dx(model, mixed_x, labels, training=True)

            real_z_score = model.dz_net(prior_z, training=True)
            fake_z_score = model.dz_net(encoded_z, training=True)
            real_x_score = call_dx(model, real_x, labels, training=True)
            fake_x_score = call_dx(model, generated_x, labels, training=True)

            dz_loss = 0.5 * (
                tf.reduce_mean(tf.square(0.9 - real_z_score))
                + tf.reduce_mean(tf.square(0.1 - fake_z_score))
            )
            dx_loss = 0.5 * (
                tf.reduce_mean(tf.square(0.9 - real_x_score))
                + tf.reduce_mean(tf.square(0.1 - fake_x_score))
            )

            grad_z = gpz_tape.gradient(mixed_z_score, mixed_z)
            grad_x = gpx_tape.gradient(mixed_x_score, mixed_x)
            grad_norm_z = tf.sqrt(
                tf.reduce_sum(tf.square(grad_z), axis=1) + 1e-12
            )
            grad_norm_x = tf.sqrt(
                tf.reduce_sum(tf.square(grad_x), axis=1) + 1e-12
            )
            gradient_penalty = tf.reduce_mean(tf.square(grad_norm_z - 1.0))
            gradient_penalty += tf.reduce_mean(tf.square(grad_norm_x - 1.0))
            total = dx_loss + dz_loss + args.gamma * gradient_penalty

        variables = model.dz_net.trainable_variables + model.dx_net.trainable_variables
        gradients = discriminator_tape.gradient(total, variables)
        del discriminator_tape
        apply_unclipped_gradients(model.d_pre_optimizer, gradients, variables)
        return dz_loss, dx_loss, total

    @tf.function(reduce_retracing=True)
    def update_generator_encoder(
        prior_z: tf.Tensor,
        real_x: tf.Tensor,
        labels: tf.Tensor,
    ):
        prior_z = tf.cast(prior_z, tf.float32)
        real_x = tf.cast(real_x, tf.float32)
        labels = tf.reshape(tf.cast(labels, tf.int32), [-1])

        with tf.GradientTape() as generator_tape:
            generated_mean, generated_variance = model.g_net(
                prior_z, labels, training=True
            )
            generated_x = model.g_net.reparameterize(
                generated_mean, generated_variance
            )
            encoded_real_z = model.e_net(real_x, training=True)
            encoded_generated_z = model.e_net(generated_x, training=True)
            reconstructed_mean, reconstructed_variance = model.g_net(
                encoded_real_z, labels, training=True
            )
            reconstructed_x = model.g_net.reparameterize(
                reconstructed_mean, reconstructed_variance
            )

            generated_x_score = call_dx(model, generated_x, labels, training=True)
            encoded_z_score = model.dz_net(encoded_real_z, training=True)
            generator_adversarial = tf.reduce_mean(
                tf.square(0.9 - generated_x_score)
            )
            encoder_adversarial = tf.reduce_mean(
                tf.square(0.9 - encoded_z_score)
            )
            cycle_x = tf.reduce_mean(tf.square(real_x - reconstructed_x))
            cycle_z = tf.reduce_mean(tf.square(prior_z - encoded_generated_z))
            variance_penalty = tf.reduce_mean(tf.square(generated_variance))
            total = (
                generator_adversarial
                + encoder_adversarial
                + 10.0 * (cycle_x + cycle_z)
                + args.alpha * variance_penalty
            )

        variables = model.g_net.trainable_variables + model.e_net.trainable_variables
        gradients = generator_tape.gradient(total, variables)
        apply_unclipped_gradients(model.g_pre_optimizer, gradients, variables)
        return (
            generator_adversarial,
            encoder_adversarial,
            cycle_z,
            cycle_x,
            variance_penalty,
            total,
        )

    return update_discriminators, update_generator_encoder


def make_fixed_conditional_prior(
    num_classes: int,
    n_per_class: int,
    z_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use identical z draws for every class so label effects are comparable."""
    rng = np.random.default_rng(seed)
    shared_z = rng.standard_normal((n_per_class, z_dim)).astype(np.float32)
    prior_z = np.concatenate([shared_z.copy() for _ in range(num_classes)], axis=0)
    labels = np.repeat(np.arange(num_classes, dtype=np.int32), n_per_class)
    return prior_z, labels


def generate_standardized_latent(
    model,
    prior_z: np.ndarray,
    labels: np.ndarray,
    sample_observation_noise: bool,
    observation_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, variance = model.g_net(
        tf.convert_to_tensor(prior_z, dtype=tf.float32),
        tf.convert_to_tensor(labels, dtype=tf.int32),
        training=False,
    )
    mean_array = mean.numpy().astype(np.float32)
    variance_array = variance.numpy().astype(np.float32)
    if sample_observation_noise:
        if observation_seed is None:
            raise ValueError("observation_seed is required when sampling BGM noise")
        rng = np.random.default_rng(observation_seed)
        generated_array = mean_array + np.sqrt(variance_array) * rng.standard_normal(
            mean_array.shape
        ).astype(np.float32)
    else:
        generated_array = mean_array
    return (
        generated_array.astype(np.float32, copy=False),
        mean_array,
        variance_array,
    )


def select_fixed_expression_by_class(
    matrix,
    labels: np.ndarray,
    num_classes: int,
    n_per_class: int,
    seed: int,
) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    result: dict[int, np.ndarray] = {}
    for class_id in range(num_classes):
        candidates = np.flatnonzero(labels == class_id)
        if len(candidates) == 0:
            continue
        selected = rng.choice(
            candidates,
            size=min(int(n_per_class), len(candidates)),
            replace=False,
        )
        result[class_id] = get_rows(matrix, selected)
    return result


def nearest_centroid_fidelity(
    train_latent: np.ndarray,
    train_y: np.ndarray,
    generated_latent: np.ndarray,
    target_y: np.ndarray,
    num_classes: int,
) -> tuple[float, dict[int, float]]:
    centroids = np.stack(
        [train_latent[train_y == class_id].mean(axis=0) for class_id in range(num_classes)],
        axis=0,
    ).astype(np.float32)
    generated_norm = np.sum(generated_latent * generated_latent, axis=1, keepdims=True)
    centroid_norm = np.sum(centroids * centroids, axis=1)[None, :]
    distances = np.maximum(
        generated_norm + centroid_norm - 2.0 * generated_latent @ centroids.T,
        0.0,
    )
    predicted = distances.argmin(axis=1).astype(np.int32)
    per_class = {
        class_id: float(np.mean(predicted[target_y == class_id] == class_id))
        for class_id in range(num_classes)
        if np.any(target_y == class_id)
    }
    macro = float(np.mean(list(per_class.values()))) if per_class else math.nan
    return macro, per_class


def safe_distribution_metrics(
    real: np.ndarray,
    generated: np.ndarray,
    n_eval: int,
    mmd_n_eval: int,
    seed: int,
    ilisi_knn_neighbors: int = 10,
) -> dict[str, float]:
    ilisi_knn_neighbors = int(ilisi_knn_neighbors)
    if ilisi_knn_neighbors <= 0:
        raise ValueError("ilisi_knn_neighbors must be positive")
    n = min(len(real), len(generated), int(n_eval))
    if n < 2:
        return {
            "eval_n_per_origin": float(n),
            "ilisi_knn_neighbors": float(ilisi_knn_neighbors),
            "ilisi_eligible": 0.0,
            "ilisi_pca20": math.nan,
            "mmd_pca50": math.nan,
            "real_zero_frac": float(np.mean(real <= 0)) if len(real) else math.nan,
            "generated_zero_frac": (
                float(np.mean(generated <= 0)) if len(generated) else math.nan
            ),
            "real_cell_sum_mean": (
                float(np.mean(np.sum(real, axis=1))) if len(real) else math.nan
            ),
            "generated_cell_sum_mean": (
                float(np.mean(np.sum(generated, axis=1)))
                if len(generated)
                else math.nan
            ),
            "gene_mean_corr": math.nan,
        }
    return distribution_metrics(
        real,
        generated,
        n,
        min(mmd_n_eval, n),
        seed,
        ilisi_knn_neighbors,
    )


def macro_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if len(finite) else math.nan


def checkpoint_selection_score(validation_macro_ilisi_pca20: float) -> float:
    """Keep the legacy score field equal to the declared selection metric."""
    value = float(validation_macro_ilisi_pca20)
    return value if np.isfinite(value) else math.nan


def evaluate_conditional_generator(
    model,
    prior_z_eval: np.ndarray,
    prior_y_eval: np.ndarray,
    generated_library_size: np.ndarray,
    train_real_by_class: Mapping[int, np.ndarray],
    validation_real_by_class: Mapping[int, np.ndarray],
    train_latent: np.ndarray,
    train_y: np.ndarray,
    latent_mean: np.ndarray,
    latent_sd: np.ndarray,
    ae,
    label_names: Sequence[str],
    args: argparse.Namespace,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    generated_standardized, _, generated_variance = generate_standardized_latent(
        model,
        prior_z_eval,
        prior_y_eval,
        sample_observation_noise=False,
    )
    generated_expression, _ = decode_bgm_latent(
        generated_standardized,
        latent_mean,
        latent_sd,
        ae,
        generated_library_size,
        args,
        args.validation_decoder_seed,
    )

    metrics: dict[str, float] = {}
    train_rows: list[dict[str, float]] = []
    validation_rows: list[dict[str, float]] = []
    for class_id, label_name in enumerate(label_names):
        generated_class = generated_expression[prior_y_eval == class_id]
        if class_id in train_real_by_class:
            class_metrics = safe_distribution_metrics(
                train_real_by_class[class_id],
                generated_class,
                args.n_eval_per_class,
                args.mmd_n_eval,
                args.seed + 1_000 + class_id,
                args.ilisi_knn_neighbors,
            )
            train_rows.append(class_metrics)
            for key, value in class_metrics.items():
                metrics[f"train_class_{class_id}_{key}"] = value
        if class_id in validation_real_by_class:
            class_metrics = safe_distribution_metrics(
                validation_real_by_class[class_id],
                generated_class,
                args.n_eval_per_class,
                args.mmd_n_eval,
                args.seed + 2_000 + class_id,
                args.ilisi_knn_neighbors,
            )
            validation_rows.append(class_metrics)
            for key, value in class_metrics.items():
                metrics[f"validation_class_{class_id}_{key}"] = value

    metric_names = (
        "ilisi_pca20",
        "mmd_pca50",
        "real_zero_frac",
        "generated_zero_frac",
        "real_cell_sum_mean",
        "generated_cell_sum_mean",
        "gene_mean_corr",
    )
    for key in metric_names:
        metrics[f"train_macro_{key}"] = macro_mean(row[key] for row in train_rows)
        metrics[f"validation_macro_{key}"] = macro_mean(
            row[key] for row in validation_rows
        )
    metrics["train_ilisi_eligible_classes"] = float(
        sum(row["ilisi_eligible"] for row in train_rows)
    )
    metrics["validation_ilisi_eligible_classes"] = float(
        sum(row["ilisi_eligible"] for row in validation_rows)
    )

    fidelity, per_class_fidelity = nearest_centroid_fidelity(
        train_latent,
        train_y,
        generated_standardized,
        prior_y_eval,
        len(label_names),
    )
    metrics["label_fidelity_macro"] = fidelity
    for class_id, value in per_class_fidelity.items():
        metrics[f"label_fidelity_class_{class_id}"] = value

    metrics["validation_selection_score"] = checkpoint_selection_score(
        metrics[SELECTION_METRIC]
    )

    variance_flat = generated_variance.ravel()
    metrics["bgm_var_q01"] = float(np.quantile(variance_flat, 0.01))
    metrics["bgm_var_q50"] = float(np.quantile(variance_flat, 0.50))
    metrics["bgm_var_q99"] = float(np.quantile(variance_flat, 0.99))
    for class_id in range(len(label_names)):
        class_variance = generated_variance[prior_y_eval == class_id].ravel()
        metrics[f"bgm_var_class_{class_id}_mean"] = float(class_variance.mean())

    return metrics, generated_expression, generated_standardized, generated_variance


def save_all_bgm_weights(model, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.g_net.save_weights(str(directory / "generator.weights.h5"))
    model.e_net.save_weights(str(directory / "encoder.weights.h5"))
    model.dz_net.save_weights(str(directory / "dz.weights.h5"))
    model.dx_net.save_weights(str(directory / "dx.weights.h5"))


def run_step1(
    model,
    train_latent: np.ndarray,
    train_y: np.ndarray,
    train_real_by_class: Mapping[int, np.ndarray],
    validation_real_by_class: Mapping[int, np.ndarray],
    latent_mean: np.ndarray,
    latent_sd: np.ndarray,
    ae,
    prior_z_eval: np.ndarray,
    prior_y_eval: np.ndarray,
    validation_generated_library_size: np.ndarray,
    label_names: Sequence[str],
    args: argparse.Namespace,
    output: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed + 300)
    sampler = BatchIndexSampler(
        train_y,
        args.step1_batch_size,
        seed=args.seed + 301,
        mode=args.batch_sampling,
    )
    update_discriminators, update_generator_encoder = make_step1_updates(model, args)

    history: list[dict[str, float]] = []
    best_score = -math.inf
    best_dir = output / "step1_best"
    checkpoint_dir = output / "step1_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    print("\nConditional BayesGM Step 1 EGM warm start ...")
    last_disc = [math.nan, math.nan, math.nan]
    for iteration in range(1, args.step1_iterations + 1):
        for _ in range(model.params["g_d_freq"]):
            indices = sampler.sample()
            batch_x = tf.convert_to_tensor(train_latent[indices], dtype=tf.float32)
            batch_y = tf.convert_to_tensor(train_y[indices], dtype=tf.int32)
            batch_z = tf.convert_to_tensor(
                rng.standard_normal((len(indices), args.bgm_z_dim)).astype(np.float32)
            )
            disc_values = update_discriminators(batch_z, batch_x, batch_y)
            last_disc = [float(value.numpy()) for value in disc_values]

        indices = sampler.sample()
        batch_x = tf.convert_to_tensor(train_latent[indices], dtype=tf.float32)
        batch_y = tf.convert_to_tensor(train_y[indices], dtype=tf.int32)
        batch_z = tf.convert_to_tensor(
            rng.standard_normal((len(indices), args.bgm_z_dim)).astype(np.float32)
        )
        gen_values = update_generator_encoder(batch_z, batch_x, batch_y)

        should_eval = (
            iteration % args.step1_eval_every == 0
            or iteration == args.step1_iterations
        )
        if not should_eval:
            continue

        gen_losses = [float(value.numpy()) for value in gen_values]
        metrics, generated, generated_latent, generated_variance = (
            evaluate_conditional_generator(
                model,
                prior_z_eval,
                prior_y_eval,
                validation_generated_library_size,
                train_real_by_class,
                validation_real_by_class,
                train_latent,
                train_y,
                latent_mean,
                latent_sd,
                ae,
                label_names,
                args,
            )
        )
        row = {
            "iteration": iteration,
            "g_loss_adv": gen_losses[0],
            "e_loss_adv": gen_losses[1],
            "l2_loss_z": gen_losses[2],
            "l2_loss_x": gen_losses[3],
            "variance_loss": gen_losses[4],
            "g_e_loss": gen_losses[5],
            "dz_loss": last_disc[0],
            "dx_loss": last_disc[1],
            "d_loss": last_disc[2],
            **metrics,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output / "step1_history.csv", index=False)
        score = metrics["validation_selection_score"]
        print(
            f"Step1 iter={iteration} | selection={score:.4f} "
            f"| val macro iLISI={metrics['validation_macro_ilisi_pca20']:.4f} "
            f"| label fidelity={metrics['label_fidelity_macro']:.4f} "
            f"| val macro MMD={metrics['validation_macro_mmd_pca50']:.4f}"
        )

        if iteration in set(args.step1_save_iters):
            directory = checkpoint_dir / f"iter_{iteration:06d}"
            save_all_bgm_weights(model, directory)
            np.save(directory / "validation_generated.npy", generated)
            np.save(directory / "validation_generated_latent.npy", generated_latent)
            np.save(directory / "validation_generated_variance.npy", generated_variance)

        if np.isfinite(score) and score > best_score:
            best_score = score
            save_all_bgm_weights(model, best_dir)
            np.save(output / "step1_validation_generated.npy", generated)
            np.save(output / "step1_validation_generated_labels.npy", prior_y_eval)
            with (output / "step1.json").open("w") as handle:
                json.dump(
                    {
                        "iteration": iteration,
                        "selection_metric": SELECTION_METRIC,
                        "validation_selection_score": score,
                        "validation_macro_ilisi_pca20": metrics["validation_macro_ilisi_pca20"],
                        "label_fidelity_macro": metrics["label_fidelity_macro"],
                    },
                    handle,
                    indent=2,
                )
            print("  new validation-selected conditional Step 1 checkpoint")

    if not (best_dir / "generator.weights.h5").exists():
        raise RuntimeError("Step 1 never produced a finite validation selection score")
    model.g_net.load_weights(str(best_dir / "generator.weights.h5"))
    model.e_net.load_weights(str(best_dir / "encoder.weights.h5"))
    model.dz_net.load_weights(str(best_dir / "dz.weights.h5"))
    model.dx_net.load_weights(str(best_dir / "dx.weights.h5"))
    return pd.DataFrame(history)


def conditional_mmd_tf(
    real_x: tf.Tensor,
    real_y: tf.Tensor,
    generated_x: tf.Tensor,
    generated_y: tf.Tensor,
    num_classes: int,
) -> tf.Tensor:
    """Macro-average MMD over classes represented in both tensors."""
    terms: list[tf.Tensor] = []
    active: list[tf.Tensor] = []
    for class_id in range(num_classes):
        real_class = tf.boolean_mask(real_x, tf.equal(real_y, class_id))
        generated_class = tf.boolean_mask(
            generated_x, tf.equal(generated_y, class_id)
        )
        count = tf.minimum(tf.shape(real_class)[0], tf.shape(generated_class)[0])

        def calculate() -> tf.Tensor:
            return adaptive_mmd_tf(real_class[:count], generated_class[:count])

        is_active = tf.greater_equal(count, 2)
        terms.append(tf.cond(is_active, calculate, lambda: tf.constant(0.0, tf.float32)))
        active.append(tf.cast(is_active, tf.float32))
    return tf.add_n(terms) / tf.maximum(tf.add_n(active), 1.0)


def make_step2_updates(
    model,
    args: argparse.Namespace,
    data_z: tf.Variable,
    anchor_variables: Sequence[tf.Tensor],
    num_classes: int,
):
    # Freeze only the BatchNorm moving statistics in the MLP variant.  For the
    # ResMLP this is an intentionally empty compatibility layer.
    model.g_net.norm_layer.trainable = False
    variance_variables = list(model.g_net.var_layer.trainable_variables)
    variance_ids = {id(variable) for variable in variance_variables}
    mean_variables = [
        variable
        for variable in model.g_net.trainable_variables
        if id(variable) not in variance_ids
    ]
    if len(mean_variables) != len(anchor_variables):
        raise RuntimeError("Step-1 anchor variable count does not match generator")

    mean_optimizer = tf.keras.optimizers.Adam(
        args.generator_lr, beta_1=0.9, beta_2=0.99
    )
    variance_optimizer = tf.keras.optimizers.Adam(
        args.variance_lr, beta_1=0.9, beta_2=0.99
    )
    latent_optimizer = model.posterior_optimizer

    @tf.function(reduce_retracing=True)
    def update_generator(
        batch_z: tf.Tensor,
        batch_x: tf.Tensor,
        batch_y: tf.Tensor,
        prior_z: tf.Tensor,
        prior_y: tf.Tensor,
    ):
        batch_z = tf.cast(batch_z, tf.float32)
        batch_x = tf.cast(batch_x, tf.float32)
        batch_y = tf.reshape(tf.cast(batch_y, tf.int32), [-1])
        prior_z = tf.cast(prior_z, tf.float32)
        prior_y = tf.reshape(tf.cast(prior_y, tf.int32), [-1])

        with tf.GradientTape(persistent=True) as tape:
            mean, predicted_variance = model.g_net(
                batch_z, batch_y, training=False
            )
            variance_for_nll = tf.clip_by_value(
                predicted_variance, args.variance_min, args.variance_max
            )
            squared_error = tf.square(batch_x - mean)
            nll = tf.reduce_mean(
                tf.reduce_sum(
                    squared_error / (2.0 * variance_for_nll)
                    + 0.5 * tf.math.log(variance_for_nll),
                    axis=1,
                )
            )
            mse = tf.reduce_mean(squared_error)

            prior_mean, _ = model.g_net(prior_z, prior_y, training=False)
            class_mmd = conditional_mmd_tf(
                batch_x, batch_y, prior_mean, prior_y, num_classes
            )

            anchor_terms = [
                tf.reduce_mean(tf.square(variable - frozen))
                for variable, frozen in zip(mean_variables, anchor_variables)
            ]
            anchor = tf.add_n(anchor_terms) / float(max(len(anchor_terms), 1))

            # NLL uses a safe bounded variance.  The regularizer deliberately
            # keeps the unclipped upper tail so values above variance_max still
            # receive a gradient pulling them back.
            regularized_variance = tf.maximum(
                predicted_variance, tf.cast(args.variance_min, tf.float32)
            )
            log_prior = tf.reduce_mean(
                tf.square(
                    tf.math.log(regularized_variance)
                    - tf.math.log(tf.cast(args.variance_init, tf.float32))
                )
            )
            bounds = tf.reduce_mean(
                tf.square(tf.nn.relu(args.variance_min - predicted_variance))
                + tf.square(tf.nn.relu(predicted_variance - args.variance_max))
            )
            variance_regularizer = (
                args.variance_log_prior_weight * log_prior
                + args.variance_bounds_weight * bounds
            )
            total = (
                nll
                + args.mmd_weight * class_mmd
                + args.anchor_weight * anchor
                + variance_regularizer
            )

        mean_gradients = tape.gradient(total, mean_variables)
        variance_gradients = tape.gradient(total, variance_variables)
        del tape
        apply_gradients_clipped(
            mean_optimizer, mean_gradients, mean_variables, args.gradient_clip
        )
        apply_gradients_clipped(
            variance_optimizer,
            variance_gradients,
            variance_variables,
            args.gradient_clip,
        )
        return total, nll, mse, class_mmd, anchor, variance_regularizer

    @tf.function(reduce_retracing=True)
    def update_latent(
        batch_indices: tf.Tensor,
        batch_x: tf.Tensor,
        batch_y: tf.Tensor,
    ):
        batch_indices = tf.cast(batch_indices, tf.int32)
        batch_x = tf.cast(batch_x, tf.float32)
        batch_y = tf.reshape(tf.cast(batch_y, tf.int32), [-1])

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(data_z)
            batch_latent = tf.gather(data_z, batch_indices)
            mean, predicted_variance = model.g_net(
                batch_latent, batch_y, training=False
            )
            variance_for_nll = tf.clip_by_value(
                predicted_variance, args.variance_min, args.variance_max
            )
            nll = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(batch_x - mean) / (2.0 * variance_for_nll)
                    + 0.5 * tf.math.log(variance_for_nll),
                    axis=1,
                )
            )
            latent_prior = tf.reduce_mean(
                0.5 * tf.reduce_sum(tf.square(batch_latent), axis=1)
            )
            total = nll + args.latent_prior_weight * latent_prior

        gradient = tape.gradient(total, data_z)
        apply_gradients_clipped(
            latent_optimizer, [gradient], [data_z], args.gradient_clip
        )
        updated = tf.gather(data_z, batch_indices)
        z_rms = tf.sqrt(tf.reduce_mean(tf.square(updated)))
        return total, nll, latent_prior, z_rms

    return update_generator, update_latent, mean_variables


def make_balanced_prior_batch(
    rng: np.random.Generator,
    num_classes: int,
    approximate_total: int,
    z_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    per_class = max(2, int(math.ceil(approximate_total / num_classes)))
    labels = np.repeat(np.arange(num_classes, dtype=np.int32), per_class)
    z = rng.standard_normal((len(labels), z_dim)).astype(np.float32)
    permutation = rng.permutation(len(labels))
    return z[permutation], labels[permutation]


def run_step2(
    model,
    train_latent: np.ndarray,
    train_y: np.ndarray,
    train_real_by_class: Mapping[int, np.ndarray],
    validation_real_by_class: Mapping[int, np.ndarray],
    latent_mean: np.ndarray,
    latent_sd: np.ndarray,
    ae,
    prior_z_eval: np.ndarray,
    prior_y_eval: np.ndarray,
    validation_generated_library_size: np.ndarray,
    label_names: Sequence[str],
    args: argparse.Namespace,
    output: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed + 400)
    data_z_initial = model.e_net(
        tf.convert_to_tensor(train_latent, dtype=tf.float32), training=False
    ).numpy().astype(np.float32)
    data_z = tf.Variable(
        data_z_initial,
        trainable=True,
        dtype=tf.float32,
        name="conditional_step2_data_z",
    )
    model.data_z = data_z

    initialize_variance_head(
        model, args.variance_init, args.generator_variance_eps
    )
    model.g_net.norm_layer.trainable = False
    variance_ids = {id(variable) for variable in model.g_net.var_layer.trainable_variables}
    mean_variables_before = [
        variable
        for variable in model.g_net.trainable_variables
        if id(variable) not in variance_ids
    ]
    anchor_variables = [tf.identity(variable) for variable in mean_variables_before]
    update_generator, update_latent, _ = make_step2_updates(
        model,
        args,
        data_z,
        anchor_variables,
        len(label_names),
    )

    batch_sampler = (
        BatchIndexSampler(
            train_y,
            args.step2_batch_size,
            seed=args.seed + 401,
            mode=args.batch_sampling,
        )
        if args.batch_sampling != "natural"
        else None
    )

    history: list[dict[str, float]] = []
    best_score = -math.inf
    best_generator = output / "generator.weights.h5"
    best_data_z = output / "data_z.npy"
    checkpoint_dir = output / "step2_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    baseline_metrics, baseline_generated, baseline_latent, baseline_variance = (
        evaluate_conditional_generator(
            model,
            prior_z_eval,
            prior_y_eval,
            validation_generated_library_size,
            train_real_by_class,
            validation_real_by_class,
            train_latent,
            train_y,
            latent_mean,
            latent_sd,
            ae,
            label_names,
            args,
        )
    )
    baseline = {"epoch": -1, **baseline_metrics}
    history.append(baseline)
    best_score = baseline_metrics["validation_selection_score"]
    model.g_net.save_weights(str(best_generator))
    np.save(best_data_z, data_z.numpy())
    np.save(output / "validation_generated.npy", baseline_generated)
    np.save(output / "validation_generated_latent.npy", baseline_latent)
    np.save(output / "validation_generated_variance.npy", baseline_variance)
    np.save(output / "validation_generated_labels.npy", prior_y_eval)
    with (output / "step2.json").open("w") as handle:
        json.dump(
            {
                "epoch": -1,
                "selection_metric": SELECTION_METRIC,
                "validation_selection_score": best_score,
                "validation_macro_ilisi_pca20": baseline_metrics["validation_macro_ilisi_pca20"],
                "label_fidelity_macro": baseline_metrics["label_fidelity_macro"],
            },
            handle,
            indent=2,
        )

    print("\nConditional BayesGM Step 2 ...")
    print(
        f"baseline selection={best_score:.4f}; "
        f"val macro iLISI={baseline_metrics['validation_macro_ilisi_pca20']:.4f}; "
        f"variance init={args.variance_init}; G lr={args.generator_lr}; "
        f"variance lr={args.variance_lr}; z lr={args.latent_lr}"
    )
    last_generator_losses = [math.nan] * 6
    last_latent_losses = [math.nan] * 4

    for epoch in range(1, args.step2_epochs + 1):
        batches = list(
            epoch_batches(
                train_y,
                args.step2_batch_size,
                rng,
                batch_sampler,
            )
        )
        progress = tqdm(batches, desc=f"Conditional Step 2 epoch {epoch}/{args.step2_epochs}")
        for indices in progress:
            batch_z = tf.gather(data_z, indices)
            batch_x = tf.convert_to_tensor(train_latent[indices], dtype=tf.float32)
            batch_y = tf.convert_to_tensor(train_y[indices], dtype=tf.int32)
            prior_z, prior_y = make_balanced_prior_batch(
                rng,
                len(label_names),
                args.mmd_samples,
                args.bgm_z_dim,
            )
            generator_values = update_generator(
                batch_z,
                batch_x,
                batch_y,
                tf.convert_to_tensor(prior_z, dtype=tf.float32),
                tf.convert_to_tensor(prior_y, dtype=tf.int32),
            )
            latent_values = update_latent(
                tf.convert_to_tensor(indices, dtype=tf.int32),
                batch_x,
                batch_y,
            )
            last_generator_losses = [float(value.numpy()) for value in generator_values]
            last_latent_losses = [float(value.numpy()) for value in latent_values]
            progress.set_postfix(
                total=f"{last_generator_losses[0]:.3f}",
                mse=f"{last_generator_losses[2]:.5f}",
                cmmd=f"{last_generator_losses[3]:.4f}",
                z=f"{last_latent_losses[3]:.3f}",
            )

        should_eval = (
            epoch % args.step2_eval_every == 0
            or epoch == args.step2_epochs
        )
        if not should_eval:
            continue

        metrics, generated, generated_latent, generated_variance = (
            evaluate_conditional_generator(
                model,
                prior_z_eval,
                prior_y_eval,
                validation_generated_library_size,
                train_real_by_class,
                validation_real_by_class,
                train_latent,
                train_y,
                latent_mean,
                latent_sd,
                ae,
                label_names,
                args,
            )
        )
        row = {
            "epoch": epoch,
            "loss_total_last": last_generator_losses[0],
            "loss_nll_last": last_generator_losses[1],
            "loss_mse_last": last_generator_losses[2],
            "conditional_mmd_last": last_generator_losses[3],
            "anchor_last": last_generator_losses[4],
            "variance_regularizer_last": last_generator_losses[5],
            "latent_loss_last": last_latent_losses[0],
            "latent_nll_last": last_latent_losses[1],
            "latent_prior_last": last_latent_losses[2],
            "z_rms_last": last_latent_losses[3],
            **metrics,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output / "step2_history.csv", index=False)
        score = metrics["validation_selection_score"]
        print(
            f"Step2 epoch={epoch} | selection={score:.4f} "
            f"| val macro iLISI={metrics['validation_macro_ilisi_pca20']:.4f} "
            f"| label fidelity={metrics['label_fidelity_macro']:.4f} "
            f"| val macro MMD={metrics['validation_macro_mmd_pca50']:.4f} "
            f"| var={metrics['bgm_var_q01']:.4f}/"
            f"{metrics['bgm_var_q50']:.4f}/{metrics['bgm_var_q99']:.4f}"
        )

        if epoch % args.step2_save_every == 0:
            directory = checkpoint_dir / f"epoch_{epoch:04d}"
            directory.mkdir(exist_ok=True)
            model.g_net.save_weights(str(directory / "generator.weights.h5"))
            np.save(directory / "data_z.npy", data_z.numpy())
            np.save(directory / "validation_generated.npy", generated)
            np.save(directory / "validation_generated_labels.npy", prior_y_eval)

        if np.isfinite(score) and score > best_score:
            best_score = score
            model.g_net.save_weights(str(best_generator))
            np.save(best_data_z, data_z.numpy())
            np.save(output / "validation_generated.npy", generated)
            np.save(output / "validation_generated_latent.npy", generated_latent)
            np.save(output / "validation_generated_variance.npy", generated_variance)
            with (output / "step2.json").open("w") as handle:
                json.dump(
                    {
                        "epoch": epoch,
                        "selection_metric": SELECTION_METRIC,
                        "validation_selection_score": score,
                        "validation_macro_ilisi_pca20": metrics["validation_macro_ilisi_pca20"],
                        "label_fidelity_macro": metrics["label_fidelity_macro"],
                    },
                    handle,
                    indent=2,
                )
            print("  new validation-selected conditional Step 2 checkpoint")

    model.g_net.load_weights(str(best_generator))
    model.data_z = tf.Variable(np.load(best_data_z), trainable=True, dtype=tf.float32)
    return pd.DataFrame(history)


def json_serializable_args(args: argparse.Namespace) -> dict[str, object]:
    result = vars(args).copy()
    for key, value in list(result.items()):
        if isinstance(value, Path):
            result[key] = str(value)
    return result


def write_config(
    args: argparse.Namespace,
    output: Path,
    train: ad.AnnData,
    validation: ad.AnnData,
    label_metadata: Mapping[str, object],
) -> None:
    config = json_serializable_args(args)
    ae_directory = args.ae_artifact_dir.expanduser().resolve()
    ae_artifact_names = (
        "metadata.json",
        "model.pt",
        "train_latent.npy",
        "validation_latent.npy",
        "train_library_size.npy",
        "validation_library_size.npy",
        "genes.npy",
    )
    config.update(
        {
            "dataset_name": str(args.dataset_name),
            "model_type": "conditional_single_cell_bayesgm",
            "conditional_parameterization": (
                "G(z + Embedding32(y); add projected label before every ResMLP block) "
                "-> (mean, diagonal_variance)"
                if args.label_injection == "additive_per_block"
                else (
                    "G(concat[z, Embedding16(y)]; add projected label before every "
                    "ResMLP block) -> (mean, diagonal_variance)"
                    if args.label_injection == "concat_per_block"
                    else "G(concat[z, Embedding(y)]) -> (mean, diagonal_variance)"
                )
            ),
            "variance_transform": "softplus(raw_variance) + generator_variance_eps",
            "step2_objective": (
                "Gaussian NLL + conditional MMD + Step1 anchor + variance regularizer; "
                "alternating per-cell z MAP update"
            ),
            "selection_metric": SELECTION_METRIC,
            "checkpoint_selection": SELECTION_METRIC,
            "label_fidelity_role": "diagnostic_only_not_used_for_selection",
            "validation_ilisi_protocol": (
                "joint_pca50_first20_scanpy_knn10_scib_ilisi_graph_knn"
            ),
            "ilisi_eligibility_rule": (
                "joint_real_generated_n - 1 >= ilisi_knn_neighbors"
            ),
            "ilisi_macro_averaging": "mean_over_eligible_classes_only",
            "validation_generation_uses_mean": True,
            "ae_observation_model": "NegativeBinomial",
            "ae_latent_l2_normalized": False,
            "generated_expression_space": (
                "NB sampled counts or NB mean, then normalize_total(target_sum)+log1p"
            ),
            "library_size_source": "training split only",
            "final_test_loaded_during_training": False,
            "n_train": int(train.n_obs),
            "n_validation": int(validation.n_obs),
            "n_genes": int(train.n_vars),
            "ae_artifact_dir_resolved": str(ae_directory),
            "ae_artifact_sha256": {
                name: file_sha256(ae_directory / name) for name in ae_artifact_names
            },
            "output_dir_resolved": str(output.resolve()),
            "gene_order_sha256": sha256_lines(train.var_names.astype(str)),
            "train_barcodes_sha256": sha256_lines(train.obs_names.astype(str)),
            "validation_barcodes_sha256": sha256_lines(validation.obs_names.astype(str)),
            **dict(label_metadata),
        }
    )
    with (output / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
    with (output / "label_map.json").open("w") as handle:
        json.dump(dict(label_metadata), handle, indent=2)


def save_final_generation(
    model,
    ae: AEBundle,
    latent_mean: np.ndarray,
    latent_sd: np.ndarray,
    train_library_size: np.ndarray,
    train_labels: np.ndarray,
    genes: Sequence[str],
    label_names: Sequence[str],
    args: argparse.Namespace,
    output: Path,
) -> None:
    prior_z, labels = make_fixed_conditional_prior(
        len(label_names),
        args.final_n_per_class,
        args.bgm_z_dim,
        args.final_generation_seed,
    )
    generated_standardized, generated_mean, generated_variance = (
        generate_standardized_latent(
            model,
            prior_z,
            labels,
            sample_observation_noise=args.sample_observation_noise,
            observation_seed=args.final_generation_seed + 1,
        )
    )
    generated_library_size = sample_library_sizes(
        train_library_size,
        train_labels,
        labels,
        args.library_mode,
        args.final_library_seed,
        args.library_shrinkage_strength,
    )
    generated_expression, decoded_counts_or_mean = decode_bgm_latent(
        generated_standardized,
        latent_mean,
        latent_sd,
        ae,
        generated_library_size,
        args,
        args.final_decoder_seed,
    )
    generated_ae_latent = (
        generated_standardized * latent_sd[None, :] + latent_mean[None, :]
    ).astype(np.float32)
    label_array = np.asarray([label_names[index] for index in labels], dtype="U")

    np.save(output / "generated.npy", generated_expression)
    np.save(output / "generated_labels.npy", labels)
    np.save(output / "generated_label_names.npy", label_array)
    np.save(output / "generated_library_size.npy", generated_library_size)
    decoded_total = np.asarray(decoded_counts_or_mean.sum(axis=1), dtype=np.float32)
    np.save(output / "generated_decoded_total.npy", decoded_total)
    np.save(output / "generated_standardized_ae_latent.npy", generated_standardized)
    np.save(output / "generated_mean_standardized_ae_latent.npy", generated_mean)
    np.save(output / "generated_variance.npy", generated_variance)
    np.save(output / "prior_z.npy", prior_z)

    generated = ad.AnnData(X=np.asarray(generated_expression, dtype=np.float32))
    generated.var_names = pd.Index(np.asarray(genes, dtype=str))
    generated.obs_names = pd.Index(
        [
            f"generated_c{int(labels[index]):02d}_{index:07d}"
            for index in range(len(labels))
        ]
    )
    generated.obs[args.label_key] = pd.Categorical(
        label_array,
        categories=list(label_names),
    )
    generated.obs["requested_condition"] = pd.Categorical(
        label_array,
        categories=list(label_names),
    )
    generated.obs["condition_id"] = labels.astype(np.int32)
    generated.obs["decoder_library_size"] = generated_library_size
    generated.obs["sampled_count_total" if args.decoder_mode == "sample" else "decoded_mean_total"] = decoded_total
    generated.obs["source"] = pd.Categorical(
        np.repeat("generated", len(labels)), categories=["generated"]
    )
    if args.decoder_mode == "sample":
        generated.layers["counts"] = sparse.csr_matrix(decoded_counts_or_mean)
    generated.obsm["X_ae_latent"] = generated_ae_latent
    generated.obsm["X_ae_latent_standardized"] = generated_standardized
    generated.obsm["bgm_diagonal_variance"] = generated_variance
    generated.uns["conditional_bgm"] = {
        "sample_observation_noise": bool(args.sample_observation_noise),
        "n_per_class": int(args.final_n_per_class),
        "seed": int(args.final_generation_seed),
        "bgm_observation_seed": int(args.final_generation_seed + 1),
        "library_seed": int(args.final_library_seed),
        "decoder_seed": int(args.final_decoder_seed),
        "library_mode": args.library_mode,
        "decoder_mode": args.decoder_mode,
        "target_sum": float(args.target_sum),
        "expression_space": (
            f"normalize_total({float(args.target_sum):g})+log1p after NB decoding"
        ),
        "label_key": args.label_key,
        "label_names": list(label_names),
    }
    generated.write_h5ad(output / "generated.h5ad", compression="gzip")


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_tensorflow()
    seed_everything(args.seed)

    output = args.output_dir.expanduser()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is non-empty: {output}. Use a new directory or "
            "pass --overwrite explicitly."
        )
    output.mkdir(parents=True, exist_ok=True)
    split_dir = args.split_dir.expanduser()
    train_path = split_dir / args.train_file
    validation_path = split_dir / args.validation_file

    print("Loading TRAIN for gradients:", train_path)
    train = load_raw_counts(train_path)
    print("Loading VALIDATION for checkpoint selection:", validation_path)
    validation = load_raw_counts(validation_path)
    assert_disjoint(train, validation)

    label_vocabulary = None
    if args.label_vocabulary_json is not None:
        label_vocabulary = load_label_vocabulary_json(args.label_vocabulary_json)
    train_y, validation_y, label_metadata = encode_labels(
        train,
        validation,
        args.label_key,
        args.min_train_cells_per_class,
        label_vocabulary,
    )
    if args.label_vocabulary_json is not None:
        label_metadata["label_vocabulary_json_resolved"] = str(
            args.label_vocabulary_json.expanduser().resolve()
        )
    label_names = list(label_metadata["id_to_label"])
    print("Conditional labels:")
    for class_id, label_name in enumerate(label_names):
        print(
            f"  {class_id}: {label_name} "
            f"(train={label_metadata['train_counts'][label_name]}, "
            f"validation={label_metadata['validation_counts'][label_name]})"
        )

    ae = load_ae_artifacts(train, validation, args)
    write_config(args, output, train, validation, label_metadata)
    np.save(output / "genes.npy", np.asarray(train.var_names.astype(str), dtype="U"))
    np.save(output / "train_labels.npy", train_y)
    np.save(output / "validation_labels.npy", validation_y)
    train_encoded = np.asarray(ae.train_latent, dtype=np.float32)
    validation_encoded = np.asarray(ae.validation_latent, dtype=np.float32)
    latent_mean = train_encoded.mean(axis=0).astype(np.float32)
    latent_sd = np.maximum(train_encoded.std(axis=0), 1e-4).astype(np.float32)
    train_latent = ((train_encoded - latent_mean) / latent_sd).astype(np.float32)
    validation_latent = (
        (validation_encoded - latent_mean) / latent_sd
    ).astype(np.float32)
    np.savez(
        output / "latent_scale.npz",
        mean=latent_mean,
        sd=latent_sd,
        train_latent_mean=train_latent.mean(axis=0),
        train_latent_sd=train_latent.std(axis=0),
    )
    np.save(output / "validation_latent.npy", validation_latent)

    # Distribution metrics use the same normalized/log1p space as final output.
    # This happens only after raw-count/AE-artifact integrity checks are complete.
    normalize_log_expression_inplace(train, args.target_sum)
    normalize_log_expression_inplace(validation, args.target_sum)

    train_real_by_class = select_fixed_expression_by_class(
        train.X,
        train_y,
        len(label_names),
        args.n_eval_per_class,
        args.seed + 11,
    )
    validation_real_by_class = select_fixed_expression_by_class(
        validation.X,
        validation_y,
        len(label_names),
        args.n_eval_per_class,
        args.seed + 12,
    )
    for class_id, values in train_real_by_class.items():
        np.save(output / f"train_real_class_{class_id}.npy", values)
    for class_id, values in validation_real_by_class.items():
        np.save(output / f"validation_real_class_{class_id}.npy", values)

    gc.collect()
    model = build_conditional_bgm(args, output, len(label_names))
    print("Conditional generator parameters:", model.g_net.count_params())
    print("Encoder parameters:", model.e_net.count_params())
    print("Conditioned Dx:", model.condition_dx)
    print("Network variant:", model.network_variant)
    print("Label injection:", model.label_injection)

    prior_z_eval, prior_y_eval = make_fixed_conditional_prior(
        len(label_names),
        args.n_eval_per_class,
        args.bgm_z_dim,
        args.validation_generation_seed,
    )
    np.save(output / "validation_prior_z.npy", prior_z_eval)
    np.save(output / "validation_prior_labels.npy", prior_y_eval)
    validation_generated_library_size = sample_library_sizes(
        ae.train_library_size,
        train_y,
        prior_y_eval,
        args.library_mode,
        args.validation_library_seed,
        args.library_shrinkage_strength,
    )
    np.save(
        output / "validation_generated_library_size.npy",
        validation_generated_library_size,
    )

    run_step1(
        model,
        train_latent,
        train_y,
        train_real_by_class,
        validation_real_by_class,
        latent_mean,
        latent_sd,
        ae,
        prior_z_eval,
        prior_y_eval,
        validation_generated_library_size,
        label_names,
        args,
        output,
    )
    run_step2(
        model,
        train_latent,
        train_y,
        train_real_by_class,
        validation_real_by_class,
        latent_mean,
        latent_sd,
        ae,
        prior_z_eval,
        prior_y_eval,
        validation_generated_library_size,
        label_names,
        args,
        output,
    )

    save_final_generation(
        model,
        ae,
        latent_mean,
        latent_sd,
        ae.train_library_size,
        train_y,
        train.var_names.astype(str),
        label_names,
        args,
        output,
    )

    print("\nConditional training complete. FINAL TEST WAS NOT LOADED OR INSPECTED.")
    print("Validation-selected generator:", output / "generator.weights.h5")
    print("Frozen conditional H5AD:", output / "generated.h5ad")
    print("Label map:", output / "label_map.json")


if __name__ == "__main__":
    main()

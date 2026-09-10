#!/usr/bin/env python3
"""Train unconditional BayesGM on a Negative-Binomial RNA autoencoder latent.

Only TRAIN and VALIDATION are opened during model development. The final-test
H5AD is deliberately not read here.

The BayesGM backbone models only the autoencoder latent. Following the
scDiffusion-X decoding design, RNA library size is handled separately: the
log-library-size mean and standard deviation are estimated from TRAIN raw counts,
a log-normal size factor is sampled independently at decoding time, and the
Negative-Binomial decoder uses

    mu = library_size * softmax(decoder(z)).

For this unconditional, label-free PBMC68k run a single global TRAIN
log-library-size distribution is used.
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
from dataclasses import dataclass
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

from autoencoder import AEBundle, decode_to_log_expression, load_ae_bundle


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/bayesgm"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-sum", type=float, default=1e4)

    # Frozen Negative-Binomial RNA autoencoder artifacts.
    parser.add_argument("--ae-artifact-dir", type=Path, default=Path("results/autoencoder"))
    parser.add_argument("--ae-latent-dim", type=int, default=128)
    parser.add_argument("--decoder-mode", choices=("sample", "mean"), default="sample")
    parser.add_argument("--decode-batch-size", type=int, default=256)
    parser.add_argument("--decoder-device", default="cpu")
    parser.add_argument("--validation-decoder-seed", type=int, default=4242)
    parser.add_argument("--validation-library-seed", type=int, default=4343)
    parser.add_argument("--final-decoder-seed", type=int, default=2028)
    parser.add_argument("--final-library-seed", type=int, default=2030)

    # BayesGM architecture.
    parser.add_argument("--bgm-z-dim", type=int, default=32)
    parser.add_argument("--g-units", type=parse_int_list, default=[512, 512, 512, 512, 512])
    parser.add_argument("--e-units", type=parse_int_list, default=[512, 512, 512, 512, 512])
    parser.add_argument("--dx-units", type=parse_int_list, default=[256, 128, 64, 16])
    parser.add_argument("--dz-units", type=parse_int_list, default=[128, 64, 32, 8])
    parser.add_argument("--network-variant", choices=("mlp", "resmlp"), default="mlp")
    parser.add_argument("--res-width", type=int, default=256)
    parser.add_argument("--res-generator-blocks", type=int, default=4)
    parser.add_argument("--res-encoder-blocks", type=int, default=3)
    parser.add_argument("--res-expansion", type=int, default=4)
    parser.add_argument("--res-dropout", type=float, default=0.0)
    parser.add_argument("--res-normalization", choices=("layernorm", "rmsnorm"), default="layernorm")

    # BayesGM Step 1.
    parser.add_argument("--step1-lr", type=float, default=1e-4)
    parser.add_argument("--step1-iterations", type=int, default=100_000)
    parser.add_argument("--step1-batch-size", type=int, default=512)
    parser.add_argument("--step1-eval-every", type=int, default=10_000)
    parser.add_argument("--step1-save-iters", type=parse_int_list, default=[20_000, 30_000, 50_000, 100_000])
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--generator-variance-eps", type=float, default=1e-3)

    # Distribution-preserving Step 2.
    parser.add_argument("--step2-epochs", type=int, default=500)
    parser.add_argument("--step2-batch-size", type=int, default=512)
    parser.add_argument("--step2-eval-every", type=int, default=20)
    parser.add_argument("--step2-save-every", type=int, default=50)
    parser.add_argument("--generator-lr", type=float, default=5e-7)
    parser.add_argument("--variance-lr", type=float, default=1e-8)
    parser.add_argument("--latent-lr", type=float, default=1e-7)
    parser.add_argument("--latent-prior-weight", type=float, default=1.0)
    parser.add_argument("--variance-init", type=float, default=0.30)
    parser.add_argument("--variance-min", type=float, default=0.15)
    parser.add_argument("--variance-max", type=float, default=0.60)
    parser.add_argument("--variance-log-prior-weight", type=float, default=5.0)
    parser.add_argument("--variance-mode", choices=("learned", "fixed"), default="learned")
    parser.add_argument("--fixed-variance", type=float, default=0.05)
    parser.add_argument(
        "--bgm-generation-mode",
        choices=("mean", "sample"),
        default="sample",
        help="Use generator mean only or sample mean + sqrt(variance)*noise before AE decoding.",
    )
    parser.add_argument("--validation-bgm-noise-seed", type=int, default=4244)
    parser.add_argument("--final-bgm-noise-seed", type=int, default=2029)
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@dataclass
class DecoderContext:
    bundle: AEBundle
    target_sum: float
    decoder_mode: str
    decode_batch_size: int
    decoder_device: str
    validation_decoder_seed: int
    validation_library_seed: int
    final_decoder_seed: int
    final_library_seed: int
    log_library_mean: float
    log_library_sd: float


@dataclass(frozen=True)
class ResidualConfig:
    width: int
    generator_blocks: int
    encoder_blocks: int
    expansion: int
    dropout: float
    normalization: str


class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(int(input_shape[-1]),),
            initializer="ones",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        inverse_rms = tf.math.rsqrt(
            tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.epsilon
        )
        return inputs * inverse_rms * self.scale


def make_res_norm(name: str, normalization: str) -> tf.keras.layers.Layer:
    if normalization == "layernorm":
        return tf.keras.layers.LayerNormalization(epsilon=1e-5, name=name)
    return RMSNorm(epsilon=1e-6, name=name)


class PreNormResidualMLPBlock(tf.keras.layers.Layer):
    """h <- h + W2(Dropout(SiLU(W1(Norm(h)))))."""

    def __init__(self, config: ResidualConfig, name: str):
        super().__init__(name=name)
        expanded = int(config.width * config.expansion)
        self.norm = make_res_norm("pre_norm", config.normalization)
        self.fc1 = tf.keras.layers.Dense(expanded, kernel_initializer="he_normal", name="expand")
        self.dropout1 = tf.keras.layers.Dropout(config.dropout, name="dropout_1")
        self.fc2 = tf.keras.layers.Dense(config.width, kernel_initializer="he_normal", name="contract")
        self.dropout2 = tf.keras.layers.Dropout(config.dropout, name="dropout_2")

    def call(self, inputs, training: bool = False):
        hidden = self.norm(inputs)
        hidden = tf.nn.silu(self.fc1(hidden))
        hidden = self.dropout1(hidden, training=training)
        hidden = self.fc2(hidden)
        hidden = self.dropout2(hidden, training=training)
        return inputs + hidden


class ResidualVariationalGenerator(tf.keras.Model):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        variance_epsilon: float,
        config: ResidualConfig,
    ):
        super().__init__(name="residual_g_net")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.variance_epsilon = float(variance_epsilon)
        self.config = config
        self.input_projection = tf.keras.layers.Dense(
            config.width, kernel_initializer="he_normal", name="input_projection"
        )
        self.blocks = [
            PreNormResidualMLPBlock(config, name=f"generator_block_{i + 1}")
            for i in range(config.generator_blocks)
        ]
        self.final_norm = make_res_norm("final_norm", config.normalization)
        self.mean_layer = tf.keras.layers.Dense(output_dim, name="mean_output")
        self.var_layer = tf.keras.layers.Dense(output_dim, name="variance_output")
        # Base Step 2 toggles this handle to freeze old BatchNorm statistics.
        self.norm_layer = tf.keras.layers.Activation(
            "linear", name="batchnorm_compatibility_handle"
        )

    def call(self, inputs, eps=None, training: bool = True):
        hidden = self.input_projection(inputs)
        for block in self.blocks:
            hidden = block(hidden, training=training)
        hidden = self.final_norm(hidden)
        mean = self.mean_layer(hidden)
        epsilon = self.variance_epsilon if eps is None else eps
        variance = tf.nn.softplus(self.var_layer(hidden)) + tf.cast(epsilon, hidden.dtype)
        return mean, variance

    @staticmethod
    def reparameterize(mean, variance):
        noise = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
        return mean + noise * tf.sqrt(variance)


class ResidualEncoder(tf.keras.Model):
    def __init__(self, input_dim: int, output_dim: int, config: ResidualConfig):
        super().__init__(name="residual_e_net")
        self.input_projection = tf.keras.layers.Dense(
            config.width, kernel_initializer="he_normal", name="input_projection"
        )
        self.blocks = [
            PreNormResidualMLPBlock(config, name=f"encoder_block_{i + 1}")
            for i in range(config.encoder_blocks)
        ]
        self.final_norm = make_res_norm("final_norm", config.normalization)
        self.output_layer = tf.keras.layers.Dense(output_dim, name="latent_output")

    def call(self, inputs, training: bool = True):
        hidden = self.input_projection(inputs)
        for block in self.blocks:
            hidden = block(hidden, training=training)
        return self.output_layer(self.final_norm(hidden))

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
) -> tuple[DecoderContext, np.ndarray, np.ndarray]:
    directory = args.ae_artifact_dir.expanduser().resolve()
    required = [
        directory / "metadata.json",
        directory / "model.pt",
        directory / "train_latent.npy",
        directory / "validation_latent.npy",
        directory / "train_library_size.npy",
        directory / "validation_library_size.npy",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing autoencoder artifact: {path}")

    bundle = load_ae_bundle(directory, device=args.decoder_device)
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
                f"AE artifact mismatch for {key}: stored={metadata.get(key)!r}, current={value!r}"
            )

    if bundle.train_latent.shape != (train.n_obs, args.ae_latent_dim):
        raise RuntimeError(f"Unexpected train latent shape: {bundle.train_latent.shape}")
    if bundle.validation_latent.shape != (validation.n_obs, args.ae_latent_dim):
        raise RuntimeError(f"Unexpected validation latent shape: {bundle.validation_latent.shape}")
    if np.any(bundle.train_library_size <= 0) or np.any(bundle.validation_library_size <= 0):
        raise RuntimeError("AE artifact contains non-positive RNA library sizes")

    train_log_library = np.log(bundle.train_library_size.astype(np.float64))
    validation_log_library = np.log(bundle.validation_library_size.astype(np.float64))
    context = DecoderContext(
        bundle=bundle,
        target_sum=float(args.target_sum),
        decoder_mode=args.decoder_mode,
        decode_batch_size=int(args.decode_batch_size),
        decoder_device=str(args.decoder_device),
        validation_decoder_seed=int(args.validation_decoder_seed),
        validation_library_seed=int(args.validation_library_seed),
        final_decoder_seed=int(args.final_decoder_seed),
        final_library_seed=int(args.final_library_seed),
        log_library_mean=float(train_log_library.mean()),
        log_library_sd=float(max(train_log_library.std(), 1e-6)),
    )

    # BayesGM models only the AE latent. Library size is a separate decoding
    # nuisance variable, matching the scDiffusion-X design.
    train_features = bundle.train_latent.copy()
    validation_features = bundle.validation_latent.copy()

    args.bgm_x_dim = int(args.ae_latent_dim)
    print("Loaded frozen Negative-Binomial RNA AE:", directory / "model.pt")
    print("AE latent dimension:", args.ae_latent_dim)
    print("BayesGM features: AE latent only")
    print("Library-size model: global TRAIN log-normal")
    print("BayesGM x dimension:", args.bgm_x_dim)
    print("Decoder mode:", args.decoder_mode)
    print(
        "Train log-library mean/sd:",
        f"{context.log_library_mean:.4f}",
        f"{context.log_library_sd:.4f}",
    )
    return context, train_features.astype(np.float32), validation_features.astype(np.float32)


def split_generated_features(
    features: np.ndarray,
    context: DecoderContext,
    library_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    latent_dim = int(context.bundle.model.latent_dim)
    if features.ndim != 2 or features.shape[1] < latent_dim:
        raise ValueError(f"Unexpected generated feature shape: {features.shape}")
    ae_latent = features[:, :latent_dim].astype(np.float32, copy=False)

    if features.shape[1] != latent_dim:
        raise ValueError(
            f"AE-latent BayesGM expects {latent_dim} columns, got {features.shape[1]}"
        )

    # scDiffusion-X samples RNA size factor separately from the generative
    # latent. In this unconditional label-free run we use one global Normal
    # fitted to TRAIN log library sizes.
    rng = np.random.default_rng(int(library_seed))
    log_library = rng.normal(
        loc=context.log_library_mean,
        scale=context.log_library_sd,
        size=len(features),
    )

    library_size = np.exp(log_library).astype(np.float32)
    return ae_latent, library_size


def decode_features_to_expression(
    features: np.ndarray,
    context: DecoderContext,
    decoder_seed: int,
    library_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ae_latent, library_size = split_generated_features(features, context, library_seed)
    expression = decode_to_log_expression(
        context.bundle.model,
        ae_latent,
        library_size,
        mode=context.decoder_mode,
        target_sum=context.target_sum,
        batch_size=context.decode_batch_size,
        device=context.decoder_device,
        seed=int(decoder_seed),
    )
    return expression, ae_latent, library_size

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
        "x_dim": int(args.bgm_x_dim),
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
        "lr_z": args.latent_lr,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "g_d_freq": 1,
        "kl_weight": 5e-5,
        "save_model": False,
        "save_res": False,
    }
    model = BGM(params=params, random_seed=args.seed)

    if args.network_variant == "resmlp":
        config = ResidualConfig(
            width=int(args.res_width),
            generator_blocks=int(args.res_generator_blocks),
            encoder_blocks=int(args.res_encoder_blocks),
            expansion=int(args.res_expansion),
            dropout=float(args.res_dropout),
            normalization=str(args.res_normalization),
        )
        model.g_net = ResidualVariationalGenerator(
            input_dim=args.bgm_z_dim,
            output_dim=args.bgm_x_dim,
            variance_epsilon=args.generator_variance_eps,
            config=config,
        )
        model.e_net = ResidualEncoder(
            input_dim=args.bgm_x_dim,
            output_dim=args.bgm_z_dim,
            config=config,
        )

    dummy_x = tf.zeros((2, args.bgm_x_dim), dtype=tf.float32)
    dummy_z = tf.zeros((2, args.bgm_z_dim), dtype=tf.float32)
    mean, variance = model.g_net(dummy_z, training=False)
    encoded = model.e_net(dummy_x, training=False)
    model.dz_net(dummy_z, training=False)
    model.dx_net(dummy_x, training=False)
    if tuple(mean.shape) != (2, args.bgm_x_dim):
        raise RuntimeError(f"Unexpected generator mean shape: {mean.shape}")
    if tuple(variance.shape) != (2, args.bgm_x_dim):
        raise RuntimeError(f"Unexpected generator variance shape: {variance.shape}")
    if tuple(encoded.shape) != (2, args.bgm_z_dim):
        raise RuntimeError(f"Unexpected encoder shape: {encoded.shape}")

    print("===== BayesGM architecture =====")
    print("network variant:", args.network_variant)
    print("BGM z dimension:", args.bgm_z_dim)
    print("BGM x dimension:", args.bgm_x_dim)
    if args.network_variant == "resmlp":
        print("ResMLP width/G blocks/E blocks:", args.res_width, args.res_generator_blocks, args.res_encoder_blocks)
        print("ResMLP expansion/norm:", args.res_expansion, args.res_normalization)
    print("G parameters:", model.g_net.count_params())
    print("E parameters:", model.e_net.count_params())
    print("Dz parameters:", model.dz_net.count_params())
    print("Dx parameters:", model.dx_net.count_params())
    return model

def effective_variance_np(variance: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.variance_mode == "fixed":
        return np.full_like(variance, np.float32(args.fixed_variance))
    return np.clip(variance, args.variance_min, args.variance_max).astype(np.float32)


def effective_variance_tf(variance: tf.Tensor, args: argparse.Namespace) -> tf.Tensor:
    if args.variance_mode == "fixed":
        return tf.fill(tf.shape(variance), tf.cast(args.fixed_variance, variance.dtype))
    return tf.clip_by_value(variance, args.variance_min, args.variance_max)


def generate_bgm_features(
    model: BGM,
    prior_z: np.ndarray,
    args: argparse.Namespace,
    noise_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, variance = model.g_net(
        tf.convert_to_tensor(prior_z, dtype=tf.float32),
        training=False,
    )
    mean_np = mean.numpy().astype(np.float32)
    variance_np = effective_variance_np(variance.numpy().astype(np.float32), args)
    if args.bgm_generation_mode == "mean":
        generated = mean_np
    else:
        rng = np.random.default_rng(int(noise_seed))
        noise = rng.standard_normal(mean_np.shape).astype(np.float32)
        generated = mean_np + noise * np.sqrt(variance_np)
    return generated.astype(np.float32), mean_np, variance_np



def decode_bgm_latent(
    standardized: np.ndarray,
    latent_mean: np.ndarray,
    latent_sd: np.ndarray,
    ae: DecoderContext,
    decoder_seed: int | None = None,
    library_seed: int | None = None,
    return_components: bool = False,
):
    raw_features = standardized * latent_sd[None, :] + latent_mean[None, :]
    if decoder_seed is None:
        decoder_seed = ae.validation_decoder_seed
    if library_seed is None:
        library_seed = ae.validation_library_seed
    expression, ae_latent, library_size = decode_features_to_expression(
        raw_features,
        ae,
        decoder_seed=int(decoder_seed),
        library_seed=int(library_seed),
    )
    if return_components:
        return expression, raw_features.astype(np.float32), ae_latent, library_size
    return expression

def evaluate_generator(
    model: BGM,
    prior_z: np.ndarray,
    train_real: np.ndarray,
    validation_real: np.ndarray,
    latent_mean: np.ndarray,
    latent_sd: np.ndarray,
    ae: DecoderContext,
    args: argparse.Namespace,
) -> tuple[dict[str, float], np.ndarray]:
    generated_standardized, generated_mean, generated_variance = generate_bgm_features(
        model, prior_z, args, args.validation_bgm_noise_seed
    )
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
    variance = generated_variance[: min(512, len(generated_variance))].ravel()
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
    ae: DecoderContext,
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
    variance_optimizer = (
        tf.keras.optimizers.Adam(args.variance_lr, beta_1=0.9, beta_2=0.99)
        if args.variance_mode == "learned"
        else None
    )

    @tf.function(reduce_retracing=True)
    def update(batch_z: tf.Tensor, batch_x: tf.Tensor, prior_z: tf.Tensor):
        with tf.GradientTape(persistent=True) as tape:
            mu, variance = model.g_net(batch_z, training=False)
            variance_safe = effective_variance_tf(variance, args)
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
            if args.variance_mode == "learned":
                log_prior = tf.reduce_mean(
                    tf.square(tf.math.log(variance_safe) - math.log(args.variance_init))
                )
                bounds = tf.reduce_mean(
                    tf.square(tf.nn.relu(args.variance_min - variance))
                    + tf.square(tf.nn.relu(variance - args.variance_max))
                )
                variance_regularizer = (
                    args.variance_log_prior_weight * log_prior + 10.0 * bounds
                )
            else:
                variance_regularizer = tf.constant(0.0, dtype=tf.float32)
            total = (
                nll
                + args.mmd_weight * latent_mmd
                + args.anchor_weight * anchor
                + variance_regularizer
            )

        mean_gradients = tape.gradient(total, mean_variables)
        variance_gradients = (
            tape.gradient(total, variance_variables)
            if args.variance_mode == "learned"
            else [None] * len(variance_variables)
        )
        del tape
        apply_gradients_clipped(mean_optimizer, mean_gradients, mean_variables, args.gradient_clip)
        if args.variance_mode == "learned":
            apply_gradients_clipped(
                variance_optimizer, variance_gradients, variance_variables, args.gradient_clip
            )
        return total, nll, mse, latent_mmd, anchor, variance_regularizer

    return update, mean_variables


def make_step2_latent_update(
    model: BGM,
    args: argparse.Namespace,
    data_z: tf.Variable,
):
    """Create the Step-2 optimizer for per-cell latent variables.

    The generator is held fixed during this update. Only the rows of
    ``data_z`` corresponding to the current mini-batch receive gradients.

    The latent objective follows the original BayesGM MAP update:

        Gaussian NLL + latent_prior_weight * 0.5 * ||z||^2
    """

    # Reuse the posterior optimizer created by the original BGM class.
    # Its learning rate is controlled by args.latent_lr through build_bgm().
    latent_optimizer = model.posterior_optimizer

    @tf.function(reduce_retracing=True)
    def update_latent(
        batch_indices: tf.Tensor,
        batch_x: tf.Tensor,
    ):
        batch_indices = tf.cast(
            batch_indices,
            tf.int32,
        )

        batch_x = tf.cast(
            batch_x,
            tf.float32,
        )

        # Do not watch the generator parameters in this step.
        # We only need gradients with respect to data_z.
        with tf.GradientTape(
            watch_accessed_variables=False
        ) as tape:
            tape.watch(data_z)

            batch_z = tf.gather(
                data_z,
                batch_indices,
            )

            mu, variance = model.g_net(
                batch_z,
                training=False,
            )

            # Use the same safe variance range as the Step-2
            # generator update.
            variance_safe = effective_variance_tf(variance, args)

            latent_nll = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(batch_x - mu)
                    / (2.0 * variance_safe)
                    + 0.5 * tf.math.log(variance_safe),
                    axis=1,
                )
            )

            latent_prior = tf.reduce_mean(
                0.5
                * tf.reduce_sum(
                    tf.square(batch_z),
                    axis=1,
                )
            )

            latent_loss = (
                latent_nll
                + args.latent_prior_weight * latent_prior
            )

        latent_gradient = tape.gradient(
            latent_loss,
            data_z,
        )

        # This gradient is normally an IndexedSlices object because
        # only the current mini-batch rows are gathered.
        apply_gradients_clipped(
            optimizer=latent_optimizer,
            gradients=[latent_gradient],
            variables=[data_z],
            clip_norm=args.gradient_clip,
        )

        updated_batch_z = tf.gather(
            data_z,
            batch_indices,
        )

        z_rms = tf.sqrt(
            tf.reduce_mean(
                tf.square(updated_batch_z)
            )
        )

        return (
            latent_loss,
            latent_nll,
            latent_prior,
            z_rms,
        )

    return update_latent

def run_step2(
    model: BGM,
    train_latent: np.ndarray,
    train_real: np.ndarray,
    validation_real: np.ndarray,
    latent_mean: np.ndarray,
    latent_sd: np.ndarray,
    ae: DecoderContext,
    prior_z_eval: np.ndarray,
    args: argparse.Namespace,
    output: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(
        args.seed + 400
    )

    # ------------------------------------------------------------
    # Initialize per-cell latent variables using the selected
    # Step-1 encoder checkpoint.
    # ------------------------------------------------------------
    data_z_initial = model.e_net(
        tf.convert_to_tensor(
            train_latent,
            dtype=tf.float32,
        ),
        training=False,
    ).numpy().astype(np.float32)

    # data_z must remain a TensorFlow Variable during Step 2.
    # Each row corresponds to one training cell.
    data_z = tf.Variable(
        data_z_initial,
        trainable=True,
        dtype=tf.float32,
        name="step2_data_z",
    )

    # Keep a reference on the BGM object, consistent with the
    # original BGM.fit() implementation.
    model.data_z = data_z

    # ------------------------------------------------------------
    # Initialize the variance head.
    # ------------------------------------------------------------
    initialize_variance_head(
        model,
        args.variance_init if args.variance_mode == "learned" else args.fixed_variance,
        args.generator_variance_eps,
    )

    # ------------------------------------------------------------
    # Snapshot the Step-1 mean-network parameters.
    # These values are used for the Step-2 anchor penalty.
    # ------------------------------------------------------------
    model.g_net.norm_layer.trainable = False

    variance_variable_ids = {
        id(variable)
        for variable in model.g_net.var_layer.trainable_variables
    }

    mean_variables_before = [
        variable
        for variable in model.g_net.trainable_variables
        if id(variable) not in variance_variable_ids
    ]

    anchor_variables = [
        tf.identity(variable)
        for variable in mean_variables_before
    ]

    # Generator parameter update:
    # NLL + MMD + Step-1 anchor + variance regularization.
    update_generator, _ = make_step2_update(
        model,
        args,
        anchor_variables,
    )

    # Per-cell latent update:
    # NLL + standard-normal latent prior.
    update_latent = make_step2_latent_update(
        model,
        args,
        data_z,
    )

    history: list[dict[str, float]] = []

    best_score = -math.inf

    best_generator = (
        output / "generator.weights.h5"
    )

    best_data_z = (
        output / "data_z.npy"
    )

    checkpoint_dir = (
        output / "step2_checkpoints"
    )

    checkpoint_dir.mkdir(
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Evaluate the initial Step-2 state before any updates.
    # ------------------------------------------------------------
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

    baseline_data_z = data_z.numpy()

    baseline = {
        "epoch": 0,
        "loss_total_last": math.nan,
        "loss_nll_last": math.nan,
        "loss_mse_last": math.nan,
        "latent_mmd_last": math.nan,
        "anchor_last": math.nan,
        "variance_regularizer_last": math.nan,
        "latent_loss_last": math.nan,
        "latent_nll_last": math.nan,
        "latent_prior_last": math.nan,
        "latent_z_rms_last": float(
            np.sqrt(
                np.mean(
                    np.square(baseline_data_z)
                )
            )
        ),
        "data_z_delta_rms": 0.0,
        "data_z_mean": float(
            baseline_data_z.mean()
        ),
        "data_z_sd": float(
            baseline_data_z.std()
        ),
        **baseline_metrics,
    }

    history.append(baseline)

    pd.DataFrame(history).to_csv(
        output / "step2_history.csv",
        index=False,
    )

    best_score = baseline_metrics[
        "validation_ilisi_pca20"
    ]

    model.g_net.save_weights(
        str(best_generator)
    )

    np.save(
        best_data_z,
        data_z.numpy(),
    )

    np.save(
        output / "validation_generated.npy",
        baseline_generated,
    )

    with (output / "step2.json").open("w") as handle:
        json.dump(
            {
                "epoch": -1,
                "validation_ilisi_pca20": best_score,
            },
            handle,
            indent=2,
        )

    print(
        "\nBayesGM distribution-preserving Step 2 "
        "with trainable per-cell latent variables ..."
    )

    print(
        f"baseline validation iLISI={best_score:.4f}; "
        f"variance mode={args.variance_mode}; "
        f"variance init/fixed={args.variance_init if args.variance_mode == 'learned' else args.fixed_variance}; "
        f"generator lr={args.generator_lr}; "
        f"variance lr={args.variance_lr}; "
        f"latent lr={args.latent_lr}"
    )

    last_generator_losses = [
        math.nan
    ] * 6

    last_latent_losses = [
        math.nan
    ] * 4

    # ------------------------------------------------------------
    # Step-2 alternating optimization.
    #
    # For every mini-batch:
    #   1. update generator parameters using current z_i
    #   2. freeze generator and update the same z_i
    # ------------------------------------------------------------
    for epoch in range(
        1,
        args.step2_epochs + 1,
    ):
        permutation = rng.permutation(
            len(train_latent)
        )

        batches = (
            len(train_latent)
            // args.step2_batch_size
        )

        progress = tqdm(
            range(batches),
            desc=(
                f"Step 2 epoch "
                f"{epoch}/{args.step2_epochs}"
            ),
        )

        for batch_number in progress:
            start = (
                batch_number
                * args.step2_batch_size
            )

            end = (
                start
                + args.step2_batch_size
            )

            indices = permutation[
                start:end
            ]

            indices_tf = tf.convert_to_tensor(
                indices,
                dtype=tf.int32,
            )

            batch_x = tf.convert_to_tensor(
                train_latent[indices],
                dtype=tf.float32,
            )

            # Read the latest latent variables for this batch.
            #
            # stop_gradient prevents the generator update from changing
            # data_z. The latent variables are updated separately below.
            batch_z = tf.stop_gradient(
                tf.gather(
                    data_z,
                    indices_tf,
                )
            )

            prior_count = min(
                args.mmd_samples,
                args.step2_batch_size,
            )

            prior_z = tf.convert_to_tensor(
                rng.standard_normal(
                    (
                        prior_count,
                        args.bgm_z_dim,
                    )
                ).astype(np.float32),
                dtype=tf.float32,
            )

            # ----------------------------------------------------
            # A. Update generator mean and variance parameters.
            # ----------------------------------------------------
            generator_values = update_generator(
                batch_z,
                batch_x,
                prior_z,
            )

            last_generator_losses = [
                float(value.numpy())
                for value in generator_values
            ]

            # ----------------------------------------------------
            # B. Update the current cells' latent variables.
            #
            # This uses the generator after the generator update,
            # matching the alternating order of the original BGM.
            # ----------------------------------------------------
            latent_values = update_latent(
                indices_tf,
                batch_x,
            )

            last_latent_losses = [
                float(value.numpy())
                for value in latent_values
            ]

            progress.set_postfix(
                total=(
                    f"{last_generator_losses[0]:.3f}"
                ),
                mse=(
                    f"{last_generator_losses[2]:.5f}"
                ),
                mmd=(
                    f"{last_generator_losses[3]:.4f}"
                ),
                zloss=(
                    f"{last_latent_losses[0]:.3f}"
                ),
                zrms=(
                    f"{last_latent_losses[3]:.3f}"
                ),
            )

        should_eval = (
            epoch % args.step2_eval_every == 0
            or epoch == args.step2_epochs
        )

        if not should_eval:
            continue

        # --------------------------------------------------------
        # Evaluate generator in decoded full-gene expression space.
        # --------------------------------------------------------
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

        current_data_z = (
            data_z.numpy()
        )

        data_z_delta_rms = float(
            np.sqrt(
                np.mean(
                    np.square(
                        current_data_z
                        - data_z_initial
                    )
                )
            )
        )

        data_z_mean = float(
            current_data_z.mean()
        )

        data_z_sd = float(
            current_data_z.std()
        )

        row = {
            "epoch": epoch,

            # Generator losses.
            "loss_total_last":
                last_generator_losses[0],
            "loss_nll_last":
                last_generator_losses[1],
            "loss_mse_last":
                last_generator_losses[2],
            "latent_mmd_last":
                last_generator_losses[3],
            "anchor_last":
                last_generator_losses[4],
            "variance_regularizer_last":
                last_generator_losses[5],

            # Per-cell latent losses.
            "latent_loss_last":
                last_latent_losses[0],
            "latent_nll_last":
                last_latent_losses[1],
            "latent_prior_last":
                last_latent_losses[2],
            "latent_z_rms_last":
                last_latent_losses[3],

            # Global data_z diagnostics.
            "data_z_delta_rms":
                data_z_delta_rms,
            "data_z_mean":
                data_z_mean,
            "data_z_sd":
                data_z_sd,

            **metrics,
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            output / "step2_history.csv",
            index=False,
        )

        score = metrics[
            "validation_ilisi_pca20"
        ]

        print(
            f"Step2 epoch={epoch} "
            f"| val_iLISI={score:.4f} "
            f"| val_MMD50="
            f"{metrics['validation_mmd_pca50']:.4f} "
            f"| z_delta={data_z_delta_rms:.6f} "
            f"| z_mean={data_z_mean:.4f} "
            f"| z_sd={data_z_sd:.4f} "
            f"| var="
            f"{metrics['bgm_var_q01']:.4f}/"
            f"{metrics['bgm_var_q50']:.4f}/"
            f"{metrics['bgm_var_q99']:.4f}"
        )

        # --------------------------------------------------------
        # Save periodic generator + latent checkpoints.
        # --------------------------------------------------------
        if epoch % args.step2_save_every == 0:
            directory = (
                checkpoint_dir
                / f"epoch_{epoch:04d}"
            )

            directory.mkdir(
                exist_ok=True
            )

            model.g_net.save_weights(
                str(
                    directory
                    / "generator.weights.h5"
                )
            )

            np.save(
                directory / "data_z.npy",
                current_data_z,
            )

            np.save(
                directory
                / "validation_generated.npy",
                generated,
            )

        # --------------------------------------------------------
        # Validation iLISI selects both generator and data_z.
        # --------------------------------------------------------
        if score > best_score:
            best_score = score

            model.g_net.save_weights(
                str(best_generator)
            )

            np.save(
                best_data_z,
                current_data_z,
            )

            np.save(
                output
                / "validation_generated.npy",
                generated,
            )

            with (
                output / "step2.json"
            ).open("w") as handle:
                json.dump(
                    {
                        "epoch": epoch,
                        "validation_ilisi_pca20":
                            score,
                        "data_z_delta_rms":
                            data_z_delta_rms,
                        "data_z_mean":
                            data_z_mean,
                        "data_z_sd":
                            data_z_sd,
                    },
                    handle,
                    indent=2,
                )

            print(
                "  new validation-selected "
                "Step 2 generator + data_z checkpoint"
            )

    # ------------------------------------------------------------
    # Restore generator and data_z from the same best epoch.
    # ------------------------------------------------------------
    model.g_net.load_weights(
        str(best_generator)
    )

    selected_data_z = np.load(
        best_data_z,
        allow_pickle=False,
    ).astype(np.float32)

    data_z.assign(
        selected_data_z
    )

    model.data_z = data_z

    print(
        "Restored validation-selected Step 2 checkpoint:"
    )

    print(
        f"  validation iLISI = {best_score:.4f}"
    )

    print(
        "  selected data_z delta RMS = "
        f"{np.sqrt(np.mean(np.square(selected_data_z - data_z_initial))):.6f}"
    )

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
            "ae_type": "single-RNA Negative-Binomial autoencoder",
            "ae_encoder_input": "log1p(raw counts)",
            "ae_decoder": "library_size * softmax(decoder(z))",
            "ae_observation_model": "Negative Binomial with gene-specific inverse dispersion",
            "generated_output_space": "NB sampled counts -> normalize_total(1e4) -> log1p" if args.decoder_mode == "sample" else "NB mean -> normalize_total(1e4) -> log1p",
            "bgm_feature_space": "AE latent only",
            "library_size_model": "global Normal(log library size) fitted on TRAIN; sampled independently at decoding",
            "bgm_x_dim": int(args.bgm_x_dim),
            "network_variant": args.network_variant,
            "variance_mode": args.variance_mode,
            "bgm_generation_mode": args.bgm_generation_mode,
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
    if args.variance_mode == "learned" and not (
        args.variance_min < args.variance_init < args.variance_max
    ):
        raise ValueError("variance-min < variance-init < variance-max is required")
    if args.fixed_variance <= 0:
        raise ValueError("fixed-variance must be positive")
    if args.res_width <= 0 or args.res_generator_blocks <= 0 or args.res_encoder_blocks <= 0:
        raise ValueError("ResMLP width/block counts must be positive")
    if args.res_expansion <= 0 or not 0.0 <= args.res_dropout < 1.0:
        raise ValueError("Invalid ResMLP expansion/dropout")
    configure_tensorflow()
    seed_everything(args.seed)
    install_generator_variance_epsilon(args.generator_variance_eps)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is non-empty; refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    split = args.split_dir.expanduser()
    train_path = split / "pbmc68k_train_raw.h5ad"
    validation_path = split / "pbmc68k_validation_raw.h5ad"

    print("Loading TRAIN for gradients:", train_path)
    train = load_log_expression(train_path, args.target_sum)
    print("Loading VALIDATION for checkpoint selection:", validation_path)
    validation = load_log_expression(validation_path, args.target_sum)
    assert_disjoint(train, validation)
    # Force a fixed-width Unicode array so eval can load with allow_pickle=False.
    np.save(output / "genes.npy", np.asarray(train.var_names.astype(str), dtype="U"))

    ae, train_encoded, validation_encoded = load_ae_artifacts(train, validation, args)
    write_run_config(args, output, train, validation)
    latent_mean = train_encoded.mean(axis=0).astype(np.float32)
    latent_sd = train_encoded.std(axis=0).astype(np.float32)
    latent_sd = np.maximum(latent_sd, 1e-4)
    train_latent = ((train_encoded - latent_mean) / latent_sd).astype(np.float32)
    validation_latent = ((validation_encoded - latent_mean) / latent_sd).astype(np.float32)
    scale_payload = dict(
        mean=latent_mean,
        sd=latent_sd,
        train_latent_mean=train_latent.mean(axis=0),
        train_latent_sd=train_latent.std(axis=0),
    )
    # feature_scale is the preferred name. latent_scale is retained for compatibility
    # with the existing unconditional evaluation/audit conventions.
    np.savez(output / "feature_scale.npz", **scale_payload)
    np.savez(output / "latent_scale.npz", **scale_payload)
    # Kept only for auditing; validation features are never used for gradients.
    np.save(output / "validation_features_standardized.npy", validation_latent)
    np.save(output / "validation_latent.npy", validation_latent)

    # Fixed expression subsets make every checkpoint directly comparable.
    train_real = select_fixed_rows(train.X, args.n_eval, args.seed + 11)
    validation_real = select_fixed_rows(validation.X, args.n_eval, args.seed + 12)
    np.save(output / "validation_real.npy", validation_real)

    gc.collect()

    model = build_bgm(args, output)
    print("BayesGM x_dim:", args.bgm_x_dim)
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
    final_standardized, final_feature_mean, final_feature_variance = generate_bgm_features(
        model, final_prior_z, args, args.final_bgm_noise_seed
    )
    (
        final_generated,
        final_features,
        final_ae_latent,
        final_library_size,
    ) = decode_bgm_latent(
        final_standardized,
        latent_mean,
        latent_sd,
        ae,
        decoder_seed=ae.final_decoder_seed,
        library_seed=ae.final_library_seed,
        return_components=True,
    )
    np.save(output / "generated.npy", final_generated)
    np.save(output / "prior_z.npy", final_prior_z)
    np.save(output / "generated_features_standardized.npy", final_standardized)
    np.save(output / "generated_feature_mean_standardized.npy", final_feature_mean)
    np.save(output / "generated_feature_variance.npy", final_feature_variance)
    np.save(output / "generated_features_unstandardized.npy", final_features)
    np.save(output / "generated_ae_latent.npy", final_ae_latent)
    np.save(output / "generated_library_size.npy", final_library_size)

    print("\nTraining complete. FINAL TEST HAS NOT BEEN LOADED OR INSPECTED.")
    print("Validation-selected generator:", output / "generator.weights.h5")
    print("Frozen generated expression:", output / "generated.npy")
    print("Run evaluate.py only after all model choices are frozen.")


if __name__ == "__main__":
    main()

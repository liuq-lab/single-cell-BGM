#!/usr/bin/env python3
"""Generate labeled conditional cells with a frozen Negative-Binomial AE.

The conditional BayesGM is reconstructed from the artifacts written by
``train.py``:

    config.json
    label_map.json
    generator.weights.h5
    latent_scale.npz
    genes.npy
    train_labels.npy
    <NB-AE artifact directory>/{metadata.json,model.pt,genes.npy,
                                train_latent.npy,train_library_size.npy}

Default conditional generator:

    e_y = Embedding(y)
    (mu_std, var_std) = G(concat([z, e_y]))

For checkpoints with ``label_injection=additive_per_block``, both z and e_y
are 32-dimensional, the input is z + e_y, and a learned projection of e_y is
added before every ResMLP block.

For ``label_injection=concat_per_block``, z is 32-dimensional, e_y is
16-dimensional, the input is concat([z, e_y]), and e_y is again projected and
added before every ResMLP block.

The generated standardized AE latent is either:

    mean mode:    h_std = mu_std
    sampled mode: h_std = mu_std + sqrt(var_std) * epsilon

It is then inverse-standardized without L2 normalization.  A library size is
sampled exclusively from the AE TRAIN split.  In ``conditional-copula`` mode,
the inverse-standardized generated latent is projected by a TRAIN-fit PCA32,
scored by distance-weighted KNN64 against global TRAIN cells, and rank-coupled
to a clipped bootstrap of the global TRAIN empirical log-library marginal.
This frozen policy is implemented locally and does not use labels.
The frozen Negative-Binomial decoder then produces either NB sampled counts or
its mean.  The decoded matrix is normalized to ``target_sum`` and ``log1p``
transformed exactly once before it is stored in ``AnnData.X``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/conditional_bgm_matplotlib")

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
import torch

from autoencoder import (
    AEBundle,
    decode_mu_matrix,
    decode_sample_counts,
    load_ae_bundle,
    normalize_total_log1p_dense,
    sha256_lines,
)


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
    ) -> None:
        super().__init__(name="conditional_g_net")
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
        self.norm_layer = tf.keras.layers.BatchNormalization(
            name="input_batchnorm"
        )
        self.all_layers = [
            tf.keras.layers.Dense(
                int(width),
                activation=None,
                name=f"hidden_{index + 1}",
            )
            for index, width in enumerate(hidden_units)
        ]
        self.mean_layer = tf.keras.layers.Dense(
            self.output_dim,
            name="mean_output",
        )
        self.var_layer = tf.keras.layers.Dense(
            self.output_dim,
            name="variance_output",
        )

    def call(
        self,
        z: tf.Tensor,
        labels: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        z = tf.cast(z, tf.float32)
        labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
        tf.debugging.assert_equal(tf.shape(z)[0], tf.shape(labels)[0])
        tf.debugging.assert_greater_equal(labels, 0)
        tf.debugging.assert_less(labels, self.num_classes)

        label_features = self.label_embedding(labels)
        hidden = tf.concat([z, label_features], axis=-1)
        hidden = self.norm_layer(hidden, training=training)
        for layer in self.all_layers:
            hidden = tf.nn.leaky_relu(layer(hidden), alpha=0.2)
        mean = self.mean_layer(hidden)
        variance = (
            tf.nn.softplus(self.var_layer(hidden))
            + tf.cast(self.variance_epsilon, mean.dtype)
        )
        return mean, variance


class RMSNorm(tf.keras.layers.Layer):
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
            tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
            + self.epsilon
        )
        return inputs * inverse_rms * self.scale


@dataclass(frozen=True)
class ResidualConfig:
    width: int = 256
    generator_blocks: int = 4
    expansion: int = 4
    dropout: float = 0.0
    normalization: str = "layernorm"


def make_norm(kind: str, name: str) -> tf.keras.layers.Layer:
    if kind == "layernorm":
        return tf.keras.layers.LayerNormalization(epsilon=1e-5, name=name)
    if kind == "rmsnorm":
        return RMSNorm(epsilon=1e-6, name=name)
    raise ValueError(f"Unknown residual normalization: {kind!r}")


class PreNormResidualMLPBlock(tf.keras.layers.Layer):
    def __init__(self, config: ResidualConfig, name: str) -> None:
        super().__init__(name=name)
        expanded = int(config.width * config.expansion)
        self.norm = make_norm(config.normalization, "pre_norm")
        self.fc1 = tf.keras.layers.Dense(
            expanded,
            kernel_initializer="he_normal",
            name="expand",
        )
        self.dropout1 = tf.keras.layers.Dropout(
            config.dropout,
            name="dropout_1",
        )
        self.fc2 = tf.keras.layers.Dense(
            config.width,
            kernel_initializer="he_normal",
            name="contract",
        )
        self.dropout2 = tf.keras.layers.Dropout(
            config.dropout,
            name="dropout_2",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden = self.norm(inputs)
        hidden = tf.nn.silu(self.fc1(hidden))
        hidden = self.dropout1(hidden, training=training)
        hidden = self.fc2(hidden)
        hidden = self.dropout2(hidden, training=training)
        return inputs + hidden


class ConditionalResidualVariationalGenerator(tf.keras.Model):
    """Conditional ResMLP generator matching train.py exactly."""

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
            input_dim=self.num_classes,
            output_dim=self.label_embed_dim,
            embeddings_initializer="glorot_uniform",
            name="label_embedding",
        )
        self.input_projection = tf.keras.layers.Dense(
            config.width,
            kernel_initializer="he_normal",
            name="input_projection",
        )
        self.blocks = [
            PreNormResidualMLPBlock(
                config,
                name=f"generator_block_{index + 1}",
            )
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
        self.final_norm = make_norm(config.normalization, "final_norm")
        self.mean_layer = tf.keras.layers.Dense(
            output_dim,
            name="mean_output",
        )
        self.var_layer = tf.keras.layers.Dense(
            output_dim,
            name="variance_output",
        )
        self.norm_layer = tf.keras.layers.Activation(
            "linear",
            name="batchnorm_compatibility_handle",
        )

    def call(
        self,
        z: tf.Tensor,
        labels: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
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
        variance = (
            tf.nn.softplus(self.var_layer(hidden))
            + tf.cast(self.variance_epsilon, mean.dtype)
        )
        return mean, variance


def build_generator(
    config: Mapping[str, Any],
    num_classes: int,
) -> tf.keras.Model:
    variant = str(config.get("network_variant", "resmlp"))
    z_dim = int(config["bgm_z_dim"])
    output_dim = int(config["ae_latent_dim"])
    label_embed_dim = int(config["label_embed_dim"])
    variance_epsilon = float(config["generator_variance_eps"])

    if variant == "resmlp":
        residual = ResidualConfig(
            width=int(config.get("res_width", 256)),
            generator_blocks=int(config.get("res_generator_blocks", 4)),
            expansion=int(config.get("res_expansion", 4)),
            dropout=float(config.get("res_dropout", 0.0)),
            normalization=str(config.get("res_normalization", "layernorm")),
        )
        generator: tf.keras.Model = ConditionalResidualVariationalGenerator(
            z_dim=z_dim,
            output_dim=output_dim,
            num_classes=num_classes,
            label_embed_dim=label_embed_dim,
            variance_epsilon=variance_epsilon,
            config=residual,
            label_injection=str(config.get("label_injection", "early_concat")),
        )
    elif variant == "mlp":
        generator = ConditionalVariationalGenerator(
            z_dim=z_dim,
            output_dim=output_dim,
            num_classes=num_classes,
            label_embed_dim=label_embed_dim,
            hidden_units=[int(value) for value in config["g_units"]],
            variance_epsilon=variance_epsilon,
        )
    else:
        raise ValueError(f"Unsupported network_variant in config.json: {variant!r}")

    dummy_z = tf.zeros((2, z_dim), dtype=tf.float32)
    dummy_y = tf.zeros((2,), dtype=tf.int32)
    mean, variance = generator(dummy_z, dummy_y, training=False)
    if tuple(mean.shape) != (2, output_dim):
        raise RuntimeError(f"Unexpected generator mean shape: {mean.shape}")
    if tuple(variance.shape) != (2, output_dim):
        raise RuntimeError(f"Unexpected generator variance shape: {variance.shape}")
    tf.debugging.assert_all_finite(mean, "Generator mean contains non-finite values")
    tf.debugging.assert_positive(variance, "Generator variance must be positive")

    # The selected checkpoint was saved/evaluated in inference mode.  For the
    # dense MLP this also prevents BatchNorm moving statistics from changing.
    generator.trainable = False
    return generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Generate conditional scRNA-seq expression from a saved "
            "train.py output directory."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing config.json and generator.weights.h5.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["all"],
        help="Label names or integer class IDs; use 'all' for every trained label.",
    )

    counts = parser.add_mutually_exclusive_group()
    counts.add_argument(
        "--n-per-label",
        type=int,
        default=None,
        help="Generate the same number for every selected label.",
    )
    counts.add_argument(
        "--counts-json",
        type=Path,
        default=None,
        help='JSON mapping label names/IDs to counts, e.g. {"0": 1000, "1": 500}.',
    )
    counts.add_argument(
        "--counts-from-h5ad",
        type=Path,
        default=None,
        help=(
            "Read counts per label from a real reference H5AD, such as the "
            "validation split."
        ),
    )
    parser.add_argument(
        "--reference-label-key",
        default=None,
        help="Label column in --counts-from-h5ad; defaults to config label_key.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=None,
        help="Optional cap applied to counts read from JSON/reference H5AD.",
    )
    parser.add_argument(
        "--sample-observation-noise",
        action="store_true",
        help="Sample h~N(mu,diag(var)); default uses conditional mean mu.",
    )
    parser.add_argument(
        "--independent-z",
        action="store_true",
        help=(
            "Use independent z draws for each label.  By default the same base "
            "z draws are reused across labels for a controlled visual comparison."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
        help=(
            "Backward-compatible base seed.  The prior, latent noise, library "
            "and NB decoder use separate deterministic streams; use their "
            "dedicated options to override them independently."
        ),
    )
    parser.add_argument("--prior-seed", type=int, default=None)
    parser.add_argument("--latent-noise-seed", type=int, default=None)
    parser.add_argument("--library-seed", type=int, default=None)
    parser.add_argument("--decoder-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--decode-batch-size",
        type=int,
        default=256,
        help="Batch size for the frozen Negative-Binomial decoder.",
    )
    parser.add_argument(
        "--ae-artifact-dir",
        type=Path,
        default=None,
        help="Override AE directory stored in config.json.",
    )
    parser.add_argument(
        "--decode-device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="PyTorch device for the frozen AE decoder.",
    )
    parser.add_argument(
        "--decoder-mode",
        choices=("sample", "mean"),
        default=None,
        help=(
            "Override the training config NB output mode. Effective package "
            "default is sample."
        ),
    )
    parser.add_argument(
        "--library-mode",
        choices=(
            "global-empirical",
            "label-empirical",
            "label-lognormal-shrunk",
            "conditional-copula",
        ),
        default=None,
        help=(
            "Override the training config TRAIN-only library-size model."
        ),
    )
    parser.add_argument(
        "--library-shrinkage-strength",
        type=float,
        default=None,
        help=(
            "Override config tau for label-lognormal-shrunk variance pooling."
        ),
    )
    parser.add_argument(
        "--library-copula-pca-components",
        type=int,
        default=32,
        help="TRAIN-fit PCA dimensionality for conditional-copula sampling.",
    )
    parser.add_argument(
        "--library-copula-knn-k",
        type=int,
        default=64,
        help="TRAIN-neighbor count for conditional-copula sampling.",
    )
    parser.add_argument(
        "--library-policy-seed",
        type=int,
        default=17,
        help="Random seed for the TRAIN-fit conditional-copula PCA.",
    )
    parser.add_argument(
        "--library-clip-quantile",
        type=float,
        default=0.001,
        help="Two-sided TRAIN log-library clipping quantile.",
    )
    parser.add_argument(
        "--library-knn-jobs",
        type=int,
        default=-1,
        help="Parallel jobs used by conditional-copula nearest-neighbor search.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .h5ad path.",
    )
    parser.add_argument(
        "--no-save-npy",
        action="store_true",
        help="Do not save the companion NPY arrays.",
    )
    parser.add_argument(
        "--list-labels",
        action="store_true",
        help="Print the labels encoded in this trained model and exit.",
    )
    return parser.parse_args()


def configure_runtime(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_generation_seeds(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> dict[str, int]:
    """Resolve four independently controllable deterministic random streams."""
    base = int(args.seed)

    def choose(argument: int | None, config_keys: Sequence[str], offset: int) -> int:
        if argument is not None:
            return int(argument)
        for key in config_keys:
            if key in config and config[key] is not None:
                return int(config[key])
        return base + offset

    seeds = {
        "prior": choose(
            args.prior_seed,
            ("final_generation_seed", "final_prior_seed", "prior_seed"),
            0,
        ),
        "latent_noise": choose(
            args.latent_noise_seed,
            ("final_latent_noise_seed", "latent_noise_seed"),
            1_000_003,
        ),
        "library": choose(
            args.library_seed,
            ("final_library_seed", "library_seed"),
            2_000_003,
        ),
        "decoder": choose(
            args.decoder_seed,
            ("final_decoder_seed", "decoder_seed"),
            3_000_003,
        ),
    }
    if len(set(seeds.values())) != len(seeds):
        raise ValueError(f"Generation seeds must be distinct; got {seeds}")
    return seeds


def validate_label_map(label_map: Mapping[str, Any]) -> list[str]:
    raw_names = label_map.get("id_to_label")
    if not isinstance(raw_names, list):
        raise TypeError("label_map.json field 'id_to_label' must be a JSON list")
    label_names = [str(value) for value in raw_names]
    if len(label_names) < 2:
        raise RuntimeError("Saved model does not contain at least two conditions")
    if len(set(label_names)) != len(label_names):
        raise RuntimeError("label_map.json contains duplicate label names")
    if any(not value.strip() for value in label_names):
        raise RuntimeError("label_map.json contains an empty label name")

    stored_inverse = label_map.get("label_to_id")
    expected_inverse = {name: index for index, name in enumerate(label_names)}
    if stored_inverse is not None:
        normalized_inverse = {
            str(name): int(class_id) for name, class_id in stored_inverse.items()
        }
        if normalized_inverse != expected_inverse:
            raise RuntimeError(
                "label_map.json label_to_id is not the exact inverse of id_to_label"
            )
    return label_names


def validate_nb_ae_artifacts(
    ae_dir: Path,
    model_genes: np.ndarray,
    config: Mapping[str, Any],
    train_labels: np.ndarray,
    label_names: Sequence[str],
    device: torch.device,
) -> tuple[AEBundle, np.ndarray]:
    required = (
        "metadata.json",
        "model.pt",
        "genes.npy",
        "train_latent.npy",
        "validation_latent.npy",
        "train_library_size.npy",
        "validation_library_size.npy",
    )
    missing = [name for name in required if not (ae_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"NB-AE artifact directory {ae_dir} is missing: {missing}"
        )
    stored_hashes = config.get("ae_artifact_sha256")
    if not isinstance(stored_hashes, dict):
        raise RuntimeError("config.json is missing the frozen AE artifact hashes")
    hash_mismatches = {
        name: {
            "stored": stored_hashes.get(name),
            "observed": file_sha256(ae_dir / name),
        }
        for name in required
        if stored_hashes.get(name) != file_sha256(ae_dir / name)
    }
    if hash_mismatches:
        raise RuntimeError(f"Frozen NB-AE artifact hash mismatch: {hash_mismatches}")

    # AENB saves pandas Index.to_numpy(), which can be object dtype.  This is a
    # locally produced, hash-verified artifact, so pickle must be allowed here.
    ae_genes = np.load(ae_dir / "genes.npy", allow_pickle=True).astype(str)
    if not np.array_equal(ae_genes, model_genes):
        raise RuntimeError(
            "Gene order mismatch between model_dir/genes.npy and NB-AE genes.npy"
        )

    bundle = load_ae_bundle(ae_dir, device=device)
    metadata = bundle.metadata
    n_genes = int(config["n_genes"])
    latent_dim = int(config["ae_latent_dim"])
    if int(metadata.get("n_genes", -1)) != n_genes:
        raise RuntimeError(
            f"NB-AE n_genes={metadata.get('n_genes')} but config n_genes={n_genes}"
        )
    if int(metadata.get("latent_dim", -1)) != latent_dim:
        raise RuntimeError(
            "NB-AE latent_dim="
            f"{metadata.get('latent_dim')} but config ae_latent_dim={latent_dim}"
        )
    if bundle.model.n_genes != n_genes or bundle.model.latent_dim != latent_dim:
        raise RuntimeError("Loaded NB-AE architecture does not match model config")
    if str(metadata.get("gene_order_sha256", "")) != sha256_lines(ae_genes):
        raise RuntimeError("NB-AE metadata gene_order_sha256 does not match genes.npy")
    likelihood = str(metadata.get("likelihood", ""))
    if "negativebinomial" not in likelihood.replace(" ", "").lower():
        raise RuntimeError(
            "AE metadata does not identify a Negative-Binomial likelihood: "
            f"{likelihood!r}"
        )
    if str(metadata.get("encoder_input", "")) != "log1p(raw_counts)":
        raise RuntimeError("NB-AE encoder_input metadata is not log1p(raw_counts)")
    if "library_size" not in str(metadata.get("decoder_mean", "")):
        raise RuntimeError("NB-AE decoder_mean metadata does not use library_size")

    train_latent = np.asarray(bundle.train_latent, dtype=np.float32)
    train_library = np.asarray(bundle.train_library_size, dtype=np.float32).reshape(-1)
    validation_latent = np.asarray(bundle.validation_latent, dtype=np.float32)
    validation_library = np.asarray(
        bundle.validation_library_size, dtype=np.float32
    ).reshape(-1)
    expected_train_n = int(metadata.get("train_n_obs", -1))
    expected_validation_n = int(metadata.get("validation_n_obs", -1))
    if train_latent.shape != (expected_train_n, latent_dim):
        raise RuntimeError(
            f"Invalid train_latent shape {train_latent.shape}; "
            f"expected {(expected_train_n, latent_dim)}"
        )
    if validation_latent.shape != (expected_validation_n, latent_dim):
        raise RuntimeError(
            f"Invalid validation_latent shape {validation_latent.shape}; "
            f"expected {(expected_validation_n, latent_dim)}"
        )
    if train_library.shape != (expected_train_n,):
        raise RuntimeError(
            f"Invalid train_library_size shape {train_library.shape}; "
            f"expected {(expected_train_n,)}"
        )
    if validation_library.shape != (expected_validation_n,):
        raise RuntimeError(
            f"Invalid validation_library_size shape {validation_library.shape}; "
            f"expected {(expected_validation_n,)}"
        )
    for name, values in (
        ("train_latent", train_latent),
        ("validation_latent", validation_latent),
        ("train_library_size", train_library),
        ("validation_library_size", validation_library),
    ):
        if not np.isfinite(values).all():
            raise FloatingPointError(f"NB-AE {name} contains NaN/Inf")
    if np.any(train_library <= 0) or np.any(validation_library <= 0):
        raise RuntimeError("NB-AE library-size arrays must be strictly positive")
    if (
        not np.allclose(train_library, np.rint(train_library), atol=1e-4, rtol=0.0)
        or not np.allclose(
            validation_library,
            np.rint(validation_library),
            atol=1e-4,
            rtol=0.0,
        )
    ):
        raise RuntimeError("NB-AE library sizes are not raw-count totals")
    theta = torch.exp(bundle.model.log_theta.detach()).cpu().numpy()
    if not np.isfinite(theta).all() or np.any(theta <= 0):
        raise FloatingPointError("NB-AE inverse-dispersion theta is invalid")

    train_labels = np.asarray(train_labels)
    if train_labels.ndim != 1 or len(train_labels) != expected_train_n:
        raise RuntimeError(
            f"train_labels.npy shape {train_labels.shape} does not match NB-AE "
            f"TRAIN cells ({expected_train_n},)"
        )
    if not np.issubdtype(train_labels.dtype, np.integer):
        if not np.all(np.equal(train_labels, np.floor(train_labels))):
            raise RuntimeError("train_labels.npy must contain integer class IDs")
    train_labels = train_labels.astype(np.int32)
    if np.any(train_labels < 0) or np.any(train_labels >= len(label_names)):
        raise RuntimeError("train_labels.npy contains an out-of-range class ID")
    observed = np.bincount(train_labels, minlength=len(label_names))
    if np.any(observed <= 0):
        absent = [label_names[i] for i in np.flatnonzero(observed <= 0)]
        raise RuntimeError(f"NB-AE TRAIN split has no cells for labels: {absent}")

    config_train_hash = config.get(
        "train_barcodes_sha256", config.get("train_obs_sha256")
    )
    ae_train_hash = metadata.get("train_obs_sha256")
    if config_train_hash is not None and str(config_train_hash) != str(ae_train_hash):
        raise RuntimeError(
            "Conditional model and NB-AE were not trained on the same ordered TRAIN split"
        )
    return bundle, train_labels


def sample_train_library_sizes(
    train_library_size: np.ndarray,
    train_labels: np.ndarray,
    requested_labels: np.ndarray,
    mode: str,
    seed: int,
    shrinkage_strength: float = 200.0,
) -> np.ndarray:
    """Sample TRAIN-only library sizes with the same model used in training."""
    train_library_size = np.asarray(train_library_size, dtype=np.float32).reshape(-1)
    train_labels = np.asarray(train_labels, dtype=np.int32).reshape(-1)
    requested_labels = np.asarray(requested_labels, dtype=np.int32).reshape(-1)
    if len(train_library_size) != len(train_labels):
        raise ValueError("TRAIN library sizes and labels have different lengths")
    if np.any(train_library_size <= 0) or not np.isfinite(train_library_size).all():
        raise ValueError("TRAIN library sizes must be finite and positive")
    if not np.isfinite(shrinkage_strength) or float(shrinkage_strength) < 0:
        raise ValueError("library shrinkage strength must be finite and non-negative")
    rng = np.random.default_rng(int(seed))
    sampled = np.empty(len(requested_labels), dtype=np.float32)
    if mode == "global-empirical":
        sampled[:] = rng.choice(
            train_library_size,
            size=len(requested_labels),
            replace=True,
        )
    elif mode == "label-empirical":
        for class_id in np.unique(requested_labels):
            target = np.flatnonzero(requested_labels == class_id)
            source = train_library_size[train_labels == class_id]
            if len(source) == 0:
                raise RuntimeError(
                    f"No TRAIN library sizes exist for requested class ID {class_id}"
                )
            sampled[target] = rng.choice(source, size=len(target), replace=True)
    elif mode == "label-lognormal-shrunk":
        log_libraries = np.log(train_library_size.astype(np.float64))
        global_var = float(np.var(log_libraries, ddof=1)) if len(log_libraries) > 1 else 0.0
        global_var = max(global_var, 1e-8)
        tau = float(shrinkage_strength)
        for class_id in np.unique(requested_labels):
            target = np.flatnonzero(requested_labels == class_id)
            local = log_libraries[train_labels == class_id]
            if len(local) == 0:
                raise RuntimeError(
                    f"No TRAIN library sizes exist for requested class ID {class_id}"
                )
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
            sampled[target] = np.exp(sampled_log).astype(np.float32)
    else:
        raise ValueError(f"Unsupported library mode: {mode!r}")
    if np.any(sampled <= 0) or not np.isfinite(sampled).all():
        raise FloatingPointError("Sampled library sizes are invalid")
    return sampled


def distance_weighted_knn_score(
    train_latent: np.ndarray,
    train_log_library: np.ndarray,
    generated_latent: np.ndarray,
    requested_components: int,
    requested_k: int,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate E[log library | latent] from TRAIN artifacts only.

    This is the embedded frozen-v1 implementation used by conditional
    generation.
    """
    train_latent = np.asarray(train_latent, dtype=np.float32)
    generated_latent = np.asarray(generated_latent, dtype=np.float32)
    train_log_library = np.asarray(train_log_library, dtype=np.float64).reshape(-1)
    if train_latent.ndim != 2 or generated_latent.ndim != 2:
        raise RuntimeError("TRAIN/generated latent arrays must be two-dimensional")
    if len(train_latent) != len(train_log_library):
        raise RuntimeError("TRAIN latent/library row counts differ")
    if train_latent.shape[1] != generated_latent.shape[1]:
        raise RuntimeError("TRAIN/generated latent widths differ")
    if len(train_latent) < 2:
        raise RuntimeError("at least two TRAIN cells are required")
    if len(generated_latent) < 1:
        raise RuntimeError("at least one generated cell is required")
    if requested_components < 1 or requested_k < 1:
        raise ValueError("conditional-copula PCA components and KNN k must be positive")
    if (
        not np.isfinite(train_latent).all()
        or not np.isfinite(generated_latent).all()
        or not np.isfinite(train_log_library).all()
    ):
        raise RuntimeError("conditional-copula inputs contain NaN/Inf")

    mean = train_latent.mean(axis=0, dtype=np.float64)
    sd = train_latent.std(axis=0, dtype=np.float64)
    sd = np.maximum(sd, 1e-4)
    train_scaled = ((train_latent - mean[None, :]) / sd[None, :]).astype(
        np.float32
    )
    generated_scaled = (
        (generated_latent - mean[None, :]) / sd[None, :]
    ).astype(np.float32)

    n_components = min(
        int(requested_components),
        train_scaled.shape[1],
        train_scaled.shape[0] - 1,
    )
    if n_components < 1:
        raise RuntimeError("conditional PCA has no valid components")
    min_shape = min(train_scaled.shape)
    solver = "randomized" if n_components < min_shape else "full"
    pca = PCA(
        n_components=n_components,
        svd_solver=solver,
        random_state=int(seed),
    )
    train_pca = pca.fit_transform(train_scaled).astype(np.float32)
    generated_pca = pca.transform(generated_scaled).astype(np.float32)

    k = min(int(requested_k), len(train_pca))
    if k < 1:
        raise RuntimeError("conditional KNN has no TRAIN neighbors")
    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=n_jobs)
    neighbors.fit(train_pca)
    distances, indices = neighbors.kneighbors(generated_pca, return_distance=True)
    local_library = train_log_library[indices]
    weights = 1.0 / np.maximum(distances.astype(np.float64), 1e-8)
    score = np.sum(weights * local_library, axis=1) / np.sum(weights, axis=1)

    exact_rows = np.any(distances <= 1e-8, axis=1)
    for row in np.flatnonzero(exact_rows):
        exact = distances[row] <= 1e-8
        score[row] = float(local_library[row, exact].mean())
    if not np.isfinite(score).all():
        raise RuntimeError("conditional KNN produced non-finite scores")

    audit = {
        "pca_components": int(n_components),
        "pca_explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "knn_k": int(k),
        "knn_distance_mean": float(distances.mean()),
        "knn_distance_q95": float(np.quantile(distances, 0.95)),
        "conditional_score_mean": float(score.mean()),
        "conditional_score_sd": float(score.std()),
    }
    return score.astype(np.float64), audit


def rank_couple(pool: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Assign one empirical pool to score ranks, preserving the pool exactly."""
    pool = np.asarray(pool, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if len(pool) != len(score):
        raise RuntimeError("rank-copula pool/score lengths differ")
    order = np.argsort(score, kind="mergesort")
    result = np.empty_like(pool)
    result[order] = np.sort(pool, kind="mergesort")
    return result


def sample_conditional_copula_library_sizes(
    train_latent: np.ndarray,
    train_library_size: np.ndarray,
    generated_latent: np.ndarray,
    library_seed: int,
    policy_seed: int = 17,
    pca_components: int = 32,
    knn_k: int = 64,
    clip_quantile: float = 0.001,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the frozen label-free conditional-copula library policy.

    Labels are deliberately not accepted: PCA, KNN and the empirical marginal
    all use the global AE TRAIN population, exactly as in the unconditioned
    six-dataset policy.
    """
    train_library_size = np.asarray(train_library_size, dtype=np.float64).reshape(-1)
    if np.any(train_library_size <= 0) or not np.isfinite(train_library_size).all():
        raise ValueError("TRAIN library sizes must be finite and positive")
    if not 0.0 < float(clip_quantile) < 0.5:
        raise ValueError("library clip quantile must be in (0, 0.5)")

    train_log_library = np.log(train_library_size)
    score, knn_audit = distance_weighted_knn_score(
        train_latent=train_latent,
        train_log_library=train_log_library,
        generated_latent=generated_latent,
        requested_components=int(pca_components),
        requested_k=int(knn_k),
        seed=int(policy_seed),
        n_jobs=int(n_jobs),
    )

    # Keep the empirical-pool RNG independent from decoder sampling.
    empirical_pool_seed = int(library_seed) + 100_000
    empirical_rng = np.random.default_rng(empirical_pool_seed)
    empirical_pool = empirical_rng.choice(
        train_log_library,
        size=len(generated_latent),
        replace=True,
    )
    low = float(np.quantile(train_log_library, clip_quantile))
    high = float(np.quantile(train_log_library, 1.0 - clip_quantile))
    coupled_log_library = np.clip(rank_couple(empirical_pool, score), low, high)
    sampled = np.exp(coupled_log_library).astype(np.float32)
    if np.any(sampled <= 0) or not np.isfinite(sampled).all():
        raise FloatingPointError("Conditional-copula library sizes are invalid")

    audit: dict[str, Any] = {
        "schema_version": 1,
        "policy": "conditional-copula",
        "implementation": "generate.py:sample_conditional_copula_library_sizes",
        "policy_version": "latent_conditioned_empirical_copula_v1",
        "policy_fit_scope": "global AE TRAIN latent and TRAIN library artifacts only",
        "labels_used_by_policy": False,
        "generated_latent_space": "inverse-standardized frozen NB-AE latent",
        "train_latent_standardization": "per-dimension TRAIN mean/sd with sd floor 1e-4",
        "pca_solver_rule": "randomized if n_components < min(TRAIN shape), else full",
        "pca_components_requested": int(pca_components),
        "knn_k_requested": int(knn_k),
        "knn_metric": "euclidean",
        "knn_weighting": "inverse distance with 1e-8 floor; exact matches only",
        "policy_seed": int(policy_seed),
        "library_seed": int(library_seed),
        "empirical_pool_seed": empirical_pool_seed,
        "empirical_pool_sampling": "TRAIN log-library bootstrap with replacement",
        "rank_coupling": "stable mergesort of score and empirical pool",
        "library_clip_quantile": float(clip_quantile),
        "library_log_clip_low": low,
        "library_log_clip_high": high,
        "train_n_obs": int(len(train_library_size)),
        "train_latent_dim": int(np.asarray(train_latent).shape[1]),
        "generated_n_obs": int(len(generated_latent)),
        "train_log_library_mean": float(train_log_library.mean()),
        "train_log_library_sd": float(train_log_library.std()),
        "generated_log_library_mean": float(coupled_log_library.mean()),
        "generated_log_library_sd": float(coupled_log_library.std()),
        **knn_audit,
    }
    return sampled, score, audit


def natural_label_order(values: Sequence[str]) -> list[str]:
    def key(value: str) -> tuple[int, Any]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted(set(map(str, values)), key=key)


def resolve_selected_ids(
    requested: Sequence[str],
    label_names: Sequence[str],
) -> list[int]:
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(range(len(label_names)))

    name_to_id = {str(name): index for index, name in enumerate(label_names)}
    selected: list[int] = []
    for token_raw in requested:
        token = str(token_raw)
        if token in name_to_id:
            class_id = name_to_id[token]
        else:
            try:
                class_id = int(token)
            except ValueError as error:
                raise ValueError(
                    f"Unknown label {token!r}. Available: {list(label_names)}"
                ) from error
            if not 0 <= class_id < len(label_names):
                raise ValueError(
                    f"Label ID {class_id} is outside [0, {len(label_names) - 1}]"
                )
        if class_id not in selected:
            selected.append(class_id)

    if not selected:
        raise ValueError("No labels were selected")
    return selected


def apply_count_cap(count: int, maximum: int | None) -> int:
    count = int(count)
    if maximum is not None:
        count = min(count, int(maximum))
    if count <= 0:
        raise ValueError(f"All requested generation counts must be positive; got {count}")
    return count


def counts_from_json(
    path: Path,
    selected_ids: Sequence[int],
    label_names: Sequence[str],
    maximum: int | None,
) -> dict[int, int]:
    payload = load_json(path.expanduser().resolve())
    result: dict[int, int] = {}
    for class_id in selected_ids:
        label_name = str(label_names[class_id])
        if label_name in payload:
            value = payload[label_name]
        elif str(class_id) in payload:
            value = payload[str(class_id)]
        else:
            raise KeyError(
                f"counts JSON has no entry for label {label_name!r} / ID {class_id}"
            )
        result[class_id] = apply_count_cap(int(value), maximum)
    return result


def counts_from_h5ad(
    path: Path,
    label_key: str,
    selected_ids: Sequence[int],
    label_names: Sequence[str],
    maximum: int | None,
) -> dict[int, int]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    reference = ad.read_h5ad(path, backed="r")
    try:
        if label_key not in reference.obs.columns:
            raise KeyError(
                f"{path} has no obs[{label_key!r}]. "
                f"Available columns: {reference.obs.columns.tolist()}"
            )
        observed_counts = reference.obs[label_key].astype(str).value_counts()
    finally:
        if getattr(reference, "isbacked", False):
            reference.file.close()

    result: dict[int, int] = {}
    for class_id in selected_ids:
        label_name = str(label_names[class_id])
        count = int(observed_counts.get(label_name, 0))
        if count <= 0:
            raise ValueError(
                f"Reference H5AD contains no cells for trained label {label_name!r}"
            )
        result[class_id] = apply_count_cap(count, maximum)
    return result


def determine_counts(
    args: argparse.Namespace,
    selected_ids: Sequence[int],
    label_names: Sequence[str],
    config: Mapping[str, Any],
) -> dict[int, int]:
    if args.counts_json is not None:
        return counts_from_json(
            args.counts_json,
            selected_ids,
            label_names,
            args.max_per_label,
        )

    if args.counts_from_h5ad is not None:
        key = str(args.reference_label_key or config.get("label_key", "condition"))
        return counts_from_h5ad(
            args.counts_from_h5ad,
            key,
            selected_ids,
            label_names,
            args.max_per_label,
        )

    n_per_label = 1000 if args.n_per_label is None else int(args.n_per_label)
    if args.max_per_label is not None:
        n_per_label = min(n_per_label, int(args.max_per_label))
    if n_per_label <= 0:
        raise ValueError("--n-per-label must be positive")
    return {class_id: n_per_label for class_id in selected_ids}


def make_conditional_prior(
    counts_by_id: Mapping[int, int],
    z_dim: int,
    seed: int,
    independent_z: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ordered_ids = list(counts_by_id)
    max_count = max(int(counts_by_id[class_id]) for class_id in ordered_ids)

    shared_z: np.ndarray | None = None
    if not independent_z:
        shared_z = rng.standard_normal((max_count, z_dim)).astype(np.float32)

    z_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for class_id in ordered_ids:
        count = int(counts_by_id[class_id])
        if independent_z:
            z_part = rng.standard_normal((count, z_dim)).astype(np.float32)
        else:
            assert shared_z is not None
            z_part = shared_z[:count].copy()
        z_parts.append(z_part)
        label_parts.append(np.full(count, class_id, dtype=np.int32))

    return np.concatenate(z_parts, axis=0), np.concatenate(label_parts, axis=0)


def generate_standardized_latent(
    generator: tf.keras.Model,
    prior_z: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    sample_observation_noise: bool,
    latent_noise_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_parts: list[np.ndarray] = []
    variance_parts: list[np.ndarray] = []

    for start in range(0, len(prior_z), batch_size):
        stop = min(start + batch_size, len(prior_z))
        mean, variance = generator(
            tf.convert_to_tensor(prior_z[start:stop], dtype=tf.float32),
            tf.convert_to_tensor(labels[start:stop], dtype=tf.int32),
            training=False,
        )
        mean_parts.append(mean.numpy().astype(np.float32))
        variance_parts.append(variance.numpy().astype(np.float32))

    standardized_mean = np.concatenate(mean_parts, axis=0)
    standardized_variance = np.concatenate(variance_parts, axis=0)

    if sample_observation_noise:
        rng = np.random.default_rng(int(latent_noise_seed))
        epsilon = rng.standard_normal(standardized_mean.shape).astype(np.float32)
        standardized_latent = (
            standardized_mean
            + np.sqrt(np.maximum(standardized_variance, 0.0)) * epsilon
        ).astype(np.float32)
    else:
        standardized_latent = standardized_mean.copy()

    if not np.isfinite(standardized_latent).all():
        raise FloatingPointError("Generated standardized latent contains NaN/Inf")
    if not np.isfinite(standardized_variance).all():
        raise FloatingPointError("Generated variance contains NaN/Inf")
    if np.any(standardized_variance <= 0):
        raise FloatingPointError("Generated variance contains non-positive values")

    return standardized_latent, standardized_mean, standardized_variance


def choose_torch_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--decode-device cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_output_path(
    requested: Path | None,
    model_dir: Path,
    sampled_latent: bool,
    decoder_mode: str,
) -> Path:
    if requested is None:
        latent_suffix = "latent_sampled" if sampled_latent else "latent_mean"
        output = model_dir / f"generated_for_umap_{latent_suffix}_nb_{decoder_mode}.h5ad"
    else:
        output = requested.expanduser()
        if output.suffix.lower() != ".h5ad":
            output = output.with_suffix(".h5ad")
    return output.resolve()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.decode_batch_size <= 0:
        raise ValueError("--decode-batch-size must be positive")
    if args.max_per_label is not None and args.max_per_label <= 0:
        raise ValueError("--max-per-label must be positive")

    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise NotADirectoryError(model_dir)

    config = load_json(model_dir / "config.json")
    label_map = load_json(model_dir / "label_map.json")
    label_names = validate_label_map(label_map)

    print("\nLabels encoded in this trained checkpoint:")
    train_counts = label_map.get("train_counts", {})
    validation_counts = label_map.get("validation_counts", {})
    for class_id, label_name in enumerate(label_names):
        print(
            f"  {class_id}: {label_name} "
            f"(train={train_counts.get(label_name, 'NA')}, "
            f"validation={validation_counts.get(label_name, 'NA')})"
        )
    print(f"Number of trained conditions: {len(label_names)}")

    if args.list_labels:
        return

    selected_ids = resolve_selected_ids(args.labels, label_names)
    counts_by_id = determine_counts(args, selected_ids, label_names, config)

    print("\nRequested generation counts:")
    for class_id, count in counts_by_id.items():
        print(f"  {class_id}: {label_names[class_id]} -> {count}")

    required_model_fields = (
        "n_genes",
        "ae_latent_dim",
        "bgm_z_dim",
        "label_embed_dim",
        "generator_variance_eps",
        "label_key",
        "num_classes",
        "ae_observation_model",
        "ae_latent_l2_normalized",
        "train_barcodes_sha256",
        "gene_order_sha256",
        "ae_artifact_sha256",
    )
    missing_fields = [key for key in required_model_fields if key not in config]
    if missing_fields:
        raise KeyError(f"config.json is missing fields: {missing_fields}")
    if not str(config["label_key"]).strip():
        raise RuntimeError("config label_key is empty")
    if int(config["num_classes"]) != len(label_names):
        raise RuntimeError("config num_classes does not match label_map.json")
    if str(config["ae_observation_model"]).lower() != "negativebinomial":
        raise RuntimeError("config ae_observation_model is not NegativeBinomial")
    if bool(config["ae_latent_l2_normalized"]):
        raise RuntimeError("NB-AE latent must not be L2-normalized before decoding")

    decoder_mode = str(args.decoder_mode or config.get("decoder_mode", "sample"))
    library_mode = str(
        args.library_mode or config.get("library_mode", "global-empirical")
    )
    if decoder_mode not in {"sample", "mean"}:
        raise ValueError(f"Invalid decoder mode in config/CLI: {decoder_mode!r}")
    if library_mode not in {
        "global-empirical",
        "label-empirical",
        "label-lognormal-shrunk",
        "conditional-copula",
    }:
        raise ValueError(f"Invalid library mode in config/CLI: {library_mode!r}")
    library_shrinkage_strength = float(
        args.library_shrinkage_strength
        if args.library_shrinkage_strength is not None
        else config.get("library_shrinkage_strength", 200.0)
    )
    if not np.isfinite(library_shrinkage_strength) or library_shrinkage_strength < 0:
        raise ValueError("Invalid library shrinkage strength")
    if library_mode == "conditional-copula":
        if args.library_copula_pca_components < 1:
            raise ValueError("--library-copula-pca-components must be positive")
        if args.library_copula_knn_k < 1:
            raise ValueError("--library-copula-knn-k must be positive")
        if not 0.0 < args.library_clip_quantile < 0.5:
            raise ValueError("--library-clip-quantile must be in (0, 0.5)")
        if args.library_knn_jobs == 0:
            raise ValueError("--library-knn-jobs cannot be zero")
    target_sum = float(config.get("target_sum", 1e4))
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError(f"Invalid target_sum in config.json: {target_sum}")
    seeds = resolve_generation_seeds(args, config)
    configure_runtime(seeds["prior"])

    generator = build_generator(config, num_classes=len(label_names))
    generator_weights = model_dir / "generator.weights.h5"
    if not generator_weights.exists():
        raise FileNotFoundError(generator_weights)
    generator.load_weights(str(generator_weights))
    print(f"\nLoaded conditional generator: {generator_weights}")
    print(f"Network variant: {config.get('network_variant', 'resmlp')}")
    print(f"Label injection: {config.get('label_injection', 'early_concat')}")

    genes_path = model_dir / "genes.npy"
    scale_path = model_dir / "latent_scale.npz"
    train_labels_path = model_dir / "train_labels.npy"
    for path in (genes_path, scale_path, train_labels_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    genes = np.load(genes_path, allow_pickle=False).astype(str)
    n_genes = int(config["n_genes"])
    ae_latent_dim = int(config["ae_latent_dim"])
    if len(genes) != n_genes:
        raise RuntimeError(
            f"genes.npy contains {len(genes)} genes, but config n_genes={n_genes}"
        )
    if len(set(genes.tolist())) != len(genes):
        raise RuntimeError("genes.npy contains duplicate gene names")
    if str(config.get("gene_order_sha256", "")) != sha256_lines(genes):
        raise RuntimeError("config gene_order_sha256 does not match genes.npy")

    with np.load(scale_path, allow_pickle=False) as scale:
        latent_mean = np.asarray(scale["mean"], dtype=np.float32)
        latent_sd = np.asarray(scale["sd"], dtype=np.float32)
    if latent_mean.shape != (ae_latent_dim,):
        raise RuntimeError(f"Unexpected latent mean shape: {latent_mean.shape}")
    if latent_sd.shape != (ae_latent_dim,):
        raise RuntimeError(f"Unexpected latent sd shape: {latent_sd.shape}")
    if not np.isfinite(latent_mean).all():
        raise RuntimeError("latent_scale.npz contains an invalid mean")
    if np.any(latent_sd <= 0) or not np.isfinite(latent_sd).all():
        raise RuntimeError("latent_scale.npz contains invalid standard deviations")

    ae_dir = (
        args.ae_artifact_dir.expanduser().resolve()
        if args.ae_artifact_dir is not None
        else Path(
            str(
                config.get(
                    "ae_artifact_dir_resolved",
                    config.get("ae_artifact_dir", ""),
                )
            )
        ).expanduser().resolve()
    )
    if str(ae_dir) == "." or not ae_dir.is_dir():
        raise FileNotFoundError(
            "Could not resolve the AE artifact directory. Pass --ae-artifact-dir."
        )

    decode_device = choose_torch_device(args.decode_device)
    train_labels = np.load(train_labels_path, allow_pickle=False)
    ae, train_labels = validate_nb_ae_artifacts(
        ae_dir=ae_dir,
        model_genes=genes,
        config=config,
        train_labels=train_labels,
        label_names=label_names,
        device=decode_device,
    )
    observed_train_counts = np.bincount(
        train_labels, minlength=len(label_names)
    )
    stored_train_counts = label_map.get("train_counts")
    if not isinstance(stored_train_counts, dict):
        raise TypeError("label_map.json train_counts must be a JSON object")
    expected_train_counts = {
        label_names[index]: int(observed_train_counts[index])
        for index in range(len(label_names))
    }
    normalized_train_counts = {
        str(name): int(count) for name, count in stored_train_counts.items()
    }
    if normalized_train_counts != expected_train_counts:
        raise RuntimeError(
            "label_map.json train_counts does not match train_labels.npy"
        )

    print(f"Loaded frozen Negative-Binomial AE: {ae_dir / 'model.pt'}")
    print(f"AE decode device: {decode_device}")
    print(f"NB decoder mode: {decoder_mode}")
    print(f"TRAIN-only library mode: {library_mode}")
    if library_mode == "label-lognormal-shrunk":
        print(f"Library variance shrinkage strength: {library_shrinkage_strength}")
    elif library_mode == "conditional-copula":
        print(
            "Conditional-copula settings: "
            f"PCA={args.library_copula_pca_components}, "
            f"KNN={args.library_copula_knn_k}, "
            f"policy_seed={args.library_policy_seed}, "
            f"clip_q={args.library_clip_quantile}"
        )
    print(f"Generation seeds: {seeds}")

    prior_z, condition_ids = make_conditional_prior(
        counts_by_id=counts_by_id,
        z_dim=int(config["bgm_z_dim"]),
        seed=seeds["prior"],
        independent_z=args.independent_z,
    )
    (
        standardized_latent,
        standardized_mean,
        standardized_variance,
    ) = generate_standardized_latent(
        generator=generator,
        prior_z=prior_z,
        labels=condition_ids,
        batch_size=args.batch_size,
        sample_observation_noise=args.sample_observation_noise,
        latent_noise_seed=seeds["latent_noise"],
    )

    ae_latent = (
        standardized_latent * latent_sd[None, :] + latent_mean[None, :]
    ).astype(np.float32)
    if not np.isfinite(ae_latent).all():
        raise FloatingPointError("Inverse-standardized AE latent contains NaN/Inf")

    library_policy_audit: dict[str, Any] | None = None
    conditional_library_score: np.ndarray | None = None
    if library_mode == "conditional-copula":
        (
            generated_library_size,
            conditional_library_score,
            library_policy_audit,
        ) = sample_conditional_copula_library_sizes(
            train_latent=ae.train_latent,
            train_library_size=ae.train_library_size,
            generated_latent=ae_latent,
            library_seed=seeds["library"],
            policy_seed=args.library_policy_seed,
            pca_components=args.library_copula_pca_components,
            knn_k=args.library_copula_knn_k,
            clip_quantile=args.library_clip_quantile,
            n_jobs=args.library_knn_jobs,
        )
    else:
        generated_library_size = sample_train_library_sizes(
            train_library_size=ae.train_library_size,
            train_labels=train_labels,
            requested_labels=condition_ids,
            mode=library_mode,
            seed=seeds["library"],
            shrinkage_strength=library_shrinkage_strength,
        )
    if decoder_mode == "sample":
        decoded_counts_or_mean = decode_sample_counts(
            ae.model,
            ae_latent,
            generated_library_size,
            batch_size=args.decode_batch_size,
            device=decode_device,
            seed=seeds["decoder"],
        )
    else:
        decoded_counts_or_mean = decode_mu_matrix(
            ae.model,
            ae_latent,
            generated_library_size,
            batch_size=args.decode_batch_size,
            device=decode_device,
        )
    decoded_counts_or_mean = np.asarray(decoded_counts_or_mean, dtype=np.float32)
    if decoded_counts_or_mean.shape != (len(condition_ids), n_genes):
        raise RuntimeError(
            f"Unexpected NB decoded shape {decoded_counts_or_mean.shape}; "
            f"expected {(len(condition_ids), n_genes)}"
        )
    if (
        not np.isfinite(decoded_counts_or_mean).all()
        or np.any(decoded_counts_or_mean < 0)
        or np.any(decoded_counts_or_mean.sum(axis=1) <= 0)
    ):
        raise FloatingPointError("NB decoded matrix contains invalid cells")
    decoded_total = decoded_counts_or_mean.sum(axis=1).astype(np.float32)
    expression = normalize_total_log1p_dense(
        decoded_counts_or_mean,
        target_sum=target_sum,
    )
    if expression.shape != (len(condition_ids), n_genes):
        raise RuntimeError(
            f"Unexpected decoded expression shape {expression.shape}; "
            f"expected {(len(condition_ids), n_genes)}"
        )
    if not np.isfinite(expression).all() or np.any(expression < 0):
        raise FloatingPointError("Decoded expression contains NaN/Inf/negative values")

    output = make_output_path(
        requested=args.output,
        model_dir=model_dir,
        sampled_latent=args.sample_observation_noise,
        decoder_mode=decoder_mode,
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    label_array = np.asarray(
        [label_names[int(class_id)] for class_id in condition_ids],
        dtype="U",
    )
    label_key = str(config["label_key"])

    generated = ad.AnnData(X=expression)
    generated.var_names = pd.Index(genes)
    generated.obs_names = pd.Index(
        [
            f"generated_c{int(condition_ids[index]):02d}_{index:07d}"
            for index in range(len(label_array))
        ]
    )
    generated.obs[label_key] = pd.Categorical(
        label_array,
        categories=label_names,
    )
    generated.obs["requested_condition"] = pd.Categorical(
        label_array,
        categories=label_names,
    )
    generated.obs["condition_id"] = condition_ids.astype(np.int32)
    generated.obs["decoder_library_size"] = generated_library_size
    if conditional_library_score is not None:
        generated.obs["conditional_library_log_score"] = conditional_library_score
    total_key = (
        "sampled_count_total" if decoder_mode == "sample" else "decoded_mean_total"
    )
    generated.obs[total_key] = decoded_total
    generated.obs["source"] = pd.Categorical(
        np.repeat("generated", len(condition_ids)),
        categories=["generated"],
    )

    if decoder_mode == "sample":
        generated.layers["counts"] = sparse.csr_matrix(decoded_counts_or_mean)

    generated.obsm["X_ae_latent"] = ae_latent
    generated.obsm["X_ae_latent_standardized"] = standardized_latent
    generated.obsm["X_ae_latent_standardized_mean"] = standardized_mean
    generated.obsm["bgm_diagonal_variance"] = standardized_variance

    generation_mode = (
        "sampled_from_diagonal_variance"
        if args.sample_observation_noise
        else "conditional_mean"
    )
    generated.uns["conditional_bgm"] = {
        "model_dir": str(model_dir),
        "generator_checkpoint": str(generator_weights),
        "ae_artifact_dir": str(ae_dir),
        "generation_mode": generation_mode,
        "sample_observation_noise": bool(args.sample_observation_noise),
        "matched_z_across_labels": not bool(args.independent_z),
        "seeds": dict(seeds),
        "label_key": label_key,
        "label_source": str(
            config.get("label_vocabulary_source", "training_split_sorted_unique")
        ),
        "label_names": list(label_names),
        "selected_labels": [label_names[class_id] for class_id in selected_ids],
        "counts_by_label": {
            label_names[class_id]: int(counts_by_id[class_id])
            for class_id in selected_ids
        },
        "network_variant": str(config.get("network_variant", "resmlp")),
        "label_injection": str(config.get("label_injection", "early_concat")),
        "label_embed_dim": int(config["label_embed_dim"]),
        "bgm_z_dim": int(config["bgm_z_dim"]),
        "ae_latent_dim": ae_latent_dim,
        "n_genes": n_genes,
        "ae_observation_model": "NegativeBinomial",
        "ae_latent_l2_normalized": False,
        "decoder_mode": decoder_mode,
        "library_mode": library_mode,
        "library_size_source": "AE training split only",
        "expression_space": "NB decode, then normalize_total(target_sum)+log1p once",
        "target_sum": target_sum,
        "raw_umi_counts": False,
        "counts_layer": "counts" if decoder_mode == "sample" else "absent_mean_mode",
        "decoded_total_obs_key": total_key,
    }
    if library_policy_audit is not None:
        generated.uns["conditional_bgm"]["library_policy_audit"] = (
            library_policy_audit
        )
        generated.uns["conditional_bgm"]["library_policy_summary"] = (
            "TRAIN-standardized inverse-scaled AE latent -> TRAIN-fit PCA -> "
            "distance-weighted KNN log-library score -> stable rank coupling "
            "to a clipped global TRAIN empirical log-library pool"
        )

    generated.write_h5ad(output, compression="gzip")

    stem = output.with_suffix("")
    if not args.no_save_npy:
        np.save(str(stem) + "_generated.npy", expression)
        np.save(str(stem) + "_genes.npy", genes)
        np.save(str(stem) + "_labels.npy", label_array)
        np.save(str(stem) + "_condition_ids.npy", condition_ids)
        np.save(str(stem) + "_library_size.npy", generated_library_size)
        np.save(str(stem) + "_decoded_total.npy", decoded_total)
        np.save(str(stem) + "_prior_z.npy", prior_z)
        np.save(
            str(stem) + "_standardized_latent.npy",
            standardized_latent,
        )
        np.save(
            str(stem) + "_standardized_mean.npy",
            standardized_mean,
        )
        np.save(
            str(stem) + "_standardized_variance.npy",
            standardized_variance,
        )
        np.save(
            str(stem) + "_ae_latent.npy",
            ae_latent,
        )
        with open(str(stem) + "_metadata.json", "w") as handle:
            json.dump(generated.uns["conditional_bgm"], handle, indent=2)

    print("\nGeneration completed.")
    print(f"Saved H5AD: {output}")
    print(f"Expression shape: {expression.shape}")
    print(f"Generation mode: {generation_mode}")
    print(f"NB decoder mode: {decoder_mode}")
    print(f"TRAIN-only library mode: {library_mode}")
    if library_mode == "label-lognormal-shrunk":
        print(f"Library variance shrinkage strength: {library_shrinkage_strength}")
    elif library_mode == "conditional-copula":
        print(json.dumps(library_policy_audit, indent=2))
    print(f"Label column: obs[{label_key!r}]")
    print("Generated counts:")
    print(generated.obs[label_key].value_counts().sort_index())
    print("Expression summary:")
    print(f"  min={float(expression.min()):.6g}")
    print(f"  mean={float(expression.mean()):.6g}")
    print(f"  max={float(expression.max()):.6g}")
    print(f"  zero_fraction={float(np.mean(expression <= 0)):.6g}")
    print("NOTE: X was normalized/log1p-transformed exactly once after NB decoding.")
    if decoder_mode == "sample":
        print("Raw NB sampled counts are stored in layers['counts'].")


if __name__ == "__main__":
    main()

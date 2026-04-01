"""Deterministic hyperdimensional tabular encoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import PreparedBatch, TabularPreprocessor
from .utils import stable_seed


class DeterministicHypervectorFactory:
    """Create reproducible bipolar hypervectors independent of encounter order."""

    def __init__(self, dim: int, seed: int) -> None:
        self.dim = dim
        self.seed = seed

    def make(self, namespace: str, token: str) -> np.ndarray:
        rng = np.random.default_rng(stable_seed(self.seed, namespace, token))
        bits = rng.integers(0, 2, size=self.dim, endpoint=False, dtype=np.int8)
        return np.where(bits == 0, -1, 1).astype(np.int8)


@dataclass(slots=True)
class EncoderFootprint:
    feature_hv_bytes: int
    lookup_bytes: int
    cache_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.feature_hv_bytes + self.lookup_bytes + self.cache_bytes


class TabularHDCEncoder:
    """Mixed-type HDC encoder using feature, level, and value hypervectors."""

    def __init__(
        self,
        preprocessor: TabularPreprocessor,
        *,
        dim: int,
        bins: int,
        seed: int,
    ) -> None:
        self.preprocessor = preprocessor
        self.dim = dim
        self.bins = bins
        self.seed = seed
        self.factory = DeterministicHypervectorFactory(dim, seed)
        self.regen_rng = np.random.default_rng(seed + 17)
        self.query_scale = float(
            max(len(preprocessor.numeric_features) + len(preprocessor.categorical_features), 1)
        )

        self.numeric_feature_hv = np.vstack(
            [self.factory.make("num_feature", feature) for feature in preprocessor.numeric_features]
        ).astype(np.int8) if preprocessor.numeric_features else np.zeros((0, dim), dtype=np.int8)
        self.categorical_feature_hv = np.vstack(
            [self.factory.make("cat_feature", feature) for feature in preprocessor.categorical_features]
        ).astype(np.int8) if preprocessor.categorical_features else np.zeros((0, dim), dtype=np.int8)
        self.level_hv = np.vstack(
            [self.factory.make("level", str(level)) for level in range(bins)]
        ).astype(np.int8)

        self.numeric_lookup = self._build_numeric_lookup()
        self.category_value_hv: dict[str, dict[str, np.ndarray]] = {}
        self.category_bound_hv: dict[str, dict[str, np.ndarray]] = {}
        for feature in preprocessor.categorical_features:
            self.category_value_hv[feature] = {}
            self.category_bound_hv[feature] = {}
            vocab = preprocessor.category_vocab.get(feature, {"__UNK__": 0})
            for value in vocab:
                self._get_category_bound_hv(feature, value)

    def _build_numeric_lookup(self) -> np.ndarray:
        if self.numeric_feature_hv.size == 0:
            return np.zeros((0, self.bins, self.dim), dtype=np.int8)
        return (self.numeric_feature_hv[:, None, :] * self.level_hv[None, :, :]).astype(np.int8)

    def _numeric_bins(self, numeric: np.ndarray) -> np.ndarray:
        normalized = (numeric + self.preprocessor.z_clip) / (2 * self.preprocessor.z_clip)
        clipped = np.clip(normalized, 0.0, 0.999999)
        return (clipped * self.bins).astype(np.int32)

    def _get_category_bound_hv(self, feature: str, value: str) -> np.ndarray:
        value_key = value if value else "__UNK__"
        if value_key not in self.category_value_hv[feature]:
            value_hv = self.factory.make(f"cat_value::{feature}", value_key)
            feature_idx = self.preprocessor.categorical_features.index(feature)
            bound = (self.categorical_feature_hv[feature_idx] * value_hv).astype(np.int8)
            self.category_value_hv[feature][value_key] = value_hv
            self.category_bound_hv[feature][value_key] = bound
        return self.category_bound_hv[feature][value_key]

    def encode_batch(self, batch: PreparedBatch) -> np.ndarray:
        encoded = np.zeros((batch.size, self.dim), dtype=np.float32)
        if batch.size == 0:
            return encoded

        if self.numeric_lookup.size:
            bin_indices = self._numeric_bins(batch.numeric)
            for feature_idx in range(bin_indices.shape[1]):
                encoded += self.numeric_lookup[feature_idx][bin_indices[:, feature_idx]].astype(np.float32)

        if batch.categorical.size:
            for col_idx, feature in enumerate(self.preprocessor.categorical_features):
                values = batch.categorical[:, col_idx]
                unique_values, inverse = np.unique(values, return_inverse=True)
                lookup = np.vstack([self._get_category_bound_hv(feature, str(value)) for value in unique_values])
                encoded += lookup[inverse].astype(np.float32)

        return encoded

    def regenerate_dimensions(self, dimension_indices: np.ndarray) -> None:
        if dimension_indices.size == 0:
            return
        for dim_index in dimension_indices:
            if self.numeric_feature_hv.size:
                self.numeric_feature_hv[:, dim_index] = self.regen_rng.choice(
                    (-1, 1), size=self.numeric_feature_hv.shape[0]
                )
            if self.categorical_feature_hv.size:
                self.categorical_feature_hv[:, dim_index] = self.regen_rng.choice(
                    (-1, 1), size=self.categorical_feature_hv.shape[0]
                )
            self.level_hv[:, dim_index] = self.regen_rng.choice((-1, 1), size=self.level_hv.shape[0])

        self.numeric_lookup = self._build_numeric_lookup()
        for feature, value_map in self.category_value_hv.items():
            feature_idx = self.preprocessor.categorical_features.index(feature)
            for value, value_hv in value_map.items():
                value_hv[dimension_indices] = self.regen_rng.choice((-1, 1), size=dimension_indices.shape[0])
                self.category_bound_hv[feature][value] = (
                    self.categorical_feature_hv[feature_idx] * value_hv
                ).astype(np.int8)

    def footprint(self) -> EncoderFootprint:
        cache_bytes = 0
        for value_map in self.category_value_hv.values():
            cache_bytes += sum(array.nbytes for array in value_map.values())
        for bound_map in self.category_bound_hv.values():
            cache_bytes += sum(array.nbytes for array in bound_map.values())
        return EncoderFootprint(
            feature_hv_bytes=int(self.numeric_feature_hv.nbytes + self.categorical_feature_hv.nbytes + self.level_hv.nbytes),
            lookup_bytes=int(self.numeric_lookup.nbytes),
            cache_bytes=int(cache_bytes),
        )


class RBFDenseEncoder:
    """Reference OnlineHD/CyberHD encoder using random Fourier-style features."""

    def __init__(self, input_dim: int, *, dim: int, seed: int) -> None:
        self.input_dim = input_dim
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.basis = self.rng.standard_normal((dim, input_dim), dtype=np.float32)
        self.base = self.rng.uniform(0.0, 2 * np.pi, size=(dim,)).astype(np.float32)

    def encode_dense(self, dense: np.ndarray, *, chunk_size: int = 1024) -> np.ndarray:
        if dense.size == 0:
            return np.zeros((dense.shape[0], self.dim), dtype=np.float32)
        encoded = np.empty((dense.shape[0], self.dim), dtype=np.float32)
        for start in range(0, dense.shape[0], chunk_size):
            stop = min(start + chunk_size, dense.shape[0])
            temp = dense[start:stop] @ self.basis.T
            encoded[start:stop] = (
                np.cos(temp + self.base).astype(np.float32, copy=False)
                * np.sin(temp).astype(np.float32, copy=False)
            )
        return encoded

    def regenerate_dimensions(self, dimension_indices: np.ndarray) -> None:
        if dimension_indices.size == 0:
            return
        self.basis[dimension_indices] = self.rng.standard_normal(
            (dimension_indices.shape[0], self.input_dim), dtype=np.float32
        )
        self.base[dimension_indices] = self.rng.uniform(0.0, 2 * np.pi, size=(dimension_indices.shape[0],)).astype(
            np.float32
        )

    def footprint(self) -> EncoderFootprint:
        return EncoderFootprint(
            feature_hv_bytes=int(self.basis.nbytes + self.base.nbytes),
            lookup_bytes=0,
            cache_bytes=0,
        )

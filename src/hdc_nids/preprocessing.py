"""Tabular preprocessing shared by HDC and baseline models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from .data import RawRecord, RawWindow
from .utils import is_floatlike


@dataclass(slots=True)
class PreparedBatch:
    dataset: str
    window_id: int
    stage_name: str
    numeric: np.ndarray
    categorical: np.ndarray
    dense: np.ndarray
    internal_labels: np.ndarray
    class_indices: np.ndarray
    binary_labels: np.ndarray

    @property
    def size(self) -> int:
        return int(self.binary_labels.shape[0])


@dataclass(slots=True)
class DenseFeatureSelector:
    """Train-only dense feature filtering for offline baselines."""

    mode: str = "none"
    variance_threshold: float = 0.0
    top_k: int = 0
    candidate_indices: np.ndarray | None = None
    preserve_indices: np.ndarray | None = None
    selected_indices: np.ndarray | None = None

    def fit(self, dense: np.ndarray, labels: np.ndarray) -> None:
        feature_count = int(dense.shape[1])
        if self.mode == "none" or feature_count == 0:
            self.selected_indices = np.arange(feature_count, dtype=np.int32)
            return

        preserve = (
            np.unique(self.preserve_indices.astype(np.int32))
            if self.preserve_indices is not None and self.preserve_indices.size > 0
            else np.zeros((0,), dtype=np.int32)
        )
        if self.candidate_indices is not None and self.candidate_indices.size > 0:
            candidate_indices = np.unique(self.candidate_indices.astype(np.int32))
        else:
            candidate_indices = np.arange(feature_count, dtype=np.int32)
        if preserve.size:
            candidate_indices = np.setdiff1d(candidate_indices, preserve, assume_unique=False)
        if candidate_indices.size == 0:
            self.selected_indices = np.sort(preserve.astype(np.int32))
            return

        candidate_mask = np.ones((feature_count,), dtype=bool)
        if "variance" in self.mode:
            variances = np.var(dense[:, candidate_indices], axis=0)
            candidate_mask = variances > float(self.variance_threshold)
            candidate_indices = candidate_indices[candidate_mask]
        if candidate_indices.size == 0:
            self.selected_indices = np.sort(preserve.astype(np.int32))
            return

        if "mi" in self.mode and self.top_k > 0 and candidate_indices.size > self.top_k:
            scores = mutual_info_classif(
                dense[:, candidate_indices],
                labels,
                discrete_features=False,
                random_state=7,
            )
            order = np.argsort(scores)[::-1]
            candidate_indices = candidate_indices[order[: self.top_k]]

        self.selected_indices = np.sort(np.concatenate([candidate_indices.astype(np.int32), preserve]))

    def transform(self, dense: np.ndarray) -> np.ndarray:
        if self.selected_indices is None:
            raise ValueError("DenseFeatureSelector must be fit before transform.")
        return dense[:, self.selected_indices]

    def transform_batch(self, batch: PreparedBatch) -> PreparedBatch:
        if self.selected_indices is None:
            raise ValueError("DenseFeatureSelector must be fit before transform.")
        return PreparedBatch(
            dataset=batch.dataset,
            window_id=batch.window_id,
            stage_name=batch.stage_name,
            numeric=batch.numeric,
            categorical=batch.categorical,
            dense=self.transform(batch.dense),
            internal_labels=batch.internal_labels,
            class_indices=batch.class_indices,
            binary_labels=batch.binary_labels,
        )


@dataclass(slots=True)
class PaperHDScaler:
    """Train-only dense scaling matching the OnlineHD encoder guidance."""

    scale_by_sqrt_features: bool = True
    row_normalize: bool = False
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, dense: np.ndarray) -> None:
        if dense.ndim != 2:
            raise ValueError("PaperHDScaler expects a 2D dense matrix.")
        self.mean_ = dense.mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)
        std = dense.std(axis=0, dtype=np.float32).astype(np.float32, copy=False)
        self.std_ = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)

    def transform(self, dense: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("PaperHDScaler must be fit before transform.")
        scaled = ((dense - self.mean_) / self.std_).astype(np.float32, copy=False)
        if self.scale_by_sqrt_features and scaled.shape[1] > 0:
            scaled /= float(np.sqrt(scaled.shape[1]))
        if self.row_normalize and scaled.shape[0] > 0:
            norms = np.linalg.norm(scaled, axis=1, keepdims=True)
            scaled = scaled / np.clip(norms, 1e-6, None)
        return scaled.astype(np.float32, copy=False)

    def transform_batch(self, batch: PreparedBatch) -> PreparedBatch:
        return PreparedBatch(
            dataset=batch.dataset,
            window_id=batch.window_id,
            stage_name=batch.stage_name,
            numeric=batch.numeric,
            categorical=batch.categorical,
            dense=self.transform(batch.dense),
            internal_labels=batch.internal_labels,
            class_indices=batch.class_indices,
            binary_labels=batch.binary_labels,
        )


class TabularPreprocessor:
    """Prepare mixed-type network records for HDC and dense baselines."""

    def __init__(
        self,
        *,
        class_labels: Sequence[str],
        benign_label: str,
        forced_categorical: set[str] | None = None,
        numeric_transform: str = "zscore",
        z_clip: float = 4.0,
    ) -> None:
        self.class_labels = list(class_labels)
        self.benign_label = benign_label
        self.class_to_index = {label: idx for idx, label in enumerate(self.class_labels)}
        self.forced_categorical = forced_categorical or set()
        self.numeric_transform = numeric_transform
        self.z_clip = z_clip

        self.feature_names: list[str] = []
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []
        self.numeric_means: np.ndarray | None = None
        self.numeric_stds: np.ndarray | None = None
        self.numeric_mins: np.ndarray | None = None
        self.numeric_maxs: np.ndarray | None = None
        self.category_vocab: dict[str, dict[str, int]] = {}
        self.dense_dim: int = 0
        self._dense_offsets: dict[str, tuple[int, int]] = {}

    def _transform_numeric_matrix(self, values: np.ndarray) -> np.ndarray:
        transformed = values.astype(np.float32, copy=True)
        if self.numeric_transform == "signed_log_zscore":
            finite_mask = np.isfinite(transformed)
            transformed[finite_mask] = (
                np.sign(transformed[finite_mask]) * np.log1p(np.abs(transformed[finite_mask]))
            )
        return transformed

    def fit(self, records: Sequence[RawRecord]) -> None:
        if not records:
            raise ValueError("Cannot fit preprocessor without warmup records.")
        self.feature_names = list(records[0].features.keys())
        feature_samples: dict[str, list[str]] = {name: [] for name in self.feature_names}
        for record in records:
            for name in self.feature_names:
                feature_samples[name].append(record.features.get(name, ""))

        self.numeric_features = []
        self.categorical_features = []
        for name in self.feature_names:
            if name in self.forced_categorical:
                self.categorical_features.append(name)
                continue
            values = [value for value in feature_samples[name] if value != ""]
            if values and all(is_floatlike(value) or value.lower() == "nan" for value in values):
                self.numeric_features.append(name)
            else:
                self.categorical_features.append(name)

        numeric_columns = []
        for feature in self.numeric_features:
            parsed = []
            for value in feature_samples[feature]:
                if value == "" or value.lower() == "nan":
                    parsed.append(np.nan)
                else:
                    parsed.append(float(value))
            numeric_columns.append(np.asarray(parsed, dtype=np.float32))

        if numeric_columns:
            numeric_matrix = self._transform_numeric_matrix(np.vstack(numeric_columns).T)
            means = np.nanmean(numeric_matrix, axis=0)
            stds = np.nanstd(numeric_matrix, axis=0)
            stds = np.where(stds < 1e-6, 1.0, stds)
            means = np.where(np.isnan(means), 0.0, means)
            mins = np.nanmin(numeric_matrix, axis=0)
            maxs = np.nanmax(numeric_matrix, axis=0)
            mins = np.where(np.isnan(mins), 0.0, mins)
            maxs = np.where(np.isnan(maxs), 1.0, maxs)
            maxs = np.where((maxs - mins) < 1e-6, mins + 1.0, maxs)
            self.numeric_means = means.astype(np.float32)
            self.numeric_stds = stds.astype(np.float32)
            self.numeric_mins = mins.astype(np.float32)
            self.numeric_maxs = maxs.astype(np.float32)
        else:
            self.numeric_means = np.zeros((0,), dtype=np.float32)
            self.numeric_stds = np.zeros((0,), dtype=np.float32)
            self.numeric_mins = np.zeros((0,), dtype=np.float32)
            self.numeric_maxs = np.zeros((0,), dtype=np.float32)

        dense_offset = 0
        self._dense_offsets = {}
        for feature in self.numeric_features:
            self._dense_offsets[feature] = (dense_offset, dense_offset + 1)
            dense_offset += 1
        self.category_vocab = {}
        for feature in self.categorical_features:
            vocab = {"__UNK__": 0}
            for value in feature_samples[feature]:
                key = value if value else "__UNK__"
                if key not in vocab:
                    vocab[key] = len(vocab)
            self.category_vocab[feature] = vocab
            self._dense_offsets[feature] = (dense_offset, dense_offset + len(vocab))
            dense_offset += len(vocab)
        self.dense_dim = dense_offset

    def _normalize_numeric(self, values: np.ndarray) -> np.ndarray:
        if self.numeric_transform == "minmax":
            scaled = (values - self.numeric_mins) / np.clip(self.numeric_maxs - self.numeric_mins, 1e-6, None)
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
            return np.clip(scaled, 0.0, 1.0).astype(np.float32)
        standardized = (values - self.numeric_means) / self.numeric_stds
        standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(standardized, -self.z_clip, self.z_clip).astype(np.float32)

    def numeric_dense_indices(self) -> np.ndarray:
        return np.arange(len(self.numeric_features), dtype=np.int32)

    def categorical_dense_indices(self) -> np.ndarray:
        indices: list[np.ndarray] = []
        for feature in self.categorical_features:
            start, end = self._dense_offsets[feature]
            indices.append(np.arange(start, end, dtype=np.int32))
        if not indices:
            return np.zeros((0,), dtype=np.int32)
        return np.concatenate(indices).astype(np.int32)

    def transform_records(
        self,
        records: Sequence[RawRecord],
        *,
        dataset: str,
        window_id: int,
        stage_name: str,
    ) -> PreparedBatch:
        numeric = np.zeros((len(records), len(self.numeric_features)), dtype=np.float32)
        categorical = np.empty((len(records), len(self.categorical_features)), dtype=object)
        dense = np.zeros((len(records), self.dense_dim), dtype=np.float32)
        internal_labels = np.empty((len(records),), dtype=object)
        class_indices = np.zeros((len(records),), dtype=np.int32)
        binary_labels = np.zeros((len(records),), dtype=np.int8)

        for row_idx, record in enumerate(records):
            internal_labels[row_idx] = record.internal_label
            class_indices[row_idx] = self.class_to_index[record.internal_label]
            binary_labels[row_idx] = int(record.binary_label)

            if self.numeric_features:
                parsed_numeric = []
                for feature in self.numeric_features:
                    raw = record.features.get(feature, "")
                    if raw == "" or raw.lower() == "nan":
                        parsed_numeric.append(np.nan)
                    else:
                        parsed_numeric.append(float(raw))
                transformed = self._transform_numeric_matrix(np.asarray([parsed_numeric], dtype=np.float32))[0]
                normalized = self._normalize_numeric(transformed)
                numeric[row_idx] = normalized
                dense[row_idx, : len(self.numeric_features)] = normalized

            for col_idx, feature in enumerate(self.categorical_features):
                value = record.features.get(feature, "") or "__UNK__"
                vocab = self.category_vocab[feature]
                if value not in vocab:
                    value = "__UNK__"
                categorical[row_idx, col_idx] = value
                start, end = self._dense_offsets[feature]
                dense[row_idx, start + vocab[value]] = 1.0

        return PreparedBatch(
            dataset=dataset,
            window_id=window_id,
            stage_name=stage_name,
            numeric=numeric,
            categorical=categorical,
            dense=dense,
            internal_labels=internal_labels,
            class_indices=class_indices,
            binary_labels=binary_labels,
        )

    def transform_window(self, window: RawWindow) -> PreparedBatch:
        return self.transform_records(
            window.records,
            dataset=window.dataset,
            window_id=window.window_id,
            stage_name=window.stage_name,
        )

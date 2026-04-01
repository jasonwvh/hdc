"""HDC model implementations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import DriftConfig, MemoryMixConfig, StagnationConfig
from .encoding import RBFDenseEncoder, TabularHDCEncoder
from .preprocessing import PaperHDScaler, PreparedBatch, TabularPreprocessor


def _cosine_similarity(queries: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    if queries.size == 0 or prototypes.size == 0:
        return np.zeros((queries.shape[0], prototypes.shape[0]), dtype=np.float32)
    query_norm = np.linalg.norm(queries, axis=1, keepdims=True)
    proto_norm = np.linalg.norm(prototypes, axis=1, keepdims=True).T
    denom = np.clip(query_norm * proto_norm, 1e-6, None)
    return (queries @ prototypes.T / denom).astype(np.float32)


@dataclass(slots=True)
class PredictionOutput:
    predicted_class_indices: np.ndarray
    predicted_labels: np.ndarray
    predicted_binary: np.ndarray
    class_scores: np.ndarray
    attack_scores: np.ndarray
    benign_scores: np.ndarray
    binary_margin: np.ndarray
    query_hv: np.ndarray
    encode_ms: float
    score_ms: float


class BaseModel:
    """Shared model interface."""

    def fit_initial(self, batch: PreparedBatch) -> None:
        raise NotImplementedError

    def predict(self, batch: PreparedBatch) -> PredictionOutput:
        raise NotImplementedError

    def update(
        self,
        batch: PreparedBatch,
        prediction: PredictionOutput,
        *,
        drift_active: bool = False,
    ) -> list[dict[str, Any]]:
        return []

    def observe_window(
        self,
        metric_history: list[dict[str, Any]],
        current_metrics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return []

    def checkpoint(self, path: str | Path) -> None:
        raise NotImplementedError

    def model_size_bytes(self) -> int:
        raise NotImplementedError


class BaseHDCModel(BaseModel):
    def __init__(
        self,
        *,
        encoder: TabularHDCEncoder,
        preprocessor: TabularPreprocessor,
        base_lr: float,
        rare_class_boost: float,
        prototype_clip: int,
    ) -> None:
        self.encoder = encoder
        self.preprocessor = preprocessor
        self.base_lr = base_lr
        self.rare_class_boost = rare_class_boost
        self.prototype_clip = prototype_clip
        self.class_labels = preprocessor.class_labels
        self.class_count = len(self.class_labels)
        self.benign_index = preprocessor.class_to_index[preprocessor.benign_label]
        self.attack_indices = [idx for idx in range(self.class_count) if idx != self.benign_index]
        self.observed_counts = np.ones((self.class_count,), dtype=np.float32)

    def _class_lr_multiplier(self, class_index: int) -> float:
        if class_index == self.benign_index:
            return 1.0
        return 1.0 + (self.rare_class_boost - 1.0) / math.sqrt(float(self.observed_counts[class_index]))

    def _combined_prototypes(self) -> np.ndarray:
        raise NotImplementedError

    def _quantized_array(self, array: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(array), -32768, 32767).astype(np.int16)

    def predict(self, batch: PreparedBatch) -> PredictionOutput:
        from .utils import monotonic_ms

        encode_start = monotonic_ms()
        query_hv = self.encoder.encode_batch(batch)
        encode_ms = monotonic_ms() - encode_start

        score_start = monotonic_ms()
        prototypes = self._combined_prototypes()
        class_scores = _cosine_similarity(query_hv, prototypes)
        predicted_class_indices = class_scores.argmax(axis=1)
        predicted_labels = np.asarray([self.class_labels[idx] for idx in predicted_class_indices], dtype=object)
        benign_scores = class_scores[:, self.benign_index]
        attack_scores = (
            class_scores[:, self.attack_indices].max(axis=1)
            if self.attack_indices
            else np.zeros((batch.size,), dtype=np.float32)
        )
        predicted_binary = (attack_scores > benign_scores).astype(np.int8)
        binary_margin = np.abs(benign_scores - attack_scores).astype(np.float32)
        score_ms = monotonic_ms() - score_start

        return PredictionOutput(
            predicted_class_indices=predicted_class_indices,
            predicted_labels=predicted_labels,
            predicted_binary=predicted_binary,
            class_scores=class_scores,
            attack_scores=attack_scores,
            benign_scores=benign_scores,
            binary_margin=binary_margin,
            query_hv=query_hv,
            encode_ms=encode_ms,
            score_ms=score_ms,
        )

    def model_size_bytes(self) -> int:
        raise NotImplementedError


class PaperOnlineHDModel(BaseModel):
    """Reference-style OnlineHD/CyberHD model for dense tabular data."""

    def __init__(
        self,
        *,
        input_dim: int,
        class_labels: list[str],
        benign_label: str,
        dim: int,
        seed: int,
        learning_rate: float = 0.035,
        bootstrap_fraction: float = 0.01,
        batch_size: int = 1024,
        scale_by_sqrt_features: bool = True,
        row_normalize: bool = False,
        regen_rate: float = 0.0,
    ) -> None:
        self.class_labels = list(class_labels)
        self.class_count = len(self.class_labels)
        self.benign_index = self.class_labels.index(benign_label)
        self.attack_indices = [idx for idx in range(self.class_count) if idx != self.benign_index]
        self.dim = dim
        self.seed = seed
        self.learning_rate = learning_rate
        self.bootstrap_fraction = bootstrap_fraction
        self.batch_size = batch_size
        self.regen_rate = regen_rate

        self.scaler = PaperHDScaler(
            scale_by_sqrt_features=scale_by_sqrt_features,
            row_normalize=row_normalize,
        )
        self.encoder = RBFDenseEncoder(input_dim, dim=dim, seed=seed)
        self.model = np.zeros((self.class_count, self.dim), dtype=np.float32)
        self.scaler_fitted = False
        self.total_regenerated_dims = 0

    def _fit_scaler_if_needed(self, dense: np.ndarray) -> None:
        if not self.scaler_fitted:
            self.scaler.fit(dense.astype(np.float32, copy=False))
            self.scaler_fitted = True

    def _transform_dense(self, dense: np.ndarray) -> np.ndarray:
        self._fit_scaler_if_needed(dense)
        return self.scaler.transform(dense.astype(np.float32, copy=False))

    def _bootstrap_count(self, sample_count: int) -> int:
        return max(1, int(math.ceil(self.bootstrap_fraction * sample_count)))

    def fit_initial(self, batch: PreparedBatch) -> None:
        dense = self._transform_dense(batch.dense)
        self._one_pass_fit_dense(dense, batch.class_indices.astype(np.int32, copy=False))

    def fit_iterative(self, batch: PreparedBatch, *, epochs: int) -> None:
        if epochs <= 0:
            return
        dense = self._transform_dense(batch.dense)
        self._iterative_fit_dense(dense, batch.class_indices.astype(np.int32, copy=False), epochs=epochs)

    def _one_pass_fit_dense(self, dense: np.ndarray, labels: np.ndarray) -> None:
        if dense.shape[0] == 0:
            return
        cut = min(self._bootstrap_count(dense.shape[0]), dense.shape[0])
        if cut > 0:
            encoded_boot = self.encoder.encode_dense(dense[:cut], chunk_size=self.batch_size)
            for lbl in np.unique(labels[:cut]):
                self.model[int(lbl)] += self.learning_rate * encoded_boot[labels[:cut] == lbl].sum(axis=0)

        for start in range(cut, dense.shape[0], self.batch_size):
            stop = min(start + self.batch_size, dense.shape[0])
            encoded = self.encoder.encode_dense(dense[start:stop], chunk_size=self.batch_size)
            for row_idx in range(encoded.shape[0]):
                sample = encoded[row_idx]
                lbl = int(labels[start + row_idx])
                scores = _cosine_similarity(sample[None, :], self.model)[0]
                pred = int(scores.argmax())
                miss_scores = 1.0 - scores
                self.model[lbl] += self.learning_rate * miss_scores[lbl] * sample
                self.model[pred] -= self.learning_rate * miss_scores[pred] * sample

    def _iterative_fit_dense(self, dense: np.ndarray, labels: np.ndarray, *, epochs: int) -> None:
        if dense.shape[0] == 0:
            return
        for _ in range(epochs):
            for start in range(0, dense.shape[0], self.batch_size):
                stop = min(start + self.batch_size, dense.shape[0])
                encoded = self.encoder.encode_dense(dense[start:stop], chunk_size=self.batch_size)
                y_true = labels[start:stop]
                scores = _cosine_similarity(encoded, self.model)
                y_pred = scores.argmax(axis=1).astype(np.int32, copy=False)
                wrong = y_true != y_pred
                if not np.any(wrong):
                    continue
                row_indices = np.arange(encoded.shape[0], dtype=np.int32)
                alpha_true = (1.0 - scores[row_indices, y_true])[:, None]
                alpha_pred = (scores[row_indices, y_pred] - 1.0)[:, None]
                touched = np.unique(np.concatenate([y_true[wrong], y_pred[wrong]]))
                for lbl in touched:
                    true_mask = wrong & (y_true == lbl)
                    pred_mask = wrong & (y_pred == lbl)
                    if np.any(true_mask):
                        self.model[int(lbl)] += self.learning_rate * (alpha_true[true_mask] * encoded[true_mask]).sum(axis=0)
                    if np.any(pred_mask):
                        self.model[int(lbl)] += self.learning_rate * (alpha_pred[pred_mask] * encoded[pred_mask]).sum(axis=0)

    def regenerate_low_variance_dimensions(self) -> np.ndarray:
        if self.regen_rate <= 0.0:
            return np.zeros((0,), dtype=np.int32)
        norms = np.linalg.norm(self.model, axis=1, keepdims=True)
        normalized = self.model / np.clip(norms, 1e-6, None)
        variances = normalized.var(axis=0)
        regen_count = max(1, int(self.dim * self.regen_rate))
        dims = np.argsort(variances)[:regen_count].astype(np.int32)
        self.encoder.regenerate_dimensions(dims)
        self.model[:, dims] = 0.0
        self.total_regenerated_dims += int(regen_count)
        return dims

    def predict(self, batch: PreparedBatch) -> PredictionOutput:
        from .utils import monotonic_ms

        encode_start = monotonic_ms()
        dense = self._transform_dense(batch.dense)
        encoded = self.encoder.encode_dense(dense, chunk_size=self.batch_size)
        encode_ms = monotonic_ms() - encode_start

        score_start = monotonic_ms()
        class_scores = _cosine_similarity(encoded, self.model)
        predicted_class_indices = class_scores.argmax(axis=1)
        predicted_labels = np.asarray([self.class_labels[idx] for idx in predicted_class_indices], dtype=object)
        benign_scores = class_scores[:, self.benign_index]
        attack_scores = (
            class_scores[:, self.attack_indices].max(axis=1)
            if self.attack_indices
            else np.zeros((batch.size,), dtype=np.float32)
        )
        predicted_binary = (attack_scores > benign_scores).astype(np.int8)
        binary_margin = np.abs(benign_scores - attack_scores).astype(np.float32)
        score_ms = monotonic_ms() - score_start

        return PredictionOutput(
            predicted_class_indices=predicted_class_indices.astype(np.int32, copy=False),
            predicted_labels=predicted_labels,
            predicted_binary=predicted_binary,
            class_scores=class_scores,
            attack_scores=attack_scores.astype(np.float32, copy=False),
            benign_scores=benign_scores.astype(np.float32, copy=False),
            binary_margin=binary_margin,
            query_hv=encoded.astype(np.float32, copy=False),
            encode_ms=encode_ms,
            score_ms=score_ms,
        )

    def update(
        self,
        batch: PreparedBatch,
        prediction: PredictionOutput,
        *,
        drift_active: bool = False,
    ) -> list[dict[str, Any]]:
        _ = prediction
        events: list[dict[str, Any]] = []
        if drift_active:
            dims = self.regenerate_low_variance_dimensions()
            if dims.size:
                events.append(
                    {
                        "window_id": batch.window_id,
                        "dataset": batch.dataset,
                        "stage_name": batch.stage_name,
                        "event_type": "dimension_regeneration",
                        "dimension_count": int(dims.size),
                    }
                )
        dense = self._transform_dense(batch.dense)
        self._iterative_fit_dense(dense, batch.class_indices.astype(np.int32, copy=False), epochs=1)
        return events

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "model": self.model.copy(),
            "basis": self.encoder.basis.copy(),
            "base": self.encoder.base.copy(),
            "scaler_mean": self.scaler.mean_.copy() if self.scaler.mean_ is not None else np.zeros((0,), dtype=np.float32),
            "scaler_std": self.scaler.std_.copy() if self.scaler.std_ is not None else np.zeros((0,), dtype=np.float32),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.model = state["model"].copy()
        self.encoder.basis = state["basis"].copy()
        self.encoder.base = state["base"].copy()
        self.scaler.mean_ = state["scaler_mean"].copy()
        self.scaler.std_ = state["scaler_std"].copy()
        self.scaler_fitted = True

    def checkpoint(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            model=self.model.astype(np.float32, copy=False),
            basis=self.encoder.basis.astype(np.float32, copy=False),
            base=self.encoder.base.astype(np.float32, copy=False),
            scaler_mean=self.scaler.mean_,
            scaler_std=self.scaler.std_,
            total_regenerated_dims=np.asarray([self.total_regenerated_dims], dtype=np.int32),
        )

    def model_size_bytes(self) -> int:
        scaler_bytes = 0
        if self.scaler.mean_ is not None:
            scaler_bytes += int(self.scaler.mean_.nbytes)
        if self.scaler.std_ is not None:
            scaler_bytes += int(self.scaler.std_.nbytes)
        return int(self.model.nbytes + self.encoder.footprint().total_bytes + scaler_bytes)


class StaticHDCModel(BaseHDCModel):
    """Single-pass HDC bundling baseline."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.prototypes = np.zeros((self.class_count, self.encoder.dim), dtype=np.float32)

    def fit_initial(self, batch: PreparedBatch) -> None:
        query_hv = self.encoder.encode_batch(batch) / self.encoder.query_scale
        for class_idx in range(self.class_count):
            mask = batch.class_indices == class_idx
            if np.any(mask):
                self.prototypes[class_idx] += query_hv[mask].sum(axis=0)
                self.observed_counts[class_idx] += float(mask.sum())

    def _combined_prototypes(self) -> np.ndarray:
        return self.prototypes

    def checkpoint(self, path: str | Path) -> None:
        np.savez_compressed(path, prototypes=self._quantized_array(self.prototypes))

    def model_size_bytes(self) -> int:
        return self._quantized_array(self.prototypes).nbytes + self.encoder.footprint().total_bytes


class OnlineHDModel(BaseHDCModel):
    """OnlineHD-style adaptive prototype learning."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.prototypes = np.zeros((self.class_count, self.encoder.dim), dtype=np.float32)

    def fit_initial(self, batch: PreparedBatch) -> None:
        prediction = self.predict(batch)
        self.update(batch, prediction, drift_active=False)

    def _combined_prototypes(self) -> np.ndarray:
        return self.prototypes

    def update(
        self,
        batch: PreparedBatch,
        prediction: PredictionOutput,
        *,
        drift_active: bool = False,
    ) -> list[dict[str, Any]]:
        _ = drift_active
        scaled_query = prediction.query_hv / self.encoder.query_scale
        for row_idx in range(batch.size):
            true_idx = int(batch.class_indices[row_idx])
            pred_idx = int(prediction.predicted_class_indices[row_idx])
            sim_true = float(prediction.class_scores[row_idx, true_idx])
            sim_pred = float(prediction.class_scores[row_idx, pred_idx])
            lr_true = self.base_lr * self._class_lr_multiplier(true_idx)
            delta_true = lr_true * (1.0 - sim_true) * scaled_query[row_idx]
            self.prototypes[true_idx] += delta_true
            if pred_idx != true_idx:
                delta_pred = self.base_lr * (1.0 - sim_pred) * scaled_query[row_idx]
                self.prototypes[pred_idx] -= delta_pred
            self.observed_counts[true_idx] += 1.0
        self.prototypes = np.clip(self.prototypes, -self.prototype_clip, self.prototype_clip)
        return []

    def checkpoint(self, path: str | Path) -> None:
        np.savez_compressed(path, prototypes=self._quantized_array(self.prototypes))

    def model_size_bytes(self) -> int:
        return self._quantized_array(self.prototypes).nbytes + self.encoder.footprint().total_bytes


class DualMemoryHDCModel(BaseHDCModel):
    """Drift-adaptive dual-memory HDC model."""

    def __init__(
        self,
        *,
        encoder: TabularHDCEncoder,
        preprocessor: TabularPreprocessor,
        base_lr: float,
        rare_class_boost: float,
        prototype_clip: int,
        memory_mix: MemoryMixConfig,
        stagnation: StagnationConfig,
        regen_rate: float,
        drift: DriftConfig,
    ) -> None:
        super().__init__(
            encoder=encoder,
            preprocessor=preprocessor,
            base_lr=base_lr,
            rare_class_boost=rare_class_boost,
            prototype_clip=prototype_clip,
        )
        self.plastic = np.zeros((self.class_count, self.encoder.dim), dtype=np.float32)
        self.stable = np.zeros((self.class_count, self.encoder.dim), dtype=np.float32)
        self.memory_mix = memory_mix
        self.stagnation = stagnation
        self.regen_rate = regen_rate
        self.drift = drift

    def fit_initial(self, batch: PreparedBatch) -> None:
        query_hv = self.encoder.encode_batch(batch) / self.encoder.query_scale
        for class_idx in range(self.class_count):
            mask = batch.class_indices == class_idx
            if np.any(mask):
                summed = query_hv[mask].sum(axis=0)
                self.plastic[class_idx] += summed
                self.stable[class_idx] += summed
                self.observed_counts[class_idx] += float(mask.sum())

    def _combined_prototypes(self) -> np.ndarray:
        combined = np.zeros_like(self.plastic)
        for class_idx in range(self.class_count):
            if class_idx == self.benign_index:
                plastic_weight = self.memory_mix.benign_plastic_weight
            else:
                plastic_weight = self.memory_mix.attack_plastic_weight
            combined[class_idx] = (
                plastic_weight * self.plastic[class_idx] + (1.0 - plastic_weight) * self.stable[class_idx]
            )
        return combined

    def update(
        self,
        batch: PreparedBatch,
        prediction: PredictionOutput,
        *,
        drift_active: bool = False,
    ) -> list[dict[str, Any]]:
        self.plastic[self.benign_index] *= self.memory_mix.benign_decay
        scaled_query = prediction.query_hv / self.encoder.query_scale
        drift_boost = self.drift.benign_lr_boost if drift_active else 1.0
        for row_idx in range(batch.size):
            true_idx = int(batch.class_indices[row_idx])
            pred_idx = int(prediction.predicted_class_indices[row_idx])
            sim_true = float(prediction.class_scores[row_idx, true_idx])
            sim_pred = float(prediction.class_scores[row_idx, pred_idx])
            lr_true = self.base_lr * self._class_lr_multiplier(true_idx)
            if true_idx == self.benign_index:
                lr_true *= drift_boost
            delta_true = lr_true * (1.0 - sim_true) * scaled_query[row_idx]
            self.plastic[true_idx] += delta_true
            if pred_idx != true_idx:
                delta_pred = self.base_lr * (1.0 - sim_pred) * scaled_query[row_idx]
                self.plastic[pred_idx] -= delta_pred

            if pred_idx == true_idx and float(prediction.binary_margin[row_idx]) >= self.memory_mix.stable_margin_threshold:
                self.stable[true_idx] += 0.5 * delta_true

            self.observed_counts[true_idx] += 1.0

        self.plastic = np.clip(self.plastic, -self.prototype_clip, self.prototype_clip)
        self.stable = np.clip(self.stable, -self.prototype_clip, self.prototype_clip)
        return []

    def observe_window(
        self,
        metric_history: list[dict[str, Any]],
        current_metrics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if len(metric_history) < self.stagnation.patience:
            return []
        recent = metric_history[-self.stagnation.patience :]
        best_recent_f1 = max(float(row["binary_f1"]) for row in recent)
        best_recent_attack = max(float(row["attack_recall_macro"]) for row in recent)
        if (
            float(current_metrics["binary_f1"]) + self.stagnation.min_improvement >= best_recent_f1
            or float(current_metrics["attack_recall_macro"]) + self.stagnation.min_improvement >= best_recent_attack
        ):
            return []

        prototypes = self._combined_prototypes()
        norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
        normalized = np.divide(prototypes, np.clip(norms, 1e-6, None))
        variances = normalized.var(axis=0)
        regen_count = max(1, int(self.encoder.dim * self.regen_rate))
        dims = np.argsort(variances)[:regen_count]
        self.encoder.regenerate_dimensions(dims.astype(np.int32))
        self.plastic[:, dims] = 0.0
        self.stable[:, dims] = 0.0
        return [
            {
                "window_id": current_metrics["window_id"],
                "dataset": current_metrics["dataset"],
                "stage_name": current_metrics["stage_name"],
                "event_type": "dimension_regeneration",
                "dimension_count": int(regen_count),
                "binary_f1": float(current_metrics["binary_f1"]),
                "attack_recall_macro": float(current_metrics["attack_recall_macro"]),
            }
        ]

    def checkpoint(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            plastic=self._quantized_array(self.plastic),
            stable=self._quantized_array(self.stable),
        )

    def model_size_bytes(self) -> int:
        return (
            self._quantized_array(self.plastic).nbytes
            + self._quantized_array(self.stable).nbytes
            + self.encoder.footprint().total_bytes
        )

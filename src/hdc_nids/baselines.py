"""Dense baseline models."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - dependency is installed in benchmark runs
    torch = None
    nn = None

from .models import BaseModel, PredictionOutput
from .preprocessing import PreparedBatch, TabularPreprocessor
from .utils import chunked, monotonic_ms


class StaticMLPModel(BaseModel):
    """Static dense MLP baseline."""

    def __init__(
        self,
        *,
        preprocessor: TabularPreprocessor,
        hidden_dim: int,
        learning_rate_init: float,
        alpha: float,
        max_iter: int,
        seed: int,
    ) -> None:
        self.preprocessor = preprocessor
        self.hidden_dim = hidden_dim
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.max_iter = max_iter
        self.seed = seed
        self.class_labels = preprocessor.class_labels
        self.benign_index = preprocessor.class_to_index[preprocessor.benign_label]
        self.attack_indices = [idx for idx in range(len(self.class_labels)) if idx != self.benign_index]
        self.classes_ = np.arange(len(self.class_labels), dtype=np.int32)
        self.estimator = MLPClassifier(
            hidden_layer_sizes=(hidden_dim,),
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=1,
            warm_start=True,
            random_state=seed,
        )

    def fit_initial(self, batch: PreparedBatch) -> None:
        for _ in range(self.max_iter):
            self.estimator.partial_fit(batch.dense, batch.class_indices, classes=self.classes_)

    def _prediction_from_proba(
        self,
        batch: PreparedBatch,
        probabilities: np.ndarray,
        *,
        encode_ms: float = 0.0,
        score_ms: float = 0.0,
    ) -> PredictionOutput:
        predicted_class_indices = probabilities.argmax(axis=1)
        predicted_labels = np.asarray(
            [self.class_labels[index] for index in predicted_class_indices],
            dtype=object,
        )
        benign_scores = probabilities[:, self.benign_index]
        attack_scores = probabilities[:, self.attack_indices].max(axis=1)
        predicted_binary = (attack_scores > benign_scores).astype(np.int8)
        binary_margin = np.abs(benign_scores - attack_scores).astype(np.float32)
        return PredictionOutput(
            predicted_class_indices=predicted_class_indices,
            predicted_labels=predicted_labels,
            predicted_binary=predicted_binary,
            class_scores=probabilities.astype(np.float32),
            attack_scores=attack_scores.astype(np.float32),
            benign_scores=benign_scores.astype(np.float32),
            binary_margin=binary_margin,
            query_hv=batch.dense.astype(np.float32),
            encode_ms=encode_ms,
            score_ms=score_ms,
        )

    def predict(self, batch: PreparedBatch) -> PredictionOutput:
        score_start = monotonic_ms()
        probabilities = self.estimator.predict_proba(batch.dense)
        score_ms = monotonic_ms() - score_start
        return self._prediction_from_proba(batch, probabilities, score_ms=score_ms)

    def checkpoint(self, path: str | Path) -> None:
        payload = {f"coef_{idx}": coef for idx, coef in enumerate(self.estimator.coefs_)}
        payload.update({f"bias_{idx}": bias for idx, bias in enumerate(self.estimator.intercepts_)})
        np.savez_compressed(path, **payload)

    def model_size_bytes(self) -> int:
        return int(
            sum(weight.nbytes for weight in self.estimator.coefs_)
            + sum(bias.nbytes for bias in self.estimator.intercepts_)
        )


class EWCMLPModel(StaticMLPModel):
    """Incremental dense MLP with anchor shrinkage approximating EWC."""

    def __init__(
        self,
        *,
        preprocessor: TabularPreprocessor,
        hidden_dim: int,
        learning_rate_init: float,
        alpha: float,
        max_iter: int,
        partial_fit_epochs: int,
        ewc_lambda: float,
        seed: int,
    ) -> None:
        super().__init__(
            preprocessor=preprocessor,
            hidden_dim=hidden_dim,
            learning_rate_init=learning_rate_init,
            alpha=alpha,
            max_iter=max_iter,
            seed=seed,
        )
        self.partial_fit_epochs = partial_fit_epochs
        self.ewc_lambda = ewc_lambda
        self.anchor_coefs: list[np.ndarray] = []
        self.anchor_intercepts: list[np.ndarray] = []
        self.importance_coefs: list[np.ndarray] = []
        self.importance_intercepts: list[np.ndarray] = []

    def fit_initial(self, batch: PreparedBatch) -> None:
        super().fit_initial(batch)
        self.anchor_coefs = [deepcopy(weight) for weight in self.estimator.coefs_]
        self.anchor_intercepts = [deepcopy(bias) for bias in self.estimator.intercepts_]
        self.importance_coefs = [np.abs(weight) + 1e-3 for weight in self.anchor_coefs]
        self.importance_intercepts = [np.abs(bias) + 1e-3 for bias in self.anchor_intercepts]

    def update(
        self,
        batch: PreparedBatch,
        prediction: PredictionOutput,
        *,
        drift_active: bool = False,
    ) -> list[dict[str, Any]]:
        _ = prediction
        _ = drift_active
        for _ in range(self.partial_fit_epochs):
            self.estimator.partial_fit(batch.dense, batch.class_indices)
        shrink = min(0.5, self.ewc_lambda / 100.0)
        for idx, weight in enumerate(self.estimator.coefs_):
            weight -= shrink * self.importance_coefs[idx] * (weight - self.anchor_coefs[idx])
        for idx, bias in enumerate(self.estimator.intercepts_):
            bias -= shrink * self.importance_intercepts[idx] * (bias - self.anchor_intercepts[idx])
        return []


class StaticSVMModel(BaseModel):
    """Static linear SVM baseline using hinge loss."""

    def __init__(
        self,
        *,
        preprocessor: TabularPreprocessor,
        alpha: float,
        max_iter: int,
        seed: int,
    ) -> None:
        self.preprocessor = preprocessor
        self.alpha = alpha
        self.max_iter = max_iter
        self.seed = seed
        self.class_labels = preprocessor.class_labels
        self.benign_index = preprocessor.class_to_index[preprocessor.benign_label]
        self.attack_indices = [idx for idx in range(len(self.class_labels)) if idx != self.benign_index]
        self.classes_ = np.arange(len(self.class_labels), dtype=np.int32)
        self.estimator = SGDClassifier(
            loss="hinge",
            alpha=alpha,
            max_iter=1,
            random_state=seed,
            average=True,
        )

    def fit_initial(self, batch: PreparedBatch) -> None:
        for _ in range(self.max_iter):
            self.estimator.partial_fit(batch.dense, batch.class_indices, classes=self.classes_)

    def _prediction_from_scores(
        self,
        batch: PreparedBatch,
        full_scores: np.ndarray,
        *,
        score_ms: float,
    ) -> PredictionOutput:
        predicted_class_indices = full_scores.argmax(axis=1)
        predicted_labels = np.asarray(
            [self.class_labels[index] for index in predicted_class_indices],
            dtype=object,
        )
        benign_scores = full_scores[:, self.benign_index]
        attack_scores = full_scores[:, self.attack_indices].max(axis=1)
        predicted_binary = (attack_scores > benign_scores).astype(np.int8)
        binary_margin = np.abs(benign_scores - attack_scores).astype(np.float32)
        return PredictionOutput(
            predicted_class_indices=predicted_class_indices,
            predicted_labels=predicted_labels,
            predicted_binary=predicted_binary,
            class_scores=full_scores,
            attack_scores=attack_scores.astype(np.float32),
            benign_scores=benign_scores.astype(np.float32),
            binary_margin=binary_margin,
            query_hv=batch.dense.astype(np.float32),
            encode_ms=0.0,
            score_ms=score_ms,
        )

    def predict(self, batch: PreparedBatch) -> PredictionOutput:
        score_start = monotonic_ms()
        decision = self.estimator.decision_function(batch.dense)
        score_ms = monotonic_ms() - score_start
        if decision.ndim == 1:
            decision = decision[:, None]
        full_scores = np.full((batch.size, len(self.class_labels)), -1e6, dtype=np.float32)
        estimator_classes = np.asarray(self.estimator.classes_, dtype=np.int32)
        full_scores[:, estimator_classes] = decision.astype(np.float32)
        return self._prediction_from_scores(batch, full_scores, score_ms=score_ms)

    def checkpoint(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            coef_=self.estimator.coef_,
            intercept_=self.estimator.intercept_,
            classes_=self.estimator.classes_,
        )

    def model_size_bytes(self) -> int:
        return int(self.estimator.coef_.nbytes + self.estimator.intercept_.nbytes)


class OnlineSVMModel(StaticSVMModel):
    """Incremental SVM baseline with binary detection and attack-family OVR heads."""

    def __init__(
        self,
        *,
        preprocessor: TabularPreprocessor,
        alpha: float,
        max_iter: int,
        partial_fit_epochs: int,
        seed: int,
    ) -> None:
        super().__init__(
            preprocessor=preprocessor,
            alpha=alpha,
            max_iter=max_iter,
            seed=seed,
        )
        self.partial_fit_epochs = partial_fit_epochs
        self.binary_classes_ = np.asarray([0, 1], dtype=np.int32)
        self.binary_estimator = SGDClassifier(
            loss="hinge",
            alpha=alpha,
            max_iter=1,
            random_state=seed,
            average=False,
        )
        self.attack_label_indices = [idx for idx in range(len(self.class_labels)) if idx != self.benign_index]
        self.attack_estimators = {
            attack_idx: SGDClassifier(
                loss="hinge",
                alpha=alpha,
                max_iter=1,
                random_state=seed + attack_idx + 1,
                average=False,
            )
            for attack_idx in self.attack_label_indices
        }

    def fit_initial(self, batch: PreparedBatch) -> None:
        binary_weight = self._sample_weights(batch.binary_labels.astype(np.int32))
        for _ in range(self.max_iter):
            self.binary_estimator.partial_fit(
                batch.dense,
                batch.binary_labels,
                classes=self.binary_classes_,
                sample_weight=binary_weight,
            )
            for attack_idx, estimator in self.attack_estimators.items():
                target = (batch.class_indices == attack_idx).astype(np.int32)
                target_weight = self._sample_weights(target)
                estimator.partial_fit(
                    batch.dense,
                    target,
                    classes=self.binary_classes_,
                    sample_weight=target_weight,
                )

    def predict(self, batch: PreparedBatch) -> PredictionOutput:
        score_start = monotonic_ms()
        binary_score = self.binary_estimator.decision_function(batch.dense).astype(np.float32)
        family_scores = np.column_stack(
            [self.attack_estimators[idx].decision_function(batch.dense) for idx in self.attack_label_indices]
        ).astype(np.float32)
        score_ms = monotonic_ms() - score_start

        predicted_binary = (binary_score > 0.0).astype(np.int8)
        predicted_class_indices = np.full((batch.size,), self.benign_index, dtype=np.int32)
        if family_scores.size:
            best_attack_offset = family_scores.argmax(axis=1)
            predicted_attack_indices = np.asarray(
                [self.attack_label_indices[offset] for offset in best_attack_offset],
                dtype=np.int32,
            )
            predicted_class_indices[predicted_binary == 1] = predicted_attack_indices[predicted_binary == 1]

        predicted_labels = np.asarray(
            [self.class_labels[index] for index in predicted_class_indices],
            dtype=object,
        )
        class_scores = np.full((batch.size, len(self.class_labels)), -1e6, dtype=np.float32)
        class_scores[:, self.benign_index] = -binary_score
        for column_idx, attack_idx in enumerate(self.attack_label_indices):
            class_scores[:, attack_idx] = family_scores[:, column_idx]
        return PredictionOutput(
            predicted_class_indices=predicted_class_indices,
            predicted_labels=predicted_labels,
            predicted_binary=predicted_binary,
            class_scores=class_scores,
            attack_scores=binary_score,
            benign_scores=(-binary_score).astype(np.float32),
            binary_margin=np.abs(binary_score).astype(np.float32),
            query_hv=batch.dense.astype(np.float32),
            encode_ms=0.0,
            score_ms=score_ms,
        )

    def _sample_weights(self, class_indices: np.ndarray) -> np.ndarray:
        counts = np.bincount(class_indices, minlength=len(self.class_labels)).astype(np.float32)
        nonzero = counts > 0
        weights = np.ones_like(counts)
        if np.any(nonzero):
            weights[nonzero] = float(class_indices.shape[0]) / counts[nonzero]
            weights[nonzero] /= float(np.mean(weights[nonzero]))
        return weights[class_indices].astype(np.float32)

    def update(
        self,
        batch: PreparedBatch,
        prediction: PredictionOutput,
        *,
        drift_active: bool = False,
    ) -> list[dict[str, Any]]:
        _ = prediction
        _ = drift_active
        binary_weight = self._sample_weights(batch.binary_labels.astype(np.int32))
        for _ in range(self.partial_fit_epochs):
            self.binary_estimator.partial_fit(
                batch.dense,
                batch.binary_labels,
                classes=self.binary_classes_,
                sample_weight=binary_weight,
            )
            for attack_idx, estimator in self.attack_estimators.items():
                target = (batch.class_indices == attack_idx).astype(np.int32)
                target_weight = self._sample_weights(target)
                estimator.partial_fit(
                    batch.dense,
                    target,
                    classes=self.binary_classes_,
                    sample_weight=target_weight,
                )
        return []

    def checkpoint(self, path: str | Path) -> None:
        payload = {
            "binary_coef_": self.binary_estimator.coef_,
            "binary_intercept_": self.binary_estimator.intercept_,
        }
        for attack_idx, estimator in self.attack_estimators.items():
            label = self.class_labels[attack_idx].replace(" ", "_")
            payload[f"{label}_coef_"] = estimator.coef_
            payload[f"{label}_intercept_"] = estimator.intercept_
        np.savez_compressed(path, **payload)

    def model_size_bytes(self) -> int:
        total = int(self.binary_estimator.coef_.nbytes + self.binary_estimator.intercept_.nbytes)
        for estimator in self.attack_estimators.values():
            total += int(estimator.coef_.nbytes + estimator.intercept_.nbytes)
        return total


class OnlineLSTMModel(BaseModel):
    """PyTorch streaming LSTM baseline over time-ordered continual windows."""

    def __init__(
        self,
        *,
        preprocessor: TabularPreprocessor,
        hidden_dim: int,
        sequence_length: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        gradient_clip: float,
        update_sample_limit: int,
        dropout: float,
        seed: int,
    ) -> None:
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for OnlineLSTMModel.")
        self.preprocessor = preprocessor
        self.class_labels = preprocessor.class_labels
        self.class_count = len(self.class_labels)
        self.benign_index = preprocessor.class_to_index[preprocessor.benign_label]
        self.attack_indices = [idx for idx in range(self.class_count) if idx != self.benign_index]
        self.input_dim = preprocessor.dense_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = max(1, sequence_length)
        self.learning_rate = learning_rate
        self.epochs = max(1, epochs)
        self.batch_size = max(1, batch_size)
        self.gradient_clip = gradient_clip
        self.update_sample_limit = update_sample_limit
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)
        self.device = torch.device("cpu")
        torch.manual_seed(seed)
        self.model = _OfflineTorchLSTM(
            input_size=self.input_dim,
            hidden_dim=self.hidden_dim,
            class_count=self.class_count,
            dropout=self.dropout,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.context_tail = np.zeros((self.sequence_length - 1, self.input_dim), dtype=np.float32)

    def _make_sequences(self, dense: np.ndarray) -> np.ndarray:
        if dense.shape[0] == 0:
            return np.zeros((0, self.sequence_length, self.input_dim), dtype=np.float32)
        prefix = self.context_tail
        extended = np.vstack([prefix, dense]).astype(np.float32)
        sequences = np.zeros((dense.shape[0], self.sequence_length, self.input_dim), dtype=np.float32)
        for row_idx in range(dense.shape[0]):
            start = row_idx
            stop = start + self.sequence_length
            sequences[row_idx] = extended[start:stop]
        return sequences

    def _advance_context(self, dense: np.ndarray) -> None:
        if self.sequence_length <= 1:
            return
        if dense.shape[0] >= self.sequence_length - 1:
            self.context_tail = dense[-(self.sequence_length - 1) :].astype(np.float32, copy=True)
            return
        combined = np.vstack([self.context_tail, dense]).astype(np.float32)
        self.context_tail = combined[-(self.sequence_length - 1) :]

    def _class_weights(self, labels: np.ndarray) -> np.ndarray:
        counts = np.bincount(labels, minlength=self.class_count).astype(np.float32)
        weights = np.ones((self.class_count,), dtype=np.float32)
        nonzero = counts > 0
        if np.any(nonzero):
            weights[nonzero] = float(labels.shape[0]) / counts[nonzero]
            weights[nonzero] /= float(np.mean(weights[nonzero]))
        return weights

    def _train_batch(self, dense: np.ndarray, labels: np.ndarray) -> None:
        if dense.shape[0] == 0:
            return
        sequences = self._make_sequences(dense)
        class_weights = torch.tensor(
            self._class_weights(labels),
            dtype=torch.float32,
            device=self.device,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        indices = np.arange(labels.shape[0])
        for _ in range(self.epochs):
            self.model.train()
            self.rng.shuffle(indices)
            for chunk in chunked(indices.tolist(), self.batch_size):
                batch_idx = np.asarray(chunk, dtype=np.int32)
                x_batch = torch.from_numpy(sequences[batch_idx]).to(self.device)
                y_batch = torch.from_numpy(labels[batch_idx].astype(np.int64, copy=False)).to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
                self.optimizer.step()

    def fit_initial(self, batch: PreparedBatch) -> None:
        dense = batch.dense.astype(np.float32, copy=False)
        self.context_tail = np.zeros((self.sequence_length - 1, self.input_dim), dtype=np.float32)
        self._train_batch(dense, batch.class_indices.astype(np.int32, copy=False))
        self._advance_context(dense)

    def _predict_probabilities(self, dense: np.ndarray) -> np.ndarray:
        sequences = self._make_sequences(dense.astype(np.float32, copy=False))
        tensor = torch.from_numpy(sequences).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.cpu().numpy().astype(np.float32, copy=False)

    def predict(self, batch: PreparedBatch) -> PredictionOutput:
        score_start = monotonic_ms()
        probabilities = self._predict_probabilities(batch.dense)
        score_ms = monotonic_ms() - score_start
        predicted_class_indices = probabilities.argmax(axis=1)
        predicted_labels = np.asarray(
            [self.class_labels[index] for index in predicted_class_indices],
            dtype=object,
        )
        benign_scores = probabilities[:, self.benign_index]
        attack_scores = probabilities[:, self.attack_indices].max(axis=1)
        predicted_binary = (attack_scores > benign_scores).astype(np.int8)
        binary_margin = np.abs(benign_scores - attack_scores).astype(np.float32)
        return PredictionOutput(
            predicted_class_indices=predicted_class_indices,
            predicted_labels=predicted_labels,
            predicted_binary=predicted_binary,
            class_scores=probabilities.astype(np.float32),
            attack_scores=attack_scores.astype(np.float32),
            benign_scores=benign_scores.astype(np.float32),
            binary_margin=binary_margin,
            query_hv=batch.dense.astype(np.float32),
            encode_ms=0.0,
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
        _ = drift_active
        full_dense = batch.dense.astype(np.float32, copy=False)
        train_dense = full_dense
        labels = batch.class_indices.astype(np.int32, copy=False)
        train_labels = labels
        if self.update_sample_limit > 0 and full_dense.shape[0] > self.update_sample_limit:
            train_dense = full_dense[-self.update_sample_limit :]
            train_labels = labels[-self.update_sample_limit :]
        self._train_batch(train_dense, train_labels)
        self._advance_context(full_dense)
        return []

    def checkpoint(self, path: str | Path) -> None:
        payload = {key: value.detach().cpu().numpy() for key, value in self.model.state_dict().items()}
        payload["context_tail"] = self.context_tail
        np.savez_compressed(path, **payload)

    def model_size_bytes(self) -> int:
        parameter_bytes = sum(parameter.nelement() * parameter.element_size() for parameter in self.model.parameters())
        return int(parameter_bytes + self.context_tail.nbytes)


class OfflineSVMRBFModel(BaseModel):
    """Conventional offline SVM baseline with RBF kernel."""

    def __init__(
        self,
        *,
        preprocessor: TabularPreprocessor,
        c_value: float,
        gamma_value: str | float,
        seed: int,
    ) -> None:
        self.preprocessor = preprocessor
        self.class_labels = preprocessor.class_labels
        self.benign_index = preprocessor.class_to_index[preprocessor.benign_label]
        self.attack_indices = [idx for idx in range(len(self.class_labels)) if idx != self.benign_index]
        self.estimator = SVC(
            C=c_value,
            gamma=gamma_value,
            kernel="rbf",
            class_weight="balanced",
            probability=False,
            decision_function_shape="ovr",
            random_state=seed,
        )

    def fit_initial(self, batch: PreparedBatch) -> None:
        self.estimator.fit(batch.dense, batch.class_indices)

    def predict(self, batch: PreparedBatch) -> PredictionOutput:
        score_start = monotonic_ms()
        predicted_class_indices = self.estimator.predict(batch.dense).astype(np.int32)
        decision = self.estimator.decision_function(batch.dense)
        score_ms = monotonic_ms() - score_start
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        probabilities = decision.astype(np.float32)
        predicted_labels = np.asarray([self.class_labels[idx] for idx in predicted_class_indices], dtype=object)
        benign_scores = probabilities[:, self.benign_index]
        attack_scores = probabilities[:, self.attack_indices].max(axis=1)
        predicted_binary = (attack_scores > benign_scores).astype(np.int8)
        binary_margin = np.abs(benign_scores - attack_scores).astype(np.float32)
        return PredictionOutput(
            predicted_class_indices=predicted_class_indices,
            predicted_labels=predicted_labels,
            predicted_binary=predicted_binary,
            class_scores=probabilities,
            attack_scores=attack_scores.astype(np.float32),
            benign_scores=benign_scores.astype(np.float32),
            binary_margin=binary_margin,
            query_hv=batch.dense.astype(np.float32),
            encode_ms=0.0,
            score_ms=score_ms,
        )

    def checkpoint(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            support_vectors_=self.estimator.support_vectors_,
            n_support_=self.estimator.n_support_,
        )

    def model_size_bytes(self) -> int:
        return int(self.estimator.support_vectors_.nbytes)


class _OfflineTorchLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, class_count: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, class_count)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(values)
        last_hidden = outputs[:, -1, :]
        return self.classifier(self.dropout(last_hidden))


class OfflineLSTMModel(BaseModel):
    """Offline PyTorch LSTM baseline for tabular IDS classification."""

    def __init__(
        self,
        *,
        preprocessor: TabularPreprocessor,
        hidden_dim: int,
        learning_rate: float,
        batch_size: int,
        dropout: float,
        max_epochs: int,
        patience: int,
        segment_count: int,
        seed: int,
    ) -> None:
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for OfflineLSTMModel.")
        self.preprocessor = preprocessor
        self.class_labels = preprocessor.class_labels
        self.benign_index = preprocessor.class_to_index[preprocessor.benign_label]
        self.attack_indices = [idx for idx in range(len(self.class_labels)) if idx != self.benign_index]
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.patience = patience
        self.segment_count = max(1, segment_count)
        self.seed = seed
        self.device = torch.device("cpu")

        self.segment_width = int(np.ceil(preprocessor.dense_dim / self.segment_count))
        self.padded_dim = self.segment_width * self.segment_count
        torch.manual_seed(seed)
        self.model = _OfflineTorchLSTM(
            input_size=self.segment_width,
            hidden_dim=hidden_dim,
            class_count=len(self.class_labels),
            dropout=dropout,
        ).to(self.device)

    def _reshape_sequences(self, dense: np.ndarray) -> np.ndarray:
        if dense.shape[1] < self.padded_dim:
            pad_width = self.padded_dim - dense.shape[1]
            dense = np.pad(dense, ((0, 0), (0, pad_width)), mode="constant")
        return dense.reshape(dense.shape[0], self.segment_count, self.segment_width).astype(np.float32)

    def _predict_probabilities(self, dense: np.ndarray) -> np.ndarray:
        sequences = self._reshape_sequences(dense)
        tensor = torch.tensor(sequences, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities.cpu().numpy().astype(np.float32)

    def fit_initial(self, batch: PreparedBatch) -> None:
        self.fit_with_validation(batch, batch)

    def fit_with_validation(self, train_batch: PreparedBatch, val_batch: PreparedBatch) -> float:
        train_sequences = self._reshape_sequences(train_batch.dense.astype(np.float32))
        train_labels = train_batch.class_indices.astype(np.int64)
        val_dense = val_batch.dense.astype(np.float32)
        class_counts = np.bincount(train_labels, minlength=len(self.class_labels)).astype(np.float32)
        class_weights = np.ones((len(self.class_labels),), dtype=np.float32)
        nonzero = class_counts > 0
        if np.any(nonzero):
            class_weights[nonzero] = float(train_labels.shape[0]) / class_counts[nonzero]
            class_weights[nonzero] /= float(np.mean(class_weights[nonzero]))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        )

        best_score = -1.0
        best_state = None
        patience_left = self.patience
        indices = np.arange(train_labels.shape[0])
        rng = np.random.default_rng(self.seed)
        for _ in range(self.max_epochs):
            self.model.train()
            rng.shuffle(indices)
            for chunk in chunked(indices.tolist(), self.batch_size):
                batch_idx = np.asarray(chunk, dtype=np.int32)
                x_batch = torch.tensor(train_sequences[batch_idx], dtype=torch.float32, device=self.device)
                y_batch = torch.tensor(train_labels[batch_idx], dtype=torch.long, device=self.device)
                optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            val_probabilities = self._predict_probabilities(val_dense)
            val_predicted = val_probabilities.argmax(axis=1)
            val_score = float(f1_score(val_batch.class_indices, val_predicted, average="macro", zero_division=0))
            if val_score > best_score + 1e-5:
                best_score = val_score
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_score

    def predict(self, batch: PreparedBatch) -> PredictionOutput:
        score_start = monotonic_ms()
        probabilities = self._predict_probabilities(batch.dense.astype(np.float32))
        score_ms = monotonic_ms() - score_start
        predicted_class_indices = probabilities.argmax(axis=1)
        predicted_labels = np.asarray([self.class_labels[idx] for idx in predicted_class_indices], dtype=object)
        benign_scores = probabilities[:, self.benign_index]
        attack_scores = probabilities[:, self.attack_indices].max(axis=1)
        predicted_binary = (attack_scores > benign_scores).astype(np.int8)
        binary_margin = np.abs(benign_scores - attack_scores).astype(np.float32)
        return PredictionOutput(
            predicted_class_indices=predicted_class_indices,
            predicted_labels=predicted_labels,
            predicted_binary=predicted_binary,
            class_scores=probabilities,
            attack_scores=attack_scores.astype(np.float32),
            benign_scores=benign_scores.astype(np.float32),
            binary_margin=binary_margin,
            query_hv=batch.dense.astype(np.float32),
            encode_ms=0.0,
            score_ms=score_ms,
        )

    def checkpoint(self, path: str | Path) -> None:
        payload = {key: value.detach().cpu().numpy() for key, value in self.model.state_dict().items()}
        np.savez_compressed(path, **payload)

    def model_size_bytes(self) -> int:
        return int(sum(parameter.nelement() * parameter.element_size() for parameter in self.model.parameters()))

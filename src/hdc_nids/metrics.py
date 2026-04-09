"""Evaluation metrics and drift detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import DriftConfig
from .preprocessing import PreparedBatch, TabularPreprocessor


@dataclass(slots=True)
class MetricBundle:
    row: dict[str, float | int | str]
    per_class_recall: dict[str, float]


@dataclass(slots=True)
class OfflineMetricBundle:
    row: dict[str, float | int | str]
    confusion: np.ndarray


@dataclass(slots=True)
class ContinualHeadlineBundle:
    row: dict[str, float | int | str]


def compute_window_metrics(
    *,
    batch: PreparedBatch,
    predicted_labels: np.ndarray,
    predicted_binary: np.ndarray,
    attack_scores: np.ndarray,
    binary_margin: np.ndarray,
    preprocessor: TabularPreprocessor,
) -> MetricBundle:
    true_binary = batch.binary_labels
    precision = precision_score(true_binary, predicted_binary, zero_division=0)
    recall = recall_score(true_binary, predicted_binary, zero_division=0)
    f1 = f1_score(true_binary, predicted_binary, zero_division=0)
    if np.unique(true_binary).size > 1:
        pr_auc = float(average_precision_score(true_binary, attack_scores))
    else:
        pr_auc = 0.0

    benign_mask = true_binary == 0
    benign_fp_rate = float(((predicted_binary == 1) & benign_mask).sum() / max(benign_mask.sum(), 1))
    attack_row_count = int(np.sum(true_binary))
    benign_row_count = int(batch.size - attack_row_count)

    per_class_recall: dict[str, float] = {}
    attack_recalls: list[float] = []
    for label in preprocessor.class_labels:
        if label == preprocessor.benign_label:
            continue
        mask = batch.internal_labels == label
        if not np.any(mask):
            continue
        class_recall = float(np.mean(predicted_labels[mask] == label))
        per_class_recall[label] = class_recall
        attack_recalls.append(class_recall)

    return MetricBundle(
        row={
            "window_id": batch.window_id,
            "dataset": batch.dataset,
            "stage_name": batch.stage_name,
            "row_count": batch.size,
            "attack_row_count": attack_row_count,
            "benign_row_count": benign_row_count,
            "has_attack_window": int(attack_row_count > 0),
            "binary_accuracy": float(accuracy_score(true_binary, predicted_binary)),
            "multiclass_accuracy": float(accuracy_score(batch.internal_labels, predicted_labels)),
            "multiclass_macro_f1": float(
                f1_score(batch.internal_labels, predicted_labels, average="macro", zero_division=0)
            ),
            "multiclass_weighted_f1": float(
                f1_score(batch.internal_labels, predicted_labels, average="weighted", zero_division=0)
            ),
            "binary_precision": float(precision),
            "binary_recall": float(recall),
            "binary_f1": float(f1),
            "binary_pr_auc": pr_auc,
            "benign_fp_rate": benign_fp_rate,
            "binary_margin_mean": float(np.mean(binary_margin)) if batch.size else 0.0,
            "attack_recall_macro": float(np.mean(attack_recalls)) if attack_recalls else 0.0,
        },
        per_class_recall=per_class_recall,
    )


def compute_continual_headline_metrics(
    *,
    true_class_indices: np.ndarray,
    predicted_class_indices: np.ndarray,
    true_binary: np.ndarray,
    predicted_binary: np.ndarray,
    attack_scores: np.ndarray,
    class_count: int,
    benign_index: int,
) -> ContinualHeadlineBundle:
    row: dict[str, float | int | str] = {
        "headline_binary_accuracy": float(accuracy_score(true_binary, predicted_binary)),
        "headline_binary_precision": float(precision_score(true_binary, predicted_binary, zero_division=0)),
        "headline_binary_recall": float(recall_score(true_binary, predicted_binary, zero_division=0)),
        "headline_binary_f1": float(f1_score(true_binary, predicted_binary, zero_division=0)),
        "headline_binary_auroc": 0.0,
        "headline_binary_auprc": 0.0,
        "headline_binary_specificity": 0.0,
        "headline_multiclass_accuracy": float(accuracy_score(true_class_indices, predicted_class_indices)),
        "headline_multiclass_macro_f1": float(
            f1_score(true_class_indices, predicted_class_indices, average="macro", zero_division=0)
        ),
        "headline_multiclass_weighted_f1": float(
            f1_score(true_class_indices, predicted_class_indices, average="weighted", zero_division=0)
        ),
        "headline_attack_recall_macro": 0.0,
        "headline_metric_basis": "aggregate_stream",
    }

    if np.unique(true_binary).size > 1:
        row["headline_binary_auroc"] = float(roc_auc_score(true_binary, attack_scores))
        row["headline_binary_auprc"] = float(average_precision_score(true_binary, attack_scores))

    benign_mask = true_binary == 0
    if np.any(benign_mask):
        row["headline_binary_specificity"] = float(np.mean(predicted_binary[benign_mask] == 0))

    attack_recalls: list[float] = []
    for class_index in range(class_count):
        if class_index == benign_index:
            continue
        mask = true_class_indices == class_index
        if np.any(mask):
            attack_recalls.append(float(np.mean(predicted_class_indices[mask] == class_index)))
    row["headline_attack_recall_macro"] = float(np.mean(attack_recalls)) if attack_recalls else 0.0
    return ContinualHeadlineBundle(row=row)


def detect_drift(
    history: Sequence[dict[str, float | int | str]],
    current: dict[str, float | int | str],
    config: DriftConfig,
) -> bool:
    if len(history) < config.history:
        return False
    recent = history[-config.history :]
    mean_f1 = float(np.mean([float(row["binary_f1"]) for row in recent]))
    mean_fp = float(np.mean([float(row["benign_fp_rate"]) for row in recent]))
    mean_margin = float(np.mean([float(row["binary_margin_mean"]) for row in recent]))
    return (
        float(current["binary_f1"]) <= mean_f1 - config.f1_drop_threshold
        and float(current["benign_fp_rate"]) >= mean_fp + config.fp_rate_rise_threshold
        and float(current["binary_margin_mean"]) <= mean_margin - config.margin_drop_threshold
    )


def compute_offline_metrics(
    *,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    true_binary: np.ndarray,
    predicted_binary: np.ndarray,
    attack_scores: np.ndarray,
    class_labels: list[str],
    benign_label: str,
    dataset: str,
    split_name: str,
    task_mode: str,
) -> OfflineMetricBundle:
    row: dict[str, float | int | str] = {
        "dataset": dataset,
        "split": split_name,
        "task_mode": task_mode,
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "binary_accuracy": float(accuracy_score(true_binary, predicted_binary)),
        "binary_precision": float(precision_score(true_binary, predicted_binary, zero_division=0)),
        "binary_recall": float(recall_score(true_binary, predicted_binary, zero_division=0)),
        "binary_f1": float(f1_score(true_binary, predicted_binary, zero_division=0)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "attack_recall_macro": 0.0,
        "binary_auroc": 0.0,
        "binary_auprc": 0.0,
    }
    if np.unique(true_binary).size > 1:
        row["binary_auroc"] = float(roc_auc_score(true_binary, attack_scores))
        row["binary_auprc"] = float(average_precision_score(true_binary, attack_scores))

    attack_recalls: list[float] = []
    for label in class_labels:
        if label == benign_label:
            continue
        mask = true_labels == label
        if np.any(mask):
            attack_recalls.append(float(np.mean(predicted_labels[mask] == label)))
    row["attack_recall_macro"] = float(np.mean(attack_recalls)) if attack_recalls else 0.0

    matrix = confusion_matrix(true_labels, predicted_labels, labels=class_labels)
    return OfflineMetricBundle(row=row, confusion=matrix.astype(np.int32))

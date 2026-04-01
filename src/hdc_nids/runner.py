"""Experiment orchestration."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import (
    EWCMLPModel,
    OfflineLSTMModel,
    OfflineSVMRBFModel,
    OnlineLSTMModel,
    OnlineSVMModel,
    StaticMLPModel,
    StaticSVMModel,
)
from .config import ExperimentConfig, load_config
from .data import DatasetStream, OfflineSplit, RawRecord, build_offline_split, build_stream
from .encoding import TabularHDCEncoder
from .metrics import compute_offline_metrics, compute_window_metrics, detect_drift
from .models import DualMemoryHDCModel, OnlineHDModel, PaperOnlineHDModel, StaticHDCModel
from .plots import plot_binary_f1, plot_drift_recovery, plot_forgetting, plot_latency
from .preprocessing import DenseFeatureSelector, PreparedBatch, TabularPreprocessor
from .utils import ensure_dir, json_dump, monotonic_ms


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _task_names(task_mode: str) -> list[str]:
    if task_mode == "binary":
        return ["binary"]
    if task_mode == "multiclass":
        return ["multiclass"]
    return ["binary", "multiclass"]


def _task_labels(task_name: str, benign_label: str, class_labels: list[str]) -> list[str]:
    if task_name == "binary":
        return [benign_label, "ATTACK"]
    return class_labels


def _collapse_records(records: list[RawRecord], task_name: str, benign_label: str) -> list[RawRecord]:
    if task_name != "binary":
        return list(records)
    collapsed: list[RawRecord] = []
    for record in records:
        collapsed.append(
            RawRecord(
                features=record.features,
                internal_label=benign_label if record.binary_label == 0 else "ATTACK",
                binary_label=record.binary_label,
                source=record.source,
                stage_name=record.stage_name,
                record_id=record.record_id,
            )
        )
    return collapsed


def _build_preprocessor(
    *,
    class_labels: list[str],
    benign_label: str,
    forced_categorical: set[str],
    numeric_transform: str,
    train_records: list[RawRecord],
) -> TabularPreprocessor:
    preprocessor = TabularPreprocessor(
        class_labels=class_labels,
        benign_label=benign_label,
        forced_categorical=forced_categorical,
        numeric_transform=numeric_transform,
    )
    preprocessor.fit(train_records)
    return preprocessor


def _build_feature_selector(config: ExperimentConfig, train_batch: PreparedBatch) -> DenseFeatureSelector:
    selector = DenseFeatureSelector(
        mode=config.feature_selection.mode,
        variance_threshold=config.feature_selection.variance_threshold,
        top_k=config.feature_selection.top_k,
    )
    selector.fit(train_batch.dense, train_batch.class_indices)
    return selector


def _build_online_model(config: ExperimentConfig, preprocessor: TabularPreprocessor) -> Any:
    if config.model_type == "continual_hdc":
        return PaperOnlineHDModel(
            input_dim=preprocessor.dense_dim,
            class_labels=preprocessor.class_labels,
            benign_label=preprocessor.benign_label,
            dim=config.hd_dim,
            seed=config.seed,
            learning_rate=config.hdc.learning_rate,
            bootstrap_fraction=config.hdc.bootstrap_fraction,
            batch_size=config.hdc.batch_size,
            scale_by_sqrt_features=config.hdc.scale_by_sqrt_features,
            row_normalize=config.hdc.row_normalize,
            regen_rate=config.regen_rate,
        )

    if config.model_type in {"static_hdc", "online_hdc", "dual_memory_hdc", "continual_hdc"}:
        encoder = TabularHDCEncoder(
            preprocessor,
            dim=config.hd_dim,
            bins=config.bins,
            seed=config.seed,
        )
        kwargs = dict(
            encoder=encoder,
            preprocessor=preprocessor,
            base_lr=config.base_lr,
            rare_class_boost=config.rare_class_boost,
            prototype_clip=config.prototype_clip,
        )
        if config.model_type == "static_hdc":
            return StaticHDCModel(**kwargs)
        if config.model_type == "online_hdc":
            return OnlineHDModel(**kwargs)
        return DualMemoryHDCModel(
            **kwargs,
            memory_mix=config.memory_mix,
            stagnation=config.stagnation,
            regen_rate=config.regen_rate,
            drift=config.drift,
        )

    if config.model_type == "static_mlp":
        return StaticMLPModel(
            preprocessor=preprocessor,
            hidden_dim=config.mlp.hidden_dim,
            learning_rate_init=config.mlp.learning_rate_init,
            alpha=config.mlp.alpha,
            max_iter=config.mlp.max_iter,
            seed=config.seed,
        )
    if config.model_type == "ewc_mlp":
        return EWCMLPModel(
            preprocessor=preprocessor,
            hidden_dim=config.mlp.hidden_dim,
            learning_rate_init=config.mlp.learning_rate_init,
            alpha=config.mlp.alpha,
            max_iter=config.mlp.max_iter,
            partial_fit_epochs=config.mlp.partial_fit_epochs,
            ewc_lambda=config.mlp.ewc_lambda,
            seed=config.seed,
        )
    if config.model_type == "static_svm":
        return StaticSVMModel(
            preprocessor=preprocessor,
            alpha=config.svm.alpha,
            max_iter=config.svm.max_iter,
            seed=config.seed,
        )
    if config.model_type in {"online_svm", "continual_svm"}:
        return OnlineSVMModel(
            preprocessor=preprocessor,
            alpha=config.svm.alpha,
            max_iter=config.svm.max_iter,
            partial_fit_epochs=config.svm.partial_fit_epochs,
            seed=config.seed,
        )
    if config.model_type in {"online_lstm", "continual_lstm"}:
        return OnlineLSTMModel(
            preprocessor=preprocessor,
            hidden_dim=config.lstm.hidden_dim,
            sequence_length=config.lstm.sequence_length,
            learning_rate=config.lstm.learning_rate,
            epochs=config.lstm.epochs,
            batch_size=config.lstm.batch_size,
            gradient_clip=config.lstm.gradient_clip,
            update_sample_limit=config.lstm.update_sample_limit,
            seed=config.seed,
        )
    raise ValueError(f"Unsupported continual model type: {config.model_type}")


def _checkpoint_if_needed(model: Any, checkpoints_dir: Path, window_id: int) -> None:
    if window_id % 10 == 0:
        model.checkpoint(checkpoints_dir / f"window_{window_id:04d}.npz")


def _run_continual_experiment(config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")

    training_start = monotonic_ms()
    stream: DatasetStream = build_stream(
        config.dataset,
        data_dir=config.data_dir,
        warmup_size=config.warmup_size,
        window_size=config.window_size,
        seed=config.seed,
    )
    preprocessor_fit_start = monotonic_ms()
    preprocessor = TabularPreprocessor(
        class_labels=stream.class_labels,
        benign_label=stream.benign_label,
        forced_categorical=stream.forced_categorical,
        numeric_transform=stream.numeric_transform,
    )
    preprocessor.fit(stream.warmup_records)
    preprocessor_fit_ms = monotonic_ms() - preprocessor_fit_start
    warmup_transform_start = monotonic_ms()
    warmup_batch = preprocessor.transform_records(
        stream.warmup_records,
        dataset=stream.dataset,
        window_id=-1,
        stage_name="warmup",
    )
    warmup_transform_ms = monotonic_ms() - warmup_transform_start

    model_build_start = monotonic_ms()
    model = _build_online_model(config, preprocessor)
    model_build_ms = monotonic_ms() - model_build_start
    initial_fit_start = monotonic_ms()
    model.fit_initial(warmup_batch)
    initial_fit_ms = monotonic_ms() - initial_fit_start
    training_total_ms = monotonic_ms() - training_start
    model.checkpoint(checkpoints_dir / "warmup.npz")

    metric_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    best_attack_recall = {label: 0.0 for label in preprocessor.class_labels if label != preprocessor.benign_label}
    drift_cooloff = 0

    for window in stream.window_factory():
        batch = preprocessor.transform_window(window)
        if batch.size == 0:
            continue

        prediction = model.predict(batch)
        bundle = compute_window_metrics(
            batch=batch,
            predicted_labels=prediction.predicted_labels,
            predicted_binary=prediction.predicted_binary,
            attack_scores=prediction.attack_scores,
            binary_margin=prediction.binary_margin,
            preprocessor=preprocessor,
        )
        metrics_row = bundle.row
        forgetting_values: list[float] = []
        for label, recall in bundle.per_class_recall.items():
            best_attack_recall[label] = max(best_attack_recall[label], recall)
            forgetting_values.append(best_attack_recall[label] - recall)
        metrics_row["benchmark_mode"] = "continual"
        metrics_row["task_mode"] = "both"
        metrics_row["avg_forgetting"] = float(np.mean(forgetting_values)) if forgetting_values else 0.0
        metrics_row["model_size_bytes"] = model.model_size_bytes()

        drift_active = False
        if config.model_type in {"dual_memory_hdc", "continual_hdc"}:
            if detect_drift(metric_rows, metrics_row, config.drift):
                drift_active = True
                drift_cooloff = config.drift.cooloff_windows
                drift_rows.append(
                    {
                        "window_id": metrics_row["window_id"],
                        "dataset": metrics_row["dataset"],
                        "stage_name": metrics_row["stage_name"],
                        "event_type": "drift_trigger",
                        "binary_f1": metrics_row["binary_f1"],
                        "benign_fp_rate": metrics_row["benign_fp_rate"],
                        "binary_margin_mean": metrics_row["binary_margin_mean"],
                    }
                )
            elif drift_cooloff > 0:
                drift_active = True
                drift_cooloff -= 1

        update_start = monotonic_ms()
        drift_rows.extend(model.update(batch, prediction, drift_active=drift_active))
        update_ms = monotonic_ms() - update_start
        metric_rows.append(metrics_row)
        drift_rows.extend(model.observe_window(metric_rows[:-1], metrics_row))
        _checkpoint_if_needed(model, checkpoints_dir, int(metrics_row["window_id"]))

        latency_rows.append(
            {
                "benchmark_mode": "continual",
                "task_mode": "both",
                "window_id": metrics_row["window_id"],
                "dataset": metrics_row["dataset"],
                "stage_name": metrics_row["stage_name"],
                "row_count": batch.size,
                "encode_ms": prediction.encode_ms,
                "score_ms": prediction.score_ms,
                "update_ms": update_ms,
                "total_ms": prediction.encode_ms + prediction.score_ms + update_ms,
                "per_1k_rows_ms": (prediction.encode_ms + prediction.score_ms + update_ms) * 1000.0 / batch.size,
                "inference_per_1k_rows_ms": (prediction.encode_ms + prediction.score_ms) * 1000.0 / batch.size,
                "model_size_bytes": model.model_size_bytes(),
            }
        )

    model.checkpoint(checkpoints_dir / "final.npz")
    _write_csv(run_dir / "metrics.csv", metric_rows)
    _write_csv(run_dir / "latency.csv", latency_rows)
    _write_csv(run_dir / "drift_events.csv", drift_rows)
    _write_csv(
        run_dir / "training.csv",
        [
            {
                "benchmark_mode": "continual",
                "task_mode": "both",
                "dataset": config.dataset,
                "experiment_name": config.experiment_name,
                "model_type": config.model_type,
                "warmup_rows": warmup_batch.size,
                "preprocessor_fit_ms": preprocessor_fit_ms,
                "warmup_transform_ms": warmup_transform_ms,
                "model_build_ms": model_build_ms,
                "initial_fit_ms": initial_fit_ms,
                "training_total_ms": training_total_ms,
                "training_per_1k_warmup_rows_ms": training_total_ms * 1000.0 / max(warmup_batch.size, 1),
            }
        ],
    )

    if metric_rows:
        plot_binary_f1(metric_rows, run_dir / "binary_f1_over_time.png")
        plot_forgetting(metric_rows, run_dir / "per_family_forgetting.png")
        plot_drift_recovery(metric_rows, drift_rows, run_dir / "drift_recovery.png")
    if latency_rows:
        plot_latency(latency_rows, run_dir / "model_size_latency.png")

    summary = {
        "benchmark_mode": "continual",
        "experiment_name": config.experiment_name,
        "dataset": config.dataset,
        "model_type": config.model_type,
        "windows": len(metric_rows),
        "mean_binary_f1": float(np.mean([row["binary_f1"] for row in metric_rows])) if metric_rows else 0.0,
        "mean_multiclass_macro_f1": float(np.mean([row["multiclass_macro_f1"] for row in metric_rows])) if metric_rows else 0.0,
        "mean_attack_recall_macro": float(np.mean([row["attack_recall_macro"] for row in metric_rows])) if metric_rows else 0.0,
        "mean_avg_forgetting": float(np.mean([row["avg_forgetting"] for row in metric_rows])) if metric_rows else 0.0,
        "mean_inference_per_1k_rows_ms": float(np.mean([row["inference_per_1k_rows_ms"] for row in latency_rows])) if latency_rows else 0.0,
        "mean_per_1k_rows_ms": float(np.mean([row["per_1k_rows_ms"] for row in latency_rows])) if latency_rows else 0.0,
        "warmup_rows": warmup_batch.size,
        "preprocessor_fit_ms": preprocessor_fit_ms,
        "warmup_transform_ms": warmup_transform_ms,
        "model_build_ms": model_build_ms,
        "initial_fit_ms": initial_fit_ms,
        "training_total_ms": training_total_ms,
        "training_per_1k_warmup_rows_ms": training_total_ms * 1000.0 / max(warmup_batch.size, 1),
        "drift_event_count": len([row for row in drift_rows if row.get("event_type") == "drift_trigger"]),
    }
    json_dump(run_dir / "summary.json", summary)
    return summary


def _build_hdc_model(
    config: ExperimentConfig,
    preprocessor: TabularPreprocessor,
    *,
    input_dim: int,
    tuned: bool,
) -> Any:
    _ = tuned
    return PaperOnlineHDModel(
        input_dim=input_dim,
        class_labels=preprocessor.class_labels,
        benign_label=preprocessor.benign_label,
        dim=config.hd_dim,
        seed=config.seed,
        learning_rate=config.hdc.learning_rate,
        bootstrap_fraction=config.hdc.bootstrap_fraction,
        batch_size=config.hdc.batch_size,
        scale_by_sqrt_features=config.hdc.scale_by_sqrt_features,
        row_normalize=config.hdc.row_normalize,
        regen_rate=config.regen_rate,
    )


def _offline_selection_score(metrics_row: dict[str, Any], task_name: str) -> float:
    if task_name == "binary":
        return float(metrics_row["binary_f1"])
    return float(metrics_row["macro_f1"])


def _evaluate_offline_batch(
    *,
    model: Any,
    batch: PreparedBatch,
    preprocessor: TabularPreprocessor,
    dataset: str,
    split_name: str,
    task_name: str,
) -> tuple[dict[str, Any], dict[str, Any], list[list[int]]]:
    prediction = model.predict(batch)
    metrics = compute_offline_metrics(
        true_labels=batch.internal_labels,
        predicted_labels=prediction.predicted_labels,
        true_binary=batch.binary_labels,
        predicted_binary=prediction.predicted_binary,
        attack_scores=prediction.attack_scores,
        class_labels=preprocessor.class_labels,
        benign_label=preprocessor.benign_label,
        dataset=dataset,
        split_name=split_name,
        task_mode=task_name,
    )
    metric_row = dict(metrics.row)
    metric_row["benchmark_mode"] = "offline"
    metric_row["row_count"] = batch.size
    metric_row["model_size_bytes"] = model.model_size_bytes()
    latency_row = {
        "benchmark_mode": "offline",
        "task_mode": task_name,
        "dataset": dataset,
        "split": split_name,
        "row_count": batch.size,
        "latency_mode": "inference_only",
        "inference_ms": prediction.encode_ms + prediction.score_ms,
        "per_1k_rows_ms": (prediction.encode_ms + prediction.score_ms) * 1000.0 / max(batch.size, 1),
        "headline_per_1k_rows_ms": (prediction.encode_ms + prediction.score_ms) * 1000.0 / max(batch.size, 1),
        "model_size_bytes": model.model_size_bytes(),
    }
    return metric_row, latency_row, metrics.confusion.tolist()


def _fit_offline_hdc(
    *,
    config: ExperimentConfig,
    preprocessor: TabularPreprocessor,
    train_batch: PreparedBatch,
    val_batch: PreparedBatch,
    task_name: str,
    tuned: bool,
) -> tuple[Any, dict[str, Any]]:
    if not tuned:
        model = _build_hdc_model(config, preprocessor, input_dim=train_batch.dense.shape[1], tuned=False)
        model.fit_initial(train_batch)
        return model, {
            "variant": "paper_onlinehd_one_pass",
            "learning_rate": config.hdc.learning_rate,
            "bootstrap_fraction": config.hdc.bootstrap_fraction,
        }

    best_model = None
    best_state = None
    best_score = -1.0
    best_epoch = 0
    best_regen_rounds = 0
    for epoch_count in config.hdc.iterative_epoch_candidates:
        model = _build_hdc_model(config, preprocessor, input_dim=train_batch.dense.shape[1], tuned=True)
        model.fit_initial(train_batch)
        if epoch_count > 0:
            model.fit_iterative(train_batch, epochs=epoch_count)

        regen_applied = 0
        for _ in range(config.hdc.regeneration_rounds):
            dims = model.regenerate_low_variance_dimensions()
            if dims.size == 0:
                break
            regen_applied += 1
            extra_epochs = max(1, epoch_count // max(config.hdc.regeneration_rounds, 1))
            model.fit_iterative(train_batch, epochs=extra_epochs)

        val_prediction = model.predict(val_batch)
        metrics = compute_offline_metrics(
            true_labels=val_batch.internal_labels,
            predicted_labels=val_prediction.predicted_labels,
            true_binary=val_batch.binary_labels,
            predicted_binary=val_prediction.predicted_binary,
            attack_scores=val_prediction.attack_scores,
            class_labels=preprocessor.class_labels,
            benign_label=preprocessor.benign_label,
            dataset=val_batch.dataset,
            split_name="val",
            task_mode=task_name,
        )
        score = _offline_selection_score(metrics.row, task_name)
        if score > best_score:
            best_model = model
            best_state = model.state_dict()
            best_score = score
            best_epoch = epoch_count
            best_regen_rounds = regen_applied

    if best_model is not None and best_state is not None:
        best_model.load_state_dict(best_state)
    return best_model, {
        "variant": "paper_cyberhd_tuned",
        "selected_epochs": best_epoch,
        "regeneration_rounds": best_regen_rounds,
        "validation_score": best_score,
        "learning_rate": config.hdc.learning_rate,
    }


def _fit_offline_svm(
    *,
    config: ExperimentConfig,
    preprocessor: TabularPreprocessor,
    train_batch: PreparedBatch,
    val_batch: PreparedBatch,
    task_name: str,
) -> tuple[Any, dict[str, Any]]:
    best_model = None
    best_score = -1.0
    best_params: dict[str, Any] = {}
    for c_value in config.svm.c_values:
        for gamma_value in config.svm.gamma_values:
            model = OfflineSVMRBFModel(
                preprocessor=preprocessor,
                c_value=float(c_value),
                gamma_value=gamma_value,
                seed=config.seed,
            )
            model.fit_initial(train_batch)
            val_prediction = model.predict(val_batch)
            metrics = compute_offline_metrics(
                true_labels=val_batch.internal_labels,
                predicted_labels=val_prediction.predicted_labels,
                true_binary=val_batch.binary_labels,
                predicted_binary=val_prediction.predicted_binary,
                attack_scores=val_prediction.attack_scores,
                class_labels=preprocessor.class_labels,
                benign_label=preprocessor.benign_label,
                dataset=val_batch.dataset,
                split_name="val",
                task_mode=task_name,
            )
            score = _offline_selection_score(metrics.row, task_name)
            if score > best_score:
                best_model = model
                best_score = score
                best_params = {"c": float(c_value), "gamma": gamma_value, "validation_score": score}
    return best_model, best_params


def _fit_offline_lstm(
    *,
    config: ExperimentConfig,
    preprocessor: TabularPreprocessor,
    train_batch: PreparedBatch,
    val_batch: PreparedBatch,
) -> tuple[Any, dict[str, Any]]:
    model = OfflineLSTMModel(
        preprocessor=preprocessor,
        hidden_dim=config.lstm.hidden_dim,
        learning_rate=config.lstm.learning_rate,
        batch_size=config.lstm.batch_size,
        dropout=config.lstm.dropout,
        max_epochs=config.lstm.max_epochs,
        patience=config.lstm.patience,
        segment_count=config.lstm.segment_count,
        seed=config.seed,
    )
    best_score = model.fit_with_validation(train_batch, val_batch)
    return model, {"hidden_dim": config.lstm.hidden_dim, "validation_score": best_score}


def _write_split_manifest(run_dir: Path, split: OfflineSplit) -> None:
    rows: list[dict[str, Any]] = []
    for split_name, records in (
        ("train", split.train_records),
        ("val", split.val_records),
        ("test", split.test_records),
    ):
        for record in records:
            rows.append(
                {
                    "record_id": record.record_id,
                    "dataset": split.dataset,
                    "split": split_name,
                    "source": record.source,
                    "label": record.internal_label,
                    "binary_label": record.binary_label,
                }
            )
    _write_csv(run_dir / "split_manifest.csv", rows)


def _apply_latency_mode(
    *,
    config: ExperimentConfig,
    latency_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
) -> None:
    training_by_task = {row["task_mode"]: row for row in training_rows}
    for row in latency_rows:
        row["latency_mode"] = config.latency_mode
        if config.latency_mode == "end_to_end":
            training_row = training_by_task.get(row["task_mode"])
            extra_ms = 0.0
            if training_row is not None:
                extra_ms = (
                    float(training_row.get("preprocessor_fit_ms", 0.0))
                    + float(training_row.get("warmup_transform_ms", 0.0))
                ) / max(float(training_row.get("warmup_rows", 1.0)), 1.0)
            row["headline_per_1k_rows_ms"] = float(row["per_1k_rows_ms"]) + extra_ms * 1000.0
        else:
            row["headline_per_1k_rows_ms"] = float(row["per_1k_rows_ms"])


def _offline_summary_headline(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    test_binary = next((row for row in metric_rows if row["split"] == "test" and row["task_mode"] == "binary"), None)
    test_multiclass = next((row for row in metric_rows if row["split"] == "test" and row["task_mode"] == "multiclass"), None)
    summary: dict[str, Any] = {}
    if test_binary:
        summary["headline_binary_accuracy"] = test_binary["binary_accuracy"]
        summary["headline_binary_f1"] = test_binary["binary_f1"]
        summary["headline_binary_auroc"] = test_binary["binary_auroc"]
    if test_multiclass:
        summary["headline_multiclass_accuracy"] = test_multiclass["accuracy"]
        summary["headline_multiclass_macro_f1"] = test_multiclass["macro_f1"]
    return summary


def _sanity_flags(config: ExperimentConfig, metric_rows: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    if config.dataset == "cicids2017":
        binary_test = next((row for row in metric_rows if row["split"] == "test" and row["task_mode"] == "binary"), None)
        if binary_test and config.model_type in {"offline_svm_rbf", "offline_lstm"} and float(binary_test["binary_accuracy"]) < 0.8:
            flags.append("offline_cicids_binary_accuracy_below_sanity_band")
    return flags


def _run_offline_experiment(config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    split = build_offline_split(
        config.dataset,
        data_dir=config.data_dir,
        validation_fraction=config.validation_fraction,
        split_strategy=config.split_strategy,
        row_limits=config.row_limits,
        seed=config.seed,
    )
    _write_split_manifest(run_dir, split)

    metric_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    confusion_payload: dict[str, Any] = {}

    for task_name in _task_names(config.task_mode):
        train_records = _collapse_records(split.train_records, task_name, split.benign_label)
        val_records = _collapse_records(split.val_records, task_name, split.benign_label)
        test_records = _collapse_records(split.test_records, task_name, split.benign_label)
        task_labels = _task_labels(task_name, split.benign_label, split.class_labels)

        training_start = monotonic_ms()
        preprocessor_fit_start = monotonic_ms()
        preprocessor = _build_preprocessor(
            class_labels=task_labels,
            benign_label=split.benign_label,
            forced_categorical=split.forced_categorical,
            numeric_transform=split.numeric_transform,
            train_records=train_records,
        )
        preprocessor_fit_ms = monotonic_ms() - preprocessor_fit_start

        transform_start = monotonic_ms()
        train_batch = preprocessor.transform_records(train_records, dataset=split.dataset, window_id=0, stage_name="train")
        val_batch = preprocessor.transform_records(val_records, dataset=split.dataset, window_id=1, stage_name="val")
        test_batch = preprocessor.transform_records(test_records, dataset=split.dataset, window_id=2, stage_name="test")
        transform_ms = monotonic_ms() - transform_start

        selector = _build_feature_selector(config, train_batch)
        if config.model_type in {"offline_svm_rbf", "offline_lstm", "offline_hdc_one_pass", "offline_hdc_tuned"} and selector.selected_indices is not None:
            train_batch = selector.transform_batch(train_batch)
            val_batch = selector.transform_batch(val_batch)
            test_batch = selector.transform_batch(test_batch)

        fit_start = monotonic_ms()
        if config.model_type == "offline_hdc_one_pass":
            model, fit_meta = _fit_offline_hdc(
                config=config,
                preprocessor=preprocessor,
                train_batch=train_batch,
                val_batch=val_batch,
                task_name=task_name,
                tuned=False,
            )
        elif config.model_type == "offline_hdc_tuned":
            model, fit_meta = _fit_offline_hdc(
                config=config,
                preprocessor=preprocessor,
                train_batch=train_batch,
                val_batch=val_batch,
                task_name=task_name,
                tuned=True,
            )
        elif config.model_type == "offline_svm_rbf":
            model, fit_meta = _fit_offline_svm(
                config=config,
                preprocessor=preprocessor,
                train_batch=train_batch,
                val_batch=val_batch,
                task_name=task_name,
            )
        elif config.model_type == "offline_lstm":
            model, fit_meta = _fit_offline_lstm(
                config=config,
                preprocessor=preprocessor,
                train_batch=train_batch,
                val_batch=val_batch,
            )
        else:
            raise ValueError(f"Unsupported offline model type: {config.model_type}")
        initial_fit_ms = monotonic_ms() - fit_start
        training_total_ms = monotonic_ms() - training_start

        model.checkpoint(checkpoints_dir / f"{task_name}_final.npz")
        for split_name, batch in (("val", val_batch), ("test", test_batch)):
            metric_row, latency_row, confusion = _evaluate_offline_batch(
                model=model,
                batch=batch,
                preprocessor=preprocessor,
                dataset=split.dataset,
                split_name=split_name,
                task_name=task_name,
            )
            metric_row["model_type"] = config.model_type
            latency_row["model_type"] = config.model_type
            metric_rows.append(metric_row)
            latency_rows.append(latency_row)
            confusion_payload[f"{task_name}_{split_name}"] = {
                "labels": preprocessor.class_labels,
                "matrix": confusion,
            }

        training_rows.append(
            {
                "benchmark_mode": "offline",
                "task_mode": task_name,
                "dataset": split.dataset,
                "experiment_name": config.experiment_name,
                "model_type": config.model_type,
                "warmup_rows": train_batch.size,
                "preprocessor_fit_ms": preprocessor_fit_ms,
                "warmup_transform_ms": transform_ms,
                "model_build_ms": 0.0,
                "initial_fit_ms": initial_fit_ms,
                "training_total_ms": training_total_ms,
                "training_per_1k_warmup_rows_ms": training_total_ms * 1000.0 / max(train_batch.size, 1),
                "feature_count": int(train_batch.dense.shape[1]),
                "selected_params": json.dumps(fit_meta, sort_keys=True),
            }
        )

    _apply_latency_mode(config=config, latency_rows=latency_rows, training_rows=training_rows)
    _write_csv(run_dir / "metrics.csv", metric_rows)
    _write_csv(run_dir / "latency.csv", latency_rows)
    _write_csv(run_dir / "training.csv", training_rows)
    json_dump(run_dir / "confusion_matrices.json", confusion_payload)

    summary = {
        "benchmark_mode": "offline",
        "experiment_name": config.experiment_name,
        "dataset": split.dataset,
        "model_type": config.model_type,
        "split_strategy": split.split_strategy,
        **_offline_summary_headline(metric_rows),
        "latency_mode": config.latency_mode,
        "mean_inference_per_1k_rows_ms": float(np.mean([row["per_1k_rows_ms"] for row in latency_rows])) if latency_rows else 0.0,
        "mean_headline_per_1k_rows_ms": float(np.mean([row["headline_per_1k_rows_ms"] for row in latency_rows])) if latency_rows else 0.0,
        "training_total_ms": float(np.mean([row["training_total_ms"] for row in training_rows])) if training_rows else 0.0,
        "training_per_1k_warmup_rows_ms": float(
            np.mean([row["training_per_1k_warmup_rows_ms"] for row in training_rows])
        ) if training_rows else 0.0,
        "sanity_flags": _sanity_flags(config, metric_rows),
    }
    json_dump(run_dir / "summary.json", summary)
    return summary


def run_experiment(config_or_path: ExperimentConfig | str | Path) -> dict[str, Any]:
    config = load_config(config_or_path) if not isinstance(config_or_path, ExperimentConfig) else config_or_path
    if config.benchmark_mode == "both":
        base_dir = ensure_dir(config.output_dir / config.experiment_name)
        offline_summary = _run_offline_experiment(deepcopy(config), ensure_dir(base_dir / "offline"))
        continual_config = deepcopy(config)
        continual_config.benchmark_mode = "continual"
        continual_summary = _run_continual_experiment(continual_config, ensure_dir(base_dir / "continual"))
        combined = {"offline": offline_summary, "continual": continual_summary}
        json_dump(base_dir / "summary.json", combined)
        return combined

    run_dir = ensure_dir(config.output_dir / config.experiment_name)
    if config.benchmark_mode == "offline":
        return _run_offline_experiment(config, run_dir)
    return _run_continual_experiment(config, run_dir)

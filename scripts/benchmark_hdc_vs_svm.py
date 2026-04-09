#!/usr/bin/env python3
"""Compare experiment outputs by benchmark track and task."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _reconstruct_binary_counts(metric_row: dict[str, str]) -> tuple[float, float, float, float]:
    row_count = float(metric_row["row_count"])
    accuracy = float(metric_row["binary_accuracy"])
    recall = float(metric_row["binary_recall"])
    benign_fp_rate = float(metric_row["benign_fp_rate"])
    denominator = recall + benign_fp_rate - 1.0

    if abs(denominator) > 1e-9:
        positive_rate = (accuracy - (1.0 - benign_fp_rate)) / denominator
    else:
        precision = float(metric_row["binary_precision"])
        fallback = recall * (1.0 - precision) + precision * benign_fp_rate
        positive_rate = _safe_ratio(precision * benign_fp_rate, fallback) if abs(fallback) > 1e-9 else 0.0

    positive_rate = min(max(positive_rate, 0.0), 1.0)
    positive_count = positive_rate * row_count
    negative_count = row_count - positive_count
    tp = recall * positive_count
    fn = positive_count - tp
    fp = benign_fp_rate * negative_count
    tn = negative_count - fp
    return tp, fp, fn, tn


def _continual_binary_fallback_scope(
    metrics_rows: list[dict[str, str]],
    latency_rows: list[dict[str, str]],
    training_row: dict[str, str],
) -> dict:
    tp = fp = fn = tn = 0.0
    attack_recall_sum = 0.0
    attack_recall_n = 0
    total_rows = 0.0
    total_inference_ms = 0.0
    total_per_1k_ms = 0.0

    for row in metrics_rows:
        row_tp, row_fp, row_fn, row_tn = _reconstruct_binary_counts(row)
        tp += row_tp
        fp += row_fp
        fn += row_fn
        tn += row_tn
        attack_recall_sum += float(row["attack_recall_macro"])
        attack_recall_n += 1
        total_rows += float(row["row_count"])

    for row in latency_rows:
        row_count = float(row["row_count"])
        total_inference_ms += float(row["inference_per_1k_rows_ms"]) * row_count / 1000.0
        total_per_1k_ms += float(row["per_1k_rows_ms"]) * row_count / 1000.0

    accuracy = _safe_ratio(tp + tn, tp + tn + fp + fn)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    binary_f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    return {
        "benchmark_mode": "continual",
        "task_mode": "binary",
        "dataset": metrics_rows[0]["dataset"] if metrics_rows else "unknown",
        "accuracy": accuracy,
        "binary_f1": binary_f1,
        "macro_f1": binary_f1,
        "weighted_f1": binary_f1,
        "attack_recall_macro": attack_recall_sum / max(attack_recall_n, 1),
        "latency_mode": "inference_only_stream_aggregate_fallback",
        "inference_per_1k_rows_ms": _safe_ratio(total_inference_ms * 1000.0, total_rows),
        "training_total_ms": float(training_row["training_total_ms"]),
        "training_per_1k_rows_ms": float(training_row["training_per_1k_warmup_rows_ms"]),
    }


def load_scope(run_dir: Path, benchmark_mode: str, task_mode: str, split_name: str) -> dict:
    metrics_rows = _load_csv(run_dir / "metrics.csv")
    latency_rows = _load_csv(run_dir / "latency.csv")
    training_rows = _load_csv(run_dir / "training.csv")
    summary = _load_json(run_dir / "summary.json")

    if benchmark_mode == "offline":
        metric_row = next(
            row for row in metrics_rows if row.get("benchmark_mode") == "offline" and row["task_mode"] == task_mode and row["split"] == split_name
        )
        latency_row = next(
            row for row in latency_rows if row.get("benchmark_mode") == "offline" and row["task_mode"] == task_mode and row["split"] == split_name
        )
        training_row = next(
            row for row in training_rows if row.get("benchmark_mode") == "offline" and row["task_mode"] == task_mode
        )
        return {
            "benchmark_mode": "offline",
            "task_mode": task_mode,
            "dataset": metric_row["dataset"],
            "accuracy": float(metric_row["binary_accuracy"] if task_mode == "binary" else metric_row["accuracy"]),
            "binary_f1": float(metric_row["binary_f1"]),
            "macro_f1": float(metric_row["macro_f1"]),
            "weighted_f1": float(metric_row["weighted_f1"]),
            "attack_recall_macro": float(metric_row["attack_recall_macro"]),
            "latency_mode": latency_row.get("latency_mode", "inference_only"),
            "inference_per_1k_rows_ms": float(latency_row.get("headline_per_1k_rows_ms", latency_row["per_1k_rows_ms"])),
            "training_total_ms": float(training_row["training_total_ms"]),
            "training_per_1k_rows_ms": float(training_row["training_per_1k_warmup_rows_ms"]),
        }

    filtered_metrics = [row for row in metrics_rows if row.get("benchmark_mode") == "continual"]
    filtered_latency = [row for row in latency_rows if row.get("benchmark_mode") == "continual"]
    training_row = next(row for row in training_rows if row.get("benchmark_mode") == "continual")

    def mean(rows: list[dict[str, str]], key: str) -> float:
        return sum(float(row[key]) for row in rows) / max(len(rows), 1)

    if summary.get("headline_metric_basis") == "aggregate_stream":
        if task_mode == "binary":
            return {
                "benchmark_mode": "continual",
                "task_mode": task_mode,
                "dataset": summary.get("dataset", "unknown"),
                "accuracy": float(summary["headline_binary_accuracy"]),
                "binary_f1": float(summary["headline_binary_f1"]),
                "macro_f1": float(summary["headline_binary_f1"]),
                "weighted_f1": float(summary["headline_binary_f1"]),
                "attack_recall_macro": float(summary["headline_attack_recall_macro"]),
                "latency_mode": "inference_only_stream_aggregate",
                "inference_per_1k_rows_ms": float(summary.get("headline_inference_per_1k_rows_ms", summary.get("mean_inference_per_1k_rows_ms", 0.0))),
                "training_total_ms": float(summary["training_total_ms"]),
                "training_per_1k_rows_ms": float(summary["training_per_1k_warmup_rows_ms"]),
            }
        return {
            "benchmark_mode": "continual",
            "task_mode": task_mode,
            "dataset": summary.get("dataset", "unknown"),
            "accuracy": float(summary["headline_multiclass_accuracy"]),
            "binary_f1": float(summary["headline_binary_f1"]),
            "macro_f1": float(summary["headline_multiclass_macro_f1"]),
            "weighted_f1": float(summary["headline_multiclass_weighted_f1"]),
            "attack_recall_macro": float(summary["headline_attack_recall_macro"]),
            "latency_mode": "inference_only_stream_aggregate",
            "inference_per_1k_rows_ms": float(summary.get("headline_inference_per_1k_rows_ms", summary.get("mean_inference_per_1k_rows_ms", 0.0))),
            "training_total_ms": float(summary["training_total_ms"]),
            "training_per_1k_rows_ms": float(summary["training_per_1k_warmup_rows_ms"]),
        }

    if task_mode == "binary":
        return _continual_binary_fallback_scope(filtered_metrics, filtered_latency, training_row)

    accuracy_key = "binary_accuracy" if task_mode == "binary" else "multiclass_accuracy"
    if filtered_metrics and accuracy_key not in filtered_metrics[0]:
        accuracy_key = "binary_f1" if task_mode == "binary" else "multiclass_macro_f1"

    return {
        "benchmark_mode": "continual",
        "task_mode": task_mode,
        "dataset": filtered_metrics[0]["dataset"] if filtered_metrics else "unknown",
        "accuracy": mean(filtered_metrics, accuracy_key),
        "binary_f1": mean(filtered_metrics, "binary_f1"),
        "macro_f1": mean(filtered_metrics, "binary_f1" if task_mode == "binary" else "multiclass_macro_f1"),
        "weighted_f1": mean(filtered_metrics, "binary_f1" if task_mode == "binary" else "multiclass_weighted_f1"),
        "attack_recall_macro": mean(filtered_metrics, "attack_recall_macro"),
        "inference_per_1k_rows_ms": mean(filtered_latency, "inference_per_1k_rows_ms"),
        "training_total_ms": float(training_row["training_total_ms"]),
        "training_per_1k_rows_ms": float(training_row["training_per_1k_warmup_rows_ms"]),
    }


def compare(hdc_dir: Path, baseline_dir: Path, benchmark_mode: str, task_mode: str, split_name: str) -> dict:
    hdc = load_scope(hdc_dir, benchmark_mode, task_mode, split_name)
    baseline = load_scope(baseline_dir, benchmark_mode, task_mode, split_name)
    result = {
        "dataset": hdc["dataset"],
        "benchmark_mode": benchmark_mode,
        "task_mode": task_mode,
        "split": split_name if benchmark_mode == "offline" else "aggregate",
        "latency_mode": hdc.get("latency_mode", "streaming"),
        "hdc_experiment": hdc_dir.name,
        "svm_experiment": baseline_dir.name,
        "hdc_accuracy": hdc["accuracy"],
        "svm_accuracy": baseline["accuracy"],
        "accuracy_delta": hdc["accuracy"] - baseline["accuracy"],
        "hdc_binary_f1": hdc["binary_f1"],
        "svm_binary_f1": baseline["binary_f1"],
        "binary_f1_delta": hdc["binary_f1"] - baseline["binary_f1"],
        "hdc_macro_f1": hdc["macro_f1"],
        "svm_macro_f1": baseline["macro_f1"],
        "macro_f1_delta": hdc["macro_f1"] - baseline["macro_f1"],
        "hdc_attack_recall_macro": hdc["attack_recall_macro"],
        "svm_attack_recall_macro": baseline["attack_recall_macro"],
        "attack_recall_delta": hdc["attack_recall_macro"] - baseline["attack_recall_macro"],
        "hdc_inference_per_1k_rows_ms": hdc["inference_per_1k_rows_ms"],
        "svm_inference_per_1k_rows_ms": baseline["inference_per_1k_rows_ms"],
        "latency_speedup_vs_svm": baseline["inference_per_1k_rows_ms"] / max(hdc["inference_per_1k_rows_ms"], 1e-9),
        "hdc_training_total_ms": hdc["training_total_ms"],
        "svm_training_total_ms": baseline["training_total_ms"],
        "training_time_delta_ms": hdc["training_total_ms"] - baseline["training_total_ms"],
        "training_speedup_vs_svm": baseline["training_total_ms"] / max(hdc["training_total_ms"], 1e-9),
        "sanity_flags": [],
    }
    if benchmark_mode == "offline" and task_mode == "binary" and hdc_dir.name.endswith("hdc_one_pass") and hdc["inference_per_1k_rows_ms"] > baseline["inference_per_1k_rows_ms"]:
        result["sanity_flags"].append("hdc_one_pass_slower_than_baseline_inference")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HDC against SVM")
    parser.add_argument("--hdc-dir", type=Path, required=True)
    parser.add_argument("--svm-dir", type=Path, required=True)
    parser.add_argument("--benchmark-mode", choices=["offline", "continual"], required=True)
    parser.add_argument("--task-mode", choices=["binary", "multiclass"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = compare(args.hdc_dir, args.svm_dir, args.benchmark_mode, args.task_mode, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

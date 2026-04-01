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


def load_scope(run_dir: Path, benchmark_mode: str, task_mode: str, split_name: str) -> dict:
    metrics_rows = _load_csv(run_dir / "metrics.csv")
    latency_rows = _load_csv(run_dir / "latency.csv")
    training_rows = _load_csv(run_dir / "training.csv")

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

#!/usr/bin/env python3
"""Compare HDC and LSTM experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_hdc_vs_svm import compare


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HDC against LSTM")
    parser.add_argument("--hdc-dir", type=Path, required=True)
    parser.add_argument("--lstm-dir", type=Path, required=True)
    parser.add_argument("--benchmark-mode", choices=["offline", "continual"], required=True)
    parser.add_argument("--task-mode", choices=["binary", "multiclass"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = compare(args.hdc_dir, args.lstm_dir, args.benchmark_mode, args.task_mode, args.split)
    result["lstm_experiment"] = result.pop("svm_experiment")
    result["lstm_accuracy"] = result.pop("svm_accuracy")
    result["lstm_binary_f1"] = result.pop("svm_binary_f1")
    result["lstm_macro_f1"] = result.pop("svm_macro_f1")
    result["lstm_attack_recall_macro"] = result.pop("svm_attack_recall_macro")
    result["lstm_inference_per_1k_rows_ms"] = result.pop("svm_inference_per_1k_rows_ms")
    result["lstm_training_total_ms"] = result.pop("svm_training_total_ms")
    result["latency_speedup_vs_lstm"] = result.pop("latency_speedup_vs_svm")
    result["training_speedup_vs_lstm"] = result.pop("training_speedup_vs_svm")
    if result["benchmark_mode"] == "offline" and result["task_mode"] == "binary" and result["hdc_experiment"].endswith("hdc_one_pass"):
        if result["hdc_inference_per_1k_rows_ms"] > result["lstm_inference_per_1k_rows_ms"]:
            result["sanity_flags"].append("hdc_one_pass_slower_than_lstm_inference")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

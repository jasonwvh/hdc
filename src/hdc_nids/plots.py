"""Plot outputs for experiment runs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_binary_f1(metrics_rows: Sequence[dict], output_path: str | Path) -> None:
    x = [int(row["window_id"]) for row in metrics_rows]
    y = [float(row["binary_f1"]) for row in metrics_rows]
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, linewidth=2, color="#1f77b4")
    plt.xlabel("Window")
    plt.ylabel("Binary F1")
    plt.title("Binary F1 Over Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_forgetting(metrics_rows: Sequence[dict], output_path: str | Path) -> None:
    x = [int(row["window_id"]) for row in metrics_rows]
    y = [float(row["avg_forgetting"]) for row in metrics_rows]
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, linewidth=2, color="#d62728")
    plt.xlabel("Window")
    plt.ylabel("Average Forgetting")
    plt.title("Per-Family Forgetting Over Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_drift_recovery(metrics_rows: Sequence[dict], drift_events: Sequence[dict], output_path: str | Path) -> None:
    x = [int(row["window_id"]) for row in metrics_rows]
    y = [float(row["binary_f1"]) for row in metrics_rows]
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, linewidth=2, color="#2ca02c", label="Binary F1")
    for event in drift_events:
        if event.get("event_type") == "drift_trigger":
            plt.axvline(int(event["window_id"]), color="#ff7f0e", linestyle="--", alpha=0.7)
    plt.xlabel("Window")
    plt.ylabel("Binary F1")
    plt.title("Drift Recovery")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_latency(latency_rows: Sequence[dict], output_path: str | Path) -> None:
    windows = [int(row["window_id"]) for row in latency_rows]
    update_ms = np.asarray([float(row["update_ms"]) for row in latency_rows], dtype=np.float32)
    infer_ms = np.asarray([float(row["encode_ms"]) + float(row["score_ms"]) for row in latency_rows], dtype=np.float32)
    model_size_mb = np.asarray([float(row["model_size_bytes"]) / (1024 * 1024) for row in latency_rows], dtype=np.float32)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(windows, infer_ms, label="Predict ms", color="#1f77b4")
    ax1.plot(windows, update_ms, label="Update ms", color="#ff7f0e")
    ax1.set_xlabel("Window")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(windows, model_size_mb, label="Model size (MB)", color="#2ca02c", linestyle=":")
    ax2.set_ylabel("Model size (MB)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

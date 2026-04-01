"""Experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DriftConfig:
    history: int = 3
    f1_drop_threshold: float = 0.08
    fp_rate_rise_threshold: float = 0.03
    margin_drop_threshold: float = 0.05
    benign_lr_boost: float = 1.5
    cooloff_windows: int = 2


@dataclass(slots=True)
class MemoryMixConfig:
    benign_plastic_weight: float = 0.7
    attack_plastic_weight: float = 0.3
    stable_margin_threshold: float = 0.15
    benign_decay: float = 0.995


@dataclass(slots=True)
class StagnationConfig:
    patience: int = 2
    min_improvement: float = 0.002


@dataclass(slots=True)
class MLPConfig:
    hidden_dim: int = 64
    learning_rate_init: float = 0.001
    alpha: float = 0.0001
    max_iter: int = 30
    partial_fit_epochs: int = 1
    ewc_lambda: float = 5.0


@dataclass(slots=True)
class SVMConfig:
    alpha: float = 0.0001
    max_iter: int = 20
    partial_fit_epochs: int = 1
    c_values: list[float] = field(default_factory=lambda: [1.0, 10.0])
    gamma_values: list[str | float] = field(default_factory=lambda: ["scale", 0.1])


@dataclass(slots=True)
class LSTMConfig:
    hidden_dim: int = 16
    sequence_length: int = 4
    learning_rate: float = 0.01
    epochs: int = 1
    batch_size: int = 128
    gradient_clip: float = 1.0
    update_sample_limit: int = 1024
    dropout: float = 0.1
    max_epochs: int = 8
    patience: int = 2
    segment_count: int = 8


@dataclass(slots=True)
class HDCConfig:
    learning_rate: float = 0.035
    bootstrap_fraction: float = 0.01
    batch_size: int = 1024
    iterative_epoch_candidates: list[int] = field(default_factory=lambda: [5, 15, 30])
    regeneration_rounds: int = 2
    scale_by_sqrt_features: bool = True
    row_normalize: bool = False


@dataclass(slots=True)
class RowLimitConfig:
    train: int = 50000
    val: int = 15000
    test: int = 15000


@dataclass(slots=True)
class FeatureSelectionConfig:
    mode: str = "none"
    variance_threshold: float = 0.0
    top_k: int = 48


@dataclass(slots=True)
class ExperimentConfig:
    dataset: str
    model_type: str
    data_dir: Path
    output_dir: Path
    experiment_name: str
    benchmark_mode: str = "continual"
    task_mode: str = "both"
    stream_mode: str = "prequential"
    split_strategy: str = "dataset_default"
    validation_fraction: float = 0.15
    latency_mode: str = "inference_only"
    window_size: int = 4096
    warmup_size: int = 16384
    hd_dim: int = 4096
    bins: int = 64
    base_lr: float = 6.0
    rare_class_boost: float = 2.0
    regen_rate: float = 0.05
    seed: int = 7
    prototype_clip: int = 4096
    memory_mix: MemoryMixConfig = field(default_factory=MemoryMixConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    stagnation: StagnationConfig = field(default_factory=StagnationConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    hdc: HDCConfig = field(default_factory=HDCConfig)
    row_limits: RowLimitConfig = field(default_factory=RowLimitConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)


def _merge_dataclass(dc_type: type, payload: dict[str, Any] | None):
    if payload is None:
        return dc_type()
    defaults = dc_type()
    data = {name: getattr(defaults, name) for name in defaults.__dataclass_fields__}
    data.update(payload)
    return dc_type(**data)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from YAML."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    data_dir = Path(raw["data_dir"]).expanduser().resolve()
    output_dir = Path(raw.get("output_dir", "outputs")).expanduser().resolve()
    experiment_name = raw.get("experiment_name", f"{raw['dataset']}_{raw['model_type']}")
    return ExperimentConfig(
        dataset=raw["dataset"],
        model_type=raw["model_type"],
        data_dir=data_dir,
        output_dir=output_dir,
        experiment_name=experiment_name,
        benchmark_mode=raw.get("benchmark_mode", "continual"),
        task_mode=raw.get("task_mode", "both"),
        stream_mode=raw.get("stream_mode", "prequential"),
        split_strategy=raw.get("split_strategy", "dataset_default"),
        validation_fraction=float(raw.get("validation_fraction", 0.15)),
        latency_mode=raw.get("latency_mode", "inference_only"),
        window_size=int(raw.get("window_size", 4096)),
        warmup_size=int(raw.get("warmup_size", 16384)),
        hd_dim=int(raw.get("hd_dim", 4096)),
        bins=int(raw.get("bins", 64)),
        base_lr=float(raw.get("base_lr", 6.0)),
        rare_class_boost=float(raw.get("rare_class_boost", 2.0)),
        regen_rate=float(raw.get("regen_rate", 0.05)),
        seed=int(raw.get("seed", 7)),
        prototype_clip=int(raw.get("prototype_clip", 4096)),
        memory_mix=_merge_dataclass(MemoryMixConfig, raw.get("memory_mix")),
        drift=_merge_dataclass(DriftConfig, raw.get("drift")),
        stagnation=_merge_dataclass(StagnationConfig, raw.get("stagnation")),
        mlp=_merge_dataclass(MLPConfig, raw.get("mlp")),
        svm=_merge_dataclass(SVMConfig, raw.get("svm")),
        lstm=_merge_dataclass(LSTMConfig, raw.get("lstm")),
        hdc=_merge_dataclass(HDCConfig, raw.get("hdc")),
        row_limits=_merge_dataclass(RowLimitConfig, raw.get("row_limits")),
        feature_selection=_merge_dataclass(FeatureSelectionConfig, raw.get("feature_selection")),
    )

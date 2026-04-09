"""Dataset loading and stream construction."""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RowLimitConfig
from .constants import (
    CICIDS_BENIGN_LABEL,
    CICIDS_CANONICAL_LABELS,
    CICIDS_CLASSES,
    CICIDS_DROP_COLUMNS,
    CICIDS_FILE_ORDER,
    CICIDS_FORCED_CATEGORICAL,
    UNSW_BENIGN_LABEL,
    UNSW_CLASSES,
    UNSW_COMMON_ATTACKS,
    UNSW_DROP_COLUMNS,
    UNSW_FORCED_CATEGORICAL,
    UNSW_RARE_ATTACKS,
)
from .utils import batched_iterable


@dataclass(slots=True)
class RawRecord:
    features: dict[str, str]
    internal_label: str
    binary_label: int
    source: str
    stage_name: str
    record_id: str = ""


@dataclass(slots=True)
class RawWindow:
    dataset: str
    window_id: int
    stage_name: str
    records: list[RawRecord]


@dataclass(slots=True)
class DatasetStream:
    dataset: str
    benign_label: str
    class_labels: list[str]
    forced_categorical: set[str]
    numeric_transform: str
    warmup_records: list[RawRecord]
    window_factory: Callable[[], Iterator[RawWindow]]


@dataclass(slots=True)
class OfflineSplit:
    dataset: str
    benign_label: str
    class_labels: list[str]
    forced_categorical: set[str]
    numeric_transform: str
    train_records: list[RawRecord]
    val_records: list[RawRecord]
    test_records: list[RawRecord]
    split_strategy: str


def _strip_keys(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): (value.strip() if isinstance(value, str) else value) for key, value in row.items()}


def _clean_numeric_string(value: str) -> str:
    if value == "":
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isinf(number):
        return "nan"
    return value


def canonicalize_cicids_label(label: str) -> str:
    cleaned = label.strip()
    cleaned = CICIDS_CANONICAL_LABELS.get(cleaned, cleaned)
    return cleaned


_PROTOCOL_MAP = {"1": "icmp", "6": "tcp", "17": "udp"}

_WELL_KNOWN_PORTS = {
    20: "ftp-data", 21: "ftp", 22: "ssh", 25: "smtp", 53: "dns",
    80: "http", 110: "pop3", 143: "imap", 443: "https", 445: "smb",
    3306: "mysql", 3389: "rdp", 8080: "http-alt",
}


def _port_bucket_vector(ports):
    result = []
    for p in ports:
        try:
            port = int(float(p))
        except (ValueError, TypeError):
            result.append("__UNK__")
            continue
        if port in _WELL_KNOWN_PORTS:
            result.append(_WELL_KNOWN_PORTS[port])
        elif port < 1024:
            result.append("system")
        elif port < 49152:
            result.append("registered")
        else:
            result.append("ephemeral")
    return result


def _load_cicids_file_as_df(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, low_memory=False, encoding="utf-8", encoding_errors="replace")
    df.columns = df.columns.str.strip()

    drop_cols = [c for c in CICIDS_DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)

    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip()
        df["Label"] = df["Label"].replace(CICIDS_CANONICAL_LABELS)
        df = df[df["Label"].notna() & (df["Label"] != "") & (df["Label"] != "nan")]

    if "Protocol" in df.columns:
        df["Protocol"] = df["Protocol"].astype(str).str.strip().map(
            lambda x: _PROTOCOL_MAP.get(x, x or "other")
        )

    for port_col, bucket_col in [("Source Port", "Src Port Bucket"), ("Destination Port", "Dst Port Bucket")]:
        if port_col in df.columns:
            df[bucket_col] = _port_bucket_vector(df[port_col])
            df = df.drop(columns=[port_col])

    df.replace([float("inf"), float("-inf")], np.nan, inplace=True)

    return df


def _load_all_cicids_dfs(data_dir: Path, per_file_limit: int | None = None) -> list[tuple[str, pd.DataFrame]]:
    base_dir = data_dir / "CICIDS2017"
    result = []
    for filename in CICIDS_FILE_ORDER:
        file_path = base_dir / filename
        df = _load_cicids_file_as_df(file_path)
        if per_file_limit is not None and len(df) > per_file_limit:
            df = df.sample(n=per_file_limit, random_state=42)
        df["_source"] = filename
        df["_stage_name"] = Path(filename).stem
        df["_row_index"] = range(len(df))
        result.append((filename, df))
    return result


def _dataframe_to_records(df: pd.DataFrame) -> list[RawRecord]:
    records = []
    cols = df.columns.tolist()
    feature_cols = [c for c in cols if not c.startswith("_")]
    for row in df.itertuples(index=False):
        row_dict = dict(zip(cols, row))
        label = row_dict.get("Label", "")
        if not label or label == "nan":
            continue
        features = {}
        for k in feature_cols:
            if k != "Label":
                v = row_dict[k]
                if isinstance(v, float) and np.isnan(v):
                    features[k] = "nan"
                else:
                    features[k] = str(v).strip() if isinstance(v, str) else str(v)
        records.append(RawRecord(
            features=features,
            internal_label=label,
            binary_label=0 if label == CICIDS_BENIGN_LABEL else 1,
            source=row_dict.get("_source", ""),
            stage_name=row_dict.get("_stage_name", ""),
            record_id=f"{row_dict.get('_source', '')}:{row_dict.get('_row_index', '')}",
        ))
    return records


def _iter_cicids_records(data_dir: Path, max_records: int | None = None, per_file_limit: int | None = None) -> Iterator[RawRecord]:
    base_dir = data_dir / "CICIDS2017"
    count = 0
    for filename in CICIDS_FILE_ORDER:
        file_path = base_dir / filename
        df = _load_cicids_file_as_df(file_path)
        file_count = 0
        cols = df.columns.tolist()
        feature_cols = [c for c in cols if c != "Label"]
        for idx, row in df.iterrows():
            if max_records is not None and count >= max_records:
                return
            if per_file_limit is not None and file_count >= per_file_limit:
                break
            label = row.get("Label", "")
            if not label or label == "nan":
                continue
            features = {k: str(row[k]) for k in feature_cols}
            yield RawRecord(
                features=features,
                internal_label=label,
                binary_label=0 if label == CICIDS_BENIGN_LABEL else 1,
                source=filename,
                stage_name=Path(filename).stem,
                record_id=f"{filename}:{idx}",
            )
            count += 1
            file_count += 1


def _collect_warmup(
    iterator: Iterable[RawRecord], warmup_size: int
) -> tuple[list[RawRecord], deque[RawRecord]]:
    warmup_records: list[RawRecord] = []
    remainder: deque[RawRecord] = deque()
    for record in iterator:
        if len(warmup_records) < warmup_size:
            warmup_records.append(record)
        else:
            remainder.append(record)
            break
    return warmup_records, remainder


def build_cicids_stream(data_dir: Path, warmup_size: int, window_size: int) -> DatasetStream:
    initial_iter = _iter_cicids_records(data_dir)
    warmup_records, prefix = _collect_warmup(initial_iter, warmup_size)

    def factory() -> Iterator[RawWindow]:
        iterator = _iter_cicids_records(data_dir)
        skipped = 0
        batch: list[RawRecord] = []
        window_id = 0
        for record in iterator:
            if skipped < warmup_size:
                skipped += 1
                continue
            batch.append(record)
            if len(batch) >= window_size:
                yield RawWindow(
                    dataset="cicids2017",
                    window_id=window_id,
                    stage_name=batch[0].stage_name,
                    records=batch,
                )
                batch = []
                window_id += 1
        if batch:
            yield RawWindow(
                dataset="cicids2017",
                window_id=window_id,
                stage_name=batch[0].stage_name,
                records=batch,
            )

    if len(warmup_records) < warmup_size:
        warmup_records.extend(list(prefix))
    return DatasetStream(
        dataset="cicids2017",
        benign_label=CICIDS_BENIGN_LABEL,
        class_labels=CICIDS_CLASSES,
        forced_categorical=set(CICIDS_FORCED_CATEGORICAL),
        numeric_transform="signed_log_zscore",
        warmup_records=warmup_records,
        window_factory=factory,
    )


def _iter_unsw_records(data_dir: Path) -> Iterator[RawRecord]:
    base_dir = data_dir / "UNSW_NB15"
    file_order = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]
    for filename in file_order:
        file_path = base_dir / filename
        df = pd.read_csv(file_path, low_memory=False)
        for row_index, row in df.iterrows():
            label = str(row.get("attack_cat", "")).strip()
            features = {
                key: str(value).strip()
                for key, value in row.items()
                if key not in UNSW_DROP_COLUMNS
            }
            yield RawRecord(
                features=features,
                internal_label=label,
                binary_label=0 if str(row.get("label", "0")) == "0" else 1,
                source=filename,
                stage_name="raw_unsw",
                record_id=f"{filename}:{row_index}",
            )


def _dedupe_records(records: list[RawRecord]) -> list[RawRecord]:
    deduped: list[RawRecord] = []
    seen: set[tuple] = set()
    for record in records:
        fingerprint = (
            tuple(record.features.values()),
            record.internal_label,
            record.binary_label,
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        deduped.append(record)
    return deduped


def _stratified_limit(records: list[RawRecord], limit: int, seed: int) -> list[RawRecord]:
    if limit <= 0 or len(records) <= limit:
        return list(records)
    labels = _safe_stratify_labels(records)
    sampled, _ = train_test_split(
        records,
        train_size=limit,
        random_state=seed,
        stratify=labels,
    )
    return list(sampled)


def _apply_row_limits(
    train_records: list[RawRecord],
    val_records: list[RawRecord],
    test_records: list[RawRecord],
    row_limits: RowLimitConfig,
    seed: int,
) -> tuple[list[RawRecord], list[RawRecord], list[RawRecord]]:
    return (
        _stratified_limit(train_records, row_limits.train, seed),
        _stratified_limit(val_records, row_limits.val, seed + 1),
        _stratified_limit(test_records, row_limits.test, seed + 2),
    )


def _safe_stratify_labels(records: list[RawRecord]) -> list[str] | None:
    labels = [record.internal_label for record in records]
    counts: dict[str, int] = defaultdict(int)
    for label in labels:
        counts[label] += 1
    if len(labels) < 2 or any(count < 2 for count in counts.values()):
        return None
    return labels


def _split_records(
    records: list[RawRecord],
    *,
    train_size: float,
    seed: int,
) -> tuple[list[RawRecord], list[RawRecord]]:
    if len(records) <= 1:
        return list(records), []
    left, right = train_test_split(
        records,
        train_size=train_size,
        random_state=seed,
        stratify=_safe_stratify_labels(records),
    )
    return list(left), list(right)


def build_offline_split(
    dataset: str,
    *,
    data_dir: Path,
    validation_fraction: float,
    split_strategy: str,
    row_limits: RowLimitConfig,
    seed: int,
) -> OfflineSplit:
    normalized = dataset.lower()
    if normalized == "cicids2017":
        all_dfs = _load_all_cicids_dfs(data_dir, per_file_limit=150000)
        combined = pd.concat([df for _, df in all_dfs], ignore_index=True)

        if split_strategy in {"dataset_default", "stratified", "stratified_70_15_15"}:
            labels = combined["Label"].values
            label_counts = pd.Series(labels).value_counts()
            min_count = label_counts.min()
            
            if min_count < 2:
                train_idx, temp_idx = train_test_split(
                    np.arange(len(combined)), train_size=0.70, random_state=seed,
                )
                val_idx, test_idx = train_test_split(
                    temp_idx, train_size=0.50, random_state=seed + 1,
                )
            else:
                train_idx, temp_idx = train_test_split(
                    np.arange(len(combined)), train_size=0.70, random_state=seed, stratify=labels,
                )
                temp_labels = labels[temp_idx]
                temp_label_counts = pd.Series(temp_labels).value_counts()
                if temp_label_counts.min() < 2:
                    val_idx, test_idx = train_test_split(
                        temp_idx, train_size=0.50, random_state=seed + 1,
                    )
                else:
                    val_idx, test_idx = train_test_split(
                        temp_idx, train_size=0.50, random_state=seed + 1, stratify=temp_labels,
                    )
            resolved_strategy = "stratified_70_15_15"
        elif split_strategy in {"chronological_day_stress", "cicids_day_stress"}:
            train_sources = {
                "Monday-WorkingHours.pcap_ISCX.csv",
                "Tuesday-WorkingHours.pcap_ISCX.csv",
                "Wednesday-workingHours.pcap_ISCX.csv",
                "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            }
            val_sources = {
                "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            }
            test_sources = {
                "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            }
            train_idx = combined.index[combined["_source"].isin(train_sources)].values
            val_idx = combined.index[combined["_source"].isin(val_sources)].values
            test_idx = combined.index[combined["_source"].isin(test_sources)].values
            resolved_strategy = "chronological_day_stress"
        else:
            raise ValueError(f"Unsupported CICIDS offline split strategy: {split_strategy}")

        train_df = combined.iloc[train_idx]
        val_df = combined.iloc[val_idx]
        test_df = combined.iloc[test_idx]

        def _stratified_sample_df(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
            if limit <= 0 or len(df) <= limit:
                return df
            labels = df["Label"].values
            counts = pd.Series(labels).value_counts()
            if counts.min() < 2:
                return df.head(limit)
            sampled, _ = train_test_split(df, train_size=limit, random_state=seed, stratify=labels)
            return sampled

        train_df = _stratified_sample_df(train_df, row_limits.train, seed)
        val_df = _stratified_sample_df(val_df, row_limits.val, seed + 1)
        test_df = _stratified_sample_df(test_df, row_limits.test, seed + 2)

        train_records = _dataframe_to_records(train_df)
        val_records = _dataframe_to_records(val_df)
        test_records = _dataframe_to_records(test_df)

        return OfflineSplit(
            dataset="cicids2017",
            benign_label=CICIDS_BENIGN_LABEL,
            class_labels=CICIDS_CLASSES,
            forced_categorical=set(CICIDS_FORCED_CATEGORICAL),
            numeric_transform="signed_log_zscore",
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            split_strategy=resolved_strategy,
        )

    if normalized == "unsw_nb15":
        records = list(_iter_unsw_records(data_dir))
        official_train = _dedupe_records([record for record in records if record.source == "UNSW_NB15_training-set.csv"])
        official_test = [record for record in records if record.source == "UNSW_NB15_testing-set.csv"]
        if split_strategy not in {"dataset_default", "official", "official_train_test_plus_val"}:
            raise ValueError(f"Unsupported UNSW offline split strategy: {split_strategy}")
        train_records, val_records = _split_records(
            official_train,
            train_size=max(1.0 - validation_fraction, 0.5),
            seed=seed,
        )
        train_records, val_records, test_records = _apply_row_limits(
            list(train_records),
            list(val_records),
            list(official_test),
            row_limits,
            seed,
        )
        return OfflineSplit(
            dataset="unsw_nb15",
            benign_label=UNSW_BENIGN_LABEL,
            class_labels=UNSW_CLASSES,
            forced_categorical=UNSW_FORCED_CATEGORICAL,
            numeric_transform="minmax",
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            split_strategy="official_train_test_plus_val",
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


def _sample_take(items: list[RawRecord], count: int) -> list[RawRecord]:
    take = min(count, len(items))
    result = items[:take]
    del items[:take]
    return result


def _interleave_groups(groups: list[list[RawRecord]]) -> list[RawRecord]:
    queues = [deque(group) for group in groups if group]
    ordered: list[RawRecord] = []
    while queues:
        next_round: list[deque[RawRecord]] = []
        for queue in queues:
            if queue:
                ordered.append(queue.popleft())
            if queue:
                next_round.append(queue)
        queues = next_round
    return ordered


def build_unsw_stream(
    data_dir: Path,
    warmup_size: int,
    window_size: int,
    seed: int,
) -> DatasetStream:
    records = list(_iter_unsw_records(data_dir))
    rng = random.Random(seed)
    by_label: dict[str, list[RawRecord]] = defaultdict(list)
    for record in records:
        by_label[record.internal_label].append(record)
    for bucket in by_label.values():
        rng.shuffle(bucket)

    warmup_pool = []
    for label in [UNSW_BENIGN_LABEL, *sorted(UNSW_COMMON_ATTACKS)]:
        warmup_pool.extend(by_label[label][:])
    rng.shuffle(warmup_pool)
    warmup_records = warmup_pool[:warmup_size]
    used_ids = {id(record) for record in warmup_records}
    for label, bucket in by_label.items():
        by_label[label] = [record for record in bucket if id(record) not in used_ids]

    benign_pool = by_label[UNSW_BENIGN_LABEL]
    common_pool = []
    for label in sorted(UNSW_COMMON_ATTACKS):
        common_pool.extend(by_label[label])
        by_label[label] = []
    rare_pool = []
    for label in sorted(UNSW_RARE_ATTACKS):
        rare_pool.extend(by_label[label])
        by_label[label] = []
    rng.shuffle(common_pool)
    rng.shuffle(rare_pool)

    benign_drift = _interleave_groups(
        [
            _sample_take(benign_pool, max(window_size * 4, len(benign_pool) // 2)),
            _sample_take(common_pool, max(window_size, len(common_pool) // 4)),
        ]
    )
    rare_attack = _interleave_groups(
        [
            _sample_take(rare_pool, len(rare_pool)),
            _sample_take(benign_pool, max(window_size * 2, len(rare_pool))),
            _sample_take(common_pool, max(window_size, len(rare_pool) // 2)),
        ]
    )
    recurrent_common = _interleave_groups([common_pool, benign_pool, rare_pool])

    staged_records: list[tuple[str, list[RawRecord]]] = [
        ("benign_drift", benign_drift),
        ("rare_attack", rare_attack),
        ("recurrent_common", recurrent_common),
    ]

    def factory() -> Iterator[RawWindow]:
        window_id = 0
        for stage_name, stage_records in staged_records:
            for batch in batched_iterable(stage_records, window_size):
                yield RawWindow(
                    dataset="unsw_nb15",
                    window_id=window_id,
                    stage_name=stage_name,
                    records=batch,
                )
                window_id += 1

    return DatasetStream(
        dataset="unsw_nb15",
        benign_label=UNSW_BENIGN_LABEL,
        class_labels=UNSW_CLASSES,
        forced_categorical=UNSW_FORCED_CATEGORICAL,
        numeric_transform="minmax",
        warmup_records=warmup_records,
        window_factory=factory,
    )


def build_stream(dataset: str, data_dir: Path, warmup_size: int, window_size: int, seed: int) -> DatasetStream:
    normalized = dataset.lower()
    if normalized == "cicids2017":
        return build_cicids_stream(data_dir, warmup_size, window_size)
    if normalized == "unsw_nb15":
        return build_unsw_stream(data_dir, warmup_size, window_size, seed)
    raise ValueError(f"Unsupported dataset: {dataset}")

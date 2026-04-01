from __future__ import annotations

import csv
from pathlib import Path

from hdc_nids.constants import CICIDS_FILE_ORDER
from hdc_nids.data import build_cicids_stream, canonicalize_cicids_label


def _write_cicids_file(path: Path, rows: list[list[str]]) -> None:
    headers = [
        "Flow ID",
        " Source IP",
        " Destination IP",
        " Timestamp",
        " Flow Duration",
        " Total Fwd Packets",
        " Label",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def test_canonicalize_cicids_web_attack_labels():
    assert canonicalize_cicids_label("Web Attack � XSS") == "Web Attack XSS"
    assert canonicalize_cicids_label("Web Attack \ufeff Sql Injection") == "Web Attack Sql Injection"


def test_cicids_stream_drops_empty_labels_and_metadata(tmp_path: Path):
    data_dir = tmp_path / "data" / "CICIDS2017"
    data_dir.mkdir(parents=True)
    benign_row = ["flow-1", "10.0.0.1", "10.0.0.2", "03/07/2017 08:55:58", "12.0", "3", "BENIGN"]
    empty_row = ["flow-2", "10.0.0.1", "10.0.0.2", "", "4.0", "1", ""]
    web_attack_row = ["flow-3", "10.0.0.1", "10.0.0.2", "6/7/2017 9:15", "7.0", "2", "Web Attack � Brute Force"]
    for filename in CICIDS_FILE_ORDER:
        rows = [benign_row]
        if filename == "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv":
            rows = [benign_row, empty_row, web_attack_row]
        _write_cicids_file(data_dir / filename, rows)

    stream = build_cicids_stream(tmp_path / "data", warmup_size=1, window_size=64)
    windows = list(stream.window_factory())
    labels = [record.internal_label for window in windows for record in window.records]
    assert "" not in labels
    assert "Web Attack Brute Force" in labels
    sample_features = windows[0].records[0].features
    assert "Flow ID" not in sample_features
    assert "Source IP" not in sample_features
    assert "Destination IP" not in sample_features
    assert "Timestamp" not in sample_features
    assert "Src Port Bucket" in sample_features
    assert "Dst Port Bucket" in sample_features
    assert stream.numeric_transform == "signed_log_zscore"
    assert "Protocol" in stream.forced_categorical

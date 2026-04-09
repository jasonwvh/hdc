from __future__ import annotations

import csv
from pathlib import Path

from hdc_nids.config import RowLimitConfig
from hdc_nids.constants import CICIDS_FILE_ORDER
from hdc_nids.data import build_offline_split


def _write_cicids_file(path: Path, rows: list[list[str]]) -> None:
    headers = [
        "Flow ID",
        " Source IP",
        " Source Port",
        " Destination IP",
        " Destination Port",
        " Protocol",
        " Timestamp",
        " Flow Duration",
        " Total Fwd Packets",
        " Label",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def test_cicids_offline_split_is_reproducible_and_disjoint(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "CICIDS2017"
    data_dir.mkdir(parents=True)
    benign = ["flow-b", "10.0.0.1", "80", "10.0.0.2", "49152", "6", "03/07/2017 08:55:58", "10.0", "3", "BENIGN"]
    attack = ["flow-a", "10.0.0.1", "21", "10.0.0.2", "49153", "6", "04/07/2017 08:55:58", "120.0", "9", "FTP-Patator"]
    for filename in CICIDS_FILE_ORDER:
        rows = [benign, attack, benign, attack, benign, attack]
        _write_cicids_file(data_dir / filename, rows)

    split_a = build_offline_split(
        "cicids2017",
        data_dir=tmp_path / "data",
        validation_fraction=0.15,
        split_strategy="dataset_default",
        row_limits=RowLimitConfig(train=0, val=0, test=0),
        seed=7,
    )
    split_b = build_offline_split(
        "cicids2017",
        data_dir=tmp_path / "data",
        validation_fraction=0.15,
        split_strategy="dataset_default",
        row_limits=RowLimitConfig(train=0, val=0, test=0),
        seed=7,
    )

    train_ids = {record.record_id for record in split_a.train_records}
    val_ids = {record.record_id for record in split_a.val_records}
    test_ids = {record.record_id for record in split_a.test_records}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids == {record.record_id for record in split_b.train_records}


def test_unsw_offline_split_respects_official_test_boundary(tmp_path: Path) -> None:
    base_dir = tmp_path / "data" / "UNSW_NB15"
    base_dir.mkdir(parents=True)
    headers = ["id", "dur", "proto", "service", "state", "attack_cat", "label"]
    with (base_dir / "UNSW_NB15_training-set.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(
            [
                ["1", "0.1", "tcp", "-", "FIN", "Normal", "0"],
                ["2", "4.2", "udp", "dns", "CON", "Generic", "1"],
                ["3", "5.0", "udp", "dns", "CON", "Exploits", "1"],
                ["4", "0.2", "tcp", "-", "FIN", "Normal", "0"],
            ]
        )
    with (base_dir / "UNSW_NB15_testing-set.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(
            [
                ["5", "0.3", "tcp", "-", "FIN", "Normal", "0"],
                ["6", "6.1", "udp", "dns", "CON", "Generic", "1"],
            ]
        )

    split = build_offline_split(
        "unsw_nb15",
        data_dir=tmp_path / "data",
        validation_fraction=0.5,
        split_strategy="dataset_default",
        row_limits=RowLimitConfig(train=0, val=0, test=0),
        seed=7,
    )
    assert all(record.source == "UNSW_NB15_training-set.csv" for record in split.train_records)
    assert all(record.source == "UNSW_NB15_training-set.csv" for record in split.val_records)
    assert all(record.source == "UNSW_NB15_testing-set.csv" for record in split.test_records)
    assert split.numeric_transform == "minmax"


def test_cicids_chronological_day_stress_split_uses_expected_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "CICIDS2017"
    data_dir.mkdir(parents=True)
    benign = ["flow-b", "10.0.0.1", "80", "10.0.0.2", "49152", "6", "03/07/2017 08:55:58", "10.0", "3", "BENIGN"]
    attack = ["flow-a", "10.0.0.1", "21", "10.0.0.2", "49153", "6", "04/07/2017 08:55:58", "120.0", "9", "FTP-Patator"]
    for filename in CICIDS_FILE_ORDER:
        _write_cicids_file(data_dir / filename, [benign, attack, benign, attack])

    split = build_offline_split(
        "cicids2017",
        data_dir=tmp_path / "data",
        validation_fraction=0.15,
        split_strategy="chronological_day_stress",
        row_limits=RowLimitConfig(train=0, val=0, test=0),
        seed=7,
    )
    assert split.split_strategy == "chronological_day_stress"
    assert all("Friday-WorkingHours-Afternoon" not in record.source for record in split.train_records)
    assert all(
        record.source in {
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        }
        for record in split.val_records
    )
    assert all("Friday-WorkingHours-Afternoon" in record.source for record in split.test_records)

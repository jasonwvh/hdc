from __future__ import annotations

import numpy as np

from hdc_nids.data import RawRecord
from hdc_nids.encoding import TabularHDCEncoder
from hdc_nids.preprocessing import TabularPreprocessor


def _make_record(protocol: str, duration: str, label: str) -> RawRecord:
    return RawRecord(
        features={"proto": protocol, "dur": duration},
        internal_label=label,
        binary_label=0 if label == "BENIGN" else 1,
        source="unit",
        stage_name="unit",
    )


def test_encoder_is_deterministic_for_seen_and_unseen_values():
    warmup = [
        _make_record("tcp", "1.0", "BENIGN"),
        _make_record("udp", "3.0", "AttackA"),
    ]
    preprocessor = TabularPreprocessor(
        class_labels=["BENIGN", "AttackA"],
        benign_label="BENIGN",
        forced_categorical={"proto"},
    )
    preprocessor.fit(warmup)
    batch = preprocessor.transform_records(
        [
            _make_record("tcp", "2.0", "BENIGN"),
            _make_record("icmp", "4.0", "AttackA"),
        ],
        dataset="unit",
        window_id=0,
        stage_name="unit",
    )
    encoder = TabularHDCEncoder(preprocessor, dim=256, bins=64, seed=11)
    encoded_first = encoder.encode_batch(batch)
    encoded_second = encoder.encode_batch(batch)

    assert np.array_equal(encoded_first, encoded_second)
    assert encoded_first.shape == (2, 256)
    assert np.any(encoded_first[1] != 0)

from __future__ import annotations

import numpy as np

from hdc_nids.data import RawRecord
from hdc_nids.preprocessing import DenseFeatureSelector, TabularPreprocessor


def _record(a: float, b: float, noisy: float, label: str) -> RawRecord:
    return RawRecord(
        features={"a": str(a), "b": str(b), "noisy": str(noisy)},
        internal_label=label,
        binary_label=0 if label == "BENIGN" else 1,
        source="unit",
        stage_name="unit",
    )


def test_dense_feature_selector_reduces_dense_dimension_train_only() -> None:
    train_records = [
        _record(-2.0, -2.0, 0.0, "BENIGN"),
        _record(-1.5, -1.4, 0.0, "BENIGN"),
        _record(1.5, 1.4, 0.0, "ATTACK"),
        _record(2.0, 2.1, 0.0, "ATTACK"),
    ]
    test_records = [
        _record(-1.8, -1.6, 99.0, "BENIGN"),
        _record(1.8, 1.7, -99.0, "ATTACK"),
    ]
    preprocessor = TabularPreprocessor(class_labels=["BENIGN", "ATTACK"], benign_label="BENIGN")
    preprocessor.fit(train_records)
    train_batch = preprocessor.transform_records(train_records, dataset="unit", window_id=0, stage_name="train")
    test_batch = preprocessor.transform_records(test_records, dataset="unit", window_id=1, stage_name="test")

    selector = DenseFeatureSelector(mode="variance_mi", variance_threshold=1e-6, top_k=2)
    selector.fit(train_batch.dense, train_batch.class_indices)
    filtered_train = selector.transform_batch(train_batch)
    filtered_test = selector.transform_batch(test_batch)

    assert filtered_train.dense.shape[1] == 2
    assert filtered_test.dense.shape[1] == 2
    assert np.array_equal(filtered_train.class_indices, train_batch.class_indices)


def test_unsw_minmax_and_selector_preserve_categorical_dimensions() -> None:
    records = [
        RawRecord(
            features={"dur": "0.0", "proto": "tcp", "service": "http", "state": "FIN"},
            internal_label="Normal",
            binary_label=0,
            source="unit",
            stage_name="unit",
        ),
        RawRecord(
            features={"dur": "10.0", "proto": "udp", "service": "dns", "state": "CON"},
            internal_label="Generic",
            binary_label=1,
            source="unit",
            stage_name="unit",
        ),
    ]
    preprocessor = TabularPreprocessor(
        class_labels=["Normal", "Generic"],
        benign_label="Normal",
        forced_categorical={"proto", "service", "state"},
        numeric_transform="minmax",
    )
    preprocessor.fit(records)
    batch = preprocessor.transform_records(records, dataset="unsw_nb15", window_id=0, stage_name="train")

    assert float(batch.numeric.min()) >= 0.0
    assert float(batch.numeric.max()) <= 1.0

    selector = DenseFeatureSelector(
        mode="variance_mi",
        variance_threshold=1e-6,
        top_k=1,
        candidate_indices=preprocessor.numeric_dense_indices(),
        preserve_indices=preprocessor.categorical_dense_indices(),
    )
    selector.fit(batch.dense, batch.class_indices)
    filtered = selector.transform_batch(batch)

    assert filtered.dense.shape[1] == 1 + preprocessor.categorical_dense_indices().shape[0]

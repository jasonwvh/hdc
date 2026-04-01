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

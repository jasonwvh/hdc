from __future__ import annotations

import numpy as np

from hdc_nids.baselines import OfflineLSTMModel, OfflineSVMRBFModel
from hdc_nids.data import RawRecord
from hdc_nids.preprocessing import TabularPreprocessor


def _record(x1: float, x2: float, label: str) -> RawRecord:
    return RawRecord(
        features={"x1": str(x1), "x2": str(x2)},
        internal_label=label,
        binary_label=0 if label == "BENIGN" else 1,
        source="unit",
        stage_name="unit",
    )


def test_offline_svm_and_lstm_learn_separable_binary_data() -> None:
    train_records = [
        _record(-2.0, -1.5, "BENIGN"),
        _record(-1.8, -1.2, "BENIGN"),
        _record(-1.5, -1.0, "BENIGN"),
        _record(1.2, 1.5, "ATTACK"),
        _record(1.5, 1.7, "ATTACK"),
        _record(1.8, 2.0, "ATTACK"),
    ]
    test_records = [
        _record(-1.7, -1.3, "BENIGN"),
        _record(1.6, 1.9, "ATTACK"),
    ]
    preprocessor = TabularPreprocessor(class_labels=["BENIGN", "ATTACK"], benign_label="BENIGN")
    preprocessor.fit(train_records)
    train_batch = preprocessor.transform_records(train_records, dataset="unit", window_id=0, stage_name="train")
    test_batch = preprocessor.transform_records(test_records, dataset="unit", window_id=1, stage_name="test")

    svm = OfflineSVMRBFModel(preprocessor=preprocessor, c_value=1.0, gamma_value="scale", seed=7)
    svm.fit_initial(train_batch)
    svm_pred = svm.predict(test_batch)
    assert float(np.mean(svm_pred.predicted_labels == test_batch.internal_labels)) >= 1.0

    lstm = OfflineLSTMModel(
        preprocessor=preprocessor,
        hidden_dim=8,
        learning_rate=0.01,
        batch_size=2,
        dropout=0.0,
        max_epochs=20,
        patience=3,
        segment_count=2,
        seed=7,
    )
    lstm.fit_with_validation(train_batch, train_batch)
    lstm_pred = lstm.predict(test_batch)
    assert float(np.mean(lstm_pred.predicted_labels == test_batch.internal_labels)) >= 0.5

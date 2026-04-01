from __future__ import annotations

import numpy as np

from hdc_nids.data import RawRecord
from hdc_nids.models import PaperOnlineHDModel
from hdc_nids.preprocessing import TabularPreprocessor


def _record(x1: float, x2: float, label: str) -> RawRecord:
    return RawRecord(
        features={"x1": str(x1), "x2": str(x2)},
        internal_label=label,
        binary_label=0 if label == "BENIGN" else 1,
        source="unit",
        stage_name="unit",
    )


def test_paper_onlinehd_learns_simple_binary_problem() -> None:
    train_records = [
        _record(-2.0, -1.8, "BENIGN"),
        _record(-1.8, -1.4, "BENIGN"),
        _record(-1.5, -1.0, "BENIGN"),
        _record(1.2, 1.3, "ATTACK"),
        _record(1.5, 1.8, "ATTACK"),
        _record(1.9, 2.1, "ATTACK"),
    ]
    test_records = [
        _record(-1.7, -1.1, "BENIGN"),
        _record(1.7, 1.9, "ATTACK"),
    ]

    preprocessor = TabularPreprocessor(class_labels=["BENIGN", "ATTACK"], benign_label="BENIGN")
    preprocessor.fit(train_records)
    train_batch = preprocessor.transform_records(train_records, dataset="unit", window_id=0, stage_name="train")
    test_batch = preprocessor.transform_records(test_records, dataset="unit", window_id=1, stage_name="test")

    model = PaperOnlineHDModel(
        input_dim=train_batch.dense.shape[1],
        class_labels=["BENIGN", "ATTACK"],
        benign_label="BENIGN",
        dim=512,
        seed=7,
        learning_rate=0.035,
        bootstrap_fraction=0.25,
        batch_size=4,
        regen_rate=0.05,
    )
    model.fit_initial(train_batch)
    model.fit_iterative(train_batch, epochs=5)
    prediction = model.predict(test_batch)
    assert float(np.mean(prediction.predicted_labels == test_batch.internal_labels)) >= 0.5

    dims = model.regenerate_low_variance_dimensions()
    assert dims.size > 0

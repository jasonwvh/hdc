from __future__ import annotations

from hdc_nids.baselines import OnlineSVMModel, StaticSVMModel
from hdc_nids.data import RawRecord
from hdc_nids.preprocessing import TabularPreprocessor


def _record(value: str, proto: str, label: str) -> RawRecord:
    return RawRecord(
        features={"dur": value, "proto": proto},
        internal_label=label,
        binary_label=0 if label == "BENIGN" else 1,
        source="unit",
        stage_name="unit",
    )


def test_static_svm_predicts_binary_and_multiclass_outputs():
    warmup = [
        _record("0.0", "tcp", "BENIGN"),
        _record("1.0", "udp", "AttackA"),
        _record("2.0", "icmp", "AttackB"),
    ]
    preprocessor = TabularPreprocessor(
        class_labels=["BENIGN", "AttackA", "AttackB"],
        benign_label="BENIGN",
        forced_categorical={"proto"},
    )
    preprocessor.fit(warmup)
    batch = preprocessor.transform_records(warmup, dataset="unit", window_id=0, stage_name="unit")
    model = StaticSVMModel(preprocessor=preprocessor, alpha=0.0001, max_iter=5, seed=7)
    model.fit_initial(batch)
    prediction = model.predict(batch)

    assert prediction.class_scores.shape == (3, 3)
    assert prediction.predicted_binary.shape == (3,)


def test_online_svm_partial_fit_update_preserves_output_shapes():
    warmup = [
        _record("0.0", "tcp", "BENIGN"),
        _record("1.0", "udp", "AttackA"),
        _record("2.0", "icmp", "AttackB"),
    ]
    preprocessor = TabularPreprocessor(
        class_labels=["BENIGN", "AttackA", "AttackB"],
        benign_label="BENIGN",
        forced_categorical={"proto"},
    )
    preprocessor.fit(warmup)
    batch = preprocessor.transform_records(warmup, dataset="unit", window_id=0, stage_name="unit")
    model = OnlineSVMModel(
        preprocessor=preprocessor,
        alpha=0.0001,
        max_iter=5,
        partial_fit_epochs=1,
        seed=7,
    )
    model.fit_initial(batch)

    before = model.predict(batch)
    events = model.update(batch, before, drift_active=False)
    after = model.predict(batch)

    assert events == []
    assert after.class_scores.shape == before.class_scores.shape
    assert after.predicted_binary.shape == before.predicted_binary.shape

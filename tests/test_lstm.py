from __future__ import annotations

from hdc_nids.baselines import OnlineLSTMModel
from hdc_nids.data import RawRecord
from hdc_nids.preprocessing import TabularPreprocessor


def _record(duration: str, proto: str, label: str) -> RawRecord:
    return RawRecord(
        features={"dur": duration, "proto": proto},
        internal_label=label,
        binary_label=0 if label == "BENIGN" else 1,
        source="unit",
        stage_name="unit",
    )


def test_online_lstm_predicts_and_updates() -> None:
    records = [
        _record("0.1", "tcp", "BENIGN"),
        _record("0.2", "tcp", "BENIGN"),
        _record("4.0", "udp", "AttackA"),
        _record("4.2", "icmp", "AttackB"),
        _record("0.3", "tcp", "BENIGN"),
        _record("3.7", "udp", "AttackA"),
    ]
    preprocessor = TabularPreprocessor(
        class_labels=["BENIGN", "AttackA", "AttackB"],
        benign_label="BENIGN",
        forced_categorical={"proto"},
    )
    preprocessor.fit(records[:4])
    warmup = preprocessor.transform_records(records[:4], dataset="unit", window_id=0, stage_name="warmup")
    batch = preprocessor.transform_records(records[2:], dataset="unit", window_id=1, stage_name="online")

    model = OnlineLSTMModel(
        preprocessor=preprocessor,
        hidden_dim=8,
        sequence_length=3,
        learning_rate=0.01,
        epochs=1,
        batch_size=2,
        gradient_clip=1.0,
        update_sample_limit=0,
        dropout=0.1,
        seed=7,
    )
    model.fit_initial(warmup)
    prediction = model.predict(batch)
    events = model.update(batch, prediction, drift_active=False)
    updated_prediction = model.predict(batch)

    assert events == []
    assert prediction.class_scores.shape == (4, 3)
    assert prediction.predicted_binary.shape == (4,)
    assert updated_prediction.class_scores.shape == (4, 3)

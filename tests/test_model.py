from __future__ import annotations

import numpy as np

from hdc_nids.data import RawRecord
from hdc_nids.encoding import TabularHDCEncoder
from hdc_nids.models import OnlineHDModel, PredictionOutput
from hdc_nids.preprocessing import TabularPreprocessor


def _record(duration: str, proto: str, label: str) -> RawRecord:
    return RawRecord(
        features={"dur": duration, "proto": proto},
        internal_label=label,
        binary_label=0 if label == "BENIGN" else 1,
        source="unit",
        stage_name="unit",
    )


def test_binary_collapse_and_online_update_touch_expected_prototypes():
    warmup = [
        _record("0.0", "tcp", "BENIGN"),
        _record("5.0", "udp", "AttackA"),
    ]
    preprocessor = TabularPreprocessor(
        class_labels=["BENIGN", "AttackA", "AttackB"],
        benign_label="BENIGN",
        forced_categorical={"proto"},
    )
    preprocessor.fit(warmup)
    encoder = TabularHDCEncoder(preprocessor, dim=256, bins=64, seed=21)
    model = OnlineHDModel(
        encoder=encoder,
        preprocessor=preprocessor,
        base_lr=4.0,
        rare_class_boost=2.0,
        prototype_clip=2048,
    )

    attack_b_batch = preprocessor.transform_records(
        [_record("7.0", "icmp", "AttackA")],
        dataset="unit",
        window_id=0,
        stage_name="unit",
    )
    query_hv = encoder.encode_batch(attack_b_batch) / encoder.query_scale
    model.prototypes[0] = -query_hv[0]
    model.prototypes[2] = query_hv[0] * 0.9

    prediction = model.predict(attack_b_batch)
    assert prediction.predicted_binary[0] == 1

    controlled_prediction = PredictionOutput(
        predicted_class_indices=np.asarray([2], dtype=np.int32),
        predicted_labels=np.asarray(["AttackB"], dtype=object),
        predicted_binary=prediction.predicted_binary,
        class_scores=np.asarray([[0.1, 0.2, 0.8]], dtype=np.float32),
        attack_scores=prediction.attack_scores,
        benign_scores=prediction.benign_scores,
        binary_margin=prediction.binary_margin,
        query_hv=prediction.query_hv,
        encode_ms=prediction.encode_ms,
        score_ms=prediction.score_ms,
    )

    before = model.prototypes.copy()
    model.update(attack_b_batch, controlled_prediction)
    changed = np.where(np.any(np.abs(model.prototypes - before) > 1e-6, axis=1))[0].tolist()
    assert set(changed) == {1, 2}

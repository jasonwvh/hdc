from __future__ import annotations

import numpy as np

from hdc_nids.metrics import compute_continual_headline_metrics


def test_compute_continual_headline_metrics_aggregates_binary_and_multiclass() -> None:
    bundle = compute_continual_headline_metrics(
        true_class_indices=np.asarray([0, 1, 2, 0, 2], dtype=np.int32),
        predicted_class_indices=np.asarray([0, 1, 0, 0, 2], dtype=np.int32),
        true_binary=np.asarray([0, 1, 1, 0, 1], dtype=np.int8),
        predicted_binary=np.asarray([0, 1, 0, 0, 1], dtype=np.int8),
        attack_scores=np.asarray([0.1, 0.9, 0.2, 0.05, 0.8], dtype=np.float32),
        class_count=3,
        benign_index=0,
    )

    row = bundle.row
    assert row["headline_metric_basis"] == "aggregate_stream"
    assert float(row["headline_binary_accuracy"]) == 0.8
    assert float(row["headline_binary_f1"]) > 0.79
    assert float(row["headline_multiclass_accuracy"]) == 0.8
    assert float(row["headline_attack_recall_macro"]) == 0.75
    assert float(row["headline_binary_specificity"]) == 1.0

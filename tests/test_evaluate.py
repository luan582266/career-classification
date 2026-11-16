"""Unit test cho src/evaluate.py:
- Confusion Matrix (đa lớp)
- Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1
- Xử lý chia cho 0 (zero_division)
- Baseline ZeroR Classifier
- Kiểm tra tính hợp lệ của cấu trúc trả về EvaluationMetrics
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import (
    ZeroRClassifier,
    accuracy_score,
    confusion_matrix,
    evaluate_predictions,
    macro_f1_score,
    macro_precision_score,
    macro_recall_score,
    precision_recall_fscore_support,
    weighted_f1_score,
)


class TestMetricsCalculation:
    def test_confusion_matrix_shape_and_values(self) -> None:
        y_true = ["N1", "N1", "N2", "N3", "N4"]
        y_pred = ["N1", "N2", "N2", "N3", "N4"]
        labels = ["N1", "N2", "N3", "N4"]

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        assert cm.shape == (4, 4)
        assert cm[0, 0] == 1  # True N1, pred N1
        assert cm[0, 1] == 1  # True N1, pred N2
        assert cm[1, 1] == 1  # True N2, pred N2
        assert cm[2, 2] == 1  # True N3, pred N3
        assert cm[3, 3] == 1  # True N4, pred N4

    def test_accuracy_score(self) -> None:
        y_true = ["N1", "N2", "N3", "N4"]
        y_pred = ["N1", "N2", "N1", "N4"]
        assert accuracy_score(y_true, y_pred) == 0.75

    def test_accuracy_score_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            accuracy_score(["N1"], ["N1", "N2"])

    def test_accuracy_score_empty(self) -> None:
        with pytest.raises(ValueError):
            accuracy_score([], [])

    def test_precision_recall_fscore(self) -> None:
        # Lớp N1: 2 sample, pred đúng cả 2 -> precision=1.0, recall=1.0, f1=1.0
        # Lớp N2: 2 sample, pred đúng 1, 1 sample đoán nhầm sang N1
        # -> N1 có 1 FP -> N1 precision = 2 / (2 + 1) = 2/3
        y_true = ["N1", "N1", "N2", "N2"]
        y_pred = ["N1", "N1", "N1", "N2"]
        metrics = precision_recall_fscore_support(y_true, y_pred, labels=["N1", "N2"])

        assert metrics["N1"]["precision"] == pytest.approx(2 / 3)
        assert metrics["N1"]["recall"] == pytest.approx(1.0)
        assert metrics["N1"]["f1"] == pytest.approx(2 * (2 / 3 * 1.0) / (2 / 3 + 1.0))
        assert metrics["N1"]["support"] == 2

        assert metrics["N2"]["precision"] == pytest.approx(1.0)
        assert metrics["N2"]["recall"] == pytest.approx(0.5)
        assert metrics["N2"]["f1"] == pytest.approx(2 * (1.0 * 0.5) / (1.0 + 0.5))
        assert metrics["N2"]["support"] == 2

    def test_zero_division_handling(self) -> None:
        # Lớp N3 hoàn toàn không được dự đoán và không có trong y_true
        y_true = ["N1", "N2"]
        y_pred = ["N1", "N2"]
        metrics = precision_recall_fscore_support(
            y_true, y_pred, labels=["N1", "N2", "N3"], zero_division=0.0
        )
        assert metrics["N3"]["precision"] == 0.0
        assert metrics["N3"]["recall"] == 0.0
        assert metrics["N3"]["f1"] == 0.0
        assert metrics["N3"]["support"] == 0

    def test_macro_scores(self) -> None:
        y_true = ["N1", "N2", "N3"]
        y_pred = ["N1", "N2", "N3"]
        labels = ["N1", "N2", "N3"]

        assert macro_precision_score(y_true, y_pred, labels=labels) == 1.0
        assert macro_recall_score(y_true, y_pred, labels=labels) == 1.0
        assert macro_f1_score(y_true, y_pred, labels=labels) == 1.0
        assert weighted_f1_score(y_true, y_pred, labels=labels) == 1.0

    def test_evaluate_predictions_dataclass(self) -> None:
        y_true = ["N1", "N2", "N3", "N4"]
        y_pred = ["N1", "N2", "N3", "N4"]
        eval_result = evaluate_predictions(y_true, y_pred, model_name="PerfectModel")

        assert eval_result.model_name == "PerfectModel"
        assert eval_result.accuracy == 1.0
        assert eval_result.macro_f1 == 1.0
        row = eval_result.summary_row()
        assert row["Model"] == "PerfectModel"
        assert row["Macro F1"] == 1.0


class TestZeroRClassifier:
    def test_fit_and_predict(self) -> None:
        zeror = ZeroRClassifier()
        rows = [{"f1": 1}, {"f1": 2}, {"f1": 3}]
        labels = ["N1", "N2", "N1"]
        zeror.fit(rows, labels)

        assert zeror.majority_label_ == "N1"
        assert zeror.predict([{"f1": 10}, {"f1": 20}]) == ["N1", "N1"]

    def test_predict_before_fit_raises(self) -> None:
        zeror = ZeroRClassifier()
        with pytest.raises(RuntimeError):
            zeror.predict([{"f1": 1}])

    def test_empty_labels_raises(self) -> None:
        zeror = ZeroRClassifier()
        with pytest.raises(ValueError):
            zeror.fit([], [])

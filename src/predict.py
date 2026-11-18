
"""Suy luận hồ sơ thô bằng bộ tiền xử lý đã fit và giải thích nhóm nghề.

Ví dụ sử dụng::

    pipeline = load_pipeline("id3")
    result = pipeline.predict(profile, top_k=3)

Chỉ nạp artifact joblib từ nguồn tin cậy. ID3 xếp hạng theo phân bố nhãn
tại nút kết thúc, Naive Bayes dùng xác suất hậu nghiệm đã chuẩn hóa.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from . import config, schema
from .id3 import ID3Classifier
from .naive_bayes import UNKNOWN_VALUE, CategoricalNaiveBayes
from .preprocess import SurveyPreprocessor


def _group_name(label: Any) -> str:
    return schema.CAREER_GROUP_NAMES.get(label, str(label))


def _trace_id3(model: ID3Classifier, record: Mapping[str, Any]):
    if model.root is None:
        raise RuntimeError("Mô hình ID3 chưa được huấn luyện.")
    node = model.root
    path = []
    unmatched_feature = None
    while not node.is_leaf:
        feature = node.feature
        if feature not in record or record[feature] not in node.children:
            unmatched_feature = feature
            break
        value = record[feature]
        path.append({"feature": feature, "value": value})
        node = node.children[value]
    return node, path, unmatched_feature


def explain_id3(model: ID3Classifier, record: Mapping[str, Any]) -> dict[str, Any]:
    """Truy vết trên bản ghi đã tiền xử lý, dùng cùng fallback với predict_one.

    Các điều kiện là giá trị thực tế mô hình sử dụng. low ứng với 1–2,
    medium ứng với 3, high ứng với 4–5. Nhánh chưa gặp được ghi riêng,
    chỉ các nhánh đã đi qua mới xuất hiện trong chuỗi điều kiện.
    """
    node, path, unmatched_feature = _trace_id3(model, record)
    outcome = f"[Nhóm nghề = {node.prediction} ({_group_name(node.prediction)})]"
    conditions = []
    for step in path:
        conditions.append(f"[{step['feature']} = {step['value']}]")

    if conditions:
        condition_text = " VÀ ".join(conditions)
        rule = f"NẾU {condition_text} THÌ {outcome}"
    else:
        rule = f"Dự đoán {outcome}"
    if unmatched_feature is not None:
        rule += (
            f". Dùng nhãn đa số tại nút hiện tại vì đặc trưng "
            f"'{unmatched_feature}' bị thiếu hoặc có giá trị chưa gặp."
        )
    return {
        "prediction": node.prediction,
        "path": path,
        "fallback": unmatched_feature is not None,
        "unmatched_feature": unmatched_feature,
        "rule": rule,
    }


def explain_naive_bayes(
    model: CategoricalNaiveBayes, record: Mapping[str, Any]
) -> dict[str, Any]:
    """Giải thích log-odds so với nhóm có điểm log-joint cao thứ hai.

    Đóng góp của đặc trưng j là log P(x_j|nhóm dự đoán) trừ
    log P(x_j|nhóm so sánh), dùng đúng Laplace và unknown của mô hình.
    top_features chứa tối đa ba đóng góp dương theo thứ tự giảm dần.
    Tổng prior_log_odds và mọi contributions bằng log_odds hậu nghiệm.
    """
    prediction = model.predict_one(record)
    competitors = [label for label in model.classes_ if label != prediction]
    if not competitors:
        return {
            "prediction": prediction,
            "comparison_label": None,
            "prior_log_odds": None,
            "log_odds": None,
            "contributions": [],
            "top_features": [],
        }
    comparison = competitors[0]
    best_score = model.compute_log_joint_score(record, comparison)
    for label in competitors[1:]:
        score = model.compute_log_joint_score(record, label)
        if score > best_score:
            comparison = label
            best_score = score

    contributions = []
    for feature in model.feature_order:
        value = record.get(feature, UNKNOWN_VALUE)
        encoded_value = value
        if value not in model.feature_known_values_[feature]:
            encoded_value = UNKNOWN_VALUE
        log_likelihoods = {}
        for label in (prediction, comparison):
            log_likelihoods[label] = model.compute_log_conditional_probability(
                value_count=model.feature_value_counts_[feature][label].get(encoded_value, 0),
                class_total=model.feature_class_totals_[feature][label],
                cardinality=model.feature_cardinality_[feature],
                alpha=model.alpha,
            )
        contributions.append({
            "feature": feature,
            "value": value,
            "log_odds": log_likelihoods[prediction] - log_likelihoods[comparison],
        })
    prior_log_odds = model.log_priors_[prediction] - model.log_priors_[comparison]
    total_log_odds = prior_log_odds
    supports = []
    for contribution in contributions:
        total_log_odds += contribution["log_odds"]
        if contribution["log_odds"] > 0:
            supports.append(contribution)
    supports.sort(key=lambda item: item["log_odds"], reverse=True)

    return {
        "prediction": prediction,
        "comparison_label": comparison,
        "prior_log_odds": prior_log_odds,
        "log_odds": total_log_odds,
        "contributions": contributions,
        "top_features": supports[:3],
    }


class PredictionPipeline:
    """Dùng lại preprocessor đã fit, trả xếp hạng và giải thích cho một hồ sơ.

    Hồ sơ cần đủ khóa đặc trưng. Giá trị None hoặc ô trống được bộ tiền
    xử lý điền theo thống kê huấn luyện. Danh mục mới được mã hóa thành -1.
    """

    def __init__(
        self,
        model: ID3Classifier | CategoricalNaiveBayes,
        preprocessor: SurveyPreprocessor,
    ) -> None:
        if not isinstance(model, (ID3Classifier, CategoricalNaiveBayes)):
            raise TypeError("Mô hình phải là ID3Classifier hoặc CategoricalNaiveBayes.")
        if not preprocessor.is_fitted:
            raise RuntimeError("Bộ tiền xử lý chưa được fit.")
        if isinstance(model, ID3Classifier):
            if model.root is None:
                raise RuntimeError("Mô hình ID3 chưa được huấn luyện.")
        elif not model.is_fitted:
            raise RuntimeError("Mô hình Naive Bayes chưa được huấn luyện.")
        available_features = set(preprocessor.numeric_features)
        available_features.update(preprocessor.categorical_features)
        if not set(model.feature_order).issubset(available_features):
            raise ValueError("Đặc trưng mô hình không khớp bộ tiền xử lý.")
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, profile: Mapping[str, Any], *, top_k: int = 3) -> dict[str, Any]:
        """Trả prediction, top_groups, explanation và bản ghi đã xử lý.

        probability của ID3 là tỷ lệ nhãn ở nút kết thúc. Danh sách vẫn
        chứa các lớp học được có tỷ lệ 0 khi top_k lớn hơn số lớp tại nút.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k phải là số nguyên dương.")
        if not isinstance(profile, Mapping):
            raise TypeError("Hồ sơ phải là một mapping tên đặc trưng tới giá trị.")
        frame = pd.DataFrame([dict(profile)])
        transformed = self.preprocessor.transform(frame)
        feature_columns = list(self.model.feature_order)
        records = transformed[feature_columns].to_dict(orient="records")
        record = records[0]
        if isinstance(self.model, ID3Classifier):
            node, _, _ = _trace_id3(self.model, record)
            total = sum(node.class_counts.values())
            if total <= 0:
                raise ValueError("Artifact ID3 thiếu thống kê nhãn tại nút kết thúc.")
            probabilities = {}
            for label in self.model.root.class_counts:
                label_count = node.class_counts.get(label, 0)
                probabilities[label] = label_count / total
            explanation = explain_id3(self.model, record)
            probability_kind = "node_class_frequency"
        else:
            probabilities = self.model.predict_proba([record])[0]
            explanation = explain_naive_bayes(self.model, record)
            probability_kind = "posterior"
        prediction = explanation["prediction"]
        # Ưu tiên nhãn predict_one chọn khi xác suất hòa để hai kết quả nhất quán.
        ranked = sorted(probabilities, key=lambda label: (
            -probabilities[label], label != prediction, str(label)
        ))
        top_groups = []
        for label in ranked[:top_k]:
            top_groups.append({
                "label": label,
                "name": _group_name(label),
                "probability": probabilities[label],
            })

        return {
            "prediction": prediction,
            "top_groups": top_groups,
            "probability_kind": probability_kind,
            "explanation": explanation,
            "processed_profile": record,
        }


def load_pipeline(
    model_name: str = "id3",
    *,
    model_path: str | Path | None = None,
    preprocessor_path: str | Path = config.PREPROCESSOR_PATH,
) -> PredictionPipeline:
    """Nạp model .joblib và preprocessor .json theo đường dẫn cấu hình.

    model_name nhận id3 hoặc naive_bayes. Có thể truyền đường dẫn artifact
    riêng khi triển khai. Hai artifact cần được tạo từ cùng lần huấn luyện.
    """
    model_types = {"id3": ID3Classifier, "naive_bayes": CategoricalNaiveBayes}
    if model_name not in model_types:
        raise ValueError("model_name phải là 'id3' hoặc 'naive_bayes'.")
    if model_path is None:
        if model_name == "id3":
            model_path = config.ID3_MODEL_PATH
        else:
            model_path = config.NAIVE_BAYES_MODEL_PATH
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy mô hình: {path}")
    model = joblib.load(path)
    if not isinstance(model, model_types[model_name]):
        raise TypeError(f"Artifact không chứa mô hình {model_name}.")
    preprocessor = SurveyPreprocessor.load(preprocessor_path)
    return PredictionPipeline(model, preprocessor)


"""Kiểm thử suy luận hồ sơ và giải thích bằng mô hình được huấn luyện thật."""

from math import log

import joblib
import pandas as pd
import pytest

from src import schema
from src.id3 import ID3Classifier
from src.naive_bayes import CategoricalNaiveBayes
from src.predict import PredictionPipeline, explain_id3, explain_naive_bayes, load_pipeline
from src.preprocess import SurveyPreprocessor


@pytest.fixture
def training_data():
    records = []
    for score in [1, 1, 2, 4, 5, 5]:
        record = {}
        for feature in schema.NUMERIC_FEATURES:
            record[feature] = score
        for feature in schema.CATEGORICAL_FEATURES:
            record[feature] = schema.VALID_CATEGORIES[feature][0]
        records.append(record)
    frame = pd.DataFrame(records)
    preprocessor = SurveyPreprocessor()
    preprocessor.fit(frame)
    rows = preprocessor.transform(frame).to_dict(orient="records")
    labels = ["N1", "N1", "N1", "N2", "N2", "N2"]
    return frame, preprocessor, rows, labels


@pytest.mark.parametrize("model_type", [ID3Classifier, CategoricalNaiveBayes])
def test_raw_profile_prediction_and_artifact_round_trip(training_data, tmp_path, model_type):
    frame, preprocessor, rows, labels = training_data
    model = model_type()
    model.fit(rows, labels)
    pipeline = PredictionPipeline(model, preprocessor)
    profile = frame.iloc[0].to_dict()
    profile[schema.NUMERIC_FEATURES[1]] = None
    profile[schema.CATEGORICAL_FEATURES[0]] = "Danh mục mới"

    result = pipeline.predict(profile, top_k=2)

    assert result["prediction"] == "N1"
    assert [item["label"] for item in result["top_groups"]] == ["N1", "N2"]
    assert result["top_groups"][0]["name"] == schema.CAREER_GROUP_NAMES["N1"]
    total_probability = 0
    for group in result["top_groups"]:
        total_probability += group["probability"]
    assert total_probability == pytest.approx(1)
    assert pipeline.predict(profile, top_k=99)["top_groups"] == result["top_groups"]
    model_path = tmp_path / "model.joblib"
    preprocessor_path = preprocessor.save(tmp_path / "preprocessor.json")
    joblib.dump(model, model_path)
    if model_type is ID3Classifier:
        model_name = "id3"
    else:
        model_name = "naive_bayes"
    loaded = load_pipeline(model_name, model_path=model_path, preprocessor_path=preprocessor_path)
    assert loaded.predict(profile, top_k=2) == result


def test_id3_path_and_local_fallback():
    rows = [
        {"a": 0, "b": 0},
        {"a": 0, "b": 0},
        {"a": 0, "b": 1},
        {"a": 1, "b": 0},
        {"a": 1, "b": 1},
        {"a": 1, "b": 2},
    ]
    labels = ["N1", "N1", "N2", "N2", "N2", "N2"]
    model = ID3Classifier(["a", "b"])
    model.fit(rows, labels)

    result = explain_id3(model, {"a": 0, "b": 1})
    assert result["prediction"] == "N2"
    assert result["path"] == [{"feature": "a", "value": 0}, {"feature": "b", "value": 1}]
    expected_rule = (
        "NẾU [a = 0] VÀ [b = 1] THÌ "
        "[Nhóm nghề = N2 (Kinh doanh - Tài chính - Quản lý)]"
    )
    assert result["rule"] == expected_rule

    for record in ({"a": 0, "b": 99}, {"a": 0}):
        fallback = explain_id3(model, record)
        assert fallback["prediction"] == "N1"
        assert fallback["fallback"] is True
        assert fallback["path"] == [{"feature": "a", "value": 0}]
        assert fallback["unmatched_feature"] == "b"
        assert "nhãn đa số" in fallback["rule"]


def test_id3_leaf_explanation_has_no_invented_condition():
    model = ID3Classifier(["a"]).fit([{"a": 1}], ["N1"])
    result = explain_id3(model, {})
    assert result["path"] == []
    assert result["fallback"] is False
    assert "NẾU" not in result["rule"]


def test_nb_log_odds_are_against_runner_up_and_only_positive_supports():
    rows = [
        {"a": 1, "b": 1, "c": 1, "d": 1},
        {"a": 1, "b": 1, "c": 1, "d": 1},
        {"a": 1, "b": 1, "c": 1, "d": 1},
        {"a": 0, "b": 0, "c": 0, "d": 0},
        {"a": 0, "b": 0, "c": 0, "d": 0},
        {"a": 0, "b": 0, "c": 0, "d": 0},
    ]
    labels = ["N1", "N1", "N1", "N2", "N2", "N2"]
    model = CategoricalNaiveBayes(["a", "b", "c", "d"])
    model.fit(rows, labels)

    result = explain_naive_bayes(model, {"a": 1, "b": 1, "c": 1, "d": 0})

    assert result["prediction"] == "N1"
    assert result["comparison_label"] == "N2"
    assert [item["feature"] for item in result["top_features"]] == ["a", "b", "c"]
    assert [item["log_odds"] for item in result["top_features"]] == pytest.approx([log(4)] * 3)
    assert result["contributions"][-1]["log_odds"] == pytest.approx(-log(4))
    assert result["log_odds"] == pytest.approx(log(16))


def test_nb_unknown_values_and_single_class():
    model = CategoricalNaiveBayes(["a"]).fit([{"a": 0}, {"a": 1}], ["N1", "N2"])
    result = explain_naive_bayes(model, {"a": 99})
    assert result["top_features"] == []
    assert result["log_odds"] == pytest.approx(0)
    single_class_model = CategoricalNaiveBayes(["a"])
    single_class_model.fit([{"a": 0}], ["N1"])
    result = explain_naive_bayes(single_class_model, {})
    assert result["comparison_label"] is None
    assert result["top_features"] == []


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k(training_data, top_k):
    frame, preprocessor, rows, labels = training_data
    pipeline = PredictionPipeline(ID3Classifier().fit(rows, labels), preprocessor)
    with pytest.raises(ValueError):
        pipeline.predict(frame.iloc[0].to_dict(), top_k=top_k)


def test_profile_validation_and_unfitted_models(training_data):
    frame, preprocessor, rows, labels = training_data
    with pytest.raises(RuntimeError):
        explain_id3(ID3Classifier(["a"]), {})
    with pytest.raises(RuntimeError):
        explain_naive_bayes(CategoricalNaiveBayes(["a"]), {})
    pipeline = PredictionPipeline(ID3Classifier().fit(rows, labels), preprocessor)
    with pytest.raises(ValueError):
        pipeline.predict({})
    profile = frame.iloc[0].to_dict()
    profile[schema.NUMERIC_FEATURES[0]] = 6
    with pytest.raises(ValueError):
        pipeline.predict(profile)
    with pytest.raises(RuntimeError):
        PredictionPipeline(ID3Classifier(), preprocessor)
    with pytest.raises(RuntimeError):
        PredictionPipeline(ID3Classifier().fit(rows, labels), SurveyPreprocessor())


def test_loading_errors(tmp_path):
    with pytest.raises(ValueError):
        load_pipeline("invalid")
    with pytest.raises(FileNotFoundError):
        load_pipeline("id3", model_path=tmp_path / "missing.joblib")

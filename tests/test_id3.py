
"""Kiểm thử tính toán, huấn luyện và dự đoán của cây ID3."""

import pytest

from src.id3 import ID3Classifier, entropy, information_gain, select_best_feature
from src.schema import FEATURES


@pytest.mark.parametrize(
    "labels, expected",
    [
        ([], 0),
        (["a", "a", "a"], 0),
        (["a", "b"], 1),
        (["a", "a", "a", "b"], 0.8112781244591328),
    ],
)
def test_entropy(labels, expected):
    assert entropy(iter(labels)) == pytest.approx(expected)


def test_information_gain_and_best_feature():
    rows = [
        {"signal": 0, "constant": 0},
        {"signal": 0, "constant": 0},
        {"signal": 1, "constant": 0},
        {"signal": 1, "constant": 0},
    ]
    labels = ["a", "a", "b", "b"]

    assert information_gain(rows, labels, "signal") == pytest.approx(1)
    assert information_gain(rows, labels, "constant") == pytest.approx(0)
    assert select_best_feature(rows, labels, ["constant", "signal"]) == "signal"


def test_default_features_follow_schema():
    rows = []
    for value in [1, 2]:
        row = {feature: value for feature in FEATURES}
        rows.append(row)

    model = ID3Classifier().fit(rows, ["a", "b"])

    assert select_best_feature(rows, ["a", "b"]) == FEATURES[0]
    assert model.predict(rows) == ["a", "b"]


def test_multilevel_branching_and_local_fallback():
    rows = [
        {"a": 0, "b": 0},
        {"a": 0, "b": 0},
        {"a": 0, "b": 1},
        {"a": 1, "b": 0},
        {"a": 1, "b": 1},
        {"a": 1, "b": 2},
    ]
    labels = ["x", "x", "y", "y", "y", "y"]
    model = ID3Classifier(["a", "b"]).fit(rows, labels)
    assert model.root.feature == "a"
    assert model.predict(rows) == labels
    assert model.predict_one({"a": 0, "b": 99}) == "x"
    assert model.predict_one({"a": 0}) == "x"
    assert model.predict_one({"a": 99}) == "y"
    assert model.predict_one({}) == "y"


@pytest.mark.parametrize(
    "options",
    [{"max_depth": 0}, {"min_samples_split": 4}, {"min_gain": 1.1}],
)
def test_training_limits_return_majority_leaf(options):
    rows = [{"a": 0}, {"a": 1}, {"a": 1}]
    labels = ["x", "y", "y"]
    model = ID3Classifier(["a"], **options)

    model.fit(rows, labels)

    assert model.root.is_leaf
    assert model.predict_one({"a": 0}) == "y"


def test_min_samples_split_allows_split_at_boundary():
    rows = [{"a": 0}, {"a": 1}]
    model = ID3Classifier(["a"], min_samples_split=2)

    model.fit(rows, ["x", "y"])

    assert model.predict(rows) == ["x", "y"]


def test_zero_gain_stops_even_with_zero_min_gain():
    rows = [
        {"a": 0, "b": 0},
        {"a": 0, "b": 1},
        {"a": 1, "b": 0},
        {"a": 1, "b": 1},
    ]
    labels = ["x", "y", "y", "x"]
    model = ID3Classifier(["a", "b"], min_gain=0)

    model.fit(rows, labels)

    assert model.root.is_leaf


def test_pure_labels_and_exhausted_features_stop():
    pure_model = ID3Classifier(["a"])
    pure_model.fit([{"a": 0}, {"a": 1}], ["x", "x"])
    assert pure_model.root.is_leaf

    model = ID3Classifier(["a"])
    model.fit([{"a": 0}, {"a": 0}, {"a": 1}], ["x", "y", "y"])
    assert model.root.children[0].is_leaf
    assert model.predict_one({"a": 0}) == "x"


def test_export_nested_tree_is_independent():
    model = ID3Classifier(["a"]).fit([{"a": 0}, {"a": 1}], ["x", "y"])
    exported = model.to_dict()
    expected_tree = {
        "prediction": "x",
        "feature": "a",
        "class_counts": {"x": 1, "y": 1},
        "depth": 0,
        "children": {
            0: {
                "prediction": "x",
                "feature": None,
                "class_counts": {"x": 1},
                "depth": 1,
                "children": {},
            },
            1: {
                "prediction": "y",
                "feature": None,
                "class_counts": {"y": 1},
                "depth": 1,
                "children": {},
            },
        },
    }
    assert exported == expected_tree

    exported["children"][0]["class_counts"]["x"] = 99
    assert model.root.children[0].class_counts == {"x": 1}


@pytest.mark.parametrize("value", [0, 1, -1, 2.5, True])
def test_invalid_min_samples_split(value):
    with pytest.raises(ValueError):
        ID3Classifier(["a"], min_samples_split=value)


def test_untrained_and_invalid_training_inputs():
    model = ID3Classifier(["a"])
    with pytest.raises(RuntimeError):
        model.predict_one({"a": 0})
    with pytest.raises(RuntimeError):
        model.to_dict()
    with pytest.raises(ValueError):
        model.fit([], [])
    with pytest.raises(ValueError):
        model.fit([{"a": 0}], [])
    with pytest.raises(KeyError):
        model.fit([{}], ["x"])

"""Unit test cho src/naive_bayes.py: xác suất tiên nghiệm, xác suất có điều
kiện (làm trơn Laplace), bảng tần suất sau fit(), và kết quả phân loại.
"""

from __future__ import annotations

import math

import pytest

from src.naive_bayes import UNKNOWN_VALUE, CategoricalNaiveBayes


# ----------------------------------------------------------------------
# Dữ liệu mẫu dùng chung: 2 đặc trưng rời rạc, 2 lớp, tách biệt rõ ràng để
# dễ kiểm chứng bằng tay.
# ----------------------------------------------------------------------
TOY_ROWS = [
    {"mon": "Toan", "ky": "A"},
    {"mon": "Toan", "ky": "A"},
    {"mon": "Toan", "ky": "B"},
    {"mon": "Van", "ky": "A"},
    {"mon": "Van", "ky": "B"},
    {"mon": "Van", "ky": "B"},
]
TOY_LABELS = ["N1", "N1", "N1", "N2", "N2", "N2"]


@pytest.fixture
def toy_model() -> CategoricalNaiveBayes:
    model = CategoricalNaiveBayes(feature_order=["mon", "ky"], alpha=1.0)
    model.fit(TOY_ROWS, TOY_LABELS)
    return model


# ----------------------------------------------------------------------
# 1. Xác suất tiên nghiệm P(c)
# ----------------------------------------------------------------------
class TestPriors:
    def test_compute_class_counts(self) -> None:
        counts = CategoricalNaiveBayes.compute_class_counts(["N1", "N1", "N2", "N3"])
        assert counts == {"N1": 2, "N2": 1, "N3": 1}

    def test_compute_class_counts_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            CategoricalNaiveBayes.compute_class_counts([])

    def test_compute_log_priors_matches_hand_computation(self) -> None:
        counts = {"N1": 3, "N2": 1}
        priors = CategoricalNaiveBayes.compute_log_priors(counts)
        assert priors["N1"] == pytest.approx(math.log(3 / 4))
        assert priors["N2"] == pytest.approx(math.log(1 / 4))

    def test_log_priors_sum_to_one_after_exponentiating(self) -> None:
        counts = {"N1": 5, "N2": 3, "N3": 2, "N4": 10}
        priors = CategoricalNaiveBayes.compute_log_priors(counts)
        total_probability = sum(math.exp(value) for value in priors.values())
        assert total_probability == pytest.approx(1.0)


# ----------------------------------------------------------------------
# 2. Xác suất có điều kiện P(x_j | c) với làm trơn Laplace
# ----------------------------------------------------------------------
class TestConditionalProbability:
    def test_matches_hand_computation(self) -> None:
        # P(x=v|c) = (3 + 1) / (10 + 1*5) = 4/15
        p = CategoricalNaiveBayes.compute_conditional_probability(
            value_count=3, class_total=10, cardinality=5, alpha=1.0
        )
        assert p == pytest.approx(4 / 15)

    def test_unseen_combination_is_nonzero(self) -> None:
        # value_count = 0 (chưa từng gặp tổ hợp (giá trị, lớp) này trong train)
        p = CategoricalNaiveBayes.compute_conditional_probability(
            value_count=0, class_total=10, cardinality=5, alpha=1.0
        )
        assert p > 0.0

    def test_zero_alpha_can_produce_zero_probability(self) -> None:
        # alpha = 0 (không làm trơn) => xác suất có thể bằng 0 cho tổ hợp
        # chưa từng gặp, đúng như lý do cần Laplace smoothing.
        p = CategoricalNaiveBayes.compute_conditional_probability(
            value_count=0, class_total=10, cardinality=5, alpha=0.0
        )
        assert p == 0.0

    def test_rejects_negative_inputs(self) -> None:
        with pytest.raises(ValueError):
            CategoricalNaiveBayes.compute_conditional_probability(
                value_count=-1, class_total=10, cardinality=5, alpha=1.0
            )
        with pytest.raises(ValueError):
            CategoricalNaiveBayes.compute_conditional_probability(
                value_count=1, class_total=10, cardinality=0, alpha=1.0
            )

    def test_log_conditional_probability_matches_log_of_probability(self) -> None:
        p = CategoricalNaiveBayes.compute_conditional_probability(
            value_count=2, class_total=8, cardinality=4, alpha=1.0
        )
        log_p = CategoricalNaiveBayes.compute_log_conditional_probability(
            value_count=2, class_total=8, cardinality=4, alpha=1.0
        )
        assert log_p == pytest.approx(math.log(p))


# ----------------------------------------------------------------------
# 3. fit(): bảng tần suất 2 chiều
# ----------------------------------------------------------------------
class TestFit:
    def test_classes_and_class_counts(self, toy_model: CategoricalNaiveBayes) -> None:
        assert toy_model.classes_ == ("N1", "N2")
        assert toy_model.class_counts_ == {"N1": 3, "N2": 3}

    def test_feature_value_counts_table(self, toy_model: CategoricalNaiveBayes) -> None:
        assert toy_model.feature_value_counts_["mon"]["N1"] == {"Toan": 3}
        assert toy_model.feature_value_counts_["mon"]["N2"] == {"Van": 3}
        assert toy_model.feature_value_counts_["ky"]["N1"] == {"A": 2, "B": 1}
        assert toy_model.feature_value_counts_["ky"]["N2"] == {"A": 1, "B": 2}

    def test_feature_class_totals(self, toy_model: CategoricalNaiveBayes) -> None:
        assert toy_model.feature_class_totals_["mon"] == {"N1": 3, "N2": 3}
        assert toy_model.feature_class_totals_["ky"] == {"N1": 3, "N2": 3}

    def test_feature_cardinality_includes_unknown_slot(
        self, toy_model: CategoricalNaiveBayes
    ) -> None:
        # "mon" có 2 giá trị quan sát được (Toan, Van) -> cardinality = 2 + 1.
        assert toy_model.feature_cardinality_["mon"] == 3
        # "ky" có 2 giá trị quan sát được (A, B) -> cardinality = 2 + 1.
        assert toy_model.feature_cardinality_["ky"] == 3

    def test_feature_known_values(self, toy_model: CategoricalNaiveBayes) -> None:
        assert toy_model.feature_known_values_["mon"] == frozenset({"Toan", "Van"})
        assert toy_model.feature_known_values_["ky"] == frozenset({"A", "B"})

    def test_rejects_mismatched_lengths(self) -> None:
        model = CategoricalNaiveBayes(feature_order=["mon"])
        with pytest.raises(ValueError):
            model.fit([{"mon": "Toan"}], ["N1", "N2"])

    def test_rejects_empty_dataset(self) -> None:
        model = CategoricalNaiveBayes(feature_order=["mon"])
        with pytest.raises(ValueError):
            model.fit([], [])

    def test_rejects_rows_missing_declared_feature(self) -> None:
        model = CategoricalNaiveBayes(feature_order=["mon", "ky"])
        with pytest.raises(KeyError):
            model.fit([{"mon": "Toan"}], ["N1"])

    def test_default_feature_order_comes_from_schema(self) -> None:
        from src.schema import FEATURE_COLUMNS

        model = CategoricalNaiveBayes()
        assert model.feature_order == tuple(FEATURE_COLUMNS)


# ----------------------------------------------------------------------
# 4. predict() / predict_proba(): phân loại dựa trên điểm log-joint
# ----------------------------------------------------------------------
class TestPredict:
    def test_predict_picks_obvious_class(self, toy_model: CategoricalNaiveBayes) -> None:
        preds = toy_model.predict([{"mon": "Toan", "ky": "A"}, {"mon": "Van", "ky": "B"}])
        assert preds == ["N1", "N2"]

    def test_predict_proba_sums_to_one(self, toy_model: CategoricalNaiveBayes) -> None:
        proba = toy_model.predict_proba([{"mon": "Toan", "ky": "A"}])[0]
        assert sum(proba.values()) == pytest.approx(1.0)
        assert set(proba) == {"N1", "N2"}

    def test_predict_proba_favours_dominant_class(
        self, toy_model: CategoricalNaiveBayes
    ) -> None:
        proba = toy_model.predict_proba([{"mon": "Toan", "ky": "A"}])[0]
        assert proba["N1"] > proba["N2"]

    def test_predict_handles_unseen_category_via_unknown(
        self, toy_model: CategoricalNaiveBayes
    ) -> None:
        # "Ly" và "C" chưa từng xuất hiện trong tập huấn luyện của 2 đặc
        # trưng này: không được phép phát sinh lỗi (KeyError/ZeroDivision),
        # và vẫn phải trả về một phân phối xác suất hợp lệ.
        proba = toy_model.predict_proba([{"mon": "Ly", "ky": "C"}])[0]
        assert sum(proba.values()) == pytest.approx(1.0)
        assert all(value > 0 for value in proba.values())
        prediction = toy_model.predict([{"mon": "Ly", "ky": "C"}])
        assert prediction[0] in toy_model.classes_

    def test_unknown_slot_is_reserved_in_cardinality_only(
        self, toy_model: CategoricalNaiveBayes
    ) -> None:
        # __unknown__ không nằm trong danh mục đã biết, nhưng đã được dự trù
        # đúng 1 slot trong cardinality (số giá trị quan sát + 1).
        for feature in ("mon", "ky"):
            known = toy_model.feature_known_values_[feature]
            assert UNKNOWN_VALUE not in known
            assert toy_model.feature_cardinality_[feature] == len(known) + 1

    def test_predict_before_fit_raises(self) -> None:
        model = CategoricalNaiveBayes(feature_order=["mon"])
        with pytest.raises(RuntimeError):
            model.predict([{"mon": "Toan"}])

    def test_tie_break_prefers_earlier_class_in_sorted_order(self) -> None:
        # Dữ liệu đối xứng tuyệt đối giữa N1 và N2 => điểm log-joint bằng
        # nhau cho mọi bản ghi => phải chọn lớp xuất hiện trước theo thứ tự
        # đã sắp xếp trong classes_ (N1 < N2), tương tự cơ chế tie-break
        # theo thứ tự đặc trưng trong id3.py.
        rows = [{"mon": "Toan"}, {"mon": "Van"}]
        labels = ["N1", "N2"]
        model = CategoricalNaiveBayes(feature_order=["mon"], alpha=1.0)
        model.fit(rows, labels)
        assert model.classes_ == ("N1", "N2")

        proba = model.predict_proba([{"mon": "Pho_thong_khong_ton_tai"}])[0]
        assert proba["N1"] == pytest.approx(proba["N2"])
        assert model.predict([{"mon": "Pho_thong_khong_ton_tai"}]) == ["N1"]


# ----------------------------------------------------------------------
# 4b. Kiểm chứng công thức bằng tay, giá trị lạ, chống underflow
# ----------------------------------------------------------------------
class TestHandComputedScores:
    # TOY: P(N1) = 1/2; |V| = 2 giá trị quan sát + 1 slot unknown = 3;
    # mỗi lớp có 3 mẫu; alpha = 1.
    def test_log_joint_score_matches_hand_computation(
        self, toy_model: CategoricalNaiveBayes
    ) -> None:
        record = {"mon": "Toan", "ky": "A"}
        # P(Toan|N1) = (3+1)/(3+3) = 4/6 ; P(A|N1) = (2+1)/(3+3) = 3/6
        expected_n1 = math.log(1 / 2) + math.log(4 / 6) + math.log(3 / 6)
        # P(Toan|N2) = (0+1)/(3+3) = 1/6 ; P(A|N2) = (1+1)/(3+3) = 2/6
        expected_n2 = math.log(1 / 2) + math.log(1 / 6) + math.log(2 / 6)
        assert toy_model.compute_log_joint_score(record, "N1") == pytest.approx(expected_n1)
        assert toy_model.compute_log_joint_score(record, "N2") == pytest.approx(expected_n2)

    def test_predict_proba_matches_hand_computation(
        self, toy_model: CategoricalNaiveBayes
    ) -> None:
        proba = toy_model.predict_proba([{"mon": "Toan", "ky": "A"}])[0]
        s1 = (1 / 2) * (4 / 6) * (3 / 6)
        s2 = (1 / 2) * (1 / 6) * (2 / 6)
        assert proba["N1"] == pytest.approx(s1 / (s1 + s2))
        assert proba["N2"] == pytest.approx(s2 / (s1 + s2))

    def test_unknown_value_uses_laplace_formula(
        self, toy_model: CategoricalNaiveBayes
    ) -> None:
        # Giá trị lạ: count = 0 => P = alpha / (n_c + alpha * (|V|+1)) = 1/6.
        record = {"mon": "Ly", "ky": "C"}
        expected = math.log(1 / 2) + math.log(1 / 6) + math.log(1 / 6)
        assert toy_model.compute_log_joint_score(record, "N1") == pytest.approx(expected)
        assert toy_model.compute_log_joint_score(record, "N2") == pytest.approx(expected)

    def test_missing_feature_at_predict_is_treated_as_unknown(
        self, toy_model: CategoricalNaiveBayes
    ) -> None:
        with_missing = toy_model.compute_log_joint_score({"mon": "Toan"}, "N1")
        explicit_unknown = toy_model.compute_log_joint_score(
            {"mon": "Toan", "ky": UNKNOWN_VALUE}, "N1"
        )
        assert with_missing == pytest.approx(explicit_unknown)

    def test_log_space_avoids_underflow_with_many_features(self) -> None:
        # 500 đặc trưng, mỗi đặc trưng có 2 giá trị; tích trực tiếp các xác suất
        # sẽ về 0.0 trong float, nhưng điểm log phải hữu hạn.
        features = [f"f{i}" for i in range(500)]
        rows = [{f: "a" for f in features}, {f: "b" for f in features}]
        model = CategoricalNaiveBayes(feature_order=features, alpha=1.0)
        model.fit(rows, ["N1", "N2"])

        score = model.compute_log_joint_score(rows[0], "N1")
        assert math.isfinite(score)
        assert score < -300  # tương đương xác suất nhỏ hơn 1e-130
        proba = model.predict_proba([rows[0]])[0]
        assert all(math.isfinite(v) for v in proba.values())
        assert sum(proba.values()) == pytest.approx(1.0)
        assert model.predict([rows[0]]) == ["N1"]


class TestValidation:
    @pytest.mark.parametrize("alpha", [0, 0.0, -1.0])
    def test_constructor_requires_positive_alpha(self, alpha: float) -> None:
        with pytest.raises(ValueError):
            CategoricalNaiveBayes(feature_order=["mon"], alpha=alpha)

    def test_predict_with_unseen_value_does_not_crash_for_small_alpha(self) -> None:
        model = CategoricalNaiveBayes(feature_order=["mon"], alpha=1e-9)
        model.fit([{"mon": "Toan"}, {"mon": "Van"}], ["N1", "N2"])
        assert model.predict([{"mon": "Ly"}])[0] in model.classes_

    def test_rejects_empty_feature_order(self) -> None:
        with pytest.raises(ValueError):
            CategoricalNaiveBayes(feature_order=[])

    def test_rejects_duplicate_feature_order(self) -> None:
        with pytest.raises(ValueError):
            CategoricalNaiveBayes(feature_order=["mon", "mon"])

    def test_compute_log_priors_rejects_zero_total(self) -> None:
        with pytest.raises(ValueError):
            CategoricalNaiveBayes.compute_log_priors({"N1": 0})

    def test_log_joint_score_rejects_unknown_class(
        self, toy_model: CategoricalNaiveBayes
    ) -> None:
        with pytest.raises(KeyError):
            toy_model.compute_log_joint_score({"mon": "Toan", "ky": "A"}, "N9")

    def test_fit_rejects_reserved_unknown_value(self) -> None:
        model = CategoricalNaiveBayes(feature_order=["mon"])
        with pytest.raises(ValueError):
            model.fit([{"mon": UNKNOWN_VALUE}, {"mon": "Van"}], ["N1", "N2"])


# ----------------------------------------------------------------------
# 5. Kiểm thử tích hợp trên dữ liệu khảo sát thực tế (data/raw/survey.csv)
# ----------------------------------------------------------------------
class TestIntegrationWithSurveyData:
    def test_fit_predict_on_real_survey_data(self) -> None:
        pd = pytest.importorskip("pandas")
        from src import schema
        from src.config import RAW_SURVEY_PATH

        df = pd.read_csv(RAW_SURVEY_PATH)
        X, y, _ref = schema.split_features_and_labels(df)
        rows = X.to_dict(orient="records")
        labels = y.tolist()

        model = CategoricalNaiveBayes(alpha=1.0)
        model.fit(rows, labels)

        assert set(model.classes_) <= set(schema.VALID_TARGET_LABELS)

        predictions = model.predict(rows[:20])
        assert len(predictions) == 20
        assert all(pred in schema.VALID_TARGET_LABELS for pred in predictions)

        proba_list = model.predict_proba(rows[:5])
        for proba in proba_list:
            assert sum(proba.values()) == pytest.approx(1.0)

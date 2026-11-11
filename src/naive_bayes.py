"""Bộ phân lớp Categorical Naive Bayes tự xây dựng từ đầu cho bài toán phân loại nhóm nghề.

Mô-đun cài đặt đầy đủ Naive Bayes rời rạc (Categorical Naive Bayes):
    - Xác suất tiên nghiệm (prior) ``P(c)`` của từng nhóm nghề (N1, N2, N3, N4).
    - Xác suất có điều kiện (likelihood) ``P(x_j | c)`` của từng đặc trưng rời
      rạc, có áp dụng hệ số làm trơn Laplace ``alpha`` để tránh xác suất
      bằng 0 khi gặp tổ hợp (giá trị, lớp) chưa từng xuất hiện trong tập
      huấn luyện.
    - Công thức tính điểm phân loại trong không gian Log để chống tràn số
      dưới (underflow) khi nhân nhiều xác suất nhỏ với nhau:

          log P(c | x) ∝ log P(c) + Σ_j log P(x_j | c)

    - ``fit``: thống kê bảng tần suất 2 chiều giữa 16 đặc trưng và 4 nhóm
      nghề N1-N4 từ dữ liệu huấn luyện.
    - ``predict`` / ``predict_proba``: suy luận nhóm nghề dựa trên điểm
      log-joint, có xử lý danh mục mới (``__unknown__``) kết hợp làm trơn
      Laplace cho những giá trị chưa từng xuất hiện lúc huấn luyện.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from math import exp, log
from typing import TypeAlias

FeatureValue: TypeAlias = Hashable
Label: TypeAlias = Hashable
Record: TypeAlias = Mapping[str, FeatureValue]

# Mã định danh dùng khi gặp một giá trị đặc trưng chưa từng xuất hiện trong
# tập huấn luyện lúc dự đoán. Giá trị này được tính là một "danh mục" riêng
# (đã được dự trù sẵn trong |V_j| lúc fit) để làm trơn Laplace vẫn áp dụng
# đúng công thức mà không làm sai lệch cardinality của các danh mục đã biết.
UNKNOWN_VALUE = "__unknown__"


def _schema_feature_order() -> tuple[str, ...]:
    """Lấy thứ tự 16 đặc trưng từ schema (import muộn để tránh phụ thuộc vòng)."""
    from .schema import FEATURES

    return tuple(FEATURES)


class CategoricalNaiveBayes:
    """Bộ phân lớp Naive Bayes cho đặc trưng rời rạc (Categorical Naive Bayes).

    Thuật toán giả định các đặc trưng độc lập có điều kiện với nhau khi biết
    nhãn lớp ``c`` (giả định "naive"):

        P(c | x_1, ..., x_n) ∝ P(c) * Π_j P(x_j | c)

    Thuộc tính sau khi huấn luyện (``fit``):
        classes_: Các nhãn lớp đã học được, sắp xếp tăng dần (ví dụ
            ``("N1", "N2", "N3", "N4")``) để đảm bảo thứ tự xác định khi
            phá vỡ thế hòa (tie-breaking) trong ``predict``.
        class_counts_: Số lượng mẫu huấn luyện của từng lớp.
        log_priors_: Log xác suất tiên nghiệm ``log P(c)`` của từng lớp.
        feature_value_counts_: Bảng tần suất 2 chiều dạng
            ``{feature: {class: {value: count}}}`` dùng để tính ``P(x_j | c)``.
        feature_class_totals_: Tổng số mẫu của từng lớp ứng với từng đặc
            trưng (mẫu số khi tính xác suất có điều kiện).
        feature_cardinality_: Số lượng giá trị khác nhau ``|V_j|`` đã quan sát
            được của mỗi đặc trưng, cộng thêm 1 để dự trù cho danh mục mới
            (``__unknown__``) có thể gặp lúc dự đoán.
        feature_known_values_: Tập các giá trị đã quan sát được của từng đặc
            trưng trong tập huấn luyện; dùng để nhận diện danh mục mới lúc
            dự đoán và ánh xạ về ``__unknown__``.
    """

    def __init__(
        self,
        feature_order: Sequence[str] | None = None,
        *,
        alpha: float = 1.0,
    ) -> None:
        self.feature_order: tuple[str, ...] = (
            tuple(feature_order) if feature_order is not None else _schema_feature_order()
        )
        if not self.feature_order:
            raise ValueError("feature_order phải chứa ít nhất một đặc trưng.")
        if len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("feature_order không được chứa đặc trưng trùng lặp.")
        if alpha < 0:
            raise ValueError("alpha (hệ số làm trơn Laplace) phải >= 0.")

        self.alpha = alpha

        # Các thuộc tính sẽ được điền bởi fit().
        self.classes_: tuple[Label, ...] = ()
        self.class_counts_: dict[Label, int] = {}
        self.log_priors_: dict[Label, float] = {}
        self.feature_value_counts_: dict[str, dict[Label, Counter[FeatureValue]]] = {}
        self.feature_class_totals_: dict[str, dict[Label, int]] = {}
        self.feature_cardinality_: dict[str, int] = {}
        self.feature_known_values_: dict[str, frozenset[FeatureValue]] = {}

    @property
    def is_fitted(self) -> bool:
        """Trả về ``True`` nếu mô hình đã được huấn luyện (đã có ``classes_``)."""
        return bool(self.classes_)

    # ------------------------------------------------------------------
    # 1. Xác suất tiên nghiệm P(c)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_class_counts(labels: Sequence[Label]) -> dict[Label, int]:
        """Đếm số lượng mẫu của từng nhóm nghề (N1, N2, N3, N4) trong ``labels``."""
        if not labels:
            raise ValueError("Không thể tính tần suất lớp từ tập nhãn rỗng.")
        return dict(Counter(labels))

    @staticmethod
    def compute_log_priors(class_counts: Mapping[Label, int]) -> dict[Label, float]:
        """Tính log xác suất tiên nghiệm ``log P(c)`` cho từng lớp.

        Công thức: ``P(c) = count(c) / N`` với ``N`` là tổng số mẫu huấn
        luyện, sau đó chuyển sang không gian Log:

            log P(c) = log(count(c)) - log(N) = log(count(c) / N)

        Không áp dụng làm trơn Laplace ở bước này vì theo thiết kế dữ liệu,
        cả 4 nhóm nghề N1-N4 đều xuất hiện trong tập huấn luyện nên không có
        vấn đề xác suất tiên nghiệm bằng 0.
        """
        total = sum(class_counts.values())
        if total == 0:
            raise ValueError("Tổng số mẫu phải lớn hơn 0 để tính xác suất tiên nghiệm.")
        return {label: log(count / total) for label, count in class_counts.items()}

    # ------------------------------------------------------------------
    # 2. Xác suất có điều kiện P(x_j | c) với làm trơn Laplace
    # ------------------------------------------------------------------
    @staticmethod
    def compute_conditional_probability(
        value_count: int,
        class_total: int,
        cardinality: int,
        alpha: float = 1.0,
    ) -> float:
        """Tính ``P(x_j = v | c)`` có làm trơn Laplace (additive smoothing).

        Công thức:

            P(x_j = v | c) = (count(x_j = v, c) + alpha) /
                              (count(c) + alpha * |V_j|)

        Trong đó:
            value_count: Số mẫu thuộc lớp ``c`` có đặc trưng ``x_j = v``,
                tức ``count(x_j = v, c)``.
            class_total: Tổng số mẫu thuộc lớp ``c``, tức ``count(c)``.
            cardinality: Số lượng giá trị khác nhau ``|V_j|`` mà đặc trưng
                ``x_j`` có thể nhận (số danh mục quan sát được trong tập
                huấn luyện, ví dụ 5 đối với thang Likert 1-5).
            alpha: Hệ số làm trơn Laplace. ``alpha = 0`` nghĩa là không làm
                trơn (có thể gây xác suất bằng 0 với tổ hợp chưa từng gặp).

        Việc cộng thêm ``alpha`` vào tử số và ``alpha * |V_j|`` vào mẫu số
        đảm bảo ``P(x_j = v | c)`` không bao giờ bằng 0, kể cả khi
        ``value_count = 0`` (tổ hợp (giá trị, lớp) chưa từng xuất hiện trong
        tập huấn luyện) — tránh làm toàn bộ tích xác suất bằng 0.
        """
        if value_count < 0 or class_total < 0:
            raise ValueError("value_count và class_total phải >= 0.")
        if cardinality <= 0:
            raise ValueError("cardinality (|V_j|) phải lớn hơn 0.")
        if alpha < 0:
            raise ValueError("alpha phải >= 0.")

        numerator = value_count + alpha
        denominator = class_total + alpha * cardinality
        return numerator / denominator

    @classmethod
    def compute_log_conditional_probability(
        cls,
        value_count: int,
        class_total: int,
        cardinality: int,
        alpha: float = 1.0,
    ) -> float:
        """Giống ``compute_conditional_probability`` nhưng trả về giá trị Log.

        Dùng khi cộng dồn nhiều xác suất có điều kiện trong không gian Log
        (xem ``compute_log_joint_score``) thay vì nhân trực tiếp.
        """
        probability = cls.compute_conditional_probability(
            value_count=value_count,
            class_total=class_total,
            cardinality=cardinality,
            alpha=alpha,
        )
        return log(probability)

    # ------------------------------------------------------------------
    # 3. Chuyển đổi công thức tính điểm phân loại sang không gian Log
    # ------------------------------------------------------------------
    def compute_log_joint_score(self, record: Record, label: Label) -> float:
        """Tính điểm log-joint cho một lớp ``label`` ứng với bản ghi ``record``.

        Thay vì nhân trực tiếp nhiều xác suất nhỏ với nhau (dễ gây tràn số
        dưới - underflow khi có 16 đặc trưng), ta cộng các giá trị Log lại
        với nhau theo công thức:

            log P(c | x) ∝ log P(c) + Σ_j log P(x_j | c)

        Lớp dự đoán cuối cùng (ở Tuần 2) sẽ là lớp có ``compute_log_joint_score``
        lớn nhất trong 4 nhóm nghề N1-N4.

        Hàm này yêu cầu mô hình đã được huấn luyện (``fit`` sẽ hoàn thiện ở
        Tuần 2) để có sẵn ``log_priors_`` và các bảng tần suất
        ``feature_value_counts_`` / ``feature_class_totals_`` /
        ``feature_cardinality_``.
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")
        if label not in self.log_priors_:
            raise KeyError(f"Lớp '{label}' không tồn tại trong mô hình đã huấn luyện.")

        log_score = self.log_priors_[label]
        for feature in self.feature_order:
            value = record.get(feature, UNKNOWN_VALUE)
            known_values = self.feature_known_values_.get(feature, frozenset())
            if value not in known_values:
                # Danh mục chưa từng xuất hiện trong tập huấn luyện: ánh xạ
                # về __unknown__ (count = 0, đã được dự trù trong cardinality
                # của feature_cardinality_ ở bước fit) thay vì phát sinh lỗi.
                value = UNKNOWN_VALUE
            value_count = self.feature_value_counts_[feature][label].get(value, 0)
            class_total = self.feature_class_totals_[feature][label]
            cardinality = self.feature_cardinality_[feature]
            log_score += self.compute_log_conditional_probability(
                value_count=value_count,
                class_total=class_total,
                cardinality=cardinality,
                alpha=self.alpha,
            )
        return log_score

    # ------------------------------------------------------------------
    # 4. Huấn luyện: thống kê bảng tần suất 2 chiều cho 16 đặc trưng x N1-N4
    # ------------------------------------------------------------------
    def fit(self, rows: Sequence[Record], labels: Sequence[Label]) -> "CategoricalNaiveBayes":
        """Huấn luyện mô hình từ các bản ghi rời rạc và nhãn nhóm nghề tương ứng.

        Với mỗi đặc trưng trong ``feature_order``, thống kê bảng tần suất 2
        chiều ``{lớp: {giá trị: số lần xuất hiện}}`` để phục vụ tính
        ``P(x_j | c)`` ở bước dự đoán, đồng thời tính xác suất tiên nghiệm
        ``log P(c)`` từ phân bố nhãn.

        Args:
            rows: Danh sách bản ghi đầu vào, mỗi bản ghi phải có đủ các khóa
                trong ``feature_order``.
            labels: Nhãn nhóm nghề tương ứng với từng bản ghi trong ``rows``.

        Returns:
            Chính đối tượng ``self`` (cho phép gọi nối tiếp kiểu
            ``model = CategoricalNaiveBayes().fit(rows, labels)``).
        """
        rows = list(rows)
        labels = list(labels)
        if len(rows) != len(labels):
            raise ValueError(
                "Số lượng bản ghi và số lượng nhãn phải bằng nhau "
                f"(nhận được {len(rows)} và {len(labels)})."
            )
        if not rows:
            raise ValueError("Không thể huấn luyện Naive Bayes với tập dữ liệu rỗng.")

        for row in rows:
            missing_features = [f for f in self.feature_order if f not in row]
            if missing_features:
                missing_text = ", ".join(missing_features)
                raise KeyError(f"Bản ghi thiếu đặc trưng: {missing_text}.")

        self.class_counts_ = self.compute_class_counts(labels)
        # Sắp xếp để classes_ có thứ tự xác định (N1 < N2 < N3 < N4), dùng
        # làm thứ tự duyệt khi phá vỡ thế hòa (tie-breaking) trong predict().
        self.classes_ = tuple(sorted(self.class_counts_, key=str))
        self.log_priors_ = self.compute_log_priors(self.class_counts_)

        self.feature_value_counts_ = {}
        self.feature_class_totals_ = {}
        self.feature_cardinality_ = {}
        self.feature_known_values_ = {}

        for feature in self.feature_order:
            value_counts: dict[Label, Counter[FeatureValue]] = {
                label: Counter() for label in self.classes_
            }
            observed_values: set[FeatureValue] = set()
            for row, label in zip(rows, labels, strict=True):
                value = row[feature]
                observed_values.add(value)
                value_counts[label][value] += 1

            self.feature_value_counts_[feature] = value_counts
            # Giả định dữ liệu đầy đủ (không có giá trị khuyết) nên tổng số
            # mẫu của một lớp là như nhau ở mọi đặc trưng, chính là
            # class_counts_[label].
            self.feature_class_totals_[feature] = dict(self.class_counts_)
            self.feature_known_values_[feature] = frozenset(observed_values)
            # +1 để dự trù một "danh mục" cho các giá trị chưa từng gặp
            # (__unknown__) sẽ xuất hiện lúc dự đoán.
            self.feature_cardinality_[feature] = len(observed_values) + 1

        return self

    # 5. Suy luận: predict_proba() và predict()
    def predict_proba(self, rows: Iterable[Record]) -> list[dict[Label, float]]:
        """Trả về xác suất hậu nghiệm (đã chuẩn hóa) của từng lớp cho mỗi bản ghi.

        Dùng kỹ thuật log-sum-exp (trừ đi điểm log lớn nhất trước khi mũ hóa
        lại bằng ``exp``) để chuẩn hóa tổng xác suất về 1 mà không gây tràn
        số khi ``exp`` của một log-score rất âm.
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        results: list[dict[Label, float]] = []
        for row in rows:
            log_scores = {
                label: self.compute_log_joint_score(row, label) for label in self.classes_
            }
            max_log_score = max(log_scores.values())
            exp_scores = {
                label: exp(score - max_log_score) for label, score in log_scores.items()
            }
            total = sum(exp_scores.values())
            results.append({label: value / total for label, value in exp_scores.items()})
        return results

    def predict_one(self, row: Record) -> Label:
        """Dự đoán nhóm nghề cho một bản ghi: lớp có điểm log-joint cao nhất.

        Nếu nhiều lớp cùng đạt điểm cao nhất (thế hòa), lớp xuất hiện trước
        trong ``classes_`` (đã sắp xếp N1 < N2 < N3 < N4) được chọn, giống
        cơ chế phá vỡ thế hòa theo thứ tự định nghĩa của ``id3.py``.
        """
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        log_scores = {
            label: self.compute_log_joint_score(row, label) for label in self.classes_
        }
        return max(self.classes_, key=lambda label: log_scores[label])

    def predict(self, rows: Iterable[Record]) -> list[Label]:
        """Dự đoán nhóm nghề có điểm log-joint cao nhất cho nhiều bản ghi."""
        return [self.predict_one(row) for row in rows]
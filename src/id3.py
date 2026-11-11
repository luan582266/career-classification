"""Cây quyết định ID3 cho dữ liệu phân loại rời rạc.

Các thuật toán không phụ thuộc vào pandas. Nếu không truyền ``feature_order``,
mô-đun dùng ``schema.FEATURES`` (thứ tự khai báo tại ``src/schema.py``).
Thứ tự này được dùng để phá vỡ thế hòa khi nhiều đặc trưng có cùng
Information Gain.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import log2
from typing import TypeAlias


FeatureValue: TypeAlias = Hashable
Label: TypeAlias = Hashable
Record: TypeAlias = Mapping[str, FeatureValue]

_EPSILON = 1e-12
_MISSING = object()


@dataclass(slots=True)
class Node:
    """Một nút của cây ID3.

    ``children`` là từ điển ``giá trị đặc trưng -> Node`` nên hỗ trợ trực tiếp
    phân nhánh đa giá trị, ví dụ Holland có thể tách thành R, I, A, S, E, C.
    ``prediction`` lưu nhãn đa số tại nút để dự phòng khi dự đoán gặp giá trị
    đặc trưng chưa xuất hiện trong dữ liệu huấn luyện.
    """

    prediction: Label
    feature: str | None = None
    children: dict[FeatureValue, "Node"] = field(default_factory=dict)
    class_counts: dict[Label, int] = field(default_factory=dict)
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        """Trả về ``True`` nếu nút không còn tách theo đặc trưng nào."""
        return self.feature is None


def entropy(labels: Iterable[Label]) -> float:
    """Tính entropy cơ số 2 của một tập nhãn.

    Với tập rỗng hoặc chỉ có một nhãn, entropy bằng 0.
    """
    label_list = list(labels)
    if not label_list:
        return 0.0

    counts = Counter(label_list)
    total = len(label_list)
    return -sum(
        (count / total) * log2(count / total)
        for count in counts.values()
        if count
    )


def information_gain(
    rows: Sequence[Record],
    labels: Sequence[Label],
    feature: str,
) -> float:
    """Tính Information Gain khi tách ``rows`` theo ``feature``.

    Công thức: ``IG(S, A) = H(S) - Σ(|S_v| / |S|) * H(S_v)``.
    """
    _validate_rows_and_labels(rows, labels)
    if not rows:
        return 0.0

    partitions: dict[FeatureValue, list[Label]] = {}
    for row, label in zip(rows, labels, strict=True):
        if feature not in row:
            raise KeyError(f"Không tìm thấy đặc trưng '{feature}' trong một bản ghi.")
        partitions.setdefault(row[feature], []).append(label)

    total = len(labels)
    weighted_entropy = sum(
        (len(partition_labels) / total) * entropy(partition_labels)
        for partition_labels in partitions.values()
    )
    return entropy(labels) - weighted_entropy


def _schema_feature_order() -> tuple[str, ...]:
    """Lấy thứ tự đặc trưng từ schema (import muộn để tránh kéo pandas khi không cần)."""
    from .schema import FEATURES

    return tuple(FEATURES)


def select_best_feature(
    rows: Sequence[Record],
    labels: Sequence[Label],
    feature_order: Sequence[str] | None = None,
) -> str:
    """Chọn đặc trưng có gain cao nhất theo thứ tự schema khi hòa điểm.

    Các đặc trưng được duyệt đúng theo ``feature_order`` (mặc định là
    ``schema.FEATURES``). Khi gain bằng nhau trong sai số số thực, đặc trưng
    xuất hiện trước sẽ được giữ lại.
    """
    return _select_best_feature_with_gain(rows, labels, feature_order)[0]


def _select_best_feature_with_gain(
    rows: Sequence[Record],
    labels: Sequence[Label],
    feature_order: Sequence[str] | None = None,
) -> tuple[str, float]:
    """Giống ``select_best_feature`` nhưng trả thêm gain để tránh tính lại."""
    _validate_rows_and_labels(rows, labels)
    ordered_features = tuple(feature_order)
    if not ordered_features:
        raise ValueError("feature_order phải chứa ít nhất một đặc trưng.")
    if len(set(ordered_features)) != len(ordered_features):
        raise ValueError("feature_order không được chứa đặc trưng trùng lặp.")

    best_feature = ordered_features[0]
    best_gain = information_gain(rows, labels, best_feature)

    for feature in ordered_features[1:]:
        gain = information_gain(rows, labels, feature)
        if gain > best_gain + _EPSILON:
            best_feature = feature
            best_gain = gain

    return best_feature, best_gain


class ID3Classifier:
    """Bộ phân lớp ID3 hỗ trợ đặc trưng phân loại và nhánh đa giá trị."""

    def __init__(
        self,
        feature_order: Sequence[str] | None = None,
        *,
        min_gain: float = _EPSILON,
        max_depth: int | None = None,
    ) -> None:
        self.feature_order = tuple(feature_order)
        if not self.feature_order:
            raise ValueError("feature_order phải chứa ít nhất một đặc trưng.")
        if len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("feature_order không được chứa đặc trưng trùng lặp.")
        if min_gain < 0:
            raise ValueError("min_gain phải lớn hơn hoặc bằng 0.")
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth phải lớn hơn hoặc bằng 0 hoặc là None.")

        self.min_gain = min_gain
        self.max_depth = max_depth
        self.root: Node | None = None

    def fit(
        self,
        rows: Sequence[Record],
        labels: Sequence[Label],
    ) -> "ID3Classifier":
        """Huấn luyện cây ID3 từ các bản ghi rời rạc và nhãn tương ứng."""
        _validate_rows_and_labels(rows, labels)
        if not rows:
            raise ValueError("Không thể huấn luyện ID3 với tập dữ liệu rỗng.")

        for row in rows:
            missing_features = [
                feature for feature in self.feature_order if feature not in row
            ]
            if missing_features:
                missing_text = ", ".join(missing_features)
                raise KeyError(f"Bản ghi thiếu đặc trưng: {missing_text}.")

        self.root = self._build_tree(
            rows=list(rows),
            labels=list(labels),
            available_features=self.feature_order,
            depth=0,
        )
        return self

    def predict_one(self, row: Record) -> Label:
        """Dự đoán nhãn cho một bản ghi.

        Khi gặp giá trị mới hoặc thiếu đặc trưng tại một nhánh, mô hình trả về
        nhãn đa số đã lưu ở nút hiện tại thay vì phát sinh lỗi.
        """
        if self.root is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        node = self.root
        while not node.is_leaf:
            assert node.feature is not None
            value = row.get(node.feature, _MISSING)
            next_node = node.children.get(value)
            if next_node is None:
                return node.prediction
            node = next_node
        return node.prediction

    def predict(self, rows: Iterable[Record]) -> list[Label]:
        """Dự đoán nhãn cho nhiều bản ghi theo thứ tự đầu vào."""
        return [self.predict_one(row) for row in rows]

    def _build_tree(
        self,
        *,
        rows: list[Record],
        labels: list[Label],
        available_features: tuple[str, ...],
        depth: int,
    ) -> Node:
        class_counts = dict(Counter(labels))
        node = Node(
            prediction=_majority_label(labels),
            class_counts=class_counts,
            depth=depth,
        )

        if (
            len(class_counts) == 1
            or not available_features
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node

        feature = select_best_feature(rows, labels, available_features)
        gain = information_gain(rows, labels, feature)
        if gain < self.min_gain + _EPSILON:
            return node

        node.feature = feature
        remaining_features = tuple(
            candidate for candidate in available_features if candidate != feature
        )
        partitions: dict[FeatureValue, tuple[list[Record], list[Label]]] = {}

        for row, label in zip(rows, labels, strict=True):
            value = row[feature]
            child_rows, child_labels = partitions.setdefault(value, ([], []))
            child_rows.append(row)
            child_labels.append(label)

        for value, (child_rows, child_labels) in partitions.items():
            node.children[value] = self._build_tree(
                rows=child_rows,
                labels=child_labels,
                available_features=remaining_features,
                depth=depth + 1,
            )
        return node


def _majority_label(labels: Sequence[Label]) -> Label:
    """Chọn nhãn nhiều nhất; nếu hòa, giữ nhãn xuất hiện sớm hơn."""
    if not labels:
        raise ValueError("Không thể chọn nhãn đa số từ tập rỗng.")

    counts = Counter(labels)
    best_label = labels[0]
    best_count = counts[best_label]
    for label in labels[1:]:
        if counts[label] > best_count:
            best_label = label
            best_count = counts[label]
    return best_label


def _validate_rows_and_labels(
    rows: Sequence[Record],
    labels: Sequence[Label],
) -> None:
    if len(rows) != len(labels):
        raise ValueError(
            "Số lượng bản ghi và số lượng nhãn phải bằng nhau "
            f"(nhận được {len(rows)} và {len(labels)})."
        )

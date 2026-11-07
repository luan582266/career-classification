
"""Cây quyết định ID3 cho dữ liệu phân loại rời rạc.

Mô-đun không phụ thuộc vào pandas hoặc schema cụ thể. Phần gọi mô-đun cần
truyền ``feature_order`` theo đúng thứ tự khai báo tại ``src/schema.py``.
Thứ tự này được dùng để phá vỡ thế hòa khi nhiều đặc trưng có cùng
Information Gain.
"""

from __future__ import annotations

# Counter giúp đếm số lần xuất hiện của từng nhãn hoặc từng lớp.
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import log2  # ID3 sử dụng log cơ số 2 để tính entropy.
from typing import TypeAlias


# Giá trị đặc trưng và nhãn phải có thể băm để dùng làm key của dictionary.
FeatureValue: TypeAlias = Hashable
Label: TypeAlias = Hashable

# Một bản ghi là dictionary dạng: {"tên_đặc_trưng": giá_trị}.
Record: TypeAlias = Mapping[str, FeatureValue]

# Sai số nhỏ dùng khi so sánh các giá trị Information Gain dạng số thực.
_EPSILON = 1e-12

# Giá trị đặc biệt biểu diễn trường hợp bản ghi dự đoán bị thiếu đặc trưng.
_MISSING = object()


@dataclass(slots=True)
class Node:
    """Một nút của cây ID3.

    ``children`` là từ điển ``giá trị đặc trưng -> Node`` nên hỗ trợ trực tiếp
    phân nhánh đa giá trị, ví dụ Holland có thể tách thành R, I, A, S, E, C.
    ``prediction`` lưu nhãn đa số tại nút để dự phòng khi dự đoán gặp giá trị
    đặc trưng chưa xuất hiện trong dữ liệu huấn luyện.
    """

    # Nhãn đa số tại nút, dùng để dự phòng khi không đi được xuống nhánh con.
    prediction: Label

    # Tên đặc trưng dùng để chia tại nút; None nghĩa là nút lá.
    feature: str | None = None

    # Ánh xạ từ giá trị đặc trưng sang nút con tương ứng.
    children: dict[FeatureValue, "Node"] = field(default_factory=dict)

    # Số lượng của từng nhãn trong tập dữ liệu tại nút này.
    class_counts: dict[Label, int] = field(default_factory=dict)

    # Độ sâu của nút trong cây; nút gốc có độ sâu bằng 0.
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        """Trả về ``True`` nếu nút không còn tách theo đặc trưng nào."""
        return self.feature is None


def entropy(labels: Iterable[Label]) -> float:
    """Tính entropy cơ số 2 của một tập nhãn.

    Với tập rỗng hoặc chỉ có một nhãn, entropy bằng 0.
    """
    # Chuyển sang list để có thể đếm số phần tử và duyệt nhiều lần.
    label_list = list(labels)
    if not label_list:
        return 0.0

    # Đếm số lần xuất hiện của mỗi nhãn để tính xác suất của từng nhãn.
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
    # Đảm bảo mỗi bản ghi có đúng một nhãn tương ứng.
    _validate_rows_and_labels(rows, labels)
    if not rows:
        return 0.0

    # Chia các nhãn thành nhiều nhóm theo từng giá trị của đặc trưng.
    # Ví dụ: "màu" có thể tạo ra các nhóm "Đỏ", "Xanh", "Vàng".
    partitions: dict[FeatureValue, list[Label]] = {}
    for row, label in zip(rows, labels, strict=True):
        if feature not in row:
            raise KeyError(f"Không tìm thấy đặc trưng '{feature}' trong một bản ghi.")
        partitions.setdefault(row[feature], []).append(label)

    total = len(labels)
    # Tính entropy trung bình có trọng số sau khi chia dữ liệu.
    weighted_entropy = sum(
        (len(partition_labels) / total) * entropy(partition_labels)
        for partition_labels in partitions.values()
    )
    # Gain càng lớn thì đặc trưng càng phù hợp để dùng phân nhánh.
    return entropy(labels) - weighted_entropy


def select_best_feature(
    rows: Sequence[Record],
    labels: Sequence[Label],
    feature_order: Sequence[str],
) -> str:
    """Chọn đặc trưng có gain cao nhất theo thứ tự schema khi hòa điểm.

    Các đặc trưng được duyệt đúng theo ``feature_order``. Khi gain bằng nhau
    trong sai số số thực, đặc trưng xuất hiện trước sẽ được giữ lại.
    """
    _validate_rows_and_labels(rows, labels)
    # Giữ nguyên thứ tự đặc trưng để xử lý trường hợp các Gain bằng nhau.
    ordered_features = tuple(feature_order)
    if not ordered_features:
        raise ValueError("feature_order phải chứa ít nhất một đặc trưng.")
    if len(set(ordered_features)) != len(ordered_features):
        raise ValueError("feature_order không được chứa đặc trưng trùng lặp.")

    # Tạm chọn đặc trưng đầu tiên làm đặc trưng tốt nhất.
    best_feature = ordered_features[0]
    best_gain = information_gain(rows, labels, best_feature)

    # So sánh Gain của các đặc trưng còn lại.
    for feature in ordered_features[1:]:
        gain = information_gain(rows, labels, feature)
        if gain > best_gain + _EPSILON:
            best_feature = feature
            best_gain = gain

    return best_feature


class ID3Classifier:
    """Bộ phân lớp ID3 hỗ trợ đặc trưng phân loại và nhánh đa giá trị."""

    def __init__(
        self,
        feature_order: Sequence[str],
        *,
        min_gain: float = _EPSILON,
        max_depth: int | None = None,
    ) -> None:
        # Lưu danh sách đặc trưng dưới dạng tuple để không bị thay đổi ngoài ý muốn.
        self.feature_order = tuple(feature_order)
        if not self.feature_order:
            raise ValueError("feature_order phải chứa ít nhất một đặc trưng.")
        if len(set(self.feature_order)) != len(self.feature_order):
            raise ValueError("feature_order không được chứa đặc trưng trùng lặp.")
        if min_gain < 0:
            raise ValueError("min_gain phải lớn hơn hoặc bằng 0.")
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth phải lớn hơn hoặc bằng 0 hoặc là None.")

        # min_gain ngăn cây chia những nhánh không đem lại nhiều thông tin.
        self.min_gain = min_gain

        # max_depth giới hạn độ sâu để tránh cây quá phức tạp.
        self.max_depth = max_depth

        # root sẽ được tạo sau khi gọi fit().
        self.root: Node | None = None

    def fit(
        self,
        rows: Sequence[Record],
        labels: Sequence[Label],
    ) -> "ID3Classifier":
        """Huấn luyện cây ID3 từ các bản ghi rời rạc và nhãn tương ứng."""
        # Kiểm tra dữ liệu trước khi bắt đầu huấn luyện.
        _validate_rows_and_labels(rows, labels)
        if not rows:
            raise ValueError("Không thể huấn luyện ID3 với tập dữ liệu rỗng.")

        # Mọi bản ghi huấn luyện phải có đầy đủ các đặc trưng đã khai báo.
        for row in rows:
            missing_features = [
                feature for feature in self.feature_order if feature not in row
            ]
            if missing_features:
                missing_text = ", ".join(missing_features)
                raise KeyError(f"Bản ghi thiếu đặc trưng: {missing_text}.")

        # Xây cây đệ quy, bắt đầu từ toàn bộ dữ liệu tại độ sâu 0.
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
        # Không thể dự đoán nếu mô hình chưa được huấn luyện.
        if self.root is None:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        # Bắt đầu duyệt cây từ nút gốc.
        node = self.root
        while not node.is_leaf:
            assert node.feature is not None
            # Nếu thiếu đặc trưng, dùng _MISSING để chắc chắn không trùng giá trị hợp lệ.
            value = row.get(node.feature, _MISSING)
            next_node = node.children.get(value)
            # Nếu gặp giá trị chưa từng xuất hiện, dùng nhãn đa số tại nút hiện tại.
            if next_node is None:
                return node.prediction
            node = next_node
        return node.prediction

    def predict(self, rows: Iterable[Record]) -> list[Label]:
        """Dự đoán nhãn cho nhiều bản ghi theo thứ tự đầu vào."""
        # Dự đoán lần lượt từng bản ghi và giữ nguyên thứ tự đầu vào.
        return [self.predict_one(row) for row in rows]

    def _build_tree(
        self,
        *,
        rows: list[Record],
        labels: list[Label],
        available_features: tuple[str, ...],
        depth: int,
    ) -> Node:
        # Thống kê số lượng từng lớp trong tập dữ liệu hiện tại.
        class_counts = dict(Counter(labels))

        # Mỗi nút luôn lưu nhãn đa số để có thể dự đoán khi cần dừng chia.
        node = Node(
            prediction=_majority_label(labels),
            class_counts=class_counts,
            depth=depth,
        )

        # Dừng nếu dữ liệu đã thuần nhất, hết đặc trưng hoặc đạt max_depth.
        if (
            len(class_counts) == 1
            or not available_features
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node

        # Chọn đặc trưng có Information Gain lớn nhất.
        feature = select_best_feature(rows, labels, available_features)
        gain = information_gain(rows, labels, feature)
        # Gain quá thấp nghĩa là chia tiếp không mang lại thông tin đáng kể.
        if gain <= self.min_gain + _EPSILON:
            return node

        # Ghi nhận đặc trưng được chọn tại nút hiện tại.
        node.feature = feature
        # Đặc trưng vừa dùng không được dùng lại trên các nút con của nhánh này.
        remaining_features = tuple(
            candidate for candidate in available_features if candidate != feature
        )
        # Tạo các nhóm bản ghi và nhãn theo từng giá trị của đặc trưng.
        partitions: dict[FeatureValue, tuple[list[Record], list[Label]]] = {}

        for row, label in zip(rows, labels, strict=True):
            value = row[feature]
            child_rows, child_labels = partitions.setdefault(value, ([], []))
            child_rows.append(row)
            child_labels.append(label)

        # Với mỗi nhóm, đệ quy xây dựng một nút con.
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

    # Đếm số lần xuất hiện của từng nhãn.
    counts = Counter(labels)
    # Chọn nhãn đầu tiên làm phương án ban đầu; nếu hòa thì giữ nhãn này.
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
    # Mỗi bản ghi phải ghép được với đúng một nhãn mục tiêu.
    if len(rows) != len(labels):
        raise ValueError(
            "Số lượng bản ghi và số lượng nhãn phải bằng nhau "
            f"(nhận được {len(rows)} và {len(labels)})."
        )

"""Mô-đun đánh giá hiệu năng mô hình phân loại nhóm nghề (Career Classification).

Cung cấp đầy đủ các hàm tính toán chỉ số đánh giá đa lớp:
    - Confusion Matrix (Ma trận nhầm lẫn)
    - Accuracy (Độ chính xác toàn cục)
    - Per-class Precision, Recall, F1-Score, Support
    - Macro Precision, Macro Recall, Macro F1
    - Weighted F1
    - Báo cáo phân loại chi tiết (Classification Report)

Đồng thời hỗ trợ:
    - Mô hình Baseline ZeroR (Majority Class Classifier) để làm chuẩn so sánh.
    - Vẽ và xuất biểu đồ ma trận nhầm lẫn (Heatmap Confusion Matrix) cho ID3 và Naive Bayes.
    - Vẽ biểu đồ cột so sánh Macro F1 giữa ID3, Naive Bayes và Baseline (ZeroR).
    - Lưu trữ số liệu đánh giá dưới dạng JSON và CSV vào thư mục reports/.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi UnicodeEncodeError
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    COMPARISON_CSV_PATH,
    FIGURES_DIR,
    ID3_CM_FIG_PATH,
    ID3_METRICS_PATH,
    ID3_MODEL_PATH,
    METRICS_DIR,
    MODEL_COMPARISON_FIG_PATH,
    NAIVE_BAYES_MODEL_PATH,
    NB_CM_FIG_PATH,
    NB_METRICS_PATH,
    PREDICTIONS_DIR,
    RANDOM_SEED,
    RAW_SURVEY_PATH,
    TEST_DATA_PATH,
    TEST_PREDICTIONS_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
)
from .id3 import ID3Classifier
from .naive_bayes import CategoricalNaiveBayes
from .schema import CAREER_GROUP_NAMES, FEATURES, TARGET_CLASSES, TARGET_COLUMN


# ==============================================================================
# 1. LỚP MÔ HÌNH BASELINE (ZeroR - Majority Class Classifier)
# ==============================================================================

class ZeroRClassifier:
    """Bộ phân lớp cơ sở Baseline (ZeroR - Majority Class Classifier).

    Thuật toán luôn dự đoán nhãn lớp chiếm đa số (tần suất xuất hiện cao nhất)
    trong tập dữ liệu huấn luyện, bỏ qua tất cả 16 đặc trưng đầu vào.
    Đây là mốc chuẩn (baseline) tối thiểu mà mọi mô hình học máy cần vượt qua.
    """

    def __init__(self, target_classes: Sequence[str] | None = None) -> None:
        self.target_classes: tuple[str, ...] = (
            tuple(target_classes) if target_classes is not None else tuple(TARGET_CLASSES)
        )
        self.majority_label_: str | None = None
        self.classes_: tuple[str, ...] = ()
        self.class_counts_: dict[str, int] = {}

    @property
    def is_fitted(self) -> bool:
        """Kiểm tra mô hình đã được huấn luyện hay chưa."""
        return self.majority_label_ is not None

    def fit(
        self,
        rows: Sequence[Mapping[str, Any]] | None,
        labels: Sequence[str],
    ) -> "ZeroRClassifier":
        """Học nhãn đa số từ tập nhãn huấn luyện.

        Args:
            rows: Tập bản ghi đặc trưng (không sử dụng trong ZeroR).
            labels: Danh sách nhãn mục tiêu tương ứng.

        Returns:
            Chính đối tượng ZeroRClassifier sau khi fit.
        """
        if not labels:
            raise ValueError("Không thể huấn luyện ZeroR với danh sách nhãn rỗng.")

        counts = Counter(labels)
        self.class_counts_ = dict(counts)
        # Sắp xếp các lớp để đảm bảo tính xác định (deterministic) khi hòa tần suất
        available_classes = sorted(counts.keys())
        self.classes_ = tuple(available_classes)

        # Chọn nhãn có số lượng lớn nhất; nếu bằng nhau chọn nhãn đầu tiên theo thứ tự
        self.majority_label_ = max(self.classes_, key=lambda c: counts[c])
        return self

    def predict_one(self, row: Mapping[str, Any] | None = None) -> str:
        """Dự đoán nhãn cho một mẫu (luôn trả về nhãn đa số đã học)."""
        if not self.is_fitted or self.majority_label_ is None:
            raise RuntimeError("Mô hình ZeroR chưa được huấn luyện. Hãy gọi fit() trước.")
        return self.majority_label_

    def predict(self, rows: Iterable[Mapping[str, Any]]) -> list[str]:
        """Dự đoán nhãn cho danh sách bản ghi."""
        if not self.is_fitted or self.majority_label_ is None:
            raise RuntimeError("Mô hình ZeroR chưa được huấn luyện. Hãy gọi fit() trước.")
        return [self.majority_label_ for _ in rows]


# ==============================================================================
# 2. CÁC HÀM TÍNH TOÁN CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRICS)
# ==============================================================================

def confusion_matrix(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Tính ma trận nhầm lẫn (Confusion Matrix) đa lớp.

    Hàng i tương ứng với nhãn thực tế (True label).
    Cột j tương ứng với nhãn dự đoán (Predicted label).

    Args:
        y_true: Danh sách nhãn thực tế.
        y_pred: Danh sách nhãn dự đoán từ mô hình.
        labels: Danh sách thứ tự các lớp. Mặc định là TARGET_CLASSES (N1-N4).

    Returns:
        Ma trận 2 chiều numpy.ndarray kích thước (K, K) với K = len(labels).
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Kích thước y_true ({len(y_true)}) và y_pred ({len(y_pred)}) không khớp nhau."
        )

    class_list = list(labels) if labels is not None else list(TARGET_CLASSES)
    label_to_index = {label: idx for idx, label in enumerate(class_list)}
    k = len(class_list)
    cm = np.zeros((k, k), dtype=np.int64)

    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        if true_label in label_to_index and pred_label in label_to_index:
            row = label_to_index[true_label]
            col = label_to_index[pred_label]
            cm[row, col] += 1

    return cm


def accuracy_score(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Tính độ chính xác tổng thể (Accuracy): tỷ lệ mẫu dự đoán đúng trên tổng số mẫu."""
    if len(y_true) != len(y_pred):
        raise ValueError("Kích thước y_true và y_pred không khớp nhau.")
    if not y_true:
        raise ValueError("Tập dữ liệu rỗng, không thể tính Accuracy.")

    correct = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt == yp)
    return float(correct / len(y_true))


def precision_recall_fscore_support(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
    zero_division: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Tính Precision, Recall, F1-Score và Support riêng cho từng lớp.

    Args:
        y_true: Danh sách nhãn thực tế.
        y_pred: Danh sách nhãn dự đoán.
        labels: Danh sách các nhãn lớp cần tính toán (mặc định TARGET_CLASSES).
        zero_division: Giá trị trả về khi mẫu số bằng 0 (mặc định 0.0).

    Returns:
        Dictionary dạng:
            {
                "N1": {"precision": ..., "recall": ..., "f1": ..., "support": ...},
                ...
            }
    """
    class_list = list(labels) if labels is not None else list(TARGET_CLASSES)
    cm = confusion_matrix(y_true, y_pred, labels=class_list)

    results: dict[str, dict[str, float]] = {}
    for i, cls in enumerate(class_list):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        support = float(cm[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        if (precision + recall) > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1 = zero_division

        results[cls] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }

    return results


def macro_precision_score(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
    zero_division: float = 0.0,
) -> float:
    """Tính trung bình cộng không trọng số của Precision qua tất cả các lớp (Macro Precision)."""
    per_class = precision_recall_fscore_support(y_true, y_pred, labels, zero_division)
    if not per_class:
        return 0.0
    return float(np.mean([metrics["precision"] for metrics in per_class.values()]))


def macro_recall_score(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
    zero_division: float = 0.0,
) -> float:
    """Tính trung bình cộng không trọng số của Recall qua tất cả các lớp (Macro Recall)."""
    per_class = precision_recall_fscore_support(y_true, y_pred, labels, zero_division)
    if not per_class:
        return 0.0
    return float(np.mean([metrics["recall"] for metrics in per_class.values()]))


def macro_f1_score(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
    zero_division: float = 0.0,
) -> float:
    """Tính trung bình cộng không trọng số của F1-Score qua tất cả các lớp (Macro F1)."""
    per_class = precision_recall_fscore_support(y_true, y_pred, labels, zero_division)
    if not per_class:
        return 0.0
    return float(np.mean([metrics["f1"] for metrics in per_class.values()]))


def weighted_f1_score(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
    zero_division: float = 0.0,
) -> float:
    """Tính F1-Score có trọng số theo tỷ lệ số mẫu thực tế của từng lớp (Weighted F1)."""
    per_class = precision_recall_fscore_support(y_true, y_pred, labels, zero_division)
    total_support = sum(m["support"] for m in per_class.values())
    if total_support == 0:
        return zero_division

    weighted_sum = sum(m["f1"] * m["support"] for m in per_class.values())
    return float(weighted_sum / total_support)


@dataclass
class EvaluationMetrics:
    """Cấu trúc dữ liệu chứa kết quả đánh giá toàn diện của một mô hình."""

    model_name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    labels: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Chuyển đổi thành từ điển có thể tuần tự hóa JSON."""
        return asdict(self)

    def summary_row(self) -> dict[str, Any]:
        """Tạo một dòng tóm tắt các chỉ số cốt lõi cho bảng so sánh."""
        return {
            "Model": self.model_name,
            "Accuracy": round(self.accuracy, 4),
            "Macro Precision": round(self.macro_precision, 4),
            "Macro Recall": round(self.macro_recall, 4),
            "Macro F1": round(self.macro_f1, 4),
            "Weighted F1": round(self.weighted_f1, 4),
        }


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    model_name: str = "Model",
    labels: Sequence[str] | None = None,
    zero_division: float = 0.0,
) -> EvaluationMetrics:
    """Thực hiện đánh giá toàn diện các dự đoán và đóng gói vào EvaluationMetrics."""
    class_list = list(labels) if labels is not None else list(TARGET_CLASSES)
    cm = confusion_matrix(y_true, y_pred, labels=class_list)
    acc = accuracy_score(y_true, y_pred)
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=class_list, zero_division=zero_division
    )
    macro_p = float(np.mean([m["precision"] for m in per_class.values()]))
    macro_r = float(np.mean([m["recall"] for m in per_class.values()]))
    macro_f1 = float(np.mean([m["f1"] for m in per_class.values()]))

    total_support = sum(m["support"] for m in per_class.values())
    weighted_f1 = (
        float(sum(m["f1"] * m["support"] for m in per_class.values()) / total_support)
        if total_support > 0
        else zero_division
    )

    return EvaluationMetrics(
        model_name=model_name,
        accuracy=acc,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class=per_class,
        confusion_matrix=cm.tolist(),
        labels=class_list,
    )


def classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
    target_names: Sequence[str] | None = None,
    digits: int = 4,
) -> str:
    """Tạo bảng báo cáo văn bản chi tiết về Precision, Recall, F1-Score theo từng lớp."""
    class_list = list(labels) if labels is not None else list(TARGET_CLASSES)
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=class_list)
    acc = accuracy_score(y_true, y_pred)
    total_support = sum(m["support"] for m in per_class.values())

    macro_p = np.mean([m["precision"] for m in per_class.values()])
    macro_r = np.mean([m["recall"] for m in per_class.values()])
    macro_f = np.mean([m["f1"] for m in per_class.values()])

    weighted_p = (
        sum(m["precision"] * m["support"] for m in per_class.values()) / total_support
        if total_support > 0
        else 0.0
    )
    weighted_r = (
        sum(m["recall"] * m["support"] for m in per_class.values()) / total_support
        if total_support > 0
        else 0.0
    )
    weighted_f = (
        sum(m["f1"] * m["support"] for m in per_class.values()) / total_support
        if total_support > 0
        else 0.0
    )

    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    row_fmt = "{:<12} {:>10} {:>10} {:>10} {:>10}\n"
    report = row_fmt.format(*headers)
    report += "-" * 56 + "\n"

    for idx, cls in enumerate(class_list):
        cls_name = target_names[idx] if target_names and idx < len(target_names) else cls
        m = per_class[cls]
        report += row_fmt.format(
            cls_name,
            f"{m['precision']:.{digits}f}",
            f"{m['recall']:.{digits}f}",
            f"{m['f1']:.{digits}f}",
            str(int(m["support"])),
        )

    report += "\n"
    report += row_fmt.format(
        "accuracy",
        "",
        "",
        f"{acc:.{digits}f}",
        str(total_support),
    )
    report += row_fmt.format(
        "macro avg",
        f"{macro_p:.{digits}f}",
        f"{macro_r:.{digits}f}",
        f"{macro_f:.{digits}f}",
        str(total_support),
    )
    report += row_fmt.format(
        "weighted avg",
        f"{weighted_p:.{digits}f}",
        f"{weighted_r:.{digits}f}",
        f"{weighted_f:.{digits}f}",
        str(total_support),
    )

    return report


# ==============================================================================
# 3. LẬP TRÌNH XUẤT BIỂU ĐỒ (VISUALIZATION)
# ==============================================================================

def _setup_plot_style() -> None:
    """Cấu hình phong cách và font chữ hỗ trợ tiếng Việt trên đồ thị."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Segoe UI",
        "Arial",
        "DejaVu Sans",
        "Tahoma",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(
    cm: np.ndarray | Sequence[Sequence[int]],
    labels: Sequence[str] | None = None,
    title: str = "Confusion Matrix",
    output_path: Path | str | None = None,
    cmap: str = "Blues",
) -> plt.Figure:
    """Vẽ biểu đồ ma trận nhầm lẫn (Heatmap Confusion Matrix) đa lớp đẹp mắt.

    Mỗi ô hiển thị cả số lượng tuyệt đối lẫn tỷ lệ phần trăm tương đối so với
    tổng số mẫu của hàng thực tế (True Class).

    Args:
        cm: Ma trận nhầm lẫn kích thước (K, K).
        labels: Nhãn các lớp tương ứng.
        title: Tiêu đề biểu đồ.
        output_path: Đường dẫn lưu tệp hình ảnh (.png).
        cmap: Bảng màu Seaborn heatmap (mặc định 'Blues').

    Returns:
        Đối tượng matplotlib.figure.Figure.
    """
    _setup_plot_style()
    cm_arr = np.array(cm, dtype=np.int64)
    class_list = list(labels) if labels is not None else list(TARGET_CLASSES)

    # Nhãn hiển thị kèm tên mô tả nhóm nghề tiếng Việt
    display_labels = [
        f"{cls}\n({CAREER_GROUP_NAMES.get(cls, cls).split(' - ')[0]})"
        for cls in class_list
    ]

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)

    # Tính toán nhãn chuỗi cho từng ô: Số lượng + Tỷ lệ % của hàng
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    annot = np.empty_like(cm_arr, dtype=object)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            count = cm_arr[i, j]
            total = row_sums[i, 0]
            pct = (count / total * 100) if total > 0 else 0.0
            annot[i, j] = f"{count}\n({pct:.1f}%)"

    sns.heatmap(
        cm_arr,
        annot=annot,
        fmt="",
        cmap=cmap,
        cbar=True,
        linewidths=1.2,
        linecolor="#e2e8f0",
        square=True,
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=16, color="#1a202c")
    ax.set_xlabel("Nhãn dự đoán (Predicted Label)", fontsize=12, fontweight="bold", labelpad=12)
    ax.set_ylabel("Nhãn thực tế (True Label)", fontsize=12, fontweight="bold", labelpad=12)

    plt.tight_layout()

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        print(f"Đã lưu biểu đồ ma trận nhầm lẫn: {out_p}")

    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path | str | None = None,
    title: str = "So sánh hiệu năng giữa ID3, Naive Bayes và Baseline (ZeroR)",
) -> plt.Figure:
    """Vẽ biểu đồ so sánh Macro F1 giữa ID3, Naive Bayes và Baseline (ZeroR).

    Bao gồm 2 khung hình (Subplots):
        - Subplot 1 (Trái): Biểu đồ cột tập trung so sánh Macro F1 cốt lõi.
        - Subplot 2 (Phải): Biểu đồ cột nhóm so sánh toàn diện 5 chỉ số
          (Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1).

    Args:
        comparison_df: DataFrame chứa các cột 'Model', 'Accuracy', 'Macro Precision',
            'Macro Recall', 'Macro F1', 'Weighted F1'.
        output_path: Đường dẫn lưu tệp PNG.
        title: Tiêu đề chung cho hình ảnh.

    Returns:
        Đối tượng matplotlib.figure.Figure.
    """
    _setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Màu sắc nhận diện từng mô hình
    palette = {
        "ID3": "#2b6cb0",          # Xanh dương đậm
        "Naive Bayes": "#2f855a",  # Xanh lục bảo
        "ZeroR": "#c53030",        # Đỏ tươi
        "Baseline (ZeroR)": "#c53030",
    }
    models = comparison_df["Model"].tolist()
    colors = [palette.get(m, "#4a5568") for m in models]

    # -------------------------------------------------------------------------
    # Subplot 1: Trọng tâm so sánh Macro F1
    # -------------------------------------------------------------------------
    bars = ax1.bar(
        comparison_df["Model"],
        comparison_df["Macro F1"],
        color=colors,
        width=0.55,
        edgecolor="#1a202c",
        linewidth=1.2,
    )

    # Hiển thị số liệu trực tiếp trên đỉnh cột
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.4f}\n({height*100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#1a202c",
        )

    ax1.set_title("Chỉ số cốt lõi: Macro F1-Score", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Macro F1", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.18)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)

    # -------------------------------------------------------------------------
    # Subplot 2: So sánh toàn diện các chỉ số (Grouped Bar Chart)
    # -------------------------------------------------------------------------
    metric_cols = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
    x = np.arange(len(metric_cols))
    num_models = len(models)
    width = 0.24

    for idx, row in comparison_df.iterrows():
        model_name = row["Model"]
        color = palette.get(model_name, "#4a5568")
        offset = (idx - (num_models - 1) / 2) * width
        rects = ax2.bar(
            x + offset,
            [row[m] for m in metric_cols],
            width,
            label=model_name,
            color=color,
            edgecolor="#2d3748",
            linewidth=0.8,
        )

    ax2.set_title("So sánh toàn diện các chỉ số phân lớp", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_cols, fontsize=10, fontweight="bold", rotation=15)
    ax2.set_ylabel("Điểm số", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.18)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, fontsize=10)
    ax2.set_axisbelow(True)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02, color="#1a202c")
    plt.tight_layout()

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        print(f"Đã lưu biểu đồ so sánh mô hình: {out_p}")

    return fig


# ==============================================================================
# 4. QUY TRÌNH ĐÁNH GIÁ ĐẦY ĐỦ TRÊN TẬP KIỂM THỬ ĐỘC LẬP
# ==============================================================================

def load_or_split_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tải tập huấn luyện và kiểm thử độc lập.

    Nếu tệp data/processed/test.csv chưa tồn tại, hàm sẽ tự động phân chia
    tập dữ liệu gốc survey.csv theo tỷ lệ cố định (Train 60% / Val 20% / Test 20%)
    với RANDOM_SEED=42 từ src/config.py để đảm bảo tính tái lập.

    Returns:
        Tuple (train_df, test_df).
    """
    if TEST_DATA_PATH.exists() and TRAIN_DATA_PATH.exists() and TEST_DATA_PATH.stat().st_size > 10:
        train_df = pd.read_csv(TRAIN_DATA_PATH, encoding="utf-8")
        test_df = pd.read_csv(TEST_DATA_PATH, encoding="utf-8")
        return train_df, test_df

    # Nếu chưa có file phân chia, tiến hành phân chia phân tầng (Stratified) từ raw survey.csv
    if not RAW_SURVEY_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu gốc tại: {RAW_SURVEY_PATH}")

    raw_df = pd.read_csv(RAW_SURVEY_PATH, encoding="utf-8")

    # Phân tầng theo cột Nhóm nghề một cách xác định và không phụ thuộc cảnh báo pandas
    groups = [
        group.sample(frac=1.0, random_state=RANDOM_SEED)
        for _, group in raw_df.groupby(TARGET_COLUMN)
    ]
    shuffled = pd.concat(groups, ignore_index=True)

    test_indices = []
    val_indices = []
    train_indices = []

    for _, group in shuffled.groupby(TARGET_COLUMN):
        n = len(group)
        n_test = int(round(n * 0.20))
        n_val = int(round(n * 0.20))
        indices = group.index.tolist()

        test_indices.extend(indices[:n_test])
        val_indices.extend(indices[n_test : n_test + n_val])
        train_indices.extend(indices[n_test + n_val :])

    test_df = shuffled.loc[test_indices].copy()
    val_df = shuffled.loc[val_indices].copy()
    train_df = shuffled.loc[train_indices].copy()

    # Lưu lại để các mô-đun khác cùng dùng đồng bộ
    TEST_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(TEST_DATA_PATH, index=False, encoding="utf-8")
    val_df.to_csv(VAL_DATA_PATH, index=False, encoding="utf-8")
    train_df.to_csv(TRAIN_DATA_PATH, index=False, encoding="utf-8")

    print(
        f"Đã tạo các tập dữ liệu: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )
    return train_df, test_df


def run_evaluation() -> tuple[pd.DataFrame, dict[str, EvaluationMetrics]]:
    """Quy trình đánh giá toàn diện các mô hình trên tập kiểm thử độc lập.

    Thực hiện:
        1. Chuẩn bị tập Train và Test độc lập.
        2. Khởi tạo và huấn luyện ID3, Categorical Naive Bayes và Baseline ZeroR.
        3. Dự đoán nhãn trên tập Test (38 mẫu).
        4. Tính toán toàn bộ chỉ số: Accuracy, Macro Precision, Macro Recall,
           Macro F1 và Weighted F1.
        5. Xuất các bảng số liệu ra thư mục reports/metrics/:
           - reports/metrics/id3_test.json
           - reports/metrics/naive_bayes_test.json
           - reports/metrics/comparison.csv
           - reports/predictions/test_predictions.csv
        6. Xuất các biểu đồ ra thư mục reports/figures/:
           - reports/figures/model_comparison.png
           - reports/figures/id3_confusion_matrix.png
           - reports/figures/nb_confusion_matrix.png

    Returns:
        Tuple (comparison_dataframe, metrics_dictionary).
    """
    train_df, test_df = load_or_split_test_data()

    X_train = train_df[FEATURES].to_dict("records")
    y_train = train_df[TARGET_COLUMN].tolist()

    X_test = test_df[FEATURES].to_dict("records")
    y_test = test_df[TARGET_COLUMN].tolist()

    print(f"=== ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP KIỂM THỬ ĐỘC LẬP (N={len(y_test)}) ===")

    # 1. Baseline ZeroR
    zeror = ZeroRClassifier(target_classes=TARGET_CLASSES)
    zeror.fit(X_train, y_train)
    zeror_preds = zeror.predict(X_test)
    zeror_metrics = evaluate_predictions(y_test, zeror_preds, model_name="Baseline (ZeroR)")

    # 2. Cây quyết định ID3
    id3 = ID3Classifier(feature_order=FEATURES)
    id3.fit(X_train, y_train)
    id3_preds = id3.predict(X_test)
    id3_metrics = evaluate_predictions(y_test, id3_preds, model_name="ID3")

    # 3. Categorical Naive Bayes
    nb = CategoricalNaiveBayes(feature_order=FEATURES, alpha=1.0)
    nb.fit(X_train, y_train)
    nb_preds = nb.predict(X_test)
    nb_metrics = evaluate_predictions(y_test, nb_preds, model_name="Naive Bayes")

    metrics_dict = {
        "ZeroR": zeror_metrics,
        "ID3": id3_metrics,
        "Naive Bayes": nb_metrics,
    }

    # Bảng so sánh tổng hợp
    comparison_df = pd.DataFrame([
        zeror_metrics.summary_row(),
        id3_metrics.summary_row(),
        nb_metrics.summary_row(),
    ])

    # -------------------------------------------------------------------------
    # Xuất các file metrics
    # -------------------------------------------------------------------------
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(ID3_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(id3_metrics.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Đã lưu chỉ số ID3: {ID3_METRICS_PATH}")

    with open(NB_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(nb_metrics.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Đã lưu chỉ số Naive Bayes: {NB_METRICS_PATH}")

    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False, encoding="utf-8")
    print(f"Đã lưu bảng so sánh: {COMPARISON_CSV_PATH}")

    # Lưu chi tiết dự đoán từng mẫu
    pred_df = test_df.copy()
    pred_df["pred_zeror"] = zeror_preds
    pred_df["pred_id3"] = id3_preds
    pred_df["pred_naive_bayes"] = nb_preds
    pred_df.to_csv(TEST_PREDICTIONS_PATH, index=False, encoding="utf-8")
    print(f"Đã lưu dự đoán chi tiết: {TEST_PREDICTIONS_PATH}")

    # -------------------------------------------------------------------------
    # Xuất các biểu đồ
    # -------------------------------------------------------------------------
    # Biểu đồ so sánh Macro F1 giữa ID3, Naive Bayes và Baseline (ZeroR)
    plot_model_comparison(
        comparison_df,
        output_path=MODEL_COMPARISON_FIG_PATH,
        title="So sánh hiệu năng giữa ID3, Naive Bayes và Baseline (ZeroR)",
    )

    # Biểu đồ Ma trận nhầm lẫn ID3
    plot_confusion_matrix(
        id3_metrics.confusion_matrix,
        labels=TARGET_CLASSES,
        title="Ma trận nhầm lẫn - Cây quyết định ID3 (Test Set)",
        output_path=ID3_CM_FIG_PATH,
        cmap="Blues",
    )

    # Biểu đồ Ma trận nhầm lẫn Naive Bayes
    plot_confusion_matrix(
        nb_metrics.confusion_matrix,
        labels=TARGET_CLASSES,
        title="Ma trận nhầm lẫn - Categorical Naive Bayes (Test Set)",
        output_path=NB_CM_FIG_PATH,
        cmap="Greens",
    )

    plt.close("all")

    # In báo cáo ra màn hình
    print("\n" + "=" * 65)
    print("BẢNG SO SÁNH HIỆU NĂNG CÁC MÔ HÌNH TRÊN TẬP TEST:")
    print("=" * 65)
    print(comparison_df.to_string(index=False))
    print("\n" + "=" * 65)
    print("BÁO CÁO PHÂN LOẠI CHI TIẾT - ID3:")
    print("=" * 65)
    print(classification_report(y_test, id3_preds, labels=TARGET_CLASSES))
    print("=" * 65)
    print("BÁO CÁO PHÂN LOẠI CHI TIẾT - NAIVE BAYES:")
    print("=" * 65)
    print(classification_report(y_test, nb_preds, labels=TARGET_CLASSES))

    return comparison_df, metrics_dict


if __name__ == "__main__":
    run_evaluation()

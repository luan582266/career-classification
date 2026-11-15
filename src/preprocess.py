"""Tiền xử lý dữ liệu khảo sát và chia tập Train / Validation / Test phân tầng.

Pipeline tiền xử lý (chống data leakage bằng cách chỉ ``fit`` trên tập Train):
    1. Kiểm tra dữ liệu thô (số nguyên 1-5, danh mục và nhãn hợp lệ). Các ô để
       trống vẫn được chấp nhận vì sẽ được điền ở bước 3.
    2. Chia dữ liệu bằng ``sklearn.model_selection.StratifiedKFold`` (5 fold:
       3 fold = 60% Train, 1 fold = 20% Validation, 1 fold = 20% Test) đảm bảo
       tỷ lệ các nhóm nghề N1-N4 đồng đều trên cả 3 tập.
    3. Xử lý giá trị thiếu: điền ``median`` (trung vị trên Train, làm tròn về
       số nguyên thấp hơn) cho 14 cột điểm Likert và điền ``mode`` (giá trị xuất
       hiện nhiều nhất trên Train) cho 2 cột ``Đối tượng công việc``, ``Môi trường``.
    4. Rời rạc hóa 14 cột điểm số thành 3 mức:
       ``low`` (<= 2), ``medium`` (= 3), ``high`` (>= 4).
    5. Mã hóa giá trị chuỗi ở 2 cột ``Đối tượng công việc`` và ``Môi trường``
       thành số nguyên (danh mục chưa từng gặp khi biến đổi được ánh xạ về
       ``UNKNOWN_CATEGORY_CODE = -1``).
    6. Xuất 3 tệp dữ liệu sạch ra ``data/processed/`` (``train.csv``,
       ``validation.csv``, ``test.csv``) chỉ gồm 16 đặc trưng và cột nhãn
       ``Nhóm nghề``; cột nhãn tham chiếu bị loại để tránh rò rỉ nhãn. Bộ tiền xử
       lý đã fit được lưu vào ``models/preprocessor.json`` để dùng lại khi dự đoán.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config, schema
from src.data_io import read_csv_safe

# Nhãn sau khi rời rạc hóa thang điểm Likert 1-5
SCORE_BINS: tuple[str, str, str] = ("low", "medium", "high")

# Mã dùng cho giá trị danh mục chưa từng xuất hiện trong tập huấn luyện
UNKNOWN_CATEGORY_CODE: int = -1

# Số fold suy ra từ tỷ lệ cấu hình: 20% Validation = 20% Test = 1 fold => 5 fold,
# còn lại (5 - 2) = 3 fold = 60% làm Train.
N_SPLITS: int = round(1 / config.TEST_RATIO)

assert abs(config.VAL_RATIO - config.TEST_RATIO) < 1e-9, (
    "Validation và Test phải có cùng tỷ lệ để mỗi tập tương ứng đúng 1 fold."
)
assert abs(config.TRAIN_RATIO - (N_SPLITS - 2) / N_SPLITS) < 1e-9, (
    "TRAIN_RATIO phải bằng (N_SPLITS - 2) / N_SPLITS."
)


def discretize_score(value: float | int) -> str:
    """Rời rạc hóa một điểm số thành ``low`` (<=2), ``medium`` (=3), ``high`` (>=4)."""
    if pd.isna(value):
        raise ValueError("Không thể rời rạc hóa giá trị NaN.")
    numeric = float(value)
    if numeric <= 2.0:
        return "low"
    if numeric >= 4.0:
        return "high"
    return "medium"


# ----------------------------------------------------------------------
# Kiểm tra dữ liệu thô
# ----------------------------------------------------------------------
def _validate_numeric_domain(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Các ô có giá trị phải là số nguyên 1-5; ô trống (NaN) được phép (sẽ được điền)."""
    for col in columns:
        raw = df[col]
        numeric = pd.to_numeric(raw, errors="coerce")

        not_numeric = raw.notna() & numeric.isna()
        if not_numeric.any():
            examples = raw[not_numeric].unique().tolist()[:5]
            raise ValueError(f"Cột '{col}' chứa giá trị không phải số: {examples}")

        present = numeric.dropna()
        invalid = present[
            (present < schema.LIKERT_MIN)
            | (present > schema.LIKERT_MAX)
            | (present != present.round())
        ]
        if not invalid.empty:
            examples = invalid.unique().tolist()[:5]
            raise ValueError(
                f"Cột '{col}' có điểm không phải số nguyên "
                f"{schema.LIKERT_MIN}-{schema.LIKERT_MAX}: {examples}"
            )


def validate_raw_survey(
    df: pd.DataFrame,
    *,
    require_target: bool = True,
    require_reference: bool = False,
) -> None:
    """Kiểm tra dữ liệu thô trước khi điền giá trị thiếu.

    Khác ``schema.validate_dataframe`` ở chỗ cho phép ô trống ở 16 đặc trưng
    (sẽ được điền median/mode), nhưng mọi ô **có giá trị** đều phải hợp lệ
    để lỗi dữ liệu không bị che giấu bởi bước điền thiếu.

    Raises:
        ValueError: Nếu thiếu cột hoặc có giá trị không hợp lệ.
    """
    required = list(schema.FEATURE_COLUMNS)
    if require_target:
        required.append(schema.TARGET_COLUMN)
    if require_reference:
        required.extend(schema.REFERENCE_COLUMNS)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing}")
    if df.empty:
        raise ValueError("Dữ liệu không có bản ghi nào.")

    _validate_numeric_domain(df, schema.NUMERIC_FEATURES)

    for col, valid_values in schema.VALID_CATEGORIES.items():
        present = df[col].dropna().astype(str).str.strip()
        present = present[present != ""]
        invalid = sorted(set(present) - set(valid_values))
        if invalid:
            raise ValueError(f"Cột '{col}' chứa danh mục không hợp lệ: {invalid}")

    if require_target:
        target = df[schema.TARGET_COLUMN]
        if target.isna().any():
            raise ValueError(f"Cột nhãn '{schema.TARGET_COLUMN}' chứa giá trị bị khuyết.")
        invalid_labels = sorted(set(target) - set(schema.VALID_TARGET_LABELS))
        if invalid_labels:
            raise ValueError(f"Nhãn mục tiêu không hợp lệ: {invalid_labels}")


# ----------------------------------------------------------------------
# Chia tập phân tầng bằng sklearn StratifiedKFold
# ----------------------------------------------------------------------
def stratified_train_val_test_split(
    df: pd.DataFrame,
    *,
    target_col: str = schema.TARGET_COLUMN,
    random_state: int | None = config.RANDOM_SEED,
    test_fold: int = 0,
    val_fold: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia ``df`` thành Train (60%), Validation (20%), Test (20%) phân tầng.

    Dùng ``StratifiedKFold(n_splits=N_SPLITS)``: lấy 1 fold làm Test, 1 fold làm
    Validation và gộp các fold còn lại làm Train. Các DataFrame trả về **giữ
    nguyên chỉ số hàng gốc** của ``df`` (chỉ số phải là duy nhất) để có thể truy
    vết và kiểm tra các tập không chồng nhau.
    """
    if target_col not in df.columns:
        raise KeyError(f"Không tìm thấy cột nhãn '{target_col}' trong dữ liệu.")
    if test_fold == val_fold or not (
        0 <= test_fold < N_SPLITS and 0 <= val_fold < N_SPLITS
    ):
        raise ValueError(
            f"test_fold và val_fold phải khác nhau và nằm trong 0..{N_SPLITS - 1}."
        )
    if not df.index.is_unique:
        raise ValueError("Chỉ số hàng của DataFrame phải là duy nhất.")

    labels = df[target_col]
    if labels.isna().any():
        raise ValueError(f"Cột nhãn '{target_col}' không được chứa giá trị thiếu.")
    smallest_class = int(labels.value_counts().min())
    if smallest_class < N_SPLITS:
        raise ValueError(
            f"Mỗi lớp cần ít nhất {N_SPLITS} mẫu để chia {N_SPLITS} fold "
            f"(lớp nhỏ nhất chỉ có {smallest_class})."
        )

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=random_state,
    )
    folds = [fold_idx for _, fold_idx in splitter.split(df, labels)]

    test_idx = folds[test_fold]
    val_idx = folds[val_fold]
    train_idx = np.sort(
        np.concatenate(
            [folds[i] for i in range(N_SPLITS) if i not in (test_fold, val_fold)]
        )
    )

    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]


# ----------------------------------------------------------------------
# Bộ tiền xử lý
# ----------------------------------------------------------------------
class SurveyPreprocessor:
    """Bộ tiền xử lý dữ liệu khảo sát (fit trên Train để chống data leakage).

    Thuộc tính sau khi gọi ``fit``:
        numeric_medians_: Trung vị (số nguyên) của 14 cột điểm Likert trên Train.
        categorical_modes_: Mode của 2 cột danh mục trên Train.
        category_mappings_: Ánh xạ ``{cột: {giá_trị_chuỗi: mã_số_nguyên}}``.
        inverse_category_mappings_: Ánh xạ ngược từ mã số nguyên về chuỗi.

    Có thể lưu / nạp lại bằng ``save`` / ``load`` (định dạng JSON) để dùng cùng
    một cách xử lý khi dự đoán dữ liệu mới.
    """

    def __init__(
        self,
        numeric_features: Sequence[str] = schema.NUMERIC_FEATURES,
        categorical_features: Sequence[str] = schema.CATEGORICAL_FEATURES,
    ) -> None:
        self.numeric_features: tuple[str, ...] = tuple(numeric_features)
        self.categorical_features: tuple[str, ...] = tuple(categorical_features)

        self.numeric_medians_: dict[str, int] = {}
        self.categorical_modes_: dict[str, str] = {}
        self.category_mappings_: dict[str, dict[str, int]] = {}
        self.inverse_category_mappings_: dict[str, dict[int, str]] = {}

    @property
    def is_fitted(self) -> bool:
        """Trả về ``True`` nếu preprocessor đã được gọi ``fit``."""
        return bool(self.numeric_medians_) or bool(self.categorical_modes_)

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        required = [*self.numeric_features, *self.categorical_features]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Dữ liệu thiếu các cột đặc trưng: {missing}")

    def fit(self, train_df: pd.DataFrame) -> "SurveyPreprocessor":
        """Học median, mode và bảng mã hóa danh mục chỉ từ ``train_df``."""
        if train_df.empty:
            raise ValueError("Không thể fit preprocessor trên tập dữ liệu rỗng.")
        self._validate_required_columns(train_df)
        _validate_numeric_domain(train_df, self.numeric_features)

        medians: dict[str, int] = {}
        for col in self.numeric_features:
            values = np.sort(pd.to_numeric(train_df[col]).dropna().to_numpy())
            if values.size == 0:
                raise ValueError(
                    f"Cột điểm '{col}' toàn bộ là giá trị thiếu trong tập train."
                )
            # Trung vị thấp (median_low): luôn là một điểm Likert nguyên có thật,
            # nên khi rời rạc hóa luôn thuộc đúng 1 trong 3 mức low/medium/high.
            medians[col] = int(values[(values.size - 1) // 2])

        modes: dict[str, str] = {}
        mappings: dict[str, dict[str, int]] = {}
        inverse_mappings: dict[str, dict[int, str]] = {}

        for col in self.categorical_features:
            clean_series = train_df[col].dropna().astype(str).str.strip()
            clean_series = clean_series[clean_series != ""]
            if clean_series.empty:
                raise ValueError(
                    f"Cột danh mục '{col}' toàn bộ là giá trị thiếu trong tập train."
                )

            # Nếu nhiều giá trị cùng xuất hiện nhiều nhất, chọn giá trị đứng đầu
            # theo thứ tự chữ cái để kết quả luôn xác định.
            modes[col] = sorted(clean_series.mode().tolist())[0]

            unique_categories = sorted(clean_series.unique().tolist())
            mapping = {cat: idx for idx, cat in enumerate(unique_categories)}
            mappings[col] = mapping
            inverse_mappings[col] = {idx: cat for cat, idx in mapping.items()}

        self.numeric_medians_ = medians
        self.categorical_modes_ = modes
        self.category_mappings_ = mappings
        self.inverse_category_mappings_ = inverse_mappings
        return self

    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Điền giá trị thiếu bằng median (số) và mode (danh mục) đã học."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor chưa được fit. Hãy gọi fit() trước.")
        self._validate_required_columns(df)
        _validate_numeric_domain(df, self.numeric_features)

        result = df.copy()
        for col in self.numeric_features:
            numeric_col = pd.to_numeric(result[col])
            result[col] = numeric_col.fillna(self.numeric_medians_[col])

        for col in self.categorical_features:
            series = result[col]
            cleaned = series.astype(str).str.strip()
            # Coi chuỗi rỗng / toàn khoảng trắng như giá trị thiếu.
            is_blank = series.isna() | (cleaned == "")
            result[col] = cleaned.where(~is_blank, self.categorical_modes_[col])

        return result

    def discretize_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rời rạc hóa 14 cột điểm số sang ``low``, ``medium``, ``high``."""
        result = df.copy()
        for col in self.numeric_features:
            if col not in result.columns:
                raise ValueError(f"Thiếu cột điểm '{col}' khi rời rạc hóa.")
            result[col] = result[col].map(discretize_score)
        return result

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mã hóa 2 cột danh mục từ chuỗi sang số nguyên theo bảng ánh xạ đã fit."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor chưa được fit. Hãy gọi fit() trước.")
        result = df.copy()
        for col in self.categorical_features:
            if col not in result.columns:
                raise ValueError(f"Thiếu cột danh mục '{col}' khi mã hóa.")
            mapping = self.category_mappings_[col]
            result[col] = (
                result[col]
                .astype(str)
                .str.strip()
                .map(lambda val, m=mapping: m.get(val, UNKNOWN_CATEGORY_CODE))
                .astype("int64")
            )
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Áp dụng đầy đủ: điền giá trị thiếu -> rời rạc hóa -> mã hóa danh mục."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor chưa được fit. Hãy gọi fit() trước.")
        imputed = self.impute_missing(df)
        discretized = self.discretize_scores(imputed)
        return self.encode_categoricals(discretized)

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Gọi ``fit(train_df)`` rồi ``transform(train_df)``."""
        return self.fit(train_df).transform(train_df)

    # ------------------------------------------------------------------
    # Lưu / nạp
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Biểu diễn trạng thái đã fit dưới dạng dict thuần (có thể ghi JSON)."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor chưa được fit. Hãy gọi fit() trước.")
        return {
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "numeric_medians": dict(self.numeric_medians_),
            "categorical_modes": dict(self.categorical_modes_),
            "category_mappings": {
                col: dict(mapping) for col, mapping in self.category_mappings_.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SurveyPreprocessor":
        """Khôi phục preprocessor đã fit từ dict do ``to_dict`` tạo ra."""
        prep = cls(
            numeric_features=data["numeric_features"],
            categorical_features=data["categorical_features"],
        )
        prep.numeric_medians_ = {k: int(v) for k, v in data["numeric_medians"].items()}
        prep.categorical_modes_ = dict(data["categorical_modes"])
        prep.category_mappings_ = {
            col: {cat: int(code) for cat, code in mapping.items()}
            for col, mapping in data["category_mappings"].items()
        }
        prep.inverse_category_mappings_ = {
            col: {code: cat for cat, code in mapping.items()}
            for col, mapping in prep.category_mappings_.items()
        }
        return prep

    def save(self, path: str | Path = config.PREPROCESSOR_PATH) -> Path:
        """Lưu preprocessor đã fit ra tệp JSON (UTF-8)."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out_path

    @classmethod
    def load(cls, path: str | Path = config.PREPROCESSOR_PATH) -> "SurveyPreprocessor":
        """Nạp preprocessor đã fit từ tệp JSON do ``save`` tạo ra."""
        in_path = Path(path)
        if not in_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy preprocessor: {in_path}")
        return cls.from_dict(json.loads(in_path.read_text(encoding="utf-8")))


# ----------------------------------------------------------------------
# Quy trình đầu-cuối
# ----------------------------------------------------------------------
def run_preprocessing(
    raw_path: str | Path = config.RAW_SURVEY_PATH,
    output_dir: str | Path = config.PROCESSED_DATA_DIR,
    *,
    preprocessor_path: str | Path | None = config.PREPROCESSOR_PATH,
    keep_reference: bool = False,
    random_state: int | None = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SurveyPreprocessor]:
    """Đọc dữ liệu thô, kiểm tra, chia StratifiedKFold 60/20/20, tiền xử lý và lưu.

    Args:
        raw_path: Đường dẫn CSV khảo sát gốc.
        output_dir: Thư mục xuất ``train.csv``, ``validation.csv``, ``test.csv``.
        preprocessor_path: Nơi lưu preprocessor đã fit (``None`` để không lưu).
        keep_reference: ``True`` để giữ thêm 2 cột nhãn tham chiếu trong tệp xuất.
            Mặc định ``False`` vì ``Tên nhóm nghề`` ánh xạ 1-1 với nhãn mục tiêu,
            giữ lại dễ gây rò rỉ nhãn khi huấn luyện.
        random_state: Seed cho ``StratifiedKFold``.

    Returns:
        Bộ 4 gồm ``(train_df, val_df, test_df, fitted_preprocessor)``; 3 DataFrame
        giữ chỉ số hàng gốc của dữ liệu thô.
    """
    # Đọc CSV an toàn (UTF-8/BOM, kiểm tra trùng cột); ô khuyết được phép nên
    # chỉ kiểm tra giá trị hợp lệ ở bước riêng bên dưới.
    raw_df = read_csv_safe(raw_path, validate=False)
    validate_raw_survey(raw_df, require_reference=keep_reference)

    # Chia tập trước khi fit preprocessor để tuyệt đối tránh data leakage.
    train_raw, val_raw, test_raw = stratified_train_val_test_split(
        raw_df,
        target_col=schema.TARGET_COLUMN,
        random_state=random_state,
    )

    preprocessor = SurveyPreprocessor()
    output_columns = [*schema.FEATURE_COLUMNS, schema.TARGET_COLUMN]
    if keep_reference:
        output_columns += list(schema.REFERENCE_COLUMNS)

    train_clean = preprocessor.fit_transform(train_raw)[output_columns]
    val_clean = preprocessor.transform(val_raw)[output_columns]
    test_clean = preprocessor.transform(test_raw)[output_columns]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_clean.to_csv(out_path / config.TRAIN_DATA_PATH.name, index=False, encoding="utf-8")
    val_clean.to_csv(out_path / config.VAL_DATA_PATH.name, index=False, encoding="utf-8")
    test_clean.to_csv(out_path / config.TEST_DATA_PATH.name, index=False, encoding="utf-8")

    if preprocessor_path is not None:
        preprocessor.save(preprocessor_path)

    return train_clean, val_clean, test_clean, preprocessor


if __name__ == "__main__":
    train_out, val_out, test_out, _prep = run_preprocessing()
    print(f"Đã xuất dữ liệu tiền xử lý vào {config.PROCESSED_DATA_DIR}:")
    for name, frame in (
        ("train.csv", train_out),
        ("validation.csv", val_out),
        ("test.csv", test_out),
    ):
        counts = frame[schema.TARGET_COLUMN].value_counts().sort_index().to_dict()
        print(f"- {name}: {len(frame)} dòng -> {counts}")
    print(f"Đã lưu preprocessor vào {config.PREPROCESSOR_PATH}")

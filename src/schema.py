"""Quy định schema dữ liệu: 16 đặc trưng đầu vào, kiểu dữ liệu, danh mục hợp lệ và nhãn tham chiếu."""

from typing import Any, Dict, List, Mapping, Sequence, Tuple
import pandas as pd

# ==============================================================================
# 1. DANH SÁCH 16 ĐẶC TRƯNG ĐẦU VÀO (INPUT FEATURES)
# ==============================================================================

# 8 đặc trưng nhóm Năng lực / Kỹ năng (thang đo Likert 1 - 5)
SKILL_FEATURES: List[str] = [
    "Khả năng tư duy logic",
    "Năng lực tính toán và phân tích số liệu",
    "Công nghệ",
    "Sáng tạo",
    "Giao tiếp",
    "Tổ chức",
    "Cẩn thận",
    "Đồng cảm",
]
SOFT_SKILL_FEATURES: List[str] = SKILL_FEATURES

# 6 đặc trưng nhóm Sở thích nghề nghiệp RIASEC (thang đo Likert 1 - 5)
RIASEC_FEATURES: List[str] = [
    "RIASEC_R_Thuc_hanh",
    "RIASEC_I_Nghien_cuu",
    "RIASEC_A_Nghe_thuat",
    "RIASEC_S_Xa_hoi",
    "RIASEC_E_Thuyet_phuc",
    "RIASEC_C_Quy_cu",
]

# 14 đặc trưng định lượng / thứ bậc (Likert 1 - 5)
NUMERIC_FEATURES: List[str] = SKILL_FEATURES + RIASEC_FEATURES
LIKERT_FEATURES: List[str] = NUMERIC_FEATURES

# 2 đặc trưng phân loại định danh (Categorical)
WORK_OBJECT_COL: str = "Đối tượng công việc"
ENVIRONMENT_COL: str = "Môi trường"

CATEGORICAL_FEATURES: List[str] = [
    WORK_OBJECT_COL,
    ENVIRONMENT_COL,
]

# Tổng hợp 16 đặc trưng đầu vào dùng để huấn luyện và dự đoán
FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
INPUT_FEATURES: List[str] = FEATURE_COLUMNS
FEATURES: List[str] = FEATURE_COLUMNS

assert len(FEATURE_COLUMNS) == 16, f"Số đặc trưng đầu vào phải là 16, hiện tại: {len(FEATURE_COLUMNS)}"

# ==============================================================================
# 2. CỘT NHÃN MỤC TIÊU & NHÃN THAM CHIẾU (TARGET & REFERENCE LABELS)
# ==============================================================================

# Nhãn mục tiêu chính dùng cho huấn luyện mô hình phân loại
TARGET_COLUMN: str = "Nhóm nghề"
TARGET_COL: str = TARGET_COLUMN
LABEL_COLUMN: str = TARGET_COLUMN

# Hai cột nhãn tham chiếu (tách riêng, không đưa vào ma trận đặc trưng X khi huấn luyện)
SPECIFIC_CAREER_COL: str = "Nghề cụ thể"
CAREER_GROUP_NAME_COL: str = "Tên nhóm nghề"

REFERENCE_COLUMNS: List[str] = [
    SPECIFIC_CAREER_COL,
    CAREER_GROUP_NAME_COL,
]
REFERENCE_LABELS: List[str] = REFERENCE_COLUMNS

# Toàn bộ 19 cột theo đúng thứ tự trong file data/raw/survey.csv
ALL_COLUMNS: List[str] = FEATURE_COLUMNS + [
    SPECIFIC_CAREER_COL,
    TARGET_COLUMN,
    CAREER_GROUP_NAME_COL,
]

# ==============================================================================
# 3. KIỂU DỮ LIỆU QUY ĐỊNH (DATA TYPES)
# ==============================================================================

FEATURE_DTYPES: Dict[str, type] = {
    **{col: int for col in NUMERIC_FEATURES},
    **{col: str for col in CATEGORICAL_FEATURES},
}

PANDAS_DTYPES: Dict[str, str] = {
    **{col: "int64" for col in NUMERIC_FEATURES},
    **{col: "object" for col in CATEGORICAL_FEATURES},
    SPECIFIC_CAREER_COL: "object",
    TARGET_COLUMN: "object",
    CAREER_GROUP_NAME_COL: "object",
}

# ==============================================================================
# 4. MIỀN GIÁ TRỊ & DANH MỤC HỢP LỆ (VALID DOMAINS & CATEGORIES)
# ==============================================================================

LIKERT_MIN: int = 1
LIKERT_MAX: int = 5
VALID_LIKERT_VALUES: List[int] = [1, 2, 3, 4, 5]

# Danh sách giá trị hợp lệ cho cột "Đối tượng công việc" (19 giá trị)
VALID_WORK_OBJECTS: List[str] = [
    "Phần mềm",
    "Dữ liệu",
    "Máy móc - thiết bị",
    "Khách hàng - thị trường",
    "Tài chính - số liệu",
    "Sổ sách - chứng từ",
    "Quy trình - hoạt động",
    "Chuỗi cung ứng - hàng hóa",
    "Sản phẩm số - Giao diện",
    "Nội dung - Truyền thông",
    "Thương hiệu - Quảng cáo",
    "Hình ảnh - Ấn phẩm",
    "Video - Âm thanh",
    "Học sinh - sinh viên",
    "Con người - tri thức",
    "Tâm lý - hành vi",
    "Cộng đồng - yếu thế",
    "Nhân sự - tổ chức",
    "Bệnh nhân - y tế",
]

# Danh sách giá trị hợp lệ cho cột "Môi trường" (6 giá trị)
VALID_ENVIRONMENTS: List[str] = [
    "Linh hoạt - sáng tạo",
    "Có quy trình",
    "Tương tác - hợp tác",
    "Cạnh tranh - mục tiêu",
    "Phân tích chuyên sâu",
    "Ổn định - rõ ràng",
]

# Các danh mục hợp lệ của các cột Categorical đầu vào
VALID_CATEGORIES: Dict[str, List[str]] = {
    WORK_OBJECT_COL: VALID_WORK_OBJECTS,
    ENVIRONMENT_COL: VALID_ENVIRONMENTS,
}

# Danh sách 4 nhóm nghề mục tiêu
VALID_TARGET_LABELS: List[str] = ["N1", "N2", "N3", "N4"]
TARGET_CLASSES: List[str] = VALID_TARGET_LABELS
VALID_LABELS: List[str] = VALID_TARGET_LABELS

# Ánh xạ mã Nhóm nghề sang Tên nhóm nghề tham chiếu
CAREER_GROUP_NAMES: Dict[str, str] = {
    "N1": "Công nghệ - Kỹ thuật - Phân tích",
    "N2": "Kinh doanh - Tài chính - Quản lý",
    "N3": "Sáng tạo - Truyền thông - Thiết kế",
    "N4": "Giáo dục – Xã hội – Hỗ trợ con người",
}

# Ánh xạ Nhóm nghề sang danh sách các Nghề cụ thể tham chiếu
CAREERS_BY_GROUP: Dict[str, List[str]] = {
    "N1": [
        "Lập trình viên",
        "Chuyên viên phân tích dữ liệu",
        "Kiểm thử phần mềm",
        "Kỹ sư điện",
        "Kỹ sư tự động hóa",
        "Kỹ sư cơ khí",
    ],
    "N2": [
        "Chuyên viên kinh doanh",
        "Chuyên viên phân tích tài chính",
        "Kế toán viên",
        "Chuyên viên vận hành",
        "Chuyên viên logistics",
    ],
    "N3": [
        "Thiết kế UI/UX",
        "Sáng tạo nội dung",
        "Digital Marketing",
        "Thiết kế đồ họa",
        "Dựng phim - Biên tập video",
    ],
    "N4": [
        "Sư phạm",
        "Tâm lý học",
        "Công tác xã hội",
        "Quản trị nhân lực",
        "Điều dưỡng",
    ],
}

VALID_SPECIFIC_CAREERS: List[str] = [
    career for group_careers in CAREERS_BY_GROUP.values() for career in group_careers
]


# ==============================================================================
# 5. HÀM TIỆN ÍCH TÁCH CỘT & KIỂM TRA SCHEMA
# ==============================================================================

def split_features_and_labels(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Tách DataFrame thành 16 đặc trưng đầu vào (X), nhãn mục tiêu (y), và nhãn tham chiếu.

    Args:
        df: DataFrame chứa dữ liệu khảo sát.

    Returns:
        Tuple (X, y, ref_df):
            - X: DataFrame gồm 16 cột đặc trưng đầu vào (FEATURE_COLUMNS).
            - y: Series chứa cột nhãn mục tiêu 'Nhóm nghề' (TARGET_COLUMN).
            - ref_df: DataFrame chứa 2 cột nhãn tham chiếu 'Nghề cụ thể' và 'Tên nhóm nghề'.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy() if TARGET_COLUMN in df.columns else pd.Series(dtype="object")
    ref_cols = [col for col in REFERENCE_COLUMNS if col in df.columns]
    ref_df = df[ref_cols].copy() if ref_cols else pd.DataFrame(index=df.index)
    return X, y, ref_df


def validate_dataframe(
    df: pd.DataFrame,
    require_target: bool = True,
    require_reference: bool = False,
) -> List[str]:
    """Kiểm tra tính hợp lệ của DataFrame theo quy định schema.

    Args:
        df: DataFrame cần kiểm tra.
        require_target: Có bắt buộc cột nhãn mục tiêu 'Nhóm nghề' hay không.
        require_reference: Có bắt buộc 2 cột tham chiếu 'Nghề cụ thể' và 'Tên nhóm nghề' hay không.

    Returns:
        Danh sách các thông báo lỗi (rỗng nếu dữ liệu hợp lệ).
    """
    errors: List[str] = []

    required_cols = list(FEATURE_COLUMNS)
    if require_target:
        required_cols.append(TARGET_COLUMN)
    if require_reference:
        required_cols.extend(REFERENCE_COLUMNS)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Thiếu các cột bắt buộc: {missing_cols}")
        return errors

    # Kiểm tra miền giá trị 1 - 5 cho 14 cột Likert
    for col in NUMERIC_FEATURES:
        if df[col].isnull().any():
            errors.append(f"Cột '{col}' chứa giá trị bị khuyết (NaN/null).")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Cột '{col}' phải có kiểu số, hiện tại: {df[col].dtype}.")
            continue
        invalid_mask = ~df[col].isin(VALID_LIKERT_VALUES)
        if invalid_mask.any():
            bad_vals = df.loc[invalid_mask, col].unique().tolist()
            errors.append(
                f"Cột '{col}' có giá trị ngoài thang đo Likert 1-5: {bad_vals}"
            )

    # Kiểm tra danh mục hợp lệ cho 2 cột phân loại
    for col, valid_vals in VALID_CATEGORIES.items():
        if df[col].isnull().any():
            errors.append(f"Cột '{col}' chứa giá trị bị khuyết (NaN/null).")
            continue
        invalid_mask = ~df[col].isin(valid_vals)
        if invalid_mask.any():
            bad_vals = df.loc[invalid_mask, col].unique().tolist()
            errors.append(f"Cột '{col}' chứa danh mục không hợp lệ: {bad_vals}")

    # Kiểm tra nhãn mục tiêu
    if require_target and TARGET_COLUMN in df.columns:
        if df[TARGET_COLUMN].isnull().any():
            errors.append(f"Cột nhãn '{TARGET_COLUMN}' chứa giá trị bị khuyết.")
        else:
            invalid_target = ~df[TARGET_COLUMN].isin(VALID_TARGET_LABELS)
            if invalid_target.any():
                bad_labels = df.loc[invalid_target, TARGET_COLUMN].unique().tolist()
                errors.append(f"Nhãn mục tiêu không hợp lệ: {bad_labels}")

    # Kiểm tra nhãn tham chiếu nếu yêu cầu
    if require_reference:
        if SPECIFIC_CAREER_COL in df.columns:
            invalid_careers = ~df[SPECIFIC_CAREER_COL].isin(VALID_SPECIFIC_CAREERS)
            if invalid_careers.any():
                bad_careers = df.loc[invalid_careers, SPECIFIC_CAREER_COL].unique().tolist()
                errors.append(f"Nghề cụ thể không hợp lệ: {bad_careers}")
        if CAREER_GROUP_NAME_COL in df.columns:
            valid_group_names = list(CAREER_GROUP_NAMES.values())
            invalid_names = ~df[CAREER_GROUP_NAME_COL].isin(valid_group_names)
            if invalid_names.any():
                bad_names = df.loc[invalid_names, CAREER_GROUP_NAME_COL].unique().tolist()
                errors.append(f"Tên nhóm nghề không hợp lệ: {bad_names}")

    return errors


def validate_input_record(record: Mapping[str, Any]) -> List[str]:
    """Kiểm tra một bản ghi đầu vào (dict) gồm 16 đặc trưng trước khi dự đoán."""
    errors: List[str] = []
    for col in FEATURE_COLUMNS:
        if col not in record:
            errors.append(f"Thiếu đặc trưng đầu vào: '{col}'")

    if errors:
        return errors

    for col in NUMERIC_FEATURES:
        val = record[col]
        try:
            int_val = int(val)
            if int_val != float(val) or int_val not in VALID_LIKERT_VALUES:
                errors.append(f"Đặc trưng '{col}' phải là số nguyên từ 1 đến 5 (nhận được: {val}).")
        except (ValueError, TypeError):
            errors.append(f"Đặc trưng '{col}' phải là số nguyên từ 1 đến 5 (nhận được: {val}).")

    for col, valid_vals in VALID_CATEGORIES.items():
        val = str(record[col]).strip() if record[col] is not None else ""
        if val not in valid_vals:
            errors.append(f"Đặc trưng '{col}' có giá trị '{val}' không nằm trong danh mục hợp lệ.")

    return errors

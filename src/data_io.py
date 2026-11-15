"""Đọc CSV an toàn và kiểm tra dữ liệu đầu vào theo ``src/schema.py``.

Hỗ trợ:
    - Mã hóa UTF-8 và UTF-8 BOM (``utf-8-sig``).
    - Phát hiện tên cột bị trùng ngay từ dòng tiêu đề thô (pandas tự đổi tên cột
      trùng thành ``col.1``, ``col.2`` nên không thể kiểm tra sau khi đọc).
    - Validate dữ liệu bằng ``schema.validate_dataframe`` (cột bắt buộc, thang
      Likert 1-5 nguyên, danh mục hợp lệ, nhãn N1-N4).
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src import schema
from src.config import RAW_SURVEY_PATH

ENCODING = "utf-8-sig"


def check_duplicate_columns(columns: Iterable[str]) -> None:
    """Báo ``ValueError`` nếu có tên cột bị trùng.

    Args:
        columns: Danh sách tên cột (nên là tên thô từ dòng tiêu đề của CSV).
    """
    counts = Counter(columns)
    duplicates = [name for name, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Phát hiện tên cột bị trùng: {duplicates}")


def validate_data(
    df: pd.DataFrame,
    *,
    require_target: bool = True,
    require_reference: bool = False,
) -> pd.DataFrame:
    """Kiểm tra ``df`` theo schema và trả về bản sao đã chuẩn hóa kiểu dữ liệu.

    Không sửa trực tiếp ``df`` đầu vào. Các cột Likert được ép về ``int64``
    (sau khi đã xác nhận toàn bộ giá trị là số nguyên 1-5).

    Args:
        df: DataFrame cần kiểm tra.
        require_target: Bắt buộc có cột nhãn ``Nhóm nghề``.
        require_reference: Bắt buộc có 2 cột nhãn tham chiếu.

    Raises:
        ValueError: Nếu dữ liệu rỗng hoặc không đúng schema.
    """
    if df.empty:
        raise ValueError("Dữ liệu không có bản ghi nào.")

    errors = schema.validate_dataframe(
        df,
        require_target=require_target,
        require_reference=require_reference,
    )
    if errors:
        raise ValueError("Dữ liệu không hợp lệ theo schema:\n- " + "\n- ".join(errors))

    result = df.copy()
    for column in schema.NUMERIC_FEATURES:
        result[column] = result[column].astype("int64")
    return result


def read_csv_safe(
    file_path: str | Path = RAW_SURVEY_PATH,
    *,
    validate: bool = True,
    require_target: bool = True,
    require_reference: bool = False,
) -> pd.DataFrame:
    """Đọc file CSV an toàn (UTF-8 / UTF-8 BOM) và validate theo schema.

    Args:
        file_path: Đường dẫn CSV; mặc định là ``data/raw/survey.csv``.
        validate: ``False`` để chỉ đọc và kiểm tra tên cột trùng.
        require_target: Bắt buộc có cột nhãn ``Nhóm nghề`` khi validate.
        require_reference: Bắt buộc có 2 cột nhãn tham chiếu khi validate.

    Raises:
        FileNotFoundError: Không tìm thấy file.
        ValueError: File rỗng, sai mã hóa, sai định dạng, trùng cột hoặc không
            đúng schema.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    try:
        # Đọc dòng tiêu đề thô để phát hiện cột trùng trước khi pandas đổi tên.
        with path.open(encoding=ENCODING, newline="") as handle:
            header = next(csv.reader(handle), None)
        if not header or not any(name.strip() for name in header):
            raise ValueError("File CSV không có dữ liệu.")

        # Loại khoảng trắng thừa ở tên cột (ví dụ " Công nghệ").
        header = [name.strip() for name in header]
        check_duplicate_columns(header)

        df = pd.read_csv(
            path,
            encoding=ENCODING,
            header=0,
            names=header,
            index_col=False,
        )
    except UnicodeDecodeError as error:
        raise ValueError("Không thể đọc file CSV với mã hóa UTF-8.") from error
    except pd.errors.EmptyDataError as error:
        raise ValueError("File CSV không có dữ liệu.") from error
    except pd.errors.ParserError as error:
        raise ValueError(f"File CSV sai định dạng: {error}") from error

    if validate:
        df = validate_data(
            df,
            require_target=require_target,
            require_reference=require_reference,
        )
    return df

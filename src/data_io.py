import pandas as pd

from src.schema import (
    REQUIRED_COLUMNS,
    SCORE_COLUMNS,
    MIN_SCORE,
    MAX_SCORE
)


def check_duplicate_columns(columns):
    """
    Kiểm tra các tên cột bị trùng.

    Parameters
    ----------
    columns : list
        Danh sách tên các cột.

    Raises
    ------
    ValueError
        Nếu phát hiện tên cột bị trùng.
    """

    duplicates = []

    for column in columns:
        if list(columns).count(column) > 1:
            if column not in duplicates:
                duplicates.append(column)

    if duplicates:
        raise ValueError(
            f"Phát hiện tên cột bị trùng: {duplicates}"
        )


def validate_data(df):
    """
    Kiểm tra dữ liệu đầu vào theo schema.py.

    Các điều kiện kiểm tra:
    - Có đầy đủ các cột bắt buộc.
    - Không có dữ liệu rỗng ở cột điểm.
    - Điểm phải là số.
    - Điểm phải nằm trong khoảng từ 1 đến 5.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Dữ liệu hợp lệ sau khi kiểm tra.

    Raises
    ------
    ValueError
        Nếu dữ liệu không đúng schema.
    """

    # Kiểm tra cột bắt buộc
    missing_columns = []

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            missing_columns.append(column)

    if missing_columns:
        raise ValueError(
            f"Thiếu các cột bắt buộc: {missing_columns}"
        )

    # Kiểm tra dữ liệu rỗng
    for column in SCORE_COLUMNS:
        if df[column].isnull().any():
            raise ValueError(
                f"Cột '{column}' có dữ liệu bị thiếu."
            )

    # Kiểm tra kiểu dữ liệu và khoảng điểm
    for column in SCORE_COLUMNS:

        # Chuyển dữ liệu sang dạng số
        scores = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # Nếu không chuyển được thành số
        if scores.isnull().any():
            raise ValueError(
                f"Cột '{column}' chứa dữ liệu không phải số."
            )

        # Kiểm tra thang điểm 1-5
        invalid_scores = scores[
            (scores < MIN_SCORE) |
            (scores > MAX_SCORE)
        ]

        if not invalid_scores.empty:
            raise ValueError(
                f"Cột '{column}' có điểm không hợp lệ. "
                f"Điểm phải nằm trong khoảng "
                f"{MIN_SCORE}-{MAX_SCORE}."
            )

        # Cập nhật lại cột sau khi chuyển sang số
        df[column] = scores

    return df


def read_csv_safe(file_path):
    """
    Đọc file CSV an toàn.

    Hỗ trợ:
    - UTF-8.
    - UTF-8 BOM (utf-8-sig).
    - Kiểm tra tên cột trùng lặp.
    - Validate dữ liệu theo schema.py.

    Parameters
    ----------
    file_path : str
        Đường dẫn đến file CSV.

    Returns
    -------
    pandas.DataFrame
        DataFrame đã được kiểm tra và xác thực.
    """

    try:
        # utf-8-sig có thể đọc được cả UTF-8 thông thường
        # và UTF-8 có BOM
        df = pd.read_csv(
            file_path,
            encoding="utf-8-sig"
        )

    except UnicodeDecodeError as error:
        raise ValueError(
            "Không thể đọc file CSV với mã hóa UTF-8."
        ) from error

    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Không tìm thấy file: {file_path}"
        ) from error

    except pd.errors.EmptyDataError as error:
        raise ValueError(
            "File CSV không có dữ liệu."
        ) from error

    # Kiểm tra tên cột bị trùng
    check_duplicate_columns(df.columns)

    # Kiểm tra dữ liệu theo schema
    validate_data(df)

    return df

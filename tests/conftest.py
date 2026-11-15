"""Fixture dữ liệu mẫu dùng chung, dựng trực tiếp từ ``src/schema.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

from src import schema


def _make_record(index: int, label: str) -> dict[str, Any]:
    """Tạo một bản ghi hợp lệ gồm 16 đặc trưng, nhãn và 2 cột tham chiếu."""
    record: dict[str, Any] = {
        column: (index + offset) % 5 + 1
        for offset, column in enumerate(schema.NUMERIC_FEATURES)
    }
    record[schema.WORK_OBJECT_COL] = schema.VALID_WORK_OBJECTS[
        index % len(schema.VALID_WORK_OBJECTS)
    ]
    record[schema.ENVIRONMENT_COL] = schema.VALID_ENVIRONMENTS[
        index % len(schema.VALID_ENVIRONMENTS)
    ]
    careers = schema.CAREERS_BY_GROUP[label]
    record[schema.SPECIFIC_CAREER_COL] = careers[index % len(careers)]
    record[schema.TARGET_COLUMN] = label
    record[schema.CAREER_GROUP_NAME_COL] = schema.CAREER_GROUP_NAMES[label]
    return record


def _make_survey_frame(rows_per_label: int = 2) -> pd.DataFrame:
    """DataFrame hợp lệ, đủ 4 nhóm nghề N1-N4, đúng thứ tự cột ``ALL_COLUMNS``."""
    records = []
    for label_index, label in enumerate(schema.VALID_TARGET_LABELS):
        for row in range(rows_per_label):
            records.append(_make_record(label_index * rows_per_label + row, label))
    return pd.DataFrame(records, columns=schema.ALL_COLUMNS)


@pytest.fixture
def valid_survey_df() -> pd.DataFrame:
    """Bảng khảo sát hợp lệ đầy đủ 19 cột (16 đặc trưng + 3 cột nhãn)."""
    return _make_survey_frame()


@pytest.fixture
def sample_student_data(valid_survey_df: pd.DataFrame) -> pd.DataFrame:
    """Chỉ 16 đặc trưng đầu vào của người dùng (không có nhãn)."""
    return valid_survey_df[schema.FEATURE_COLUMNS].copy()


@pytest.fixture
def sample_career_data() -> pd.DataFrame:
    """Bảng tham chiếu nhóm nghề N1-N4 lấy từ schema."""
    return pd.DataFrame(
        {
            schema.TARGET_COLUMN: schema.VALID_TARGET_LABELS,
            schema.CAREER_GROUP_NAME_COL: [
                schema.CAREER_GROUP_NAMES[label] for label in schema.VALID_TARGET_LABELS
            ],
            schema.SPECIFIC_CAREER_COL: [
                ", ".join(schema.CAREERS_BY_GROUP[label])
                for label in schema.VALID_TARGET_LABELS
            ],
        }
    )


@pytest.fixture
def valid_input_record(valid_survey_df: pd.DataFrame) -> dict[str, Any]:
    """Một bản ghi 16 đặc trưng hợp lệ (dùng cho ``validate_input_record``)."""
    row = valid_survey_df.iloc[0][schema.FEATURE_COLUMNS]
    return {
        column: (int(value) if column in schema.NUMERIC_FEATURES else value)
        for column, value in row.items()
    }


@pytest.fixture
def invalid_student_data(valid_survey_df: pd.DataFrame) -> pd.DataFrame:
    """Dữ liệu có điểm Likert ngoài thang 1-5 (6 và 0)."""
    df = valid_survey_df.copy()
    df.loc[0, schema.NUMERIC_FEATURES[0]] = 6
    df.loc[1, schema.NUMERIC_FEATURES[1]] = 0
    return df


@pytest.fixture
def invalid_category_df(valid_survey_df: pd.DataFrame) -> pd.DataFrame:
    """Dữ liệu có danh mục "Môi trường" không nằm trong danh sách hợp lệ."""
    df = valid_survey_df.copy()
    df.loc[0, schema.ENVIRONMENT_COL] = "Môi trường không tồn tại"
    return df


@pytest.fixture
def missing_column_df(valid_survey_df: pd.DataFrame) -> pd.DataFrame:
    """Dữ liệu thiếu một cột đặc trưng bắt buộc."""
    return valid_survey_df.drop(columns=[schema.NUMERIC_FEATURES[0]])


@pytest.fixture
def write_csv(tmp_path: Path) -> Callable[..., Path]:
    """Factory ghi DataFrame (hoặc chuỗi thô) ra file CSV tạm.

    ``encoding="utf-8-sig"`` tạo file có BOM; ``"utf-8"`` tạo file không BOM.
    """

    def _write(
        data: pd.DataFrame | str,
        name: str = "data.csv",
        encoding: str = "utf-8",
    ) -> Path:
        path = tmp_path / name
        if isinstance(data, str):
            path.write_bytes(data.encode(encoding))
        else:
            data.to_csv(path, index=False, encoding=encoding)
        return path

    return _write

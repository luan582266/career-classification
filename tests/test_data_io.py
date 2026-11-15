"""Unit test cho src/data_io.py: đọc CSV UTF-8 / BOM, cột trùng, validate theo schema."""

from __future__ import annotations

import pandas as pd
import pytest

from src import schema
from src.config import RAW_SURVEY_PATH
from src.data_io import check_duplicate_columns, read_csv_safe, validate_data


class TestCheckDuplicateColumns:
    def test_accepts_unique_columns(self) -> None:
        check_duplicate_columns(["a", "b", "c"])

    def test_rejects_duplicates_and_reports_names(self) -> None:
        with pytest.raises(ValueError, match="'a'"):
            check_duplicate_columns(["a", "b", "a"])


class TestValidateData:
    def test_valid_frame_passes(self, valid_survey_df: pd.DataFrame) -> None:
        result = validate_data(valid_survey_df)
        assert len(result) == len(valid_survey_df)

    def test_does_not_mutate_input(self, valid_survey_df: pd.DataFrame) -> None:
        float_df = valid_survey_df.copy()
        float_df[schema.NUMERIC_FEATURES] = float_df[schema.NUMERIC_FEATURES].astype(float)
        before = float_df.copy()
        result = validate_data(float_df)
        pd.testing.assert_frame_equal(float_df, before)
        assert all(result[c].dtype == "int64" for c in schema.NUMERIC_FEATURES)

    def test_rejects_missing_column(self, missing_column_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Thiếu các cột bắt buộc"):
            validate_data(missing_column_df)

    def test_rejects_out_of_range_scores(self, invalid_student_data: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Likert"):
            validate_data(invalid_student_data)

    def test_rejects_decimal_scores(self, valid_survey_df: pd.DataFrame) -> None:
        df = valid_survey_df.astype({schema.NUMERIC_FEATURES[0]: float})
        df.loc[0, schema.NUMERIC_FEATURES[0]] = 2.5
        with pytest.raises(ValueError):
            validate_data(df)

    def test_rejects_missing_values(self, valid_survey_df: pd.DataFrame) -> None:
        df = valid_survey_df.astype({schema.NUMERIC_FEATURES[0]: float})
        df.loc[0, schema.NUMERIC_FEATURES[0]] = None
        with pytest.raises(ValueError, match="khuyết"):
            validate_data(df)

    def test_rejects_non_numeric_scores(self, valid_survey_df: pd.DataFrame) -> None:
        df = valid_survey_df.astype({schema.NUMERIC_FEATURES[0]: object})
        df.loc[0, schema.NUMERIC_FEATURES[0]] = "abc"
        with pytest.raises(ValueError, match="kiểu số"):
            validate_data(df)

    def test_rejects_invalid_category(self, invalid_category_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="danh mục không hợp lệ"):
            validate_data(invalid_category_df)

    def test_rejects_invalid_target_label(self, valid_survey_df: pd.DataFrame) -> None:
        valid_survey_df.loc[0, schema.TARGET_COLUMN] = "N9"
        with pytest.raises(ValueError, match="Nhãn mục tiêu"):
            validate_data(valid_survey_df)

    def test_target_optional_when_not_required(self, valid_survey_df: pd.DataFrame) -> None:
        features_only = valid_survey_df[schema.FEATURE_COLUMNS]
        validate_data(features_only, require_target=False)
        with pytest.raises(ValueError):
            validate_data(features_only, require_target=True)

    def test_rejects_empty_frame(self, valid_survey_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="không có bản ghi"):
            validate_data(valid_survey_df.iloc[0:0])


class TestReadCsvSafe:
    @pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
    def test_reads_utf8_with_and_without_bom(
        self, write_csv, valid_survey_df: pd.DataFrame, encoding: str
    ) -> None:
        path = write_csv(valid_survey_df, encoding=encoding)
        df = read_csv_safe(path)
        # Với file có BOM, tên cột đầu tiên không được dính ký tự \ufeff.
        assert list(df.columns) == schema.ALL_COLUMNS
        assert df[schema.WORK_OBJECT_COL].tolist() == valid_survey_df[schema.WORK_OBJECT_COL].tolist()

    def test_accepts_str_and_path(self, write_csv, valid_survey_df: pd.DataFrame) -> None:
        path = write_csv(valid_survey_df)
        assert len(read_csv_safe(str(path))) == len(read_csv_safe(path))

    def test_strips_whitespace_in_column_names(
        self, write_csv, valid_survey_df: pd.DataFrame
    ) -> None:
        padded = valid_survey_df.rename(
            columns={schema.NUMERIC_FEATURES[0]: f" {schema.NUMERIC_FEATURES[0]} "}
        )
        df = read_csv_safe(write_csv(padded))
        assert schema.NUMERIC_FEATURES[0] in df.columns

    def test_detects_duplicate_column_names_in_raw_header(
        self, write_csv, valid_survey_df: pd.DataFrame
    ) -> None:
        # pandas sẽ tự đổi cột trùng thành "col.1"; hàm phải bắt được từ header thô.
        header = list(valid_survey_df.columns)
        header[1] = header[0]
        lines = [",".join(header)]
        for _, row in valid_survey_df.iterrows():
            lines.append(",".join(str(v) for v in row))
        path = write_csv("\n".join(lines) + "\n")
        with pytest.raises(ValueError, match="bị trùng"):
            read_csv_safe(path)

    def test_rejects_missing_column(self, write_csv, missing_column_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Thiếu các cột bắt buộc"):
            read_csv_safe(write_csv(missing_column_df))

    def test_rejects_out_of_range_scores(
        self, write_csv, invalid_student_data: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="Likert"):
            read_csv_safe(write_csv(invalid_student_data))

    def test_rejects_empty_cell_in_score_column(
        self, write_csv, valid_survey_df: pd.DataFrame
    ) -> None:
        df = valid_survey_df.astype({schema.NUMERIC_FEATURES[0]: float})
        df.loc[0, schema.NUMERIC_FEATURES[0]] = None
        with pytest.raises(ValueError, match="khuyết"):
            read_csv_safe(write_csv(df))

    def test_missing_file_raises_file_not_found(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            read_csv_safe(tmp_path / "khong_ton_tai.csv")

    def test_empty_file_raises_value_error(self, write_csv) -> None:
        with pytest.raises(ValueError, match="không có dữ liệu"):
            read_csv_safe(write_csv("", name="empty.csv"))

    def test_header_only_file_raises_value_error(
        self, write_csv, valid_survey_df: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="không có bản ghi"):
            read_csv_safe(write_csv(valid_survey_df.iloc[0:0]))

    def test_invalid_encoding_raises_value_error(self, write_csv) -> None:
        path = write_csv("a,b\n\xe9,2\n", name="latin1.csv", encoding="latin-1")
        with pytest.raises(ValueError, match="UTF-8"):
            read_csv_safe(path)

    def test_validate_false_skips_schema_check(self, write_csv) -> None:
        path = write_csv("x,y\n1,2\n")
        df = read_csv_safe(path, validate=False)
        assert list(df.columns) == ["x", "y"]

    def test_default_path_reads_real_survey(self) -> None:
        if not RAW_SURVEY_PATH.exists():
            pytest.skip("Không có data/raw/survey.csv")
        df = read_csv_safe()
        assert list(df.columns) == schema.ALL_COLUMNS
        assert len(df) > 0

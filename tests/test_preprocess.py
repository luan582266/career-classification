"""Unit test cho src/preprocess.py: chia tập StratifiedKFold, tiền xử lý, chống data leakage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, schema
from src.preprocess import (
    N_SPLITS,
    SCORE_BINS,
    UNKNOWN_CATEGORY_CODE,
    SurveyPreprocessor,
    discretize_score,
    run_preprocessing,
    stratified_train_val_test_split,
    validate_raw_survey,
)


def _make_balanced_df(rows_per_class: int = 10) -> pd.DataFrame:
    """Tạo DataFrame có đủ 16 đặc trưng, 4 lớp N1-N4 cân bằng để test chia tập."""
    records = []
    for label_idx, label in enumerate(schema.VALID_TARGET_LABELS):
        for i in range(rows_per_class):
            rec = {
                col: ((label_idx + i + j) % 5) + 1
                for j, col in enumerate(schema.NUMERIC_FEATURES)
            }
            rec[schema.WORK_OBJECT_COL] = schema.VALID_WORK_OBJECTS[
                (label_idx + i) % len(schema.VALID_WORK_OBJECTS)
            ]
            rec[schema.ENVIRONMENT_COL] = schema.VALID_ENVIRONMENTS[
                (label_idx + i) % len(schema.VALID_ENVIRONMENTS)
            ]
            rec[schema.TARGET_COLUMN] = label
            careers = schema.CAREERS_BY_GROUP[label]
            rec[schema.SPECIFIC_CAREER_COL] = careers[i % len(careers)]
            rec[schema.CAREER_GROUP_NAME_COL] = schema.CAREER_GROUP_NAMES[label]
            records.append(rec)
    return pd.DataFrame(records, columns=schema.ALL_COLUMNS)


def _fitted_preprocessor(df: pd.DataFrame | None = None) -> SurveyPreprocessor:
    return SurveyPreprocessor().fit(_make_balanced_df(10) if df is None else df)


class TestDiscretizeScore:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [(1, "low"), (2, "low"), (3, "medium"), (4, "high"), (5, "high")],
    )
    def test_likert_bins(self, score: int, expected: str) -> None:
        assert discretize_score(score) == expected

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError):
            discretize_score(float("nan"))


class TestValidateRawSurvey:
    def test_valid_data_with_blank_cells_passes(self) -> None:
        df = _make_balanced_df(5).astype({schema.NUMERIC_FEATURES[0]: float})
        df.loc[0, schema.NUMERIC_FEATURES[0]] = np.nan
        df.loc[1, schema.WORK_OBJECT_COL] = None
        validate_raw_survey(df)  # ô trống được phép, sẽ được điền ở bước sau

    def test_rejects_out_of_range_score(self) -> None:
        df = _make_balanced_df(5)
        df.loc[0, schema.NUMERIC_FEATURES[0]] = 7
        with pytest.raises(ValueError, match="1-5"):
            validate_raw_survey(df)

    def test_rejects_decimal_score(self) -> None:
        df = _make_balanced_df(5).astype({schema.NUMERIC_FEATURES[0]: float})
        df.loc[0, schema.NUMERIC_FEATURES[0]] = 2.5
        with pytest.raises(ValueError, match="số nguyên"):
            validate_raw_survey(df)

    def test_rejects_text_in_score_column(self) -> None:
        df = _make_balanced_df(5).astype({schema.NUMERIC_FEATURES[0]: object})
        df.loc[0, schema.NUMERIC_FEATURES[0]] = "abc"
        with pytest.raises(ValueError, match="không phải số"):
            validate_raw_survey(df)

    def test_rejects_invalid_category_and_label(self) -> None:
        bad_category = _make_balanced_df(5)
        bad_category.loc[0, schema.ENVIRONMENT_COL] = "Môi trường lạ"
        with pytest.raises(ValueError, match="danh mục không hợp lệ"):
            validate_raw_survey(bad_category)

        bad_label = _make_balanced_df(5)
        bad_label.loc[0, schema.TARGET_COLUMN] = "N9"
        with pytest.raises(ValueError, match="Nhãn mục tiêu"):
            validate_raw_survey(bad_label)

    def test_rejects_missing_label_and_missing_column(self) -> None:
        missing_label = _make_balanced_df(5)
        missing_label.loc[0, schema.TARGET_COLUMN] = None
        with pytest.raises(ValueError, match="khuyết"):
            validate_raw_survey(missing_label)

        with pytest.raises(ValueError, match="Thiếu các cột"):
            validate_raw_survey(_make_balanced_df(5).drop(columns=[schema.NUMERIC_FEATURES[0]]))


class TestStratifiedSplit:
    def test_sizes_and_stratification_60_20_20(self) -> None:
        df = _make_balanced_df(rows_per_class=15)  # 60 mẫu: 15 mỗi lớp
        train_df, val_df, test_df = stratified_train_val_test_split(df, random_state=42)

        assert (len(train_df), len(val_df), len(test_df)) == (36, 12, 12)
        assert train_df[schema.TARGET_COLUMN].value_counts().to_dict() == {
            "N1": 9, "N2": 9, "N3": 9, "N4": 9
        }
        for part in (val_df, test_df):
            assert part[schema.TARGET_COLUMN].value_counts().to_dict() == {
                "N1": 3, "N2": 3, "N3": 3, "N4": 3
            }

    def test_splits_are_disjoint_and_cover_all_rows(self) -> None:
        df = _make_balanced_df(rows_per_class=15)
        train_df, val_df, test_df = stratified_train_val_test_split(df, random_state=42)

        train_ids, val_ids, test_ids = set(train_df.index), set(val_df.index), set(test_df.index)
        assert not (train_ids & val_ids)
        assert not (train_ids & test_ids)
        assert not (val_ids & test_ids)
        assert train_ids | val_ids | test_ids == set(df.index)

    def test_reproducible_with_same_seed_and_differs_with_other_seed(self) -> None:
        df = _make_balanced_df(rows_per_class=15)
        first = stratified_train_val_test_split(df, random_state=42)
        second = stratified_train_val_test_split(df, random_state=42)
        other = stratified_train_val_test_split(df, random_state=7)
        for a, b in zip(first, second):
            assert a.index.tolist() == b.index.tolist()
        assert first[2].index.tolist() != other[2].index.tolist()

    def test_n_splits_matches_configured_ratios(self) -> None:
        assert N_SPLITS == 5
        assert config.TRAIN_RATIO == pytest.approx((N_SPLITS - 2) / N_SPLITS)

    @pytest.mark.parametrize(
        ("test_fold", "val_fold"), [(0, 0), (-1, 1), (0, 5), (5, 1)]
    )
    def test_rejects_invalid_fold_arguments(self, test_fold: int, val_fold: int) -> None:
        df = _make_balanced_df(rows_per_class=10)
        with pytest.raises(ValueError):
            stratified_train_val_test_split(df, test_fold=test_fold, val_fold=val_fold)

    def test_rejects_too_small_class_missing_label_and_duplicate_index(self) -> None:
        with pytest.raises(ValueError, match="ít nhất"):
            stratified_train_val_test_split(_make_balanced_df(rows_per_class=3))

        with_nan = _make_balanced_df(rows_per_class=10)
        with_nan.loc[0, schema.TARGET_COLUMN] = None
        with pytest.raises(ValueError, match="thiếu"):
            stratified_train_val_test_split(with_nan)

        duplicated_index = _make_balanced_df(rows_per_class=10)
        duplicated_index.index = [0] * len(duplicated_index)
        with pytest.raises(ValueError, match="duy nhất"):
            stratified_train_val_test_split(duplicated_index)

        with pytest.raises(KeyError):
            stratified_train_val_test_split(_make_balanced_df(10), target_col="khong_co")


class TestSurveyPreprocessor:
    def test_imputes_median_and_mode_without_data_leakage(self) -> None:
        df = _make_balanced_df(rows_per_class=2)  # 8 dòng
        train_df = df.iloc[:5].reset_index(drop=True)
        val_df = df.iloc[5:].reset_index(drop=True)

        num_col = schema.NUMERIC_FEATURES[0]
        cat_col = schema.WORK_OBJECT_COL

        # Train có [1, 1, 2, 5, NaN]; median_low của [1, 1, 2, 5] = 1 -> "low"
        train_df[num_col] = [1.0, 1.0, 2.0, 5.0, np.nan]
        # Val toàn giá trị cao [5, 5] không được làm lệch median học từ train
        val_df[num_col] = [5.0, np.nan, 5.0]

        # Train: mode là "Dữ liệu" (3 lần vs "Phần mềm" 1 lần)
        train_df[cat_col] = ["Dữ liệu", "Dữ liệu", "Dữ liệu", "Phần mềm", None]
        # Val có "Máy móc - thiết bị" (chưa có trong train)
        val_df[cat_col] = ["Máy móc - thiết bị", None, "Dữ liệu"]

        prep = SurveyPreprocessor()
        train_clean = prep.fit_transform(train_df)
        val_clean = prep.transform(val_df)

        assert prep.numeric_medians_[num_col] == 1
        assert prep.categorical_modes_[cat_col] == "Dữ liệu"

        # Ô thiếu ở train (dòng 4) và val (dòng 1) đều dùng median/mode của train
        assert train_clean.loc[4, num_col] == "low"
        assert val_clean.loc[1, num_col] == "low"

        expected_mode_code = prep.category_mappings_[cat_col]["Dữ liệu"]
        assert train_clean.loc[4, cat_col] == expected_mode_code
        assert val_clean.loc[1, cat_col] == expected_mode_code
        # Danh mục chỉ có trong val (chưa gặp ở train) -> UNKNOWN_CATEGORY_CODE
        assert val_clean.loc[0, cat_col] == UNKNOWN_CATEGORY_CODE

    def test_validation_data_does_not_change_fitted_statistics(self) -> None:
        train_df = _make_balanced_df(rows_per_class=5)
        prep = SurveyPreprocessor().fit(train_df)
        before = (dict(prep.numeric_medians_), dict(prep.categorical_modes_))

        extreme = train_df.copy()
        extreme[schema.NUMERIC_FEATURES] = 5
        prep.transform(extreme)

        assert (prep.numeric_medians_, prep.categorical_modes_) == before

    @pytest.mark.parametrize(
        ("values", "expected_median", "expected_bin"),
        [
            ([1, 2, 3, 4], 2, "low"),       # hai số giữa 2 và 3 -> lấy 2 (median_low)
            ([3, 3, 4, 5], 3, "medium"),
            ([4, 4, 5, 5], 4, "high"),
            ([1, 3, 5], 3, "medium"),
        ],
    )
    def test_median_is_integer_likert_value(
        self, values: list[int], expected_median: int, expected_bin: str
    ) -> None:
        df = _make_balanced_df(rows_per_class=2).iloc[: len(values)].reset_index(drop=True)
        col = schema.NUMERIC_FEATURES[0]
        df[col] = values
        prep = SurveyPreprocessor().fit(df)

        assert prep.numeric_medians_[col] == expected_median
        assert isinstance(prep.numeric_medians_[col], int)
        assert discretize_score(prep.numeric_medians_[col]) == expected_bin

    def test_mode_tie_is_broken_alphabetically(self) -> None:
        df = _make_balanced_df(rows_per_class=2).iloc[:4].reset_index(drop=True)
        df[schema.WORK_OBJECT_COL] = ["Dữ liệu", "Phần mềm", "Dữ liệu", "Phần mềm"]
        prep = SurveyPreprocessor().fit(df)
        assert prep.categorical_modes_[schema.WORK_OBJECT_COL] == sorted(
            ["Dữ liệu", "Phần mềm"]
        )[0]

    def test_discretize_and_encode_outputs_expected_domains(
        self, valid_survey_df: pd.DataFrame
    ) -> None:
        clean = SurveyPreprocessor().fit_transform(valid_survey_df)

        for col in schema.NUMERIC_FEATURES:
            assert set(clean[col].unique()) <= set(SCORE_BINS)
        for col in schema.CATEGORICAL_FEATURES:
            assert pd.api.types.is_integer_dtype(clean[col])
            assert (clean[col] >= 0).all()
        assert set(clean[schema.TARGET_COLUMN]) == set(schema.VALID_TARGET_LABELS)

    def test_transform_rejects_invalid_scores(self) -> None:
        prep = _fitted_preprocessor()
        bad = _make_balanced_df(rows_per_class=2)
        bad.loc[0, schema.NUMERIC_FEATURES[0]] = 9
        with pytest.raises(ValueError, match="1-5"):
            prep.transform(bad)

        bad_text = _make_balanced_df(rows_per_class=2).astype({schema.NUMERIC_FEATURES[0]: object})
        bad_text.loc[0, schema.NUMERIC_FEATURES[0]] = "abc"
        with pytest.raises(ValueError, match="không phải số"):
            prep.transform(bad_text)

    def test_fit_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="rỗng"):
            SurveyPreprocessor().fit(_make_balanced_df(2).iloc[0:0])

        with pytest.raises(ValueError, match="thiếu các cột"):
            SurveyPreprocessor().fit(_make_balanced_df(2).drop(columns=[schema.ENVIRONMENT_COL]))

        all_missing = _make_balanced_df(2).astype({schema.NUMERIC_FEATURES[0]: float})
        all_missing[schema.NUMERIC_FEATURES[0]] = np.nan
        with pytest.raises(ValueError, match="toàn bộ là giá trị thiếu"):
            SurveyPreprocessor().fit(all_missing)

        all_missing_cat = _make_balanced_df(2)
        all_missing_cat[schema.ENVIRONMENT_COL] = None
        with pytest.raises(ValueError, match="toàn bộ là giá trị thiếu"):
            SurveyPreprocessor().fit(all_missing_cat)

    def test_transform_before_fit_raises(self, valid_survey_df: pd.DataFrame) -> None:
        with pytest.raises(RuntimeError):
            SurveyPreprocessor().transform(valid_survey_df)


class TestSaveLoad:
    def test_roundtrip_gives_identical_transform_and_inverse_mapping(
        self, tmp_path: Path
    ) -> None:
        train_df = _make_balanced_df(rows_per_class=10)
        prep = SurveyPreprocessor().fit(train_df)
        path = prep.save(tmp_path / "sub" / "preprocessor.json")

        loaded = SurveyPreprocessor.load(path)
        sample = _make_balanced_df(rows_per_class=3)
        pd.testing.assert_frame_equal(prep.transform(sample), loaded.transform(sample))

        assert loaded.numeric_medians_ == prep.numeric_medians_
        assert loaded.categorical_modes_ == prep.categorical_modes_
        assert loaded.category_mappings_ == prep.category_mappings_
        # Bảng mã hóa phải đọc ngược được từ mã số về chuỗi gốc.
        env = schema.ENVIRONMENT_COL
        for category, code in loaded.category_mappings_[env].items():
            assert loaded.inverse_category_mappings_[env][code] == category

    def test_save_before_fit_and_load_missing_file_raise(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            SurveyPreprocessor().save(tmp_path / "x.json")
        with pytest.raises(FileNotFoundError):
            SurveyPreprocessor.load(tmp_path / "khong_ton_tai.json")


class TestRunPreprocessing:
    @staticmethod
    def _write_raw(tmp_path: Path, df: pd.DataFrame) -> Path:
        path = tmp_path / "raw.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        return path

    def test_outputs_exclude_reference_columns_and_match_files(self, tmp_path: Path) -> None:
        raw_path = self._write_raw(tmp_path, _make_balanced_df(rows_per_class=20))
        out_dir = tmp_path / "processed"
        prep_path = tmp_path / "models" / "preprocessor.json"

        train_df, val_df, test_df, prep = run_preprocessing(
            raw_path, out_dir, preprocessor_path=prep_path
        )

        expected_columns = [*schema.FEATURE_COLUMNS, schema.TARGET_COLUMN]
        for name, frame in (
            ("train.csv", train_df), ("validation.csv", val_df), ("test.csv", test_df)
        ):
            assert list(frame.columns) == expected_columns
            saved = pd.read_csv(out_dir / name)
            assert list(saved.columns) == expected_columns
            pd.testing.assert_frame_equal(
                saved.reset_index(drop=True),
                frame.reset_index(drop=True),
                check_dtype=False,
            )

        # 3 tập không chồng nhau theo chỉ số hàng gốc và phủ hết dữ liệu
        ids = [set(train_df.index), set(val_df.index), set(test_df.index)]
        assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
        assert ids[0] | ids[1] | ids[2] == set(range(80))

        # Preprocessor đã được lưu và nạp lại cho kết quả giống hệt
        loaded = SurveyPreprocessor.load(prep_path)
        assert loaded.category_mappings_ == prep.category_mappings_

    def test_keep_reference_option_keeps_all_columns(self, tmp_path: Path) -> None:
        raw_path = self._write_raw(tmp_path, _make_balanced_df(rows_per_class=10))
        train_df, *_ = run_preprocessing(
            raw_path, tmp_path / "out", preprocessor_path=None, keep_reference=True
        )
        # Thứ tự cột: 16 đặc trưng, nhãn mục tiêu, rồi 2 cột tham chiếu.
        assert list(train_df.columns) == [
            *schema.FEATURE_COLUMNS,
            schema.TARGET_COLUMN,
            *schema.REFERENCE_COLUMNS,
        ]
        assert set(train_df.columns) == set(schema.ALL_COLUMNS)
        assert not (tmp_path / "models").exists()

    def test_rejects_invalid_raw_data(self, tmp_path: Path) -> None:
        df = _make_balanced_df(rows_per_class=10)
        df.loc[0, schema.NUMERIC_FEATURES[0]] = 7
        raw_path = self._write_raw(tmp_path, df)
        with pytest.raises(ValueError, match="1-5"):
            run_preprocessing(raw_path, tmp_path / "out", preprocessor_path=None)

    def test_fills_blank_cells_instead_of_failing(self, tmp_path: Path) -> None:
        df = _make_balanced_df(rows_per_class=10).astype({schema.NUMERIC_FEATURES[0]: float})
        df.loc[0, schema.NUMERIC_FEATURES[0]] = np.nan
        df.loc[1, schema.WORK_OBJECT_COL] = None
        raw_path = self._write_raw(tmp_path, df)
        train_df, val_df, test_df, _ = run_preprocessing(
            raw_path, tmp_path / "out", preprocessor_path=None
        )
        for frame in (train_df, val_df, test_df):
            assert not frame.isna().any().any()


class TestRealSurveyData:
    def test_run_preprocessing_on_real_survey(self, tmp_path: Path) -> None:
        if not config.RAW_SURVEY_PATH.exists():
            pytest.skip("Không có data/raw/survey.csv")

        train_df, val_df, test_df, prep = run_preprocessing(
            config.RAW_SURVEY_PATH,
            tmp_path,
            preprocessor_path=tmp_path / "preprocessor.json",
        )

        assert (len(train_df), len(val_df), len(test_df)) == (2400, 800, 800)
        for frame, per_class in ((train_df, 600), (val_df, 200), (test_df, 200)):
            assert frame[schema.TARGET_COLUMN].value_counts().to_dict() == {
                label: per_class for label in schema.VALID_TARGET_LABELS
            }
            assert not frame.isna().any().any()
            assert list(frame.columns) == [*schema.FEATURE_COLUMNS, schema.TARGET_COLUMN]
            for col in schema.NUMERIC_FEATURES:
                assert set(frame[col].unique()) <= set(SCORE_BINS)
            for col in schema.CATEGORICAL_FEATURES:
                assert pd.api.types.is_integer_dtype(frame[col])
                assert (frame[col] >= 0).all()

        assert len(prep.category_mappings_[schema.WORK_OBJECT_COL]) == len(
            schema.VALID_WORK_OBJECTS
        )
        assert len(prep.category_mappings_[schema.ENVIRONMENT_COL]) == len(
            schema.VALID_ENVIRONMENTS
        )

    def test_exported_processed_files_are_clean(self) -> None:
        paths = {
            "train": (config.TRAIN_DATA_PATH, 2400),
            "validation": (config.VAL_DATA_PATH, 800),
            "test": (config.TEST_DATA_PATH, 800),
        }
        if not all(path.exists() for path, _ in paths.values()):
            pytest.skip("Chưa chạy `python -m src.preprocess` để tạo data/processed/")

        expected_columns = [*schema.FEATURE_COLUMNS, schema.TARGET_COLUMN]
        for path, size in paths.values():
            frame = pd.read_csv(path)
            assert list(frame.columns) == expected_columns
            assert len(frame) == size
            assert not frame.isna().any().any()
            counts = frame[schema.TARGET_COLUMN].value_counts()
            assert set(counts.index) == set(schema.VALID_TARGET_LABELS)
            assert counts.nunique() == 1  # phân bố đều giữa N1-N4
            for col in schema.NUMERIC_FEATURES:
                assert set(frame[col].unique()) <= set(SCORE_BINS)

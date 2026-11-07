"""Cấu hình đường dẫn thư mục, tham số ngẫu nhiên và tỷ lệ chia tập dữ liệu."""

from pathlib import Path

# Thư mục gốc của dự án
BASE_DIR: Path = Path(__file__).resolve().parent.parent
ROOT_DIR: Path = BASE_DIR

# Đường dẫn thư mục dữ liệu
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
DATA_DICTIONARY_PATH: Path = DATA_DIR / "data_dictionary.md"

# Đường dẫn các tệp dữ liệu
RAW_SURVEY_PATH: Path = RAW_DATA_DIR / "survey.csv"
RAW_DATA_PATH: Path = RAW_SURVEY_PATH

TRAIN_DATA_PATH: Path = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH: Path = PROCESSED_DATA_DIR / "validation.csv"
VALIDATION_DATA_PATH: Path = VAL_DATA_PATH
TEST_DATA_PATH: Path = PROCESSED_DATA_DIR / "test.csv"

TRAIN_PATH: Path = TRAIN_DATA_PATH
VAL_PATH: Path = VAL_DATA_PATH
TEST_PATH: Path = TEST_DATA_PATH

# Đường dẫn thư mục và tệp mô hình
MODELS_DIR: Path = BASE_DIR / "models"
ID3_MODEL_PATH: Path = MODELS_DIR / "id3.joblib"
NAIVE_BAYES_MODEL_PATH: Path = MODELS_DIR / "naive_bayes.joblib"
NB_MODEL_PATH: Path = NAIVE_BAYES_MODEL_PATH
MANIFEST_PATH: Path = MODELS_DIR / "manifest.json"

# Đường dẫn thư mục báo cáo và kết quả đánh giá
REPORTS_DIR: Path = BASE_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_DIR: Path = REPORTS_DIR / "metrics"
PREDICTIONS_DIR: Path = REPORTS_DIR / "predictions"

ID3_CM_FIG_PATH: Path = FIGURES_DIR / "id3_confusion_matrix.png"
NB_CM_FIG_PATH: Path = FIGURES_DIR / "nb_confusion_matrix.png"
MODEL_COMPARISON_FIG_PATH: Path = FIGURES_DIR / "model_comparison.png"

ID3_METRICS_PATH: Path = METRICS_DIR / "id3_test.json"
NB_METRICS_PATH: Path = METRICS_DIR / "naive_bayes_test.json"
COMPARISON_CSV_PATH: Path = METRICS_DIR / "comparison.csv"
TEST_PREDICTIONS_PATH: Path = PREDICTIONS_DIR / "test_predictions.csv"

# Seed ngẫu nhiên cố định để tái lập kết quả
RANDOM_SEED: int = 42
RANDOM_STATE: int = RANDOM_SEED
SEED: int = RANDOM_SEED

# Tỷ lệ phân chia tập dữ liệu (Train / Validation / Test: 60% / 20% / 20%)
TRAIN_RATIO: float = 0.60
VAL_RATIO: float = 0.20
VALIDATION_RATIO: float = VAL_RATIO
TEST_RATIO: float = 0.20

assert abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) < 1e-9, (
    "Tổng tỷ lệ TRAIN_RATIO + VAL_RATIO + TEST_RATIO phải bằng 1.0"
)

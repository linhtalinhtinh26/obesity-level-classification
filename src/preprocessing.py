"""Pipeline tiền xử lý học tham số từ train, dùng lại để transform validation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data_loader import CATEGORIES, FEATURES, LABELS, NUMERIC_LIMITS, TARGET, is_missing, parse_number, read_csv


NUMERIC_FEATURES = tuple(NUMERIC_LIMITS)
CATEGORICAL_FEATURES = tuple(CATEGORIES)


def prepare_features(X):
    """Trả DataFrame mới theo đúng 16 cột, chuyển missing values về np.nan.

    X phải có đúng schema đặc trưng, không chứa ID hoặc nhãn. Cột số được
    đọc bằng quy tắc Issue #2. Số không hữu hạn/sai kiểu gây ValueError.
    Số hữu hạn ngoài ngưỡng được giữ nguyên để tránh tự loại outlier.
    Category lạ được giữ để encoder đã fit xử lý theo handle_unknown.
    Hàm không học tham số và không đổi index hay dữ liệu đầu vào.
    """
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique or set(X.columns) != set(FEATURES):
        raise ValueError("X must contain exactly the 16 feature columns")
    result = X.loc[:, list(FEATURES)].copy()
    for column in FEATURES:
        values = []
        for value in result[column]:
            if pd.isna(value) or is_missing(str(value)):
                values.append(np.nan)
            elif column in NUMERIC_LIMITS:
                number = parse_number(str(value))
                if number is None:
                    raise ValueError(f"Invalid or nonfinite numeric value in {column}")
                values.append(number)
            else:
                values.append(str(value))
        result[column] = values
    return result


def split_features_target(frame):
    """Tách DataFrame train đúng schema thành (X, y), giữ cùng thứ tự và index.

    Từ chối thiếu/thừa cột, nhãn thiếu/lạ bằng ValueError. X là bản sao
    của 16 đặc trưng, y là Series nhãn, không sửa frame hoặc CSV nguồn.
    """
    expected = {"id", *FEATURES, TARGET}
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or set(frame.columns) != expected:
        raise ValueError("Training data must match the Issue #2 schema")
    if not frame[TARGET].isin(LABELS).all():
        raise ValueError("Target contains unknown or missing labels")
    return prepare_features(frame.loc[:, list(FEATURES)]), frame[TARGET].copy()


def load_features_target(path=Path("data/train.csv")):
    """Đọc CSV bằng loader Issue #2 rồi trả (X, y) đúng schema.

    Lỗi đường dẫn/cấu trúc CSV giữ nguyên từ read_csv. API không tự xóa
    duplicate hoặc kiểm tra việc chia tập, người gọi đối chiếu validation.
    """
    columns, rows = read_csv(path)
    return split_features_target(pd.DataFrame(rows, columns=columns))


def _schema_feature_names(transformer, input_features):
    """Khai báo đúng thứ tự cột do prepare_features tạo ra, độc lập thứ tự đầu vào."""
    return np.asarray(FEATURES, dtype=object)


def build_preprocessor():
    """Tạo Pipeline CHƯA fit mới cho mỗi lần gọi.

    Nhánh numeric dùng median của train và StandardScaler. Cột train toàn
    thiếu được giữ với giá trị dự phòng 0. Nhánh category dùng hằng
    __MISSING__ rồi OneHotEncoder, category chưa gặp trở thành vector 0
    trong block tương ứng. Đầu ra dense float64, numeric trước category.
    Bước schema không học tham số, giữ train/validation cùng thứ tự cột.
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    category = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)),
    ])
    columns = ColumnTransformer([
        ("numeric", numeric, list(NUMERIC_FEATURES)),
        ("category", category, list(CATEGORICAL_FEATURES)),
    ], remainder="drop")
    return Pipeline([
        ("schema", FunctionTransformer(prepare_features, feature_names_out=_schema_feature_names)),
        ("columns", columns),
    ])


def transform_split(X_train, X_validation):
    """Trả (pipeline, train_processed, validation_processed) cho một split.

    Hai đầu vào là DataFrame chỉ chứa đặc trưng. Hàm tạo pipeline riêng,
    chỉ fit_transform train, validation chỉ transform, không nhận y.
    Hai DataFrame đầu ra dùng tên cột từ ColumnTransformer, giữ index
    của tập tương ứng. Từ chối kết quả không hữu hạn bằng ValueError.
    Người gọi Issue #6 chịu trách nhiệm chia stratified với seed 42,
    giữ y khớp index và kiểm tra trùng nội dung giữa các tập.
    """
    pipeline = build_preprocessor()
    train_values = pipeline.fit_transform(X_train)
    validation_values = pipeline.transform(X_validation)
    if not np.isfinite(train_values).all() or not np.isfinite(validation_values).all():
        raise ValueError("Preprocessing produced nonfinite values")
    names = pipeline.named_steps["columns"].get_feature_names_out()
    train = pd.DataFrame(train_values, columns=names, index=X_train.index)
    validation = pd.DataFrame(validation_values, columns=names, index=X_validation.index)
    return pipeline, train, validation

"""Kiểm chứng split và pipeline Issue #5, không sửa dữ liệu hoặc học mô hình."""

import argparse
import hashlib
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from .data_loader import FEATURES, LABELS
from .preprocessing import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_features_target, transform_split,
)

SPLITS = {"80_20": 0.2, "70_30": 0.3, "60_40": 0.4}
SEED = 42


def file_sha256(path):
    """Nhận đường dẫn và trả SHA-256 của toàn bộ byte để nhận diện phiên bản CSV."""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def make_split(X, y, validation_size, seed=SEED):
    """Trả hai mảng vị trí dòng có thứ tự, chia stratified và không sửa X/y.

    Vị trí dùng với iloc, khác với giá trị index pandas hoặc cột id Kaggle.
    Từ chối index không duy nhất/không căn chỉnh để tránh kiểm tra nhập nhằng.
    """
    if set(X.columns) != set(FEATURES) or not X.columns.is_unique:
        raise ValueError("X must contain exactly the 16 features, without id or target")
    if not X.index.is_unique or not X.index.equals(y.index):
        raise ValueError("X/y require unique, aligned indexes")
    if not y.isin(LABELS).all() or set(y) != set(LABELS):
        raise ValueError("All seven known labels must be present")
    return train_test_split(
        np.arange(len(X)), test_size=validation_size, stratify=y, random_state=seed,
    )


def duplicate_content(X_train, X_validation):
    """Đếm trùng chính xác 16 đặc trưng, bỏ id/nhãn ngay cả khi đầu vào có chúng.

    Trả số mẫu nội dung chung, số dòng mỗi tập tham gia và số cặp dòng trùng.
    Missing ở cùng vị trí được coi bằng nhau; không dùng hash xấp xỉ hay làm tròn.
    """
    def keys(frame):
        """Tạo bộ đếm tuple với missing được chuẩn hóa về None để so sánh ổn định."""
        return Counter(
            tuple(None if pd.isna(value) else value for value in row)
            for row in frame.loc[:, list(FEATURES)].itertuples(index=False, name=None)
        )

    train, validation = keys(X_train), keys(X_validation)
    common = train.keys() & validation.keys()
    return {
        "shared_feature_patterns": len(common),
        "train_rows_in_shared_patterns": sum(train[key] for key in common),
        "validation_rows_in_shared_patterns": sum(validation[key] for key in common),
        "cross_split_duplicate_pairs": sum(train[key] * validation[key] for key in common),
    }


def audit_learned_parameters(pipeline, X_train):
    """Đối chiếu imputer/scaler/encoder với tham chiếu chỉ tính từ X_train.

    Không nhận validation hoặc nhãn. Cột numeric toàn thiếu có fallback 0
    theo API Issue #5; scaler dùng variance tổng thể ddof=0.
    """
    columns = pipeline.named_steps["columns"].named_transformers_
    numeric = columns["numeric"].named_steps
    values = X_train.loc[:, list(NUMERIC_FEATURES)].to_numpy(dtype=float)
    medians = np.array([
        np.median(column[~np.isnan(column)]) if (~np.isnan(column)).any() else 0.0
        for column in values.T
    ])
    filled = np.where(np.isnan(values), medians, values)
    mean, variance = filled.mean(axis=0), filled.var(axis=0)
    # StandardScaler cũng dùng scale=1 với cột hằng/gần hằng ở độ chính xác máy.
    count = len(filled)
    eps = np.finfo(float).eps
    constant = variance <= count * eps * variance + (count * mean * eps) ** 2
    scale = np.where(constant, 1.0, np.sqrt(variance))
    encoder = columns["category"].named_steps["encoder"]
    categories_match = all(
        np.array_equal(learned, np.sort(X_train[column].astype(object).fillna("__MISSING__").unique()))
        for column, learned in zip(CATEGORICAL_FEATURES, encoder.categories_)
    ) and len(encoder.categories_) == len(CATEGORICAL_FEATURES)
    return {
        "imputer_from_train": bool(np.allclose(numeric["imputer"].statistics_, medians)),
        "scaler_mean_from_train": bool(np.allclose(numeric["scaler"].mean_, mean)),
        "scaler_variance_from_train": bool(np.allclose(numeric["scaler"].var_, variance)),
        "scaler_scale_from_train": bool(np.allclose(numeric["scaler"].scale_, scale)),
        "scaler_sample_count_from_train": bool(np.all(numeric["scaler"].n_samples_seen_ == count)),
        "encoder_categories_from_train": categories_match,
    }


def validate_split(X, y, validation_size, seed=SEED):
    """Nhận X/y đúng schema, trả kết quả kiểm tra, bảng lớp, vị trí và pipeline.

    Dùng lại transform_split của Issue #5 hai lần để kiểm tra tái lập.
    Các lỗi schema/pipeline được ném ra; kiểm tra không đạt được ghi False.
    """
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    train_pos, valid_pos = make_split(X, y, validation_size, seed)
    X_train, X_valid = X.iloc[train_pos], X.iloc[valid_pos]
    y_train, y_valid = y.iloc[train_pos], y.iloc[valid_pos]
    pipeline, train, valid = transform_split(X_train, X_valid)
    # Transform thêm lần nữa phải giữ nguyên tham số đã học, kể cả encoder.
    numeric = pipeline.named_steps["columns"].named_transformers_["numeric"].named_steps
    encoder = pipeline.named_steps["columns"].named_transformers_["category"].named_steps["encoder"]
    learned_before = [numeric["imputer"].statistics_.copy(), numeric["scaler"].mean_.copy(),
                      numeric["scaler"].var_.copy(), numeric["scaler"].scale_.copy(),
                      np.asarray(numeric["scaler"].n_samples_seen_).copy(),
                      *[category.copy() for category in encoder.categories_]]
    pipeline.transform(X_valid)
    learned_after = [numeric["imputer"].statistics_, numeric["scaler"].mean_,
                     numeric["scaler"].var_, numeric["scaler"].scale_,
                     np.asarray(numeric["scaler"].n_samples_seen_), *encoder.categories_]
    again_train_pos, again_valid_pos = make_split(X, y, validation_size, seed)
    repeated, train_again, valid_again = transform_split(
        X.iloc[again_train_pos], X.iloc[again_valid_pos],
    )
    duplicates = duplicate_content(X_train, X_valid)
    checks = {
        "index_disjoint": not X_train.index.isin(X_valid.index).any(),
        "row_count_preserved": len(train_pos) + len(valid_pos) == len(X),
        "all_classes_present": set(y_train) == set(y_valid) == set(LABELS),
        "processed_row_count_preserved": len(train) == len(X_train) and len(valid) == len(X_valid),
        "schema_matches": train.columns.equals(valid.columns) and train.columns.is_unique,
        "pipeline_feature_names_match": list(train.columns) == list(pipeline.get_feature_names_out()),
        "finite_output": bool(np.isfinite(train.to_numpy()).all() and np.isfinite(valid.to_numpy()).all()),
        "label_alignment": train.index.equals(y_train.index) and valid.index.equals(y_valid.index),
        "split_reproducible": np.array_equal(train_pos, again_train_pos) and np.array_equal(valid_pos, again_valid_pos),
        "processed_reproducible": train.equals(train_again) and valid.equals(valid_again),
        "fresh_pipeline_on_repeat": pipeline is not repeated,
        "validation_transform_preserves_parameters": all(
            np.array_equal(before, after) for before, after in zip(learned_before, learned_after)
        ),
        "input_unchanged": X.equals(original_X) and y.equals(original_y),
        "content_duplicate_free": duplicates["shared_feature_patterns"] == 0,
        **audit_learned_parameters(pipeline, X_train),
    }
    distribution = []
    for subset, labels in (("train", y_train), ("validation", y_valid)):
        for label in LABELS:
            count = int((labels == label).sum())
            overall = float((y == label).mean())
            distribution.append({
                "subset": subset, "label": label, "count": count,
                "proportion": count / len(labels), "overall_proportion": overall,
                "difference_percentage_points": 100 * (count / len(labels) - overall),
            })
    summary = {
        "train_rows": len(train_pos), "validation_rows": len(valid_pos),
        "input_features": X.shape[1], "output_features": train.shape[1],
        **duplicates, "all_checks_passed": all(checks.values()),
    }
    positions = pd.DataFrame({
        "subset": ["train"] * len(train_pos) + ["validation"] * len(valid_pos),
        "order_in_subset": list(range(len(train_pos))) + list(range(len(valid_pos))),
        "row_position": np.concatenate([train_pos, valid_pos]),
    })
    return {"summary": summary, "checks": checks, "distribution": distribution,
            "positions": positions, "pipeline": pipeline}


def load_saved_positions(source_path, positions_path, metadata_path, split):
    """Đọc lại vị trí có thứ tự sau khi xác minh checksum nguồn và file chỉ số.

    Trả (train_positions, validation_positions); người gọi dùng X.iloc và
    y.iloc. Không chấp nhận CSV khác phiên bản, chỉ số sửa đổi hoặc thiếu dòng.
    """
    metadata = pd.read_csv(metadata_path, dtype=str)
    row = metadata.loc[metadata["split"] == split]
    if len(row) != 1:
        raise ValueError("Expected exactly one split metadata record")
    row = row.iloc[0]
    if file_sha256(source_path) != row["source_sha256"]:
        raise ValueError("Source checksum does not match saved split")
    if file_sha256(positions_path) != row["positions_sha256"]:
        raise ValueError("Positions checksum does not match metadata")
    table = pd.read_csv(positions_path)
    if set(table["subset"]) != {"train", "validation"}:
        raise ValueError("Invalid subset names")
    arrays = []
    for subset in ("train", "validation"):
        part = table.loc[table["subset"] == subset].sort_values("order_in_subset")
        if not np.array_equal(part["order_in_subset"], np.arange(len(part))):
            raise ValueError("Invalid row order")
        arrays.append(part["row_position"].to_numpy(dtype=int))
    if not np.array_equal(np.sort(np.concatenate(arrays)), np.arange(int(row["source_rows"]))):
        raise ValueError("Positions must partition the entire source")
    return tuple(arrays)


def run_validation(source_path, output_dir, expected_rows=20_758):
    """Đọc nguồn, kiểm chứng ba split và ghi CSV tổng hợp/chỉ số, không ghi X/y.

    Trả từ điển các bảng. Nếu số dòng sai sẽ báo False; schema sai gây lỗi.
    Kiểm tra checksum trước/sau và chặn mọi đích ghi trùng tệp nguồn.
    """
    source_path, output_dir = Path(source_path), Path(output_dir)
    names = [f"issue_6_{name}.csv" for name in ("summary", "class_distribution", "checks", "metadata")]
    names += [f"issue_6_positions_{split}.csv" for split in SPLITS]
    for name in names:
        destination = output_dir / name
        if destination.resolve() == source_path.resolve() or (
            destination.exists() and destination.samefile(source_path)
        ):
            raise ValueError("Report would overwrite source data")
    checksum = file_sha256(source_path)
    X, y = load_features_target(source_path)
    summaries, distributions, checks, metadata, pipelines, positions = [], [], [], [], [], {}
    for split, size in SPLITS.items():
        result = validate_split(X, y, size)
        pipelines.append(result["pipeline"])
        result["checks"]["expected_source_rows"] = len(X) == expected_rows
        summaries.append({"split": split, "seed": SEED, **result["summary"]})
        distributions.extend({"split": split, **row} for row in result["distribution"])
        checks.extend({"split": split, "check": key, "passed": bool(value)}
                      for key, value in result["checks"].items())
        positions[split] = result["positions"]
        metadata.append({
            "split": split, "seed": SEED, "validation_fraction": size,
            "source_sha256": checksum, "source_rows": len(X),
            "positions_file": f"issue_6_positions_{split}.csv",
            "position_kind": "zero_based_iloc_in_source_file_order",
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        })
    independent = len({id(pipeline) for pipeline in pipelines}) == len(SPLITS)
    unchanged = file_sha256(source_path) == checksum
    if not unchanged:
        raise ValueError("Source file changed during validation; no reports written")
    for row in summaries:
        checks.extend([
            {"split": row["split"], "check": "pipelines_independent_across_splits", "passed": independent},
            {"split": row["split"], "check": "source_file_unchanged", "passed": unchanged},
        ])
        row["all_checks_passed"] = all(c["passed"] for c in checks if c["split"] == row["split"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in metadata:
        path = output_dir / row["positions_file"]
        positions[row["split"]].to_csv(path, index=False, lineterminator="\n")
        row["positions_sha256"] = file_sha256(path)
    tables = {"summary": pd.DataFrame(summaries), "class_distribution": pd.DataFrame(distributions),
              "checks": pd.DataFrame(checks), "metadata": pd.DataFrame(metadata)}
    for name, table in tables.items():
        table.to_csv(output_dir / f"issue_6_{name}.csv", index=False, lineterminator="\n")
    return tables


def main():
    """Chạy CLI từ gốc dự án; trả 0 nếu đạt, 1 nếu cần review, 2 nếu lỗi thực thi."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    args = parser.parse_args()
    try:
        tables = run_validation(args.input, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(tables["summary"].to_string(index=False))
    return int(not tables["checks"]["passed"].all())


if __name__ == "__main__":
    raise SystemExit(main())

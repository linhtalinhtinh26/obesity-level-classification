"""Kiểm thử Issue #6 bằng fixture nhỏ và lỗi được tạo có chủ đích."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data_loader import CATEGORIES, LABELS, NUMERIC_LIMITS
from src.preprocessing import build_preprocessor, split_features_target, transform_split
from src.split_validation import (
    SPLITS, audit_learned_parameters, duplicate_content, file_sha256,
    load_saved_positions, make_split, run_validation, validate_split,
)


def fixture():
    """Tạo 140 dòng đủ bảy lớp, ID khác vị trí/index và đặc trưng có missing."""
    rows = []
    for i in range(140):
        row = {name: (low + high) / 2 for name, (low, high) in NUMERIC_LIMITS.items()}
        row.update({name: values[i % len(values)] for name, values in CATEGORIES.items()})
        row.update(id=1000 + i, Age=10 + i / 10, NObeyesdad=LABELS[i % 7])
        if i % 11 == 0:
            row["Age"] = None
        if i % 13 == 0:
            row["Gender"] = None
        rows.append(row)
    return pd.DataFrame(rows, index=np.arange(500, 640))


class SplitValidationTest(unittest.TestCase):
    """Kiểm tra partition, tham số train, dữ liệu hỏng và báo cáo tái sử dụng."""

    def setUp(self):
        """Tách fixture bằng API hiện có của Issue #5."""
        self.frame = fixture()
        self.X, self.y = split_features_target(self.frame)

    def test_three_ratios_stratification_and_repeatability(self):
        """Từng lớp có đúng tỷ lệ trong fixture cân bằng; iloc khác nhãn index."""
        for size in SPLITS.values():
            with self.subTest(size=size):
                train, valid = make_split(self.X, self.y, size)
                other_train, other_valid = make_split(self.X, self.y, size)
                self.assertEqual(len(valid), round(140 * size))
                self.assertEqual(len(train) + len(valid), 140)
                self.assertFalse(set(train) & set(valid))
                self.assertEqual(set(train) | set(valid), set(range(140)))
                np.testing.assert_array_equal(train, other_train)
                np.testing.assert_array_equal(valid, other_valid)
                for part in (train, valid):
                    self.assertEqual(set(self.y.iloc[part]), set(LABELS))
                    self.assertTrue((self.y.iloc[part].value_counts() == len(part) // 7).all())

    def test_duplicate_content_ignores_id_label_and_counts_pairs(self):
        """Hai dòng train và ba validation cùng đặc trưng cho đúng sáu cặp trùng."""
        train = pd.concat([self.frame.iloc[:1]] * 2).assign(id=[1, 2])
        valid = pd.concat([self.frame.iloc[:1]] * 3).assign(id=[7, 8, 9], NObeyesdad=LABELS[2])
        result = duplicate_content(train, valid)
        self.assertEqual(result, {"shared_feature_patterns": 1, "train_rows_in_shared_patterns": 2,
                                 "validation_rows_in_shared_patterns": 3, "cross_split_duplicate_pairs": 6})
        valid["Weight"] = 999
        self.assertEqual(duplicate_content(train, valid)["shared_feature_patterns"], 0)

    def test_reject_misaligned_or_duplicate_indexes_and_wrong_schema(self):
        """Không cho phép chia với index nhập nhằng, X chứa ID hay thiếu nhãn."""
        for X, y in ((self.X, self.y.iloc[::-1]),
                     (self.X.set_axis([0] * 140), self.y.set_axis([0] * 140)),
                     (self.X.assign(id=1), self.y),
                     (self.X, self.y.replace(LABELS[0], LABELS[1]))):
            with self.assertRaises(ValueError):
                make_split(X, y, 0.2)

    def test_processed_invariants_input_integrity_and_independence(self):
        """Kiểm tra bảo toàn dòng/nhãn/schema, hữu hạn và không giả định số cột."""
        original_X, original_y = self.X.copy(deep=True), self.y.copy(deep=True)
        results = [validate_split(self.X, self.y, size) for size in SPLITS.values()]
        self.assertEqual(len({id(result["pipeline"]) for result in results}), 3)
        for result in results:
            self.assertTrue(all(result["checks"].values()), result["checks"])
            self.assertEqual(len(result["distribution"]), 14)
        pd.testing.assert_frame_equal(self.X, original_X)
        pd.testing.assert_series_equal(self.y, original_y)
        narrow = self.X.copy()
        for column in CATEGORIES:
            narrow[column] = "single"
        result = validate_split(narrow, self.y, 0.2)
        self.assertEqual(result["summary"]["output_features"], 16)

    def test_train_only_parameters_extreme_and_unseen_validation(self):
        """Validation cực trị/category lạ không làm đổi median/mean/variance/categories."""
        train = self.X.iloc[:10].copy()
        train["Age"] = [10, 20, np.nan, 30, 40, 50, 60, 70, 80, 90]
        valid = self.X.iloc[10:12].copy()
        valid["Age"] = 1e9
        valid["Gender"] = "validation_only"
        pipeline, _, output = transform_split(train, valid)
        self.assertTrue(all(audit_learned_parameters(pipeline, train).values()))
        before = audit_learned_parameters(pipeline, train)
        pipeline.transform(valid)
        self.assertEqual(before, audit_learned_parameters(pipeline, train))
        self.assertTrue(np.isfinite(output.to_numpy()).all())
        self.assertNotIn("category__Gender_validation_only", output)
        # Mô phỏng lỗi fit trên dữ liệu gộp: kiểm toán phải phát hiện.
        leaked = build_preprocessor().fit(pd.concat([train, valid]))
        checks = audit_learned_parameters(leaked, train)
        self.assertFalse(checks["scaler_mean_from_train"])
        self.assertFalse(checks["scaler_sample_count_from_train"])
        self.assertFalse(checks["encoder_categories_from_train"])

    def test_detect_imputer_and_scaler_parameter_corruption(self):
        """Thay tham số đã học có chủ đích phải bị phát hiện, gồm median và variance."""
        pipeline, _, _ = transform_split(self.X.iloc[:20], self.X.iloc[20:25])
        numeric = pipeline.named_steps["columns"].named_transformers_["numeric"].named_steps
        numeric["imputer"].statistics_[0] += 100
        numeric["scaler"].var_[0] += 100
        checks = audit_learned_parameters(pipeline, self.X.iloc[:20])
        self.assertFalse(checks["imputer_from_train"])
        self.assertFalse(checks["scaler_variance_from_train"])

    def test_all_missing_numeric_and_category_remain_finite(self):
        """Fallback train toàn thiếu không học giá trị thay thế từ validation."""
        train = self.X.iloc[:20].copy()
        train["Age"], train["Gender"] = np.nan, np.nan
        pipeline, processed, valid = transform_split(train, self.X.iloc[20:25])
        self.assertTrue(all(audit_learned_parameters(pipeline, train).values()))
        self.assertTrue(np.isfinite(valid.to_numpy()).all())
        self.assertEqual(processed.shape[1], valid.shape[1])

    def test_nonfinite_input_is_rejected(self):
        """Inf không được impute âm thầm như missing hợp lệ."""
        bad = self.X.copy()
        bad.iloc[0, bad.columns.get_loc("Weight")] = np.inf
        with self.assertRaises(ValueError):
            validate_split(bad, self.y, 0.2)

    def test_detect_broken_processed_outputs(self):
        """Giả lập schema lệch, Inf và index nhãn lệch để chứng minh kiểm tra bắt lỗi."""
        real = transform_split

        def broken(train, valid):
            """Cố ý làm hỏng đầu ra validation, giữ nguyên API của pipeline."""
            pipeline, a, b = real(train, valid)
            b = b.rename(columns={b.columns[0]: "wrong_column"})
            b.iloc[0, 0] = np.inf
            b.index = b.index[::-1]
            return pipeline, a, b

        with patch("src.split_validation.transform_split", side_effect=broken):
            checks = validate_split(self.X, self.y, 0.2)["checks"]
        for key in ("schema_matches", "finite_output", "label_alignment"):
            self.assertFalse(checks[key], key)

    def test_reports_reuse_checksum_and_source_unchanged(self):
        """CSV chỉ chứa thống kê/vị trí; checksum chặn dùng lại trên dữ liệu bị đổi."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.csv"
            output = Path(directory) / "tables"
            self.frame.to_csv(source, index=False)
            checksum = file_sha256(source)
            tables = run_validation(source, output, expected_rows=140)
            self.assertTrue(tables["checks"]["passed"].all())
            self.assertEqual(checksum, file_sha256(source))
            X, y = split_features_target(self.frame.reset_index(drop=True))
            for split, size in SPLITS.items():
                paths = (source, output / f"issue_6_positions_{split}.csv", output / "issue_6_metadata.csv", split)
                actual = load_saved_positions(*paths)
                expected = make_split(X, y, size)
                for a, b in zip(actual, expected):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(set(pd.read_csv(paths[1]).columns), {"subset", "order_in_subset", "row_position"})
            positions = output / "issue_6_positions_80_20.csv"
            positions.write_text(positions.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Positions checksum"):
                load_saved_positions(source, positions, output / "issue_6_metadata.csv", "80_20")
            source.write_text(source.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Source checksum"):
                load_saved_positions(source, positions, output / "issue_6_metadata.csv", "80_20")

    def test_wrong_source_count_is_reported(self):
        """Không tuyên bố fixture nhỏ có 20.758 dòng; kiểm tra số dòng phải FAIL."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.csv"
            self.frame.to_csv(source, index=False)
            tables = run_validation(source, Path(directory) / "out")
            checks = tables["checks"].query("check == 'expected_source_rows'")
            self.assertFalse(checks["passed"].any())
            self.assertFalse(tables["summary"]["all_checks_passed"].any())

    def test_output_cannot_overwrite_source(self):
        """Chặn đường dẫn đầu vào trùng với tên báo cáo trước khi ghi."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "issue_6_summary.csv"
            self.frame.to_csv(source, index=False)
            checksum = file_sha256(source)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                run_validation(source, Path(directory))
            self.assertEqual(checksum, file_sha256(source))


if __name__ == "__main__":
    unittest.main()

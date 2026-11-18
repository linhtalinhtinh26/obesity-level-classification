"""Fixture độc lập kiểm tra schema và tham số học chỉ từ train."""

import copy
import unittest

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from src.preprocessing import build_preprocessor, prepare_features, split_features_target, transform_split


SAMPLE = {
    "id": "0", "Gender": "Female", "Age": "20", "Height": "1.6", "Weight": "60",
    "family_history_with_overweight": "yes", "FAVC": "no", "FCVC": "2", "NCP": "3",
    "CAEC": "Sometimes", "SMOKE": "no", "CH2O": "2", "SCC": "no", "FAF": "1",
    "TUE": "1", "CALC": "no", "MTRANS": "Walking", "NObeyesdad": "Normal_Weight",
}


class PreprocessingTest(unittest.TestCase):
    """Kiểm tra đầu ra, missing values, category lạ và tính độc lập giữa các lần fit."""

    def setUp(self):
        """Ba dòng train có median Age=30, dùng chỉ số riêng để theo dõi căn chỉnh."""
        self.frame = pd.DataFrame([SAMPLE, dict(SAMPLE, id="1", Age="40"),
                                   dict(SAMPLE, id="2", Age="NA")], index=[10, 11, 12])
        self.X, self.y = split_features_target(self.frame)

    def test_schema_and_target_alignment(self):
        """X có 16 đặc trưng, giữ index và từ chối schema sai hoặc nhãn thiếu/lạ."""
        self.assertEqual(self.X.shape, (3, 16))
        self.assertNotIn("id", self.X)
        self.assertNotIn("NObeyesdad", self.X)
        self.assertEqual(list(self.y.index), [10, 11, 12])
        for bad in (self.frame.drop(columns="Height"), self.frame.assign(extra=1),
                    self.frame.assign(NObeyesdad="other"), self.frame.assign(NObeyesdad=None)):
            with self.assertRaises(ValueError):
                split_features_target(bad)
        with self.assertRaises(ValueError):
            prepare_features(self.X.assign(id=1))

    def test_missing_unknown_and_no_leakage(self):
        """Median và mean phải lấy từ train, validation cực trị và category lạ không đổi fit."""
        validation = self.X.iloc[:1].copy()
        validation.loc[10, "Age"] = 1000.0
        validation.loc[10, "Gender"] = "unseen"
        pipeline, train, valid = transform_split(self.X, validation)
        numeric = pipeline.named_steps["columns"].named_transformers_["numeric"]
        self.assertEqual(numeric.named_steps["imputer"].statistics_[0], 30)
        self.assertEqual(numeric.named_steps["scaler"].mean_[0], 30)
        self.assertAlmostEqual(train.iloc[2, 0], 0)
        self.assertAlmostEqual(valid.iloc[0, 0], 970 / np.sqrt(200 / 3))
        gender = [c for c in valid if c.startswith("category__Gender_")]
        self.assertEqual(valid[gender].to_numpy().sum(), 0)
        self.assertEqual(list(train.columns), list(valid.columns))
        self.assertEqual(list(train.index), [10, 11, 12])
        self.assertTrue(np.isfinite(valid.to_numpy()).all())
        before = numeric.named_steps["scaler"].mean_.copy()
        pipeline.transform(validation)
        np.testing.assert_array_equal(before, numeric.named_steps["scaler"].mean_)

    def test_all_missing_and_nonfinite(self):
        """Giữ cột toàn thiếu, chuẩn hóa dấu hiệu thiếu và từ chối số không hữu hạn/sai kiểu."""
        train = self.X.copy()
        train["Age"] = None
        train["Gender"] = pd.NA
        pipeline, result, valid = transform_split(train, train.iloc[:1])
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        self.assertEqual(result["numeric__Age"].tolist(), [0, 0, 0])
        self.assertIn("category__Gender___MISSING__", result.columns)
        self.assertEqual(result.shape[1], valid.shape[1])
        for value in ("inf", "-inf", "bad", np.inf):
            with self.assertRaises(ValueError):
                prepare_features(self.X.assign(Age=value))

    def test_independent_fit_repeatability_and_input_integrity(self):
        """Mỗi lời gọi tạo pipeline riêng, kết quả lặp lại và đầu vào giữ nguyên."""
        original = copy.deepcopy(self.X)
        first, a, _ = transform_split(self.X, self.X.iloc[:1])
        second, b, _ = transform_split(self.X, self.X.iloc[:1])
        self.assertIsNot(first, second)
        pd.testing.assert_frame_equal(a, b)
        other = self.X.copy()
        other["Age"] = "100"
        third, _, _ = transform_split(other, other.iloc[:1])
        self.assertEqual(third.named_steps["columns"].named_transformers_["numeric"].named_steps["scaler"].mean_[0], 100)
        self.assertEqual(first.named_steps["columns"].named_transformers_["numeric"].named_steps["scaler"].mean_[0], 30)
        pd.testing.assert_frame_equal(self.X, original)

    def test_transform_requires_fit_and_column_order_is_stable(self):
        """Từ chối transform trước fit, đầu vào đảo cột vẫn được chuẩn hóa về schema chuẩn."""
        with self.assertRaises(NotFittedError):
            build_preprocessor().transform(self.X)
        pipeline, _, expected = transform_split(self.X, self.X.iloc[:1])
        actual = pipeline.transform(self.X.iloc[:1, ::-1])
        np.testing.assert_allclose(actual, expected.to_numpy())
        with self.assertRaises(ValueError):
            pipeline.transform(self.X.drop(columns="Age"))

    def test_fit_and_transform_column_orders_and_feature_names(self):
        """Fit hai thứ tự train và transform hai thứ tự validation phải tương đương.

        Expected dùng pipeline train chuẩn làm mốc giá trị. Tên cột lấy từ
        DataFrame trả về phải khớp metadata của toàn bộ pipeline, kể cả khi
        thứ tự cột ban đầu khác thứ tự chuẩn mà bước schema tạo ra.
        """
        validation = self.X.iloc[:2].copy()
        reference, expected_train, expected_valid = transform_split(self.X, validation)
        expected_names = list(expected_train.columns)
        self.assertEqual(list(reference.get_feature_names_out()), expected_names)
        for reverse_train in (False, True):
            train = self.X.iloc[:, ::-1] if reverse_train else self.X
            for reverse_valid in (False, True):
                valid = validation.iloc[:, ::-1] if reverse_valid else validation
                with self.subTest(train_reversed=reverse_train, validation_reversed=reverse_valid):
                    pipeline, actual_train, actual_valid = transform_split(train, valid)
                    pd.testing.assert_frame_equal(actual_train, expected_train)
                    pd.testing.assert_frame_equal(actual_valid, expected_valid)
                    self.assertEqual(list(pipeline.get_feature_names_out()), expected_names)
                    self.assertEqual(list(pipeline.named_steps["schema"].get_feature_names_out()),
                                     list(self.X.columns))
                    # Kiểm tra cả API build_preprocessor trực tiếp để tránh wrapper che lỗi.
                    direct = build_preprocessor()
                    np.testing.assert_allclose(direct.fit_transform(train), expected_train.to_numpy())
                    np.testing.assert_allclose(direct.transform(valid), expected_valid.to_numpy())
                    self.assertEqual(list(direct.get_feature_names_out()), expected_names)
                    for bad in (valid.drop(columns="Age"), valid.assign(extra=1),
                                valid.rename(columns={"Age": "Height"})):
                        with self.assertRaises(ValueError):
                            direct.transform(bad)
                        with self.assertRaises(ValueError):
                            build_preprocessor().fit_transform(bad)

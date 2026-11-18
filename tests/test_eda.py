"""Kiểm thử thống kê bằng fixture độc lập có thể tính tay."""

import copy
import csv
import os
import tempfile
import unittest
from pathlib import Path

from src.eda import build_tables, category_summary, export_tables, numeric_summary, percentile


SAMPLE = {
    "id": "0", "Gender": "Female", "Age": "20", "Height": "1.6", "Weight": "60",
    "family_history_with_overweight": "yes", "FAVC": "no", "FCVC": "2", "NCP": "3",
    "CAEC": "Sometimes", "SMOKE": "no", "CH2O": "2", "SCC": "no", "FAF": "1",
    "TUE": "1", "CALC": "no", "MTRANS": "Walking", "NObeyesdad": "Normal_Weight",
}


class EDATest(unittest.TestCase):
    """Bảo vệ cách tính mẫu số, phân vị, nhóm theo nhãn và xuất bảng."""

    def test_numeric_statistics_and_invalid_values(self):
        """Bốn số hữu hạn có mean 2,5 và Q1 1,75, tách riêng missing và invalid."""
        rows = [{"Age": value} for value in ("1", "2", "3", "4", "NA", "", "inf", "bad")]
        result = numeric_summary(rows, "Age")
        self.assertEqual([result[k] for k in ("row_count", "valid_count", "missing_count", "invalid_count")],
                         [8, 4, 2, 2])
        self.assertEqual([result[k] for k in ("mean", "min", "q25", "median", "q75", "max")],
                         [2.5, 1, 1.75, 2.5, 3.25, 4])
        self.assertAlmostEqual(result["std"], (5 / 3) ** .5)
        self.assertIsNone(numeric_summary([], "Age")["mean"])
        self.assertIsNone(numeric_summary([{"Age": "1"}], "Age")["std"])
        self.assertIsNone(numeric_summary([{"Age": "null"}], "Age")["median"])
        self.assertEqual(percentile([5], .25), 5)
        with self.assertRaises(ValueError):
            percentile([1], 2)

    def test_category_denominator_and_missing(self):
        """Tỷ lệ chia cho ba ô hợp lệ, không chia cho bốn dòng gồm ô thiếu."""
        rows = [{"Gender": value} for value in ("Female", "Female", "Male", "NA")]
        result = {row["category"]: row for row in category_summary(rows, "Gender")}
        self.assertEqual(result["Female"]["count"], 2)
        self.assertEqual(result["Female"]["denominator"], 3)
        self.assertEqual(result["Female"]["missing_count"], 1)
        self.assertAlmostEqual(result["Female"]["proportion"], 2 / 3)
        self.assertEqual(result["Male"]["n_categories"], 2)
        self.assertTrue(all(row["proportion"] is None for row in category_summary([], "Gender")))

    def test_grouping_labels_and_input_unchanged(self):
        """Giữ đủ bảy hàng nhãn, phân nhóm đúng, và không làm đổi fixture đầu vào."""
        rows = [SAMPLE.copy(), dict(SAMPLE, id="1", Age="40", NObeyesdad="Obesity_Type_I")]
        original = copy.deepcopy(rows)
        tables = build_tables(list(SAMPLE), rows)
        labels = tables["label_distribution"]
        self.assertEqual(len(labels), 7)
        self.assertEqual(sum(row["count"] for row in labels), 2)
        self.assertEqual(sum(row["proportion"] for row in labels), 1)
        self.assertTrue(all(row["imbalance_ratio"] is None for row in labels))
        age = [row for row in tables["numeric_by_label"] if row["feature"] == "Age"]
        self.assertEqual(next(row["mean"] for row in age if row["label"] == "Normal_Weight"), 20)
        categories = [row for row in tables["category_by_label"]
                      if row["label"] == "Normal_Weight" and row["feature"] == "Gender"]
        self.assertEqual(sum(row["count"] for row in categories), 1)
        self.assertEqual(rows, original)
        self.assertEqual(tables, build_tables(list(SAMPLE), rows))

    def test_schema_labels_and_empty_dataset(self):
        """Thiếu cột hoặc nhãn lạ phải dừng, dữ liệu rỗng cho tỷ lệ không xác định."""
        for columns, rows in ((["Age"], []), (list(SAMPLE), [dict(SAMPLE, NObeyesdad="")]),
                              (list(SAMPLE), [dict(SAMPLE, NObeyesdad="other")])):
            with self.assertRaises(ValueError):
                build_tables(columns, rows)
        tables = build_tables(list(SAMPLE), [])
        self.assertTrue(all(row["proportion"] is None for row in tables["label_distribution"]))

    def test_imbalance_ratio(self):
        """Bảy lớp có số lượng 1,1,1,1,1,1,2 phải cho tỷ số mất cân bằng bằng 2."""
        labels = ("Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II",
                  "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III", "Obesity_Type_III")
        rows = [dict(SAMPLE, NObeyesdad=label) for label in labels]
        result = build_tables(list(SAMPLE), rows)["label_distribution"]
        self.assertTrue(all(row["imbalance_ratio"] == 2 for row in result))

    def test_export_repeatability_and_source_protection(self):
        """Hai lần xuất giống byte, bảng chứa số đã tính, hard link không làm hỏng nguồn."""
        tables = build_tables(list(SAMPLE), [SAMPLE])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.csv"
            source.write_bytes(b"source unchanged")
            output = root / "tables"
            export_tables(tables, output, source)
            first = {path.name: path.read_bytes() for path in output.glob("*.csv")}
            self.assertEqual(len(first), 5)
            with (output / "eda_numeric_summary.csv").open(newline="", encoding="utf-8") as stream:
                age = next(row for row in csv.DictReader(stream) if row["feature"] == "Age")
            self.assertEqual(age["mean"], "20.0")
            self.assertEqual(age["std"], "")
            export_tables(tables, output, source)
            self.assertEqual(first, {path.name: path.read_bytes() for path in output.glob("*.csv")})
            self.assertEqual(source.read_bytes(), b"source unchanged")
            alias_dir = root / "alias"
            alias_dir.mkdir()
            os.link(source, alias_dir / "eda_label_distribution.csv")
            with self.assertRaises(ValueError):
                export_tables(tables, alias_dir, source)
            self.assertFalse((alias_dir / "eda_numeric_summary.csv").exists())
            self.assertEqual(source.read_bytes(), b"source unchanged")


    def test_all_exported_headers_and_expected_rows(self):
        """Đối chiếu năm header và các hàng CSV với giá trị tính tay của fixture một mẫu.

        Expected được khai báo trực tiếp để phát hiện sai thứ tự cột, ánh xạ
        giá trị hoặc xuất thiếu category có số đếm bằng 0.
        """
        numeric_header = [
            "feature", "row_count", "valid_count", "missing_count", "invalid_count",
            "mean", "std", "min", "q25", "median", "q75", "max",
        ]
        category_header = [
            "feature", "category", "n_categories", "count", "denominator",
            "proportion", "missing_count",
        ]
        # Với một mẫu Age=20, std không xác định và mọi phân vị đều bằng 20.
        numeric_row = ["Age", "1", "1", "0", "0", "20.0", "", "20.0", "20.0", "20.0", "20.0", "20.0"]
        female = ["Gender", "Female", "1", "1", "1", "1.0", "0"]
        male = ["Gender", "Male", "1", "0", "1", "0.0", "0"]
        expected = {
            "numeric_summary": (numeric_header, [numeric_row]),
            "category_summary": (category_header, [female, male]),
            "label_distribution": (
                ["label", "count", "denominator", "proportion", "imbalance_ratio"],
                [
                    ["Insufficient_Weight", "0", "1", "0.0", ""],
                    ["Normal_Weight", "1", "1", "1.0", ""],
                    ["Overweight_Level_I", "0", "1", "0.0", ""],
                    ["Overweight_Level_II", "0", "1", "0.0", ""],
                    ["Obesity_Type_I", "0", "1", "0.0", ""],
                    ["Obesity_Type_II", "0", "1", "0.0", ""],
                    ["Obesity_Type_III", "0", "1", "0.0", ""],
                ],
            ),
            "numeric_by_label": (["label", *numeric_header], [["Normal_Weight", *numeric_row]]),
            "category_by_label": (
                ["label", *category_header],
                [["Normal_Weight", *female], ["Normal_Weight", *male]],
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.csv"
            source.write_bytes(b"fixture source")
            output = Path(directory) / "tables"
            export_tables(build_tables(list(SAMPLE), [SAMPLE]), output, source)
            for name, (header, expected_rows) in expected.items():
                with self.subTest(table=name):
                    with (output / f"eda_{name}.csv").open(encoding="utf-8", newline="") as stream:
                        actual = list(csv.reader(stream))
                    self.assertEqual(actual[0], header)
                    if name == "label_distribution":
                        self.assertEqual(actual[1:], expected_rows)
                    else:
                        for row in expected_rows:
                            self.assertEqual(actual[1:].count(row), 1)

    def test_grouped_missing_denominators_and_quantiles(self):
        """Hai lớp cùng bốn dòng có số ô hợp lệ khác nhau ở Age và Gender.

        Age lớp Normal là [1,3,5], lớp Obesity là [10,14].
        Gender dùng mẫu số riêng lần lượt 3 và 1, tránh chia nhầm cho
        số dòng lớp hoặc tổng số dòng toàn tập.
        """
        rows = [
            dict(SAMPLE, id=str(i), Age=age, Gender=gender, NObeyesdad=label)
            for i, (age, gender, label) in enumerate([
                ("1", "Female", "Normal_Weight"),
                ("3", "Female", "Normal_Weight"),
                ("5", "Male", "Normal_Weight"),
                ("NA", "", "Normal_Weight"),
                ("10", "Female", "Obesity_Type_I"),
                ("14", "NA", "Obesity_Type_I"),
                ("", "", "Obesity_Type_I"),
                ("null", "null", "Obesity_Type_I"),
            ])
        ]
        tables = build_tables(list(SAMPLE), rows)
        expected = {
            "Normal_Weight": (3, 1, 3, 2, 3, 4, 2),
            "Obesity_Type_I": (2, 2, 12, 11, 12, 13, 8 ** .5),
        }
        for label, (valid, missing, average, q25, median, q75, std) in expected.items():
            with self.subTest(label=label):
                age = next(row for row in tables["numeric_by_label"]
                           if row["label"] == label and row["feature"] == "Age")
                self.assertEqual(age["row_count"], 4)
                self.assertEqual(age["valid_count"], valid)
                self.assertEqual(age["missing_count"], missing)
                self.assertEqual(age["invalid_count"], 0)
                self.assertEqual([age[key] for key in ("mean", "q25", "median", "q75")],
                                 [average, q25, median, q75])
                self.assertAlmostEqual(age["std"], std)
        for label, denominator, missing, count, proportion in (
            ("Normal_Weight", 3, 1, 2, 2 / 3),
            ("Obesity_Type_I", 1, 3, 1, 1),
        ):
            group = [row for row in tables["category_by_label"]
                     if row["label"] == label and row["feature"] == "Gender"]
            female = next(row for row in group if row["category"] == "Female")
            self.assertEqual(female["denominator"], denominator)
            self.assertEqual(female["missing_count"], missing)
            self.assertEqual(female["count"], count)
            self.assertAlmostEqual(female["proportion"], proportion)
            self.assertAlmostEqual(sum(row["proportion"] for row in group), 1)
        self.assertEqual([row["count"] for row in tables["label_distribution"]], [0, 4, 0, 0, 4, 0, 0])

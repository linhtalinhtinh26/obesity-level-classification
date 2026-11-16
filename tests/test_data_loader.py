import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.data_loader import read_csv, validate_dataset, write_report


TARGET = "NObeyesdad"
LABELS = (
    "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
    "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
)
SAMPLE_FEATURES = {
    "Gender": "Female", "Age": "22.5", "Height": "1.7", "Weight": "64",
    "family_history_with_overweight": "yes", "FAVC": "no", "FCVC": "2.3",
    "NCP": "3.1", "CAEC": "Sometimes", "SMOKE": "no", "CH2O": "2.2",
    "SCC": "no", "FAF": "1.2", "TUE": "0.8", "CALC": "Frequently",
    "MTRANS": "Public_Transportation",
}
CHECKS = "dataset_validation_checks.csv"
DISTRIBUTION = "dataset_validation_labels.csv"


class DatasetValidationTest(unittest.TestCase):
    """Kiểm tra validation với dữ liệu giả lập độc lập với các hằng số của module."""
    def setUp(self):
        """Tạo fixture mới cho mỗi test, gồm bảy bản ghi có ID riêng và đủ bảy nhãn."""
        self.columns = ["id", *SAMPLE_FEATURES, TARGET]
        self.rows = [
            {"id": str(index), **SAMPLE_FEATURES, TARGET: label}
            for index, label in enumerate(LABELS)
        ]

    def write_input(self, source):
        """Ghi fixture vào đường dẫn tạm source để kiểm tra quá trình đọc và xuất CSV."""
        source.parent.mkdir(parents=True, exist_ok=True)
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.rows)

    def test_valid_csv_and_export(self):
        """CSV hợp lệ phải cho đúng nội dung hai bảng và giữ nguyên từng byte của tệp nguồn."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.csv"
            self.write_input(source)
            original = source.read_bytes()
            checks, labels = validate_dataset(*read_csv(source), expected_rows=len(self.rows))
            self.assertTrue(all(row[-1] for row in checks))
            write_report(checks, labels, source.parent / "tables", source_path=source)
            report = source.parent / "tables" / "dataset_validation_labels.csv"
            with report.open(encoding="utf-8", newline="") as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(exported, [
                {"label": label, "status": "expected", "count": "1"} for label in LABELS
            ] + [{"label": "", "status": "missing", "count": "0"}])
            with report.with_name(CHECKS).open(encoding="utf-8", newline="") as stream:
                results = {(row["check"], row["column"]): row for row in csv.DictReader(stream)}
            for key, count in {
                ("column_count", ""): "18", ("feature_count", ""): "16",
                ("row_count", ""): "7", ("label_total", TARGET): "7",
                ("missing", "Age"): "0", ("duplicate_id", "id"): "0",
                ("duplicate_full_row", ""): "0", ("duplicate_without_id", ""): "0",
                ("out_of_range", "Weight"): "0", ("unexpected_category", "Gender"): "0",
            }.items():
                self.assertEqual(results[key]["count"], count)
                self.assertEqual(results[key]["passed"], "True")
            self.assertEqual(source.read_bytes(), original)

    def test_anomalies_and_distinct_duplicate_counts(self):
        """Trộn giá trị thiếu, sai kiểu, ngoài ngưỡng và category sai vào fixture.
        Ba trường hợp trùng phải cho số đếm 2 ID, 1 dòng đầy đủ và 2 dòng khi bỏ ID.
        Tổng phân bố vẫn phải bao gồm cả bản ghi có nhãn lạ hoặc thiếu.
        """
        first = self.rows[0]
        self.rows.extend([first.copy(), dict(first, id="99"), dict(first, Age="NaN")])
        self.rows[1].update(Height="inf", Weight="-1", Gender="female", id="-1")
        self.rows[2][TARGET] = "unknown"
        self.rows[3][TARGET] = ""
        checks, labels = validate_dataset(self.columns, self.rows)
        counts = {(check, column): count for check, column, count, _, _ in checks}
        self.assertEqual(counts["duplicate_id", "id"], 2)
        self.assertEqual(counts["duplicate_full_row", ""], 1)
        self.assertEqual(counts["duplicate_without_id", ""], 2)
        for key in (("missing", "Age"), ("invalid_numeric", "Height"),
                    ("out_of_range", "Weight"), ("unexpected_category", "Gender"),
                    ("invalid_id", "id"), ("absent_label", LABELS[2])):
            self.assertEqual(counts[key], 1)
        self.assertEqual(sum(count for _, _, count in labels), len(self.rows))

    def test_wrong_schema_and_malformed_csv(self):
        """Schema sai phải được báo trong validation. CSV trùng tiêu đề hoặc
        có dòng sai số trường phải bị từ chối khi đọc bằng ValueError.
        """
        checks, _ = validate_dataset(["id", "extra"], [{"id": "0", "extra": ""}])
        failures = {(check, column) for check, column, _, _, passed in checks if not passed}
        self.assertTrue({("row_count", ""), ("missing_column", TARGET),
                         ("unexpected_column", "extra")} <= failures)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.csv"
            for malformed in ("id,id\n0,1\n", "id,Age\n0\n"):
                source.write_text(malformed, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_csv(source)


    def test_missing_id_and_target(self):
        """Hai dòng giống nhau vẫn được đếm khi thiếu cột ID và nhãn.
        Phép đếm ID trùng và tổng nhãn phải được đánh dấu là không xác định.
        """
        columns = ["Age"]
        rows = [{"Age": "21"}, {"Age": "21"}]
        checks, labels = validate_dataset(columns, rows, expected_rows=2)
        results = {(name, column): (count, passed) for name, column, count, _, passed in checks}
        self.assertEqual(results["missing_column", "id"], (1, False))
        self.assertEqual(results["missing_column", TARGET], (1, False))
        self.assertEqual(results["duplicate_full_row", ""], (1, False))
        self.assertEqual(results["duplicate_without_id", ""], (1, False))
        self.assertEqual(results["duplicate_id", "id"], (None, None))
        self.assertEqual(results["label_total", TARGET], (None, None))
        self.assertFalse(any(name == "absent_label" for name, _ in results))
        self.assertEqual(labels, [("", "target_column_unavailable", 2)])

    def test_output_preflight_rejects_same_path_and_hard_link(self):
        """Thử đích trùng trực tiếp với nguồn và một hard link cùng định danh tệp.
        Cả hai phải bị từ chối trước khi tạo bảng nào hoặc làm đổi byte nguồn.
        """
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / DISTRIBUTION
            self.write_input(source)
            original = source.read_bytes()
            alias = output / "train.csv"
            os.link(source, alias)
            for input_path in (source, alias):
                with self.subTest(input_path=input_path):
                    with self.assertRaisesRegex(ValueError, "overwrite the input"):
                        write_report([], [], output, source_path=input_path)
                    self.assertEqual(input_path.read_bytes(), original)
                    self.assertFalse((output / CHECKS).exists())

    def test_cli_defaults_and_repeatability(self):
        """Chạy hai lần với đường dẫn mặc định, dùng fixture nhỏ hơn 20.758 dòng.
        Mã thoát 1 phản ánh sai lệch số dòng. Hai bộ báo cáo phải giống nhau
        từng byte và CSV nguồn phải giữ nguyên.
        """
        script = Path(__file__).resolve().parents[1] / "src" / "data_loader.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data" / "train.csv"
            self.write_input(source)
            original = source.read_bytes()
            output = root / "outputs" / "tables"

            def run():
                result = subprocess.run(
                    [sys.executable, "-B", str(script)], cwd=root, capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(source.read_bytes(), original)
                return [(output / name).read_bytes() for name in (CHECKS, DISTRIBUTION)]

            first = run()
            self.assertIn(b"row_count,,7,Expected 20758,False", first[0])
            self.assertEqual(first, run())

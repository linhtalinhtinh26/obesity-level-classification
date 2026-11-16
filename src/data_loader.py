"""Read and audit the Kaggle S4E2 training CSV without modifying it."""

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path


TARGET = "NObeyesdad"
LABELS = (
    "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
    "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II",
    "Obesity_Type_III",
)
# Khoảng sàng lọc theo đơn vị gốc, gồm cả hai đầu mút và chấp nhận số thập phân.
NUMERIC_LIMITS = {
    "Age": (0, 120), "Height": (0.5, 3), "Weight": (2, 700),
    "FCVC": (1, 3), "NCP": (1, 4), "CH2O": (1, 3),
    "FAF": (0, 3), "TUE": (0, 2),
}
CATEGORIES = {
    "Gender": ("Female", "Male"),
    "family_history_with_overweight": ("yes", "no"),
    "FAVC": ("yes", "no"),
    "CAEC": ("no", "Sometimes", "Frequently", "Always"),
    "SMOKE": ("yes", "no"),
    "SCC": ("yes", "no"),
    "CALC": ("no", "Sometimes", "Frequently", "Always"),
    "MTRANS": ("Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"),
}
FEATURES = (*NUMERIC_LIMITS, *CATEGORIES)


def read_csv(path):
    """Đọc CSV từ đường dẫn path và trả về (columns, rows).
    columns giữ thứ tự tiêu đề. Mỗi phần tử trong rows là từ điển tên cột
    sang chuỗi gốc để validation phát hiện sai khác về chữ hoa hoặc khoảng trắng.
    Ném ValueError nếu thiếu tiêu đề, trùng tên cột hoặc sai số trường.
    Lỗi đọc tệp và lỗi cú pháp CSV lần lượt có thể phát sinh OSError và csv.Error.
    """
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        columns = next(reader, [])
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("CSV requires a nonempty header with unique column names")
        rows = []
        for values in reader:
            if len(values) != len(columns):
                raise ValueError(f"CSV line {reader.line_num} has an unexpected field count")
            rows.append(dict(zip(columns, values)))
    return columns, rows


def is_missing(value):
    """Kiểm tra chuỗi value có biểu thị missing values hay không.
    Bỏ khoảng trắng hai đầu và so khớp các dấu hiệu thiếu không phân biệt
    chữ hoa hay chữ thường.
    """
    return value.strip().casefold() in {"", "na", "n/a", "nan", "null"}


def parse_number(value):
    """Đổi chuỗi value thành float hữu hạn. Trả về None nếu không đọc được hoặc số không hữu hạn."""
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def validate_dataset(columns, rows, expected_rows=20_758):
    """Kiểm tra schema và chất lượng dữ liệu từ kết quả read_csv.
    columns là danh sách tên cột, rows là danh sách bản ghi dạng từ điển.
    expected_rows là số dòng tham chiếu, mặc định theo tập train Kaggle S4E2.
    Trả về (checks, distribution), lần lượt là các hàng của bảng kiểm tra
    và bảng đếm nhãn. Phép kiểm tra không thể thực hiện có kết quả None.
    """
    expected = {"id", *FEATURES, TARGET}
    checks = [
        ("row_count", "", len(rows), f"Expected {expected_rows}", len(rows) == expected_rows),
        ("column_count", "", len(columns), "Expected 18", len(columns) == 18),
        ("feature_count", "", len(set(columns) - {"id", TARGET}), "Expected 16",
         len(set(columns) - {"id", TARGET}) == 16),
    ]

    def record(check, column, count, criterion):
        """Ghi số trường hợp vi phạm và điều kiện tương ứng. Phép kiểm tra đạt khi count bằng 0."""
        checks.append((check, column, count, criterion, count == 0))

    # Đếm đúng số cột vẫn có thể che giấu một cột bị thay thế. Đối chiếu cả tên cột.
    for column in sorted(expected - set(columns)):
        record("missing_column", column, 1, "Required column")
    for column in sorted(set(columns) - expected):
        record("unexpected_column", column, 1, "Column outside training schema")
    for column in columns:
        values = [row[column] for row in rows]
        # Tách missing values trước để chúng không bị đếm lại là sai kiểu hoặc sai category.
        present = [value for value in values if not is_missing(value)]
        record("missing", column, len(values) - len(present), "Blank, NA, N/A, NaN or NULL")
        if column == "id":
            # Đếm duplicate trên ID hợp lệ sau khi đổi sang số nguyên.
            ids = [int(value) for value in present if re.fullmatch(r"[0-9]{1,18}", value)]
            record("invalid_id", column, len(present) - len(ids),
                   "Nonnegative decimal integer, at most 18 digits")
            record("duplicate_id", column, len(ids) - len(set(ids)),
                   "Repeated valid integer ID, excluding first occurrence")
        elif column in NUMERIC_LIMITS:
            low, high = NUMERIC_LIMITS[column]
            numbers = [parse_number(value) for value in present]
            record("invalid_numeric", column, numbers.count(None), "Finite number required")
            record("out_of_range", column,
                   sum(not low <= n <= high for n in numbers if n is not None),
                   f"Inclusive screening range [{low}, {high}]")
        elif column in CATEGORIES or column == TARGET:
            # So khớp chính xác để phát hiện sai chính tả, chữ hoa và khoảng trắng.
            allowed = LABELS if column == TARGET else CATEGORIES[column]
            record("unexpected_category", column, sum(v not in allowed for v in present),
                   "Exact match to " + " | ".join(allowed))

    # Mỗi bộ giá trị khác nhau giữ lại một lần. Phần còn lại là số lần lặp.
    # Khi bỏ ID, phép so sánh vẫn dùng nhãn nếu cột nhãn có mặt.
    for check, fields in (
        ("duplicate_full_row", columns),
        ("duplicate_without_id", [column for column in columns if column != "id"]),
    ):
        if fields:
            keys = {tuple(row[field] for field in fields) for row in rows}
            record(check, "", len(rows) - len(keys),
                   "Exact CSV field equality, excluding first occurrence")
        else:
            checks.append((check, "", None, "No columns available", None))

    if "id" not in columns:
        checks.append(("duplicate_id", "id", None, "ID column unavailable", None))
    if TARGET not in columns:
        checks.append(("label_total", TARGET, None, "Target column unavailable", None))
        return checks, [("", "target_column_unavailable", len(rows))]

    counts = Counter(row[TARGET] for row in rows)
    # Chỉ bảy nhãn hợp lệ góp vào tổng đối chiếu. Nhãn lạ và thiếu được báo riêng.
    known = sum(counts[label] for label in LABELS)
    checks.append(("label_total", TARGET, known, f"Expected {len(rows)} known labels",
                   known == len(rows)))
    distribution = [(label, "expected", counts[label]) for label in LABELS]
    for label in LABELS:
        record("absent_label", label, int(counts[label] == 0), "Expected label occurs")
    unexpected = sorted(set(counts) - set(LABELS))
    distribution.extend((value, "unexpected", counts[value])
                        for value in unexpected if not is_missing(value))
    distribution.append(("", "missing", sum(counts[v] for v in unexpected if is_missing(v))))
    return checks, distribution


def write_report(checks, distribution, output_dir, *, source_path):
    """Ghi checks và distribution thành hai bảng CSV UTF-8 có tiêu đề.
    output_dir là thư mục kết quả. source_path là đường dẫn CSV đầu vào để
    đối chiếu với cả hai tệp đích. Hàm trả về None, ném ValueError nếu đích
    trỏ tới tệp nguồn và có thể ném OSError nếu thao tác tệp thất bại.
    """
    output_dir = Path(output_dir)
    source = Path(source_path).resolve()
    reports = (
        ("dataset_validation_checks.csv", ("check", "column", "count", "criterion", "passed"), checks),
        ("dataset_validation_labels.csv", ("label", "status", "count"), distribution),
    )
    # Kiểm tra cả hai đích trước khi mở tệp ở chế độ ghi. resolve xử lý
    # đường dẫn tương đương. samefile phát hiện hard link, tức hai tên
    # khác nhau cùng trỏ đến một tệp vật lý.
    for name, _, _ in reports:
        destination = output_dir / name
        if destination.resolve() == source or (destination.exists() and destination.samefile(source)):
            raise ValueError(f"Report would overwrite the input CSV: {destination}")

    # Mỗi bảng dùng bộ tiêu đề riêng, sau đó ghi các hàng thống kê tương ứng.
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, header, rows in reports:
        with (output_dir / name).open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(header)
            writer.writerows(rows)


def main():
    """Đọc tham số dòng lệnh, chạy validation và xuất báo cáo.
    Trả về 0 khi mọi phép kiểm tra đạt, hoặc 1 khi có mục được đánh dấu.
    Lỗi đọc, kiểm tra cấu trúc CSV hoặc xuất báo cáo kết thúc với mã 2.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    args = parser.parse_args()
    try:
        checks, distribution = validate_dataset(*read_csv(args.input))
        write_report(checks, distribution, args.output_dir, source_path=args.input)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"{error}\n")
    return int(any(not passed for _, _, _, _, passed in checks))


if __name__ == "__main__":
    raise SystemExit(main())

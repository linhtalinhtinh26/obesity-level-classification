"""Thống kê EDA từ CSV gốc bằng thư viện chuẩn Python 3.11."""

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

from .data_loader import (
    CATEGORIES, FEATURES, LABELS, NUMERIC_LIMITS, TARGET,
    is_missing, parse_number, read_csv, validate_dataset,
)


NUMERIC_FIELDS = (
    "feature", "row_count", "valid_count", "missing_count", "invalid_count",
    "mean", "std", "min", "q25", "median", "q75", "max",
)
CATEGORY_FIELDS = (
    "feature", "category", "n_categories", "count", "denominator",
    "proportion", "missing_count",
)
SCHEMAS = {
    "numeric_summary": NUMERIC_FIELDS,
    "category_summary": CATEGORY_FIELDS,
    "label_distribution": ("label", "count", "denominator", "proportion", "imbalance_ratio"),
    "numeric_by_label": ("label", *NUMERIC_FIELDS),
    "category_by_label": ("label", *CATEGORY_FIELDS),
}


def percentile(values, probability):
    """Nội suy tuyến tính phân vị trên dãy đã sắp xếp, theo vị trí (n−1)×p.

    Trả về None nếu dãy rỗng. probability phải nằm trong khoảng [0, 1].
    """
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be in [0, 1]")
    if not values:
        return None
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def numeric_summary(rows, feature):
    """Tóm tắt một cột số, giữ riêng số ô thiếu và số không đọc được.

    std dùng độ lệch chuẩn mẫu (n−1), không xác định khi n < 2.
    Các số hữu hạn ngoài ngưỡng sàng lọc vẫn được thống kê, tránh âm thầm lọc dữ liệu.
    """
    present = [row[feature] for row in rows if not is_missing(row[feature])]
    values = sorted(number for value in present if (number := parse_number(value)) is not None)
    return dict(zip(NUMERIC_FIELDS, (
        feature, len(rows), len(values), len(rows) - len(present), len(present) - len(values),
        mean(values) if values else None, stdev(values) if len(values) > 1 else None,
        values[0] if values else None, percentile(values, .25), percentile(values, .5),
        percentile(values, .75), values[-1] if values else None,
    )))


def category_summary(rows, feature):
    """Đếm category quan sát, giữ nguyên chính tả, với mẫu số là số ô không thiếu.

    Hợp danh sách category dự kiến với category thực tế để có cả số đếm 0.
    None biểu thị tỷ lệ không xác định khi mẫu số bằng 0.
    """
    counts = Counter(row[feature] for row in rows if not is_missing(row[feature]))
    denominator = sum(counts.values())
    categories = sorted(set(CATEGORIES[feature]) | set(counts))
    return [dict(zip(CATEGORY_FIELDS, (
        feature, category, len(counts), counts[category], denominator,
        counts[category] / denominator if denominator else None, len(rows) - denominator,
    ))) for category in categories]


def build_tables(columns, rows):
    """Trả năm bảng dạng danh sách từ điển mà không đổi columns hoặc rows.

    Schema thiếu/thừa cột hoặc nhãn lạ/thiếu gây ValueError để tránh làm mất
    mẫu khi nhóm theo nhãn. Một lớp vắng mặt vẫn có hàng thống kê rỗng.
    """
    if len(columns) != len(set(columns)) or set(columns) != {"id", *FEATURES, TARGET}:
        raise ValueError("EDA requires the training schema")
    if any(row[TARGET] not in LABELS for row in rows):
        raise ValueError("EDA requires known, nonmissing labels")
    groups = {label: [row for row in rows if row[TARGET] == label] for label in LABELS}
    counts = [len(groups[label]) for label in LABELS]
    # Tỷ số lớn nhất/nhỏ nhất chỉ xác định khi đủ cả bảy lớp.
    imbalance = max(counts) / min(counts) if min(counts) else None
    return {
        "numeric_summary": [numeric_summary(rows, feature) for feature in NUMERIC_LIMITS],
        "category_summary": [item for feature in CATEGORIES for item in category_summary(rows, feature)],
        "label_distribution": [dict(zip(SCHEMAS["label_distribution"], (
            label, len(groups[label]), len(rows), len(groups[label]) / len(rows) if rows else None,
            imbalance,
        ))) for label in LABELS],
        "numeric_by_label": [{"label": label, **numeric_summary(group, feature)}
                             for label, group in groups.items() for feature in NUMERIC_LIMITS],
        "category_by_label": [{"label": label, **item} for label, group in groups.items()
                              for feature in CATEGORIES for item in category_summary(group, feature)],
    }


def export_tables(tables, output_dir, source_path):
    """Xuất năm CSV UTF-8 theo schema cố định, chặn đích trùng nguồn qua hard link.

    Không tạo cột chỉ số. Giá trị None được ghi thành ô trống.
    """
    output_dir = Path(output_dir)
    source = Path(source_path).resolve()
    destinations = {name: output_dir / f"eda_{name}.csv" for name in SCHEMAS}
    for destination in destinations.values():
        if destination.resolve() == source or (destination.exists() and destination.samefile(source)):
            raise ValueError("Report would overwrite source CSV")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, destination in destinations.items():
        with destination.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SCHEMAS[name])
            writer.writeheader()
            writer.writerows(tables[name])


def main():
    """Đọc đường dẫn dòng lệnh, validation, xuất bảng và kiểm tra SHA-256 nguồn.

    Các mục validation không đạt được in ra để người đọc cân nhắc khi diễn giải.
    Lỗi schema/nhãn hoặc lỗi tệp làm dừng lệnh trước khi xuất thống kê sai.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tables"))
    args = parser.parse_args()
    before = hashlib.sha256(args.input.read_bytes()).hexdigest()
    columns, rows = read_csv(args.input)
    checks, _ = validate_dataset(columns, rows)
    for check in checks:
        if not check[-1]:
            print("Validation flag:", check)
    export_tables(build_tables(columns, rows), args.output_dir, args.input)
    after = hashlib.sha256(args.input.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError("Source CSV changed during EDA")
    print(f"Rows={len(rows)}, features={len(FEATURES)}, SHA-256={after}")


if __name__ == "__main__":
    main()

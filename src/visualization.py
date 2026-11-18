
"""Trực quan hóa dữ liệu EDA cho dự án phân loại mức độ béo phì."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import LABELS


# Xác định đường dẫn từ thư mục gốc dự án.
project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "data" / "train.csv"
figures_dir = project_root / "outputs" / "figures"


def plot_label_distribution(train_df):
    """Vẽ biểu đồ số lượng mẫu thuộc 7 lớp cân nặng."""

    # Đếm số mẫu từng lớp, theo đúng thứ tự nhãn của nhóm.
    label_counts = (
        train_df["NObeyesdad"]
        .value_counts()
        .reindex(LABELS, fill_value=0)
    )

    # Tạo khung biểu đồ.
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    # Vẽ biểu đồ cột ngang.
    sns.barplot(
        x=label_counts.values,
        y=label_counts.index,
        color="steelblue",
        ax=ax,
    )


    # Khoảng cách chữ bằng 2% số lượng của lớp lớn nhất.
    label_offset = label_counts.max() * 0.02

    # Hiển thị số lượng ở cuối mỗi cột.
    for bar in ax.patches:
        count = int(bar.get_width())
        ax.text(
            count + label_offset,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
        )


    # Bổ sung tiêu đề và nhãn trục.
    ax.set_title("Phân bố 7 lớp mức độ cân nặng")
    ax.set_xlabel("Số lượng mẫu")
    ax.set_ylabel("Mức độ cân nặng")
    ax.set_xlim(0, label_counts.max() * 1.15)

    # Tạo thư mục đầu ra và lưu hình PNG.
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "01_label_distribution.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Đã lưu biểu đồ: {output_path}")


def plot_age_distribution(train_df):
    """Vẽ biểu đồ phân bố độ tuổi của các mẫu dữ liệu."""

    # Tạo khung biểu đồ.
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Chia độ tuổi thành 25 khoảng và vẽ biểu đồ tần suất.
    sns.histplot(
        data=train_df,
        x="Age",
        bins=25,
        kde=True,
        color="steelblue",
        ax=ax,
    )

    # Đặt tiêu đề và nhãn cho biểu đồ.
    ax.set_title("Phân bố độ tuổi trong bộ dữ liệu")
    ax.set_xlabel("Tuổi (năm)")
    ax.set_ylabel("Số lượng mẫu")

    # Lưu biểu đồ thành file PNG.
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "02_age_distribution.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Đã lưu biểu đồ: {output_path}")


def plot_height_distribution(train_df):
    """Vẽ biểu đồ phân bố chiều cao của các mẫu dữ liệu."""

    # Tạo khung biểu đồ.
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Vẽ histogram và đường mật độ KDE.
    sns.histplot(
        data=train_df,
        x="Height",
        bins=25,
        kde=True,
        color="steelblue",
        ax=ax,
    )

    # Đặt tiêu đề và đơn vị của các trục.
    ax.set_title("Phân bố chiều cao trong bộ dữ liệu")
    ax.set_xlabel("Chiều cao (m)")
    ax.set_ylabel("Số lượng mẫu")

    # Lưu hình PNG vào thư mục outputs/figures.
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "03_height_distribution.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Đã lưu biểu đồ: {output_path}")

def plot_weight_distribution(train_df):
    """Vẽ biểu đồ phân bố cân nặng của các mẫu dữ liệu."""

    # Tạo khung biểu đồ.
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Vẽ histogram và đường mật độ KDE.
    sns.histplot(
        data=train_df,
        x="Weight",
        bins=25,
        kde=True,
        color="steelblue",
        ax=ax,
    )

    # Đặt tiêu đề và đơn vị của các trục.
    ax.set_title("Phân bố cân nặng trong bộ dữ liệu")
    ax.set_xlabel("Cân nặng (kg)")
    ax.set_ylabel("Số lượng mẫu")

    # Lưu hình PNG vào thư mục outputs/figures.
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "04_weight_distribution.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Đã lưu biểu đồ: {output_path}")


def plot_categorical_distributions(train_df):
    """Vẽ phân bố của ba đặc trưng phân loại quan trọng."""

    # Danh sách các cột cần vẽ và tiêu đề tương ứng.
    features = [
        ("Gender", "Giới tính"),
        ("family_history_with_overweight", "Tiền sử gia đình thừa cân"),
        ("FAVC", "Thường xuyên ăn thực phẩm nhiều calo"),
    ]

    # Tạo một hàng gồm ba biểu đồ nhỏ.
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Lần lượt vẽ từng đặc trưng phân loại.
    for ax, (column, title) in zip(axes, features):
        sns.countplot(
            data=train_df,
            x=column,
            color="steelblue",
            ax=ax,
        )

        ax.set_title(title)
        ax.set_xlabel("Nhóm")
        ax.set_ylabel("Số lượng mẫu")

    # Tạo thư mục và xác định tên file hình.
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "05_categorical_distributions.png"

    # Lưu biểu đồ.
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Đã lưu biểu đồ: {output_path}")


def plot_weight_by_label(train_df):
    """So sánh phân bố cân nặng giữa bảy lớp bằng Boxplot."""

    # Bước 1: Tạo khung hình.
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    # Bước 2: Vẽ Boxplot cho từng lớp cân nặng.
    sns.boxplot(
        data=train_df,
        x="Weight",
        y="NObeyesdad",
        order=LABELS,
        color="steelblue",
        ax=ax,
    )

    # Bước 3: Đặt tiêu đề và đơn vị trục.
    ax.set_title("Phân bố cân nặng theo 7 mức độ cân nặng")
    ax.set_xlabel("Cân nặng (kg)")
    ax.set_ylabel("Mức độ cân nặng")

    # Bước 4: Lưu biểu đồ.
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "06_weight_by_label_boxplot.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Đã lưu biểu đồ: {output_path}")


def plot_height_weight_scatter(train_df):
    """Vẽ quan hệ giữa chiều cao, cân nặng và bảy nhãn."""

    # Bước 1: Tạo khung biểu đồ.
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    # Bước 2: Vẽ các điểm dữ liệu.
    sns.scatterplot(
        data=train_df,
        x="Height",
        y="Weight",
        hue="NObeyesdad",
        hue_order=LABELS,
        palette="tab10",
        alpha=0.6,
        s=18,
        ax=ax,
    )

    # Bước 3: Đặt tiêu đề và nhãn trục.
    ax.set_title("Quan hệ chiều cao và cân nặng theo 7 lớp")
    ax.set_xlabel("Chiều cao (m)")
    ax.set_ylabel("Cân nặng (kg)")

    # Bước 4: Hiển thị chú giải các lớp.
    ax.legend(
        title="Mức độ cân nặng",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    # Bước 5: Lưu hình.
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "07_height_weight_scatter.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Đã lưu biểu đồ: {output_path}")


def main():
    """Đọc dữ liệu train và gọi hàm vẽ biểu đồ."""
    train_df = pd.read_csv(data_path)

    # Biểu đồ 1: Phân bố nhãn.
    plot_label_distribution(train_df)

    # Biểu đồ 2: Phân bố độ tuổi.
    plot_age_distribution(train_df)

    # Biểu đồ 3: Phân bố chiều cao.
    plot_height_distribution(train_df)

    # Biểu đồ 4: Phân bố cân nặng.
    plot_weight_distribution(train_df)

    # Biểu đồ 5: Phân bố các đặc trưng phân loại.
    plot_categorical_distributions(train_df)

    # Biểu đồ 6: So sánh cân nặng giữa 7 lớp.
    plot_weight_by_label(train_df)

    # Biểu đồ 7: Quan hệ chiều cao và cân nặng theo nhãn.
    plot_height_weight_scatter(train_df)

if __name__ == "__main__":
    main()

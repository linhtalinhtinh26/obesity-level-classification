
"""Kiểm thử các hàm trực quan hóa dữ liệu của Issue #4."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data_loader import LABELS
from src.visualization import (
    plot_label_distribution,
    plot_age_distribution,
    plot_height_distribution,
    plot_weight_distribution,
    plot_categorical_distributions,
    plot_weight_by_label,
    plot_height_weight_scatter,
)


class VisualizationTest(unittest.TestCase):
    """Kiểm tra việc tạo biểu đồ và bảo vệ dữ liệu đầu vào."""

    def test_label_distribution_creates_png(self):
        """Biểu đồ phân bố nhãn phải tạo PNG và giữ nguyên dữ liệu."""

        # Bước 1: Tạo dữ liệu giả lập gồm đủ bảy lớp.
        train_df = pd.DataFrame({
            "NObeyesdad": list(LABELS)
        })

        # Lưu bản sao để kiểm tra dữ liệu có bị thay đổi không.
        original_df = train_df.copy(deep=True)

        # Bước 2: Tạo thư mục tạm cho kết quả kiểm thử.
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            # Bước 3: Tạm chuyển nơi lưu hình sang thư mục test.
            with patch("src.visualization.figures_dir", output_dir):
                plot_label_distribution(train_df)

            # Bước 4: Kiểm tra hình PNG được tạo.
            output_path = output_dir / "01_label_distribution.png"

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

            # Kiểm tra chữ ký nhận diện của định dạng PNG.
            with output_path.open("rb") as image_file:
                self.assertEqual(
                    image_file.read(8),
                    b"\x89PNG\r\n\x1a\n"
                )

        # Bước 5: Kiểm tra dữ liệu đầu vào không thay đổi.
        pd.testing.assert_frame_equal(train_df, original_df)


    def test_numeric_distributions_create_pngs(self):
        """Ba biểu đồ dữ liệu số phải tạo PNG và giữ nguyên dữ liệu."""

        # Tạo dữ liệu giả lập với các giá trị số khác nhau.
        train_df = pd.DataFrame({
            "Age": [18, 19, 21, 23, 25, 28, 32, 40],
            "Height": [1.55, 1.60, 1.65, 1.68, 1.72, 1.75, 1.80, 1.85],
            "Weight": [45, 52, 60, 68, 75, 82, 90, 105],
        })

        # Lưu bản sao để kiểm tra dữ liệu có bị thay đổi không.
        original_df = train_df.copy(deep=True)

        # Danh sách các hàm vẽ và tên PNG mong muốn.
        chart_cases = [
            (plot_age_distribution, "02_age_distribution.png"),
            (plot_height_distribution, "03_height_distribution.png"),
            (plot_weight_distribution, "04_weight_distribution.png"),
        ]

        # Tạo thư mục tạm để không ghi đè các biểu đồ thật.
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            with patch("src.visualization.figures_dir", output_dir):

                # Kiểm tra lần lượt từng loại biểu đồ.
                for plot_function, filename in chart_cases:
                    with self.subTest(filename=filename):
                        plot_function(train_df)

                        output_path = output_dir / filename

                        # File PNG phải tồn tại và không rỗng.
                        self.assertTrue(output_path.is_file())
                        self.assertGreater(output_path.stat().st_size, 0)

                        # Kiểm tra định dạng PNG.
                        with output_path.open("rb") as image_file:
                            self.assertEqual(
                                image_file.read(8),
                                b"\x89PNG\r\n\x1a\n"
                            )

                        # Hàm vẽ không được làm thay đổi DataFrame.
                        pd.testing.assert_frame_equal(
                            train_df,
                            original_df,
                        )


    def test_categorical_and_relationship_charts_create_pngs(self):
        """Ba biểu đồ phân loại và quan hệ phải tạo PNG đúng định dạng."""

        # Bước 1: Tạo dữ liệu giả lập.
        # Mỗi lớp cân nặng có hai mẫu để kiểm tra Boxplot.
        train_df = pd.DataFrame({
            "NObeyesdad": list(LABELS) * 2,
            "Height": [
                1.50, 1.58, 1.62, 1.68, 1.72, 1.78, 1.83,
                1.53, 1.60, 1.64, 1.70, 1.74, 1.80, 1.85,
            ],
            "Weight": [
                45, 58, 69, 75, 89, 110, 130,
                47, 60, 71, 78, 92, 115, 135,
            ],
            "Gender": ["Female", "Male"] * 7,
            "family_history_with_overweight": ["yes", "no"] * 7,
            "FAVC": ["yes", "no"] * 7,
        })

        # Giữ bản sao để phát hiện thay đổi ngoài ý muốn.
        original_df = train_df.copy(deep=True)

        # Bước 2: Khai báo các trường hợp cần kiểm tra.
        chart_cases = [
            (
                plot_categorical_distributions,
                "05_categorical_distributions.png",
            ),
            (
                plot_weight_by_label,
                "06_weight_by_label_boxplot.png",
            ),
            (
                plot_height_weight_scatter,
                "07_height_weight_scatter.png",
            ),
        ]

        # Bước 3: Tạo thư mục tạm để bảo vệ hình thật.
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            with patch("src.visualization.figures_dir", output_dir):

                # Bước 4: Chạy và kiểm tra từng hàm vẽ.
                for plot_function, filename in chart_cases:
                    with self.subTest(filename=filename):
                        plot_function(train_df)

                        output_path = output_dir / filename

                        # File phải tồn tại và có nội dung.
                        self.assertTrue(output_path.is_file())
                        self.assertGreater(output_path.stat().st_size, 0)

                        # File phải có chữ ký PNG hợp lệ.
                        with output_path.open("rb") as image_file:
                            self.assertEqual(
                                image_file.read(8),
                                b"\x89PNG\r\n\x1a\n",
                            )

                        # Không được thay đổi dữ liệu đầu vào.
                        pd.testing.assert_frame_equal(
                            train_df,
                            original_df,
                        )

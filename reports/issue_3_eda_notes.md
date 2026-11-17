# Kết quả EDA — Issue #3

## Phạm vi và cách tái tạo

Dữ liệu dùng là data/train.csv local, theo nguồn Kaggle Playground Series S4E2. Nguồn tải thủ công đã ghi trong PR #7, checksum chưa đối chiếu độc lập với bản tải Kaggle đã xác thực.
SHA-256 trước và sau hai lần chạy đều là `fec835cc642ffbc3274a026abfe465d0aaceedfe2f7b35d987029f676d7dae6d`.
Python 3.11.18, chỉ dùng thư viện chuẩn. Chạy từ gốc repository bằng `python -B -m src.eda`. Chạy toàn bộ test bằng `python -B -m unittest discover -s tests -v`.

## Quy ước thống kê và năm bảng bàn giao

- eda_numeric_summary.csv có 8 hàng tương ứng 8 đặc trưng số. row_count là số dòng của nhóm, valid_count là số hữu hạn đọc được, missing_count là số ô thiếu, invalid_count là số còn lại không đọc thành số hữu hạn. mean và std là trung bình và độ lệch chuẩn mẫu dùng n−1. std trống nếu n < 2. q25, median, q75 dùng nội suy tuyến tính tại vị trí (n−1)×p của dãy đã sắp xếp. Số ngoài ngưỡng sàng lọc vẫn được đưa vào thống kê.
- eda_category_summary.csv thống kê 8 đặc trưng category. n_categories đếm category quan sát không thiếu, count đếm từng category, denominator là số ô không thiếu của đặc trưng, proportion = count/denominator. missing_count lặp ở mỗi hàng để người đọc biết mẫu số đã loại bao nhiêu ô thiếu. Category dự kiến không xuất hiện có count bằng 0.
- eda_label_distribution.csv có đủ 7 hàng. denominator là số dòng train. imbalance_ratio là tỷ số số lượng lớp lớn nhất/nhỏ nhất, lặp ở cả bảy hàng để tiện sử dụng. Khi thiếu lớp, tỷ số để trống.
- eda_numeric_by_label.csv dùng cùng schema số và thêm label, gồm 56 hàng. Mỗi nhóm được tính độc lập.
- eda_category_by_label.csv thêm label. denominator tính riêng trong từng lớp và từng đặc trưng, nên proportion biểu thị phân bố category bên trong lớp. Category thiếu không được ghép thành category giả. Tỷ lệ trống khi mẫu số bằng 0.

Chuỗi category giữ nguyên chính tả. Giá trị không xác định được xuất thành ô trống. Các bảng UTF-8 có header, không có cột index, giữ thứ tự đặc trưng và nhãn theo loader. Có thể đọc bằng csv.DictReader hoặc pandas.read_csv khi môi trường có pandas.

## Kích thước và phân bố lớp

Validation xác nhận 20.758 dòng, 18 cột, 16 đặc trưng. Các phép kiểm tra missing, duplicate và bất thường hiện tại đều đạt. Tổng số mẫu của bảy lớp bằng 20.758.

| Nhãn | Số mẫu | Tỷ lệ |
|---|---:|---:|
| Insufficient_Weight | 2523 | 12.15% |
| Normal_Weight | 3082 | 14.85% |
| Overweight_Level_I | 2427 | 11.69% |
| Overweight_Level_II | 2522 | 12.15% |
| Obesity_Type_I | 2910 | 14.02% |
| Obesity_Type_II | 3248 | 15.65% |
| Obesity_Type_III | 4046 | 19.49% |

Lớp lớn nhất là Obesity_Type_III (4046 mẫu), nhỏ nhất là Overweight_Level_I (2427 mẫu). Tỷ số lớn nhất/nhỏ nhất bằng 1.6671. Cần giữ stratify khi chia tập và dùng macro-F1 bên cạnh accuracy để quan sát hiệu quả giữa các lớp.

## Thống kê số

| Đặc trưng | Mean | Std mẫu | Min | Q1 | Median | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Age | 23.8418 | 5.6881 | 14.0000 | 20.0000 | 22.8154 | 26.0000 | 61.0000 |
| Height | 1.7002 | 0.0873 | 1.4500 | 1.6319 | 1.7000 | 1.7629 | 1.9757 |
| Weight | 87.8878 | 26.3794 | 39.0000 | 66.0000 | 84.0649 | 111.6006 | 165.0573 |
| FCVC | 2.4459 | 0.5332 | 1.0000 | 2.0000 | 2.3938 | 3.0000 | 3.0000 |
| NCP | 2.7613 | 0.7054 | 1.0000 | 3.0000 | 3.0000 | 3.0000 | 4.0000 |
| CH2O | 2.0294 | 0.6085 | 1.0000 | 1.7920 | 2.0000 | 2.5496 | 3.0000 |
| FAF | 0.9817 | 0.8383 | 0.0000 | 0.0080 | 1.0000 | 1.5874 | 3.0000 |
| TUE | 0.6168 | 0.6021 | 0.0000 | 0.0000 | 0.5739 | 1.0000 | 2.0000 |

## Weight và Height theo nhãn

Khoảng Q1–Q3 trình bày hai đầu mút chứa 50% giá trị trung tâm. Độ rộng khoảng tứ phân vị được tính bằng **IQR = Q3 − Q1**, có cùng đơn vị với đặc trưng. Các cột Q1–Q3 bên dưới ghi hai đầu mút; có thể lấy hiệu để biết độ phân tán của nửa dữ liệu trung tâm.

| Nhãn | Weight median (kg) | Weight Q1–Q3 (kg) | Height median (m) | Height Q1–Q3 (m) |
|---|---:|---|---:|---|
| Insufficient_Weight | 50.000 | 45.000–52.290 | 1.700 | 1.600–1.752 |
| Normal_Weight | 60.000 | 55.000–68.000 | 1.660 | 1.600–1.740 |
| Overweight_Level_I | 75.000 | 68.867–79.932 | 1.700 | 1.620–1.750 |
| Overweight_Level_II | 81.627 | 79.349–85.000 | 1.700 | 1.680–1.743 |
| Obesity_Type_I | 90.116 | 82.000–104.000 | 1.695 | 1.623–1.780 |
| Obesity_Type_II | 118.333 | 112.741–120.920 | 1.774 | 1.754–1.829 |
| Obesity_Type_III | 111.914 | 109.900–132.326 | 1.643 | 1.629–1.736 |

So sánh median và IQR cho thấy Weight thay đổi rõ theo các nhóm nhãn. Height có các khoảng chồng lấp giữa lớp và cần xem cùng Weight khi diễn giải. Các bảng là mô tả quan hệ quan sát trong dữ liệu tổng hợp. Chưa thể xác định tác động nhân quả hoặc chất lượng dự đoán ngoài mẫu từ các khác biệt này.

## Category và quan hệ với nhãn

| Đặc trưng | Số category quan sát | Category phổ biến nhất | Số mẫu | Tỷ lệ |
|---|---:|---|---:|---:|
| Gender | 2 | Female | 10422 | 50.21% |
| family_history_with_overweight | 2 | yes | 17014 | 81.96% |
| FAVC | 2 | yes | 18982 | 91.44% |
| CAEC | 4 | Sometimes | 17529 | 84.44% |
| SMOKE | 2 | no | 20513 | 98.82% |
| SCC | 2 | no | 20071 | 96.69% |
| CALC | 3 | Sometimes | 15066 | 72.58% |
| MTRANS | 5 | Public_Transportation | 16687 | 80.39% |

Tỷ lệ category dưới đây được so trong từng lớp, tránh nhầm tần suất lớn do một lớp có nhiều mẫu hơn.

| Nhãn | Female | family_history_with_overweight=yes | FAVC=yes |
|---|---:|---:|---:|
| Insufficient_Weight | 64.25% | 42.57% | 85.33% |
| Normal_Weight | 53.86% | 58.08% | 83.94% |
| Overweight_Level_I | 44.09% | 68.85% | 90.77% |
| Overweight_Level_II | 29.94% | 92.35% | 78.63% |
| Obesity_Type_I | 43.54% | 98.35% | 96.80% |
| Obesity_Type_II | 0.25% | 99.85% | 98.34% |
| Obesity_Type_III | 99.88% | 99.98% | 99.98% |

## Bàn giao preprocessing và phối hợp Issue #4

Thang đo Weight, Age và Height khác nhau như bảng số cho thấy, cần fit scaler trên phần train trong bước pipeline sau. Các thuộc tính như FCVC, NCP, FAF và TUE có giá trị thập phân, cần giữ độ chính xác khi đọc. Các category hiếm có thể thiếu trong một split, cần cân nhắc encoder xử lý category chưa gặp. Không áp dụng encoder, scaler, imputer hoặc huấn luyện mô hình ở Issue #3.

Dữ liệu hiện tại không có missing values theo quy tắc loader, vẫn cần bảo vệ pipeline khi dùng đầu vào khác. Không loại dòng hoặc sửa category trong EDA. Weight/Height có liên hệ với nhãn, cần trình bày giới hạn dữ liệu tổng hợp và tránh suy diễn nhân quả.

Taho có thể dùng năm bảng này và loader chung cho các biểu đồ Issue #4. Thứ tự nhãn lấy từ LABELS. build_tables(columns, rows) trả dict gồm năm danh sách bản ghi, tên khóa tương ứng hậu tố tên CSV. Nếu dùng DataFrame, có thể đặt tên train_df, numeric_features, categorical_features, target_col, label_order theo quy ước đề xuất. Đây là phương án bàn giao chờ Taho và nhóm trưởng xác nhận.

Notebook 01_eda.ipynb được giữ nguyên. Chờ review PR #7 và thống nhất người tích hợp, sau đó đưa bảng/nhận xét và biểu đồ vào notebook cuối, chạy tuần tự từ kernel mới. Tiêu chí notebook tích hợp và thỏa thuận hai bên còn chờ thực hiện.

## Bằng chứng kiểm chứng

14 PASS / 0 FAIL, gồm 6 tests Issue #2 và 8 tests Issue #3. Hai lần chạy CLI trên dữ liệu thật tạo năm CSV giống nhau từng byte. Mỗi ô CSV đã đối chiếu với kết quả tính lại. Tổng các count và proportion của category đúng theo mẫu số toàn tập hoặc từng lớp.

| Bảng | Số hàng dữ liệu | SHA-256 |
|---|---:|---|
| eda_category_by_label.csv | 161 | `e7ac2b883d4976a8f5e9ca0a0d8ccd47c94445e8733b3bcb7a4681ee03a7c26e` |
| eda_category_summary.csv | 23 | `5a7a93936c70a38fabdf9cc517c3086e1df8d6c89e382b06b9fbea6cce123a76` |
| eda_label_distribution.csv | 7 | `0719fadcd4fea918a599cb3db7b4b0282e118bb3533f01bf908c2f6b51cc4896` |
| eda_numeric_by_label.csv | 56 | `624fa673f75e7eb2b8908351e5769d76d4070e4d7d239983a943985b27d13da3` |
| eda_numeric_summary.csv | 8 | `6aadbe197cfc4e3889edca3ae2fdac7c489ae5e56f1902023e2b426337380d87` |

# Pipeline tiền xử lý — Issue #5

## Quyết định dựa trên EDA

EDA ghi nhận 20.758 dòng và không có missing values theo quy tắc Issue #2. API vẫn xử lý missing để dùng được với đầu vào sau này. Các quy tắc dưới đây được chọn trước khi xem kết quả validation của mô hình.

- Tám cột số lấy từ NUMERIC_LIMITS của loader, tám cột category lấy từ CATEGORIES. ID và NObeyesdad được tách khỏi X. Schema sai hoặc nhãn thiếu/lạ gây ValueError.
- Numeric dùng SimpleImputer với median học từ train, sau đó StandardScaler học mean và variance từ train. Median hạn chế ảnh hưởng của giá trị cực trị khi cần điền thiếu. Thang đo Age, Weight và Height khác nhau trong EDA nên cần scaling cho các mô hình nhạy với thang đo trong bước sau.
- Nếu một cột số train hoàn toàn thiếu, keep_empty_features=True giữ cột và điền 0 theo hành vi SimpleImputer. Đây là dự phòng kỹ thuật cần được reviewer cân nhắc khi gặp dữ liệu đó. Không suy ra median từ validation.
- Category thiếu dùng SimpleImputer với hằng __MISSING__, sau đó OneHotEncoder(handle_unknown="ignore"). Hằng này là quy ước biểu diễn missing, không ước lượng từ validation. Category lạ ở validation có block toàn 0. Nếu train không có missing nhưng validation có, hằng missing cũng có thể chưa được encoder học và được xử lý như category lạ.
- Các dấu hiệu thiếu từ loader, None và missing scalar của pandas được chuẩn hóa thành np.nan. Chuỗi category giữ nguyên chính tả để không tự sửa dữ liệu không nhất quán. Giá trị số không đọc được hoặc không hữu hạn gây ValueError. Số hữu hạn ngoài ngưỡng sàng lọc được giữ nguyên, cần đối chiếu báo cáo validation trước khi diễn giải.
- Không xóa duplicate hoặc outlier. Không fit scaler/encoder trên dữ liệu gộp. API transform_split không nhận y nên không dùng nhãn validation để quyết định phương pháp hoặc tham số.

## API bàn giao Issue #6

Chạy từ thư mục gốc bằng Python 3.11 của .venv. Các API nằm trong src.preprocessing.

```python
from src.preprocessing import load_features_target, transform_split

X, y = load_features_target()  # Mặc định data/train.csv
# train_index và validation_index do Issue #6 tạo, là các vị trí dòng.
pipeline, X_train_processed, X_validation_processed = transform_split(
    X.iloc[train_index], X.iloc[validation_index]
)
y_train = y.iloc[train_index]
y_validation = y.iloc[validation_index]
```

split_features_target(frame) nhận DataFrame đủ 18 cột và trả (X, y), giữ index và thứ tự dòng. prepare_features(X) là bước kiểm tra/chuyển kiểu không học tham số, trả bản sao 16 đặc trưng. build_preprocessor() tạo Pipeline mới chưa fit để người gọi có thể dùng trong pipeline thí nghiệm. transform_split(X_train, X_validation) tạo pipeline riêng, fit_transform train và chỉ transform validation, trả pipeline cùng hai DataFrame đã xử lý.

Đầu ra là dense float64, numeric trước category. Tên cột lấy từ ColumnTransformer với tiền tố numeric__ hoặc category__. Có thể đọc tên bằng pipeline.named_steps["columns"].get_feature_names_out(). Hai tập của cùng một split có cùng tên và thứ tự cột. Số cột giữa các split có thể khác khi train có category khác nhau. Phải dùng lại đúng pipeline của từng split cho các mô hình sau.

Issue #6 chịu trách nhiệm chia stratified 80/20, 70/30, 60/40 với seed 42, lưu chỉ số dùng chung cho mô hình, kiểm tra chỉ số và nhãn căn chỉnh, lập bảng phân bố lớp và sai lệch do làm tròn. Hướng dẫn này chờ Taho review, không thay thế deliverable Issue #6.

## Kiểm chứng local

Python 3.11.18, numpy 2.4.6, pandas 3.1.1, scikit-learn 1.9.1, scipy 1.17.1. Lệnh regression tests là `.venv/Scripts/python.exe -B -m unittest discover -s tests -v`, đạt 20 PASS / 0 FAIL, gồm 14 test Issue #2–#3 và 6 test preprocessing.

Kiểm chứng dùng train_test_split trên các vị trí dòng trong bộ nhớ với stratify=y và random_state=42. Mỗi tỷ lệ gọi transform_split hai lần để kiểm tra tính lặp lại và dùng pipeline riêng. Các kiểm tra đối chiếu median imputer và mean scaler với đúng phần train đều đạt.

| Train/validation | Dòng train | Dòng validation | Số cột sau xử lý | Dòng validation có nội dung đặc trưng trùng train |
|---|---:|---:|---:|---:|
| 80/20 | 16.606 | 4.152 | 30 | 0 |
| 70/30 | 14.530 | 6.228 | 30 | 0 |
| 60/40 | 12.454 | 8.304 | 30 | 0 |

Phép đếm trùng so khớp chính xác tuple 16 đặc trưng đã chuyển kiểu trước imputation, encoding và scaling, bỏ ID và nhãn. Đây là phép kiểm tra nội dung riêng với việc đối chiếu chỉ số dòng. Tệp local hiện không có missing nên kết quả so khớp không chịu ảnh hưởng của cách so sánh NaN. Nếu phát hiện trùng ở dữ liệu khác, cần báo cáo và xin quyết định xử lý trước khi đánh giá mô hình.

Cả ba tỷ lệ có cùng schema trong từng cặp train/validation, không còn missing hoặc giá trị không hữu hạn, giữ index đầu vào và kết quả lặp lại giống nhau. Pipeline của ba tỷ lệ có định danh đối tượng khác nhau. Test riêng dùng giá trị validation cực trị và category lạ để kiểm tra tham số train không thay đổi sau transform, kể cả cột train toàn thiếu.

SHA-256 nguồn trước/sau kiểm chứng đều là `fec835cc642ffbc3274a026abfe465d0aaceedfe2f7b35d987029f676d7dae6d`. Mã chỉ đọc CSV bằng loader Issue #2. Giới hạn nguồn tải thủ công và checksum chưa đối chiếu độc lập với bản tải Kaggle đã xác thực vẫn kế thừa từ báo cáo EDA.

## Phần chờ review

Cần nhóm trưởng duyệt các lựa chọn median, category missing hằng, fallback 0 khi cột số train toàn thiếu và cách xử lý category lạ. Chờ Taho xác nhận API và hoàn tất kiểm chứng/báo cáo Issue #6. Chưa đánh giá mô hình hoặc triển khai giảm chiều.

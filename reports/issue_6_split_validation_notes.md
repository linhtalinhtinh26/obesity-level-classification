# Issue #6 — Kiểm tra chia dữ liệu và tiền xử lý

## Mục tiêu và phạm vi

Issue #6 kiểm tra xem dữ liệu dùng để học và dữ liệu dùng để đánh giá có được
tách đúng, giữ nguyên nhãn và xử lý nhất quán hay không. Không huấn luyện mô hình
và không xây dựng lại pipeline. Mã gọi API của `src/preprocessing.py` thuộc Issue #5.

Nhánh kiểm tra: `feat/issue-6-split-validation`; working tree sạch trước khi làm.
Đã đọc Issue #6, hai tài liệu đã duyệt, loader, preprocessing, test preprocessing
và ghi chú Issue #5. Không sửa các file này. Không thực hiện Git ghi, sửa Issue,
commit, push, chuyển nhánh hay thao tác PR.

## File và luồng xử lý

- `src/split_validation.py`: hàm kiểm tra tái sử dụng, xuất bảng và CLI.
- `tests/test_split_validation.py`: 12 unit tests bằng fixture nhỏ, độc lập Kaggle.
- `outputs/tables/issue_6_*.csv`: 4 bảng tổng hợp và 3 bảng vị trí chia.
- Tài liệu này: kết quả, cách đọc, cách chạy lại, giới hạn và giải thích học tập.

Luồng: `train.csv` → `load_features_target()` → X gồm 16 đặc trưng và y gồm nhãn
→ chia vị trí dòng ba tỷ lệ → lấy train/validation bằng `.iloc` → gọi
`transform_split(X_train, X_validation)` → kiểm tra kết quả và tham số đã học
→ lặp lại để đối chiếu → xuất thống kê/chỉ số. Không xuất các giá trị đặc trưng
hoặc từng nhãn gốc ra báo cáo.

`split_features_target()` là API đã dùng bên trong loader Issue #5 và trong tests.
`build_preprocessor()` được gọi bên trong `transform_split()`; không cần tạo thêm
pipeline khác trong luồng kiểm chứng. Test dùng trực tiếp builder để mô phỏng lỗi
fit trên dữ liệu gộp và chứng minh kiểm tra phát hiện được lỗi đó.

## Môi trường và kết quả thực tế

Chạy local bằng `.venv`: Python 3.11.9, numpy 2.4.6, pandas 3.1.1,
scikit-learn 1.9.1. Phiên bản Python patch khác ghi chú Issue #5 (3.11.18), vẫn
đúng yêu cầu Python 3.11; metadata ghi phiên bản thực tế, không sửa ghi chú đồng đội.

Dữ liệu: 20.758 dòng; X đúng 16 cột, không có `id` hoặc `NObeyesdad`.
SHA-256 trước/sau đều:
`fec835cc642ffbc3274a026abfe465d0aaceedfe2f7b35d987029f676d7dae6d`.

| Train/validation | Train | Validation | Chiều trước/sau | Giao index | Mẫu nội dung chung | Lệch tỷ lệ lớp lớn nhất (điểm %) |
|---|---:|---:|---|---:|---:|---:|
| 80/20 | 16.606 | 4.152 | 16 → 30 | 0 | 0 | 0,013020 |
| 70/30 | 14.530 | 6.228 | 16 → 30 | 0 | 0 | 0,007930 |
| 60/40 | 12.454 | 8.304 | 16 → 30 | 0 | 0 | 0,005348 |

Cả bảy lớp có mặt ở cả hai tập của từng split. Sai lệch rất nhỏ do số dòng phải
là số nguyên: không phải lúc nào cũng có thể lấy chính xác 20%, 30% hay 40% của
mỗi lớp. Bảng phân bố ghi đủ số lượng và tỷ lệ, không chỉ kết luận bằng mắt.

69/69 kiểm tra trên dữ liệu thực đạt. Ba pipeline là ba đối tượng độc lập.
Mỗi lần lặp có cùng chỉ số chia và DataFrame đã xử lý (so sánh bằng `equals`).
Tên/thứ tự cột train và validation khớp trong từng split; số dòng và index nhãn
giữ nguyên; không NaN/Inf. Mã không hard-code 30 cột: fixture có một category
cho mỗi cột cho 16 cột sau xử lý và vẫn kiểm tra được.

## Fit, transform, stratify và leakage có nghĩa gì?

- `train_test_split`: chia các vị trí `0..20757` thành hai mảng, không xóa dòng
  nào khỏi CSV. Tỷ lệ 80/20 cho 16.606 vị trí train và 4.152 validation.
- `stratify=y`: dùng nhãn để giữ gần đúng tỷ lệ bảy lớp ở cả hai tập. Dùng y để
  chia stratified không đồng nghĩa dùng nhãn validation để fit tiền xử lý.
- `random_state=42`: cố định phép chia ngẫu nhiên; tái lập còn phụ thuộc đúng
  nội dung/thứ tự dữ liệu và phiên bản thư viện, nên phải lưu chỉ số/checksum.
- `fit`: học thông số từ train. Ví dụ Age train `[10, 20, thiếu]` có median 15;
  sau điền thiếu, scaler học mean 15 và variance từ `[10, 20, 15]`.
- `transform`: dùng thông số đó để xử lý dữ liệu mới. Nếu Age validation là
  1.000 thì không tính lại median/mean bằng giá trị 1.000 này.
- Data leakage: thông tin validation lọt vào bước học/chọn phương pháp, làm
  đánh giá thiếu độc lập. Ví dụ fit scaler trên train cộng validation là sai.

`transform_split()` không nhận y. Mã kiểm tra chỉ dùng y cho stratification,
bảng lớp và căn chỉnh; không lựa chọn cấu hình dựa trên nhãn validation.
Không thay đổi phương pháp tiền xử lý sau khi xem các thống kê validation.

## Kiểm tra tham số được học

`audit_learned_parameters()` tính tham chiếu độc lập từ X_train:

- Numeric imputer: median từng cột; cột toàn thiếu dùng fallback 0 của Issue #5.
- Scaler: mean, variance tổng thể (`ddof=0`), scale và số mẫu đã thấy sau impute.
- Encoder: tập category đúng của train, có thay missing bằng `__MISSING__`.

Các tham số số được so với `np.allclose` (sai số float mặc định); category so
chính xác. Một lần transform validation bổ sung phải giữ nguyên các tham số.
Test cố ý fit trên dữ liệu gộp, thay median/variance, thêm category chỉ có ở
validation để xác nhận kiểm tra báo lỗi. Đây là bằng chứng kiểm thử và rà soát
luồng API hiện tại, không phải chứng minh tổng quát mọi loại leakage trong dự án.

## Phân biệt hai loại trùng

1. Giao index: cùng dòng nguồn xuất hiện ở cả train và validation. Phải bằng 0.
2. Trùng nội dung: hai dòng nguồn khác nhau có đúng cùng 16 đặc trưng. So sánh
   trước imputation/scaling/encoding, không dùng id hoặc nhãn. Các ô missing ở
   cùng vị trí được xem là bằng nhau. Không tìm near-duplicate hoặc làm tròn số.

Quy ước đếm trong summary:

- `shared_feature_patterns`: số tuple 16 đặc trưng khác nhau có ở cả hai tập.
- `train_rows_in_shared_patterns`, `validation_rows_in_shared_patterns`: số dòng
  mỗi tập tham gia các tuple chung (mỗi dòng tính một lần).
- `cross_split_duplicate_pairs`: tổng `số dòng train × số dòng validation` của
  từng tuple chung. Hai dòng train và ba validation cùng nội dung tạo 6 cặp,
  nhưng chỉ 1 mẫu nội dung chung.

Test có ID và nhãn khác nhau vẫn phát hiện trùng đặc trưng. Nếu dữ liệu tương
lai có trùng, mã ghi kiểm tra không đạt và CLI trả 1 để nhóm review; không xóa
dòng. Nhóm cần quyết định cách tách nhóm trùng hoặc giao thức đánh giá trước
thực nghiệm mô hình. Giá trị số ngoài ngưỡng hữu hạn vẫn được giữ theo Issue #5;
cần đọc báo cáo chất lượng dữ liệu/EDA, không hiểu PASS ở đây là mọi giá trị đều
hợp lý về mặt sức khỏe.

## Cách đọc các bảng

| File (trong outputs/tables/) | Nội dung và cách đọc |
|---|---|
| `issue_6_summary.csv` | Một dòng mỗi tỷ lệ; kích thước, chiều, số trùng; `all_checks_passed=True` nghĩa là tất cả kiểm tra tương ứng đều đạt |
| `issue_6_class_distribution.csv` | 42 dòng = 3 split × 2 tập × 7 lớp; `proportion` trong [0,1], `difference_percentage_points` là độ lệch với toàn bộ nguồn nhân 100 |
| `issue_6_checks.csv` | 69 dòng kiểm tra; `passed=False` cần xem tên `check`, không bỏ qua vì summary trông hợp lý |
| `issue_6_metadata.csv` | Seed, tỷ lệ, checksum nguồn/file chỉ số, số dòng và phiên bản môi trường |
| `issue_6_positions_80_20.csv` | Vị trí dòng train/validation 80/20, có thứ tự; không phải ID Kaggle |
| `issue_6_positions_70_30.csv` | Tương tự cho 70/30 |
| `issue_6_positions_60_40.csv` | Tương tự cho 60/40 |

## API quan trọng và cách tái sử dụng

- `make_split(X, y, validation_size, seed=42)` → hai mảng vị trí train/validation.
  Ví dụ fixture 140 dòng, `validation_size=0.2` → 112/28 dòng. Index X/y phải
  duy nhất và căn chỉnh; đủ bảy nhãn đã biết; X đúng 16 đặc trưng.
- `duplicate_content(X_train, X_validation)` → từ điển bốn số đếm trên.
- `validate_split(X, y, 0.2)` → dict có `summary`, `checks`, `distribution`,
  `positions`, `pipeline`. Không ghi file; pipeline là đối tượng đã fit train.
- `run_validation(source_path, output_dir)` → dict bốn DataFrame tổng hợp và
  ghi bảy CSV. Mặc định đối chiếu 20.758 dòng; test fixture truyền số dòng riêng.
- `load_saved_positions(source_path, positions_path, metadata_path, split)` →
  hai mảng vị trí có thứ tự, sau khi xác minh checksum và partition nguồn.

Ví dụ dùng lại đúng tập mẫu cho Issue sau (chạy từ thư mục gốc):

```python
from src.preprocessing import load_features_target, transform_split
from src.split_validation import load_saved_positions

train_pos, valid_pos = load_saved_positions(
    "data/train.csv",
    "outputs/tables/issue_6_positions_80_20.csv",
    "outputs/tables/issue_6_metadata.csv",
    "80_20",
)
X, y = load_features_target("data/train.csv")
X_train, X_valid = X.iloc[train_pos], X.iloc[valid_pos]
y_train, y_valid = y.iloc[train_pos], y.iloc[valid_pos]
pipeline, train_processed, valid_processed = transform_split(X_train, X_valid)
```

Không sắp xếp/đổi dòng CSV giữa bước xác minh và nạp dữ liệu. Nếu checksum không
khớp, phải tạo bộ chỉ số/metadata mới sau review, không bỏ qua kiểm tra. Dùng
`order_in_subset` để khôi phục đúng thứ tự khi đọc bảng vị trí, không tự sort theo
`row_position`. Đã thử đọc lại cả ba file chỉ số thực và đối chiếu đúng với phép
chia tái tạo trong môi trường hiện tại.

## Lệnh đã chạy và kiểm tra

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -m src.split_validation
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
git -c safe.directory=C:/Workspace/ML/ML_Prj/Obesity-level-classification diff --check
git -c safe.directory=C:/Workspace/ML/ML_Prj/Obesity-level-classification status --short
```

Tham số safe.directory chỉ phục vụ lệnh Git đọc trong sandbox Windows, không
được ghi vào cấu hình Git. Còn chạy kiểm tra SHA-256, kiểm tra bảng CSV và đoạn
Python đọc lại chỉ số cho cả ba split. Lần chạy tests cuối: **32 PASS, 0 FAIL,
0 ERROR**, gồm 20 tests cũ và 12 tests mới; không còn cảnh báo phát sinh từ mã mới.
CLI dữ liệu thực trả exit code 0, **69/69 kiểm tra PASS**.

## Giới hạn và việc cần review

- Checksum chứng minh file local không đổi trong lượt kiểm chứng, không xác thực
  độc lập file này với bản tải Kaggle chính thức; giới hạn nguồn kế thừa Issue #5.
- Chưa kiểm chứng trên máy khác, phiên bản thư viện khác hoặc dữ liệu mới.
- Các lựa chọn median, missing category hằng, fallback 0 và category lạ vẫn là
  lựa chọn của Issue #5 cần nhóm review; Issue #6 kiểm tra đúng cách áp dụng.
- Chưa có mô hình, metric phân loại hay PCA/SVD; không kết luận hiệu quả dự đoán.
- Dữ liệu invalid gây ValueError trước khi xuất báo cáo; không tự sửa để chạy tiếp.
- Các file output đã có được tái tạo tại đúng tên Issue #6 khi chạy CLI lại;
  không ghi đè nguồn, kể cả đường dẫn nguồn trùng hoặc hard link trùng đích.
- Không sửa file đồng đội và chưa thực hiện bất kỳ thao tác Git ghi nào.

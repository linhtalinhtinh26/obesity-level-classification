# Obesity Level Classification

Phân loại 7 mức độ cân nặng dựa trên đặc điểm thể chất và thói quen sinh hoạt bằng Machine Learning. Dự án học thuật; kết quả không dùng để chẩn đoán.

## Tài liệu và công việc

- [Kiến trúc đã duyệt](SYSTEM_ARCHITECTURE.md)
- [Kế hoạch 16–29/11/2026](PLAN_14_DAYS.md)
- [GitHub Issues](https://github.com/linhtalinhtinh26/obesity-level-classification/issues)
- Thành viên: `linhtalinhtinh26`, `doonipon`.

## Bắt đầu trên Windows / PowerShell

Cài Python **3.11** trước khi tạo môi trường. Kiểm tra phiên bản; không dùng Python khác thay thế mà chưa thống nhất.

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m jupyter notebook
```

Không cần kích hoạt môi trường hoặc thay đổi execution policy. Trong VS Code chọn Python interpreter `.venv/Scripts/python.exe`; notebook chọn kernel của môi trường này.

Các dependency hiện khai báo theo tên thư viện được duyệt. Chưa kiểm thử môi trường Python 3.11; sẽ cố định phiên bản sau khi kiểm thử theo kế hoạch.

## Dữ liệu

Làm theo [data/README.md](data/README.md). Không commit CSV gốc hoặc thông tin đăng nhập Kaggle. Notebook và mã ML chưa được triển khai; thư mục hiện là bộ khung.

## Tổ chức

```text
data/                 Hướng dẫn và CSV cục bộ
notebooks/            Notebook trình bày thực nghiệm
src/                  Hàm Python tái sử dụng
outputs/figures/      Hình kết quả
outputs/tables/       Bảng kết quả
reports/              Báo cáo
slides/               Slide
```

Tên file triển khai tuân theo SYSTEM_ARCHITECTURE.md. `.gitkeep` chỉ giữ các thư mục trống trên Git.

## Quy tắc thực nghiệm

- Dùng `train.csv` có nhãn; bỏ `id` khỏi đặc trưng, tách `NObeyesdad` khỏi X.
- Chia stratified 80/20, 70/30, 60/40 với seed 42; dùng cùng chỉ số split cho các mô hình so sánh.
- Chỉ fit imputer, encoder, scaler và các bước học tham số trên train; validation chỉ transform.
- Giai đoạn đầu: cấu trúc/môi trường, kiểm tra dữ liệu, EDA, pipeline và kiểm tra split theo Issues #1–#6.

## Làm việc bằng Git

Tạo nhánh theo công việc, ví dụ `git switch -c task/2-dataset-validation`. Kiểm tra `git status`, chọn đúng file bằng `git add <file>`, commit rồi push nhánh và tạo Pull Request để kiểm tra chéo. Không thêm dữ liệu gốc, token hoặc file môi trường.

Danh tính commit và tài khoản xác thực GitHub là hai cấu hình khác nhau. Kiểm tra cả hai khi làm việc với nhiều tài khoản; remote dự án này phải là `https://github.com/linhtalinhtinh26/obesity-level-classification.git`.

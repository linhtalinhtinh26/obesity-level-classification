# Chuẩn bị dữ liệu

Nguồn: [Kaggle Playground Series S4E2](https://www.kaggle.com/competitions/playground-series-s4e2/data).

1. Đăng nhập Kaggle, mở trang Data; hoàn tất điều kiện truy cập của Kaggle nếu được yêu cầu.
2. Tải bộ dữ liệu và giải nén trên máy.
3. Đặt `train.csv` vào `data/train.csv`. Có thể đặt `test.csv` vào `data/test.csv` để kiểm tra schema; file này không có nhãn, không dùng đánh giá độ chính xác.
4. Không đưa CSV gốc, file zip hoặc thông tin xác thực lên GitHub.

Quy mô dự kiến cần xác minh bằng file thực: 20.758 dòng train; 16 đặc trưng đầu vào, `id`, nhãn `NObeyesdad` với 7 lớp. Không xem các con số này là kết quả kiểm tra đã thực hiện.

Kiểm tra thiếu, trùng, kiểu dữ liệu và bất thường theo Issue #2 trước EDA. Không sửa dữ liệu gốc. Chạy notebook từ môi trường Python 3.11 của dự án; dùng đường dẫn phù hợp với thư mục làm việc của notebook.

# Kế hoạch triển khai 14 ngày — Dự án Machine Learning

**Tên dự án:** `obesity-level-classification`  
**Đề tài:** Phân loại mức độ béo phì dựa trên đặc điểm thể chất và thói quen sinh hoạt  
**Thời gian:** 16/11/2026 – 29/11/2026 (14 ngày liên tiếp)  
**Thành viên:** linhtalinhtinh26; doonipon  
**Tài liệu liên quan:** `SYSTEM_ARCHITECTURE.md`  
**Dữ liệu:** https://www.kaggle.com/competitions/playground-series-s4e2/data

## 1. Mục tiêu và đầu ra

Hoàn thành bộ thực nghiệm phân loại 7 mức độ cân nặng, gồm khảo sát dữ liệu, tiền xử lý, hai phương pháp giảm chiều, một phương pháp phân cụm, ba mô hình phân loại và hai mô hình hồi quy score. Bàn giao mã nguồn, notebook, bảng/hình kết quả, hướng dẫn chạy, báo cáo 15–20 trang và slide trình bày.

**Các kịch bản tối thiểu:**

- Phân loại: **Naive Bayes, Logistic Regression, SVM** × **80/20, 70/30, 60/40** × **full, PCA, TruncatedSVD** = **27 cấu hình**.
- Giảm chiều: PCA và TruncatedSVD; trực quan 4–6 thành phần, biểu diễn các cặp và đối chiếu phương sai giải thích.
- Phân cụm: K-Means trên đặc trưng không có nhãn; thử nhiều `k`, silhouette, ARI/NMI và biểu đồ.
- Hồi quy score: **Linear Regression và KNN Regressor** × ba tỷ lệ chia × ít nhất hai biểu diễn (full và giảm còn xấp xỉ 1/3 chiều).
- Đánh giá: accuracy, macro-F1, weighted-F1, confusion matrix; train/validation gap; MAE, RMSE, R² cho hồi quy.

## 2. Lịch trình chi tiết

| Ngày | Ngày thực hiện | Công việc chính | Phân công | Sản phẩm/nghiệm thu |
|---|---|---|---|---|
| **1** | **16/11** | Khởi tạo cấu trúc repository, môi trường Python, `requirements.txt`, `.gitignore`; tải dữ liệu từ Kaggle, xác minh tên cột/số dòng và 7 nhãn | **linhtalinhtinh26:** cấu trúc, môi trường, tài liệu tải dữ liệu. **doonipon:** kiểm tra schema và tính hợp lệ | Repository có cấu trúc chuẩn; dữ liệu đọc được tại máy; `data/README.md` |
| **2** | **17/11** | EDA: mô tả đặc trưng, thống kê thiếu/trùng, phân bố nhãn, biểu đồ tuổi, chiều cao, cân nặng, thuộc tính category | **doonipon:** EDA và thống kê. **linhtalinhtinh26:** biểu đồ và kiểm tra chéo | `01_eda.ipynb`; bảng thống kê, ≥6 hình có chú thích |
| **3** | **18/11** | Tiền xử lý: tách X/y, loại `id`, chia stratified 80/20–70/30–60/40, xử lý khuyết, one-hot, scale bằng pipeline | **doonipon:** `preprocessing.py`. **linhtalinhtinh26:** kiểm tra 3 split và kiểm thử schema | Pipeline chạy cho ba tỷ lệ, không fit trên validation; mô tả trước/sau tiền xử lý |
| **4** | **19/11** | PCA và TruncatedSVD, đồ thị 4–6 thành phần, explained variance, so sánh với nhãn, xác định số chiều ~1/3 | **linhtalinhtinh26:** hai phương pháp và đồ thị. **doonipon:** kiểm tra thiết kế pipeline và giải thích kết quả | `02_preprocessing_reduction.ipynb`; bảng số chiều/thông tin giữ lại |
| **5** | **20/11** | Baseline Naive Bayes và Logistic Regression với split 80/20, full features; thống nhất metrics và lưu dự đoán | **linhtalinhtinh26:** Naive Bayes. **doonipon:** Logistic Regression | Hai mô hình chạy; accuracy, F1, confusion matrix, cấu hình lưu lại |
| **6** | **21/11** | Huấn luyện SVM (ưu tiên tuyến tính), so sánh baseline của ba mô hình trên cùng split | **doonipon:** SVM. **linhtalinhtinh26:** tổng hợp bảng baseline, đối chiếu nhầm lẫn | Đủ ba classifier và bảng so sánh đầu tiên |
| **7** | **22/11** | Chạy ma trận 27 cấu hình cho full/PCA/SVD với ba tỷ lệ chia; ghi seed, thời gian chạy và metrics | **doonipon:** Logistic Regression, SVM. **linhtalinhtinh26:** Naive Bayes, kết xuất bảng chung | Bảng thí nghiệm đầy đủ và biểu đồ so sánh |
| **8** | **23/11** | Phân tích overfitting; thử regularization/tuning trong train; đối chiếu trước/sau hiệu chỉnh | **doonipon:** tinh chỉnh Logistic/SVM. **linhtalinhtinh26:** phân tích Naive Bayes và giảm chiều, đồ thị | Bảng train–validation, kết luận có số liệu và ghi rõ cách tuning |
| **9** | **24/11** | K-Means: lựa chọn `k`, đánh giá silhouette, ARI/NMI, hiển thị cụm và đối chiếu lớp | **linhtalinhtinh26:** K-Means, hình. **doonipon:** thống kê lớp trong cụm, giải thích | `04_clustering.ipynb`; chỉ số định lượng, biểu đồ |
| **10** | **25/11** | Thiết kế hồi quy score: chọn lớp, fit teacher trên train, tạo score cho train/validation, chốt giao thức tránh leakage | **doonipon:** teacher và hàm xuất score. **linhtalinhtinh26:** kiểm tra phân phối score và giao thức | Notebook/script tạo target score, mô tả rõ ý nghĩa và giới hạn |
| **11** | **26/11** | Linear Regression, KNN Regressor; full vs giảm khoảng 1/3 chiều, ba tỷ lệ chia; so sánh sai số | **doonipon:** Linear Regression. **linhtalinhtinh26:** KNN Regressor | Bảng MAE/RMSE/R², hình predicted-vs-score, nhận xét |
| **12** | **27/11** | Tổng hợp thí nghiệm, viết báo cáo theo bố cục đề bài, chọn biểu đồ và bảng chính | **doonipon:** dữ liệu/tiền xử lý/phân loại. **linhtalinhtinh26:** giảm chiều/phân cụm/hồi quy; **cả hai:** kết luận | Bản nháp báo cáo đủ các phần, liên kết số liệu đúng với kết quả thực chạy |
| **13** | **28/11** | Hoàn thiện báo cáo, slide, README và hướng dẫn tái lập; kiểm thử từ môi trường sạch | **doonipon:** biên tập báo cáo. **linhtalinhtinh26:** slide và README. **Cả hai:** kiểm thử | Báo cáo 15–20 trang, slide, hướng dẫn tải/chạy, mã nguồn chạy lại được |
| **14** | **29/11** | Rà soát toàn bộ yêu cầu, chạy lại kịch bản quan trọng, kiểm tra trích dẫn/biểu đồ; diễn tập trình bày | **Cả hai:** kiểm tra chéo, hoàn thiện và đóng gói | Bộ nộp cuối gồm report, slide, code, liên kết dữ liệu và hướng dẫn |

## 3. Mốc nghiệm thu

| Mốc | Hạn | Tiêu chí |
|---|---|---|
| **M1 — Dữ liệu & tiền xử lý** | **18/11 (ngày 3)** | Schema, EDA, pipeline và ba phép chia dữ liệu |
| **M2 — Giảm chiều & phân loại baseline** | **21/11 (ngày 6)** | PCA, SVD, ba mô hình phân loại trên 80/20 |
| **M3 — So sánh phân loại** | **23/11 (ngày 8)** | 27 cấu hình, confusion matrix, overfit, tuning |
| **M4 — Toàn bộ thực nghiệm** | **26/11 (ngày 11)** | K-Means, hai hồi quy score, bảng/biểu đồ đánh giá |
| **M5 — Hồ sơ hoàn thiện** | **29/11 (ngày 14)** | Báo cáo, slide, code và hướng dẫn tái lập |

## 4. Nguyên tắc triển khai

1. Duy trì **một pipeline tiền xử lý chung** để kết quả giữa các mô hình có thể so sánh.
2. Trong mỗi tỷ lệ chia, fit encoder/scaler/PCA/SVD trên **train**, transform validation. Không dùng nhãn validation cho các bước fit hoặc tuning.
3. Lưu thí nghiệm với cấu hình mô hình, seed (`42`), tỷ lệ chia, số chiều và chỉ số đánh giá; dùng bảng kết quả chung thay vì chép số liệu thủ công.
4. Các kết quả chính phải đi kèm mã nguồn, bảng/biểu đồ và nhận xét dựa trên số liệu.
5. Với SVM, kiểm tra thời gian chạy trên quy mô nhỏ trước, rồi xác định cấu hình chạy trên tập đầy đủ; mọi giới hạn phải được ghi rõ.
6. Score hồi quy là **đầu ra liên tục của classifier nguồn**, không phải đại lượng sức khỏe thực đo. Mỗi thí nghiệm phải ghi rõ cách sinh score và tập dùng để fit teacher/regressor.
7. Chỉ công bố dữ liệu thông qua **liên kết Kaggle** và hướng dẫn chuẩn bị; không đưa CSV vào repository.
8. Các phần viết báo cáo được cập nhật song song với thực nghiệm để đảm bảo thông số trong tài liệu trùng khớp số liệu cuối.

## 5. Danh sách đầu ra theo thư mục

| Thư mục/tệp | Nội dung |
|---|---|
| `SYSTEM_ARCHITECTURE.md` | Mục tiêu, thiết kế hệ thống, quy tắc thí nghiệm |
| `PLAN_14_DAYS.md` | Lịch triển khai và mốc nghiệm thu |
| `README.md`, `data/README.md` | Cài đặt, nguồn dữ liệu, tổ chức thư mục và cách chạy |
| `requirements.txt`, `.gitignore` | Môi trường và quy tắc lưu trữ |
| `src/` | Hàm đọc dữ liệu, preprocessing, giảm chiều, mô hình và đánh giá |
| `notebooks/01_eda.ipynb` | Khảo sát và biểu đồ dữ liệu |
| `notebooks/02_preprocessing_reduction.ipynb` | Mã hóa, chuẩn hóa, PCA, SVD |
| `notebooks/03_classification.ipynb` | Naive Bayes, Logistic, SVM và so sánh |
| `notebooks/04_clustering.ipynb` | K-Means và đánh giá cụm |
| `notebooks/05_score_regression.ipynb` | Teacher score và hai mô hình hồi quy |
| `outputs/tables/`, `outputs/figures/` | Bảng metrics, confusion matrices, các đồ thị |
| `reports/`, `slides/` | Báo cáo và slide hoàn thiện |

## 6. Kiểm tra yêu cầu trước bàn giao

- [ ] Dữ liệu không trùng đề tài/bài thực hành theo đăng ký; xác minh nguồn Kaggle 2024, 16 thuộc tính, ≥300 mẫu và 7 nhãn.
- [ ] Có thống kê dữ liệu, trực quan, xử lý các trường hợp thiếu/trùng/bất thường, one-hot và chuẩn hóa khi phù hợp.
- [ ] Có **PCA và TruncatedSVD**, 4–6 thành phần để trực quan, phương sai giải thích và so sánh.
- [ ] Có **K-Means**, chỉ số đánh giá định lượng và so sánh cụm–nhãn.
- [ ] Có **Naive Bayes, Logistic Regression, SVM**, đủ ba tỷ lệ chia và dữ liệu đầy đủ/giảm chiều.
- [ ] Có đánh giá train/validation, confusion matrix, overfit và thử regularization cho mô hình phù hợp.
- [ ] Có **hai mô hình hồi quy score**, so sánh full và giảm về xấp xỉ 1/3 chiều, ba tỷ lệ chia.
- [ ] Có báo cáo **15–20 trang, A4, Times New Roman 13, căn đều, giãn dòng 1.3–1.5**, lý thuyết, dữ liệu, thực nghiệm, kết luận, tài liệu tham khảo.
- [ ] Có slide, chương trình nguồn, `README.md`, hướng dẫn lấy/tổ chức dữ liệu và chạy lại kịch bản; không gửi kèm dữ liệu gốc.
- [ ] Kết quả giữa bảng, hình, notebook và báo cáo nhất quán; mỗi thành viên có phần mô hình thực nghiệm cụ thể.

## 7. Quản lý thay đổi

Theo dõi từng công việc bằng GitHub Issue với mục tiêu, checklist và tiêu chí hoàn thành. Mã nguồn, kết quả và tài liệu được kiểm tra trước khi tích hợp; các thay đổi ảnh hưởng đến phạm vi đề tài hoặc giao thức đánh giá cần được cập nhật đồng thời trong hai tài liệu.
# Hướng dẫn chạy – NV3 (Khám phá dữ liệu) & NV4 (Tiền xử lý)

Đề tài: Sử dụng thuật toán Naive Bayes để phân loại tin nhắn lừa đảo trực tuyến
Dataset: D1 – SMS Phishing Dataset (Mishra & Soni, Mendeley Data, CC BY 4.0)

## 1. Cấu trúc thư mục
```
.
├── 01_kham_pha_du_lieu.ipynb     # NV3: khám phá, thống kê, biểu đồ, nhận xét
├── 02_tien_xu_ly.ipynb           # NV4: làm sạch, bảng trước/sau, chia train/test
├── split_train_test.py           # code chia train/test chạy độc lập (tái lập)
├── requirements.txt
├── data/
│   ├── raw/Dataset_5971.csv      # <-- BẠN ĐẶT FILE GỐC VÀO ĐÂY (không sửa)
│   └── processed/                # tự sinh: dataset sạch, train, test, split_info.json
└── outputs/
    ├── tables/                   # tự sinh: các bảng thống kê (CSV, mở được bằng Excel)
    └── figures/                  # tự sinh: biểu đồ PNG
```

## 2. Các bước chạy
1. Cài thư viện: `pip install -r requirements.txt`
2. Chép `Dataset_5971.csv` (giải nén từ `Dataset_5971.zip` đã tải ở Phần 2) vào `data/raw/`.
3. Mở `01_kham_pha_du_lieu.ipynb` → **Run All**.
4. Mở `02_tien_xu_ly.ipynb` → **Run All**.
5. (Tùy chọn) `python split_train_test.py` để chia lại từ dataset sạch bằng script.

Mẹo: nên kiểm tra SHA256 của file gốc (ghi ở Phần 2) trước khi chạy. Cuối notebook 1 có bảng "đối chiếu với Phần 2": nếu cột Kết quả toàn `KHỚP` nghĩa là bạn đang dùng đúng file gốc.

## 3. Đối chiếu sản phẩm bàn giao
| Sản phẩm yêu cầu | Nằm ở đâu |
|---|---|
| Notebook khám phá dữ liệu | `01_kham_pha_du_lieu.ipynb` |
| Bảng thống kê | `outputs/tables/01…07_*.csv` |
| Biểu đồ phân bố nhãn | `outputs/figures/01_phan_bo_nhan_goc.png` (+ 02, 03 bổ sung) |
| Notebook tiền xử lý | `02_tien_xu_ly.ipynb` |
| Dataset sạch | `data/processed/sms_phishing_clean.csv` |
| Code chia train/test | Bước 8 notebook 02 + `split_train_test.py` |
| Bảng số mẫu trước/sau xử lý | `outputs/tables/09_*.csv` (theo từng bước), `10_*.csv` (gọn: gốc vs sạch) |

## 4. Các quyết định cần cả nhóm thống nhất (đổi ở ô cấu hình đầu notebook 02)
| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `CONFLICT_POLICY` | `"drop"` | Cách xử lý nội dung bị gán cả spam và smishing |
| `TEST_SIZE` | `0.20` | Tỷ lệ tập test |
| `RANDOM_STATE` | `42` | Hạt giống – ghi vào báo cáo |

## 5. Xử lý sự cố thường gặp
* `AssertionError: Không thấy file ...` → chưa đặt file vào `data/raw/` hoặc sai tên.
* Bảng đối chiếu có dòng `LỆCH` → file không phải bản gốc, hoặc bị Excel mở rồi lưu lại. Tải lại từ Mendeley.
* Mở CSV bằng Excel bị lỗi chữ → dùng *Data → From Text/CSV* và chọn UTF-8.

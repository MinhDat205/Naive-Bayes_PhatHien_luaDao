================================================================================
 BỘ DỮ LIỆU – NHÓM 7 – ĐỀ TÀI NCKH
 "Sử dụng thuật toán Naive Bayes để phân loại tin nhắn lừa đảo trực tuyến"
 Trường Đại học Giao thông Vận tải TP. Hồ Chí Minh
================================================================================

Người lập: Phạm Đức Tuấn            Ngày lập: 14/11/2026
Người tải và khảo sát dữ liệu: Phan Tấn Tài
Người nhận bàn giao: Lê Trung Trực (khảo sát, tiền xử lý, chia train/test)

--------------------------------------------------------------------------------
 1. TÓM TẮT
--------------------------------------------------------------------------------

  Thư mục   Dataset                         Kênh    Vai trò
  -------   -----------------------------   -----   ---------------------------
  D1        SMS Phishing Dataset            SMS     DATASET CHÍNH (huấn luyện,
            (Mishra & Soni, 2022)                   kiểm thử)
  D2        SpamAssassin Public Corpus      Email   Chỉ dùng để so sánh trong
                                                    báo cáo. KHÔNG huấn luyện.
  D3        SMS Spam Collection (UCI)       SMS     Chỉ dùng làm đối chứng
                                                    trong báo cáo. KHÔNG
                                                    huấn luyện.

  Quyết định chọn D1: xem "Báo cáo Giai đoạn 3 – Chốt dataset" và
  "Biên bản họp nhóm – Quyết định chọn dataset".

  QUY TẮC CHUNG:
  - Không sửa, không ghi đè bất kỳ file nào trong các thư mục này.
  - Mọi bước làm sạch / tiền xử lý làm trên BẢN SAO, lưu ở thư mục khác.
  - Không trộn D1 với D2 hoặc D3.

--------------------------------------------------------------------------------
 2. CẤU TRÚC THƯ MỤC
--------------------------------------------------------------------------------

  Dataset/
  ├── D1 – SMS Phishing Dataset (Mishra & Soni)/
  │   ├── Dataset_5971.zip        <- file dữ liệu gốc (chứa Dataset_5971.csv)
  │   ├── data_conversion.pdf     <- tài liệu kèm theo của tác giả
  │   ├── data_visual.pdf         <- tài liệu kèm theo của tác giả
  │   ├── Frequency.pdf           <- tài liệu kèm theo của tác giả
  │   ├── Email_extract.py        <- script trích cột EMAIL của tác giả
  │   ├── Http_extract.py         <- script trích cột URL của tác giả
  │   └── Phone_extract.py        <- script trích cột PHONE của tác giả
  ├── D2 – SpamAssassin Public Corpus/
  │   ├── easy_ham/    (2.500 email + file cmds)
  │   ├── easy_ham_2/  (1.400 email + file cmds)
  │   ├── hard_ham/    (  250 email + file cmds)
  │   ├── spam/        (  500 email + file cmds)
  │   └── spam_2/      (1.397 email + file cmds)
  └── D3 – SMS Spam Collection (UCI)/
      ├── SMSSpamCollection       <- file dữ liệu
      └── readme                  <- mô tả của tác giả

--------------------------------------------------------------------------------
 3. D1 – SMS PHISHING DATASET (DATASET CHÍNH)
--------------------------------------------------------------------------------

  Tên đầy đủ : SMS Phishing Dataset for Machine Learning and Pattern
               Recognition, Version 1
  Tác giả    : Sandhya Mishra, Devpriya Soni – Jaypee Institute of
               Information Technology (Ấn Độ)
  Công bố    : 20/06/2022, Mendeley Data
  Link       : https://data.mendeley.com/datasets/f45bkkt8pr/1
  DOI        : 10.17632/f45bkkt8pr.1
  Giấy phép  : CC BY 4.0 (được sử dụng, phải ghi tên tác giả)
  Ngày tải   : 13/11/2026
  Ngôn ngữ   : Tiếng Anh

  File và mã SHA256 (đã kiểm tra lại ngày 14/11/2026 – KHỚP):
    Dataset_5971.zip   244.698 byte
      9bbf3188fdad81495d8e82825648b9b63b53fc86841a3d26c02629990b233cc3
    Dataset_5971.csv   595.987 byte (nằm trong file zip)
      649844f1c62a6b27e145eaf17a65f7010c56c390e11a794ea8329993a05ba71e

  Cấu trúc file CSV: UTF-8, 5.971 dòng dữ liệu + 1 dòng tiêu đề, 5 cột
    LABEL : nhãn – ham / spam / smishing
    TEXT  : nội dung tin nhắn
    URL   : tin có chứa đường link không (yes / No)
    EMAIL : tin có chứa địa chỉ email không (yes / No)
    PHONE : tin có chứa số điện thoại không (yes / No)

  Số mẫu theo nhãn (đếm trên file gốc, chưa chuẩn hóa):
    ham       4.844
    Smishing    616  ┐ smishing: 638
    smishing     22  ┘
    spam        466  ┐ spam: 489
    Spam         23  ┘
    Tổng      5.971

  VẤN ĐỀ CẦN XỬ LÝ (theo báo cáo Phần 2 của Tài, đếm trên dữ liệu thô):
    - Nhãn viết hoa/thường lẫn lộn -> chuẩn hóa về chữ thường.
    - 254 dòng TEXT có khoảng trắng/tab thừa ở đầu hoặc cuối.
    - 106 dòng trùng nội dung và cùng nhãn -> giữ 1 bản.
    - 34 nội dung (68 dòng) bị gán cả spam lẫn smishing -> xử lý theo
      Quyết định 3 trong Biên bản họp nhóm.
    - 145 dòng có ký tự lỗi mã hóa (�) -> giữ nguyên, ghi vào hạn chế.
    - Phải loại trùng TRƯỚC khi chia train/test.
    Các số trên có thể thay đổi sau khi bỏ khoảng trắng thừa; ghi lại số
    thực tế sau khi chạy code.

  Đọc bằng Pandas:
    import pandas as pd
    df = pd.read_csv("Dataset_5971.csv", encoding="utf-8")

  Trích dẫn:
    Mishra, S., & Soni, D. (2022). SMS Phishing Dataset for Machine Learning
    and Pattern Recognition (Version 1) [Data set]. Mendeley Data.
    https://doi.org/10.17632/f45bkkt8pr.1

--------------------------------------------------------------------------------
 4. D2 – SPAMASSASSIN PUBLIC CORPUS (SO SÁNH, KHÔNG HUẤN LUYỆN)
--------------------------------------------------------------------------------

  Đơn vị     : Apache SpamAssassin Project
  Link       : https://spamassassin.apache.org/old/publiccorpus/
  Giấy phép  : Trang chính thức không nêu giấy phép. Bản quyền nội dung thư
               thuộc người gửi gốc; không gửi các thư này vào hệ thống email
               đang hoạt động.
  Ngày tải   : 13/11/2026
  Ngôn ngữ   : Chủ yếu tiếng Anh
  Định dạng  : Mỗi email là 1 file riêng, giữ đủ header. Nhãn lấy theo tên
               thư mục: spam, spam_2 = spam; easy_ham, easy_ham_2,
               hard_ham = ham. File "cmds" là file phụ, không phải email.

  Số email trong thư mục (đếm ngày 14/11/2026):
    easy_ham    2.500   ham
    easy_ham_2  1.400   ham
    hard_ham      250   ham
    spam          500   spam
    spam_2      1.397   spam
    Tổng        6.047   (ham 4.150, spam 1.897). Không có email trùng nội dung.

  LƯU Ý:
    - Báo cáo Phần 2 ghi spam_2 = 1.396 (tổng 6.046). Số đếm trong thư mục
      này là 1.397. Cần Tài đếm lại và thống nhất số liệu trong báo cáo.
    - Thư mục này chứa email ĐÃ GIẢI NÉN, không có 5 file .tar.bz2 gốc nên
      không đối chiếu được mã SHA256 dưới đây. Nếu cần chứng minh nguồn,
      tải lại các file .tar.bz2 từ link trên.

  Mã SHA256 các file gốc (do Tài ghi khi tải, ngày 13/11/2026):
    20030228_spam.tar.bz2         1.183.768 byte
      c08debc32413804949a866be45ef78195cec2cbafd1da744ed76cdb860589743
    20030228_easy_ham.tar.bz2     1.612.216 byte
      2b7b65904bcfcc31d2b5f51946f2d261370b257402cbbd62930b46ab83367438
    20030228_hard_ham.tar.bz2     1.029.898 byte
      ce2ce67880643dbde65ea7f85bffbfe4417349c4bd80b6b0de56262ae6b0a9c9
    20030228_easy_ham_2.tar.bz2   1.077.892 byte
      b4bd3dc5ae5b40f38e99a0e41ad7d16b428b56e818e3a6ffa07330d62004dd38
    20050311_spam_2.tar.bz2       2.059.029 byte
      44280a0e28bf7645b2279e8e42659271ad82c6f40bf8d364dbda7f1df344a765

--------------------------------------------------------------------------------
 5. D3 – SMS SPAM COLLECTION (ĐỐI CHỨNG, KHÔNG HUẤN LUYỆN)
--------------------------------------------------------------------------------

  Tác giả    : Tiago A. Almeida, José María Gómez Hidalgo
  Công bố    : 2012, UCI Machine Learning Repository
  Link       : https://archive.ics.uci.edu/dataset/228/
  DOI        : 10.24432/C5CC84
  Giấy phép  : CC BY 4.0
  Ngày tải   : ……/……/2026   (người tải điền)
  Ngôn ngữ   : Tiếng Anh

  File:
    SMSSpamCollection   477.907 byte
      SHA256: ................................................................
      (người tải điền: Get-FileHash .\SMSSpamCollection -Algorithm SHA256)
    readme              5.868 byte

  Định dạng: văn bản thuần, mỗi dòng = nhãn + ký tự TAB + nội dung tin nhắn.
  Nhãn: ham / spam.
  Số mẫu theo trang UCI: 5.574 tin – ham 4.827 (86,6%), spam 747 (13,4%).
  (Chưa đếm lại trên file trong thư mục này.)

  Đọc bằng Pandas:
    import pandas as pd
    df = pd.read_csv("SMSSpamCollection", sep="\t", header=None,
                     names=["label", "text"], encoding="utf-8")

  Trích dẫn:
    Almeida, T. A., & Gómez Hidalgo, J. M. (2012). SMS Spam Collection
    [Data set]. UCI Machine Learning Repository.
    https://doi.org/10.24432/C5CC84

--------------------------------------------------------------------------------
 6. CÁCH KIỂM TRA FILE CÓ ĐÚNG BẢN GỐC (SHA256)
--------------------------------------------------------------------------------

  SHA256 là "dấu vân tay" của file: chỉ cần nội dung thay đổi 1 ký tự thì
  mã sẽ khác hoàn toàn. Nếu mã tính ra trùng với mã ghi trong file này,
  file chính là bản gốc chưa bị chỉnh sửa.

  Trên Windows (PowerShell), mở thư mục chứa file rồi chạy:
    Get-FileHash .\Dataset_5971.zip -Algorithm SHA256

  So sánh cột "Hash" với mã ở mục 3 (không phân biệt chữ hoa/thường).

================================================================================

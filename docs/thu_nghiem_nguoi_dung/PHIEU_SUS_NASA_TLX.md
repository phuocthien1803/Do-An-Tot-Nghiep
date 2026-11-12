# Phiếu đánh giá sau mỗi lượt đi

Mã người tham gia: ______  Điều kiện: ☐ G  ☐ G + bản gốc  ☐ G + bản cải tiến  Bố trí: ☐ A ☐ B ☐ C

## A. SUS — Thang đo khả năng sử dụng hệ thống (1 = rất không đồng ý … 5 = rất đồng ý)

Người hỏi đọc to từng câu; người tham gia trả lời bằng số.

| # | Câu hỏi | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 1 | Tôi nghĩ tôi muốn dùng hệ thống này thường xuyên. | | | | | |
| 2 | Tôi thấy hệ thống phức tạp không cần thiết. | | | | | |
| 3 | Tôi thấy hệ thống dễ dùng. | | | | | |
| 4 | Tôi nghĩ tôi cần người kỹ thuật hỗ trợ mới dùng được. | | | | | |
| 5 | Các chức năng của hệ thống kết hợp với nhau tốt. | | | | | |
| 6 | Tôi thấy hệ thống có quá nhiều điểm không nhất quán. | | | | | |
| 7 | Tôi nghĩ hầu hết mọi người sẽ học dùng hệ thống rất nhanh. | | | | | |
| 8 | Tôi thấy hệ thống rất rườm rà, khó dùng. | | | | | |
| 9 | Tôi cảm thấy tự tin khi dùng hệ thống. | | | | | |
| 10 | Tôi phải học nhiều thứ trước khi dùng được hệ thống. | | | | | |

Cách tính: SUS = 2,5 × [(tổng câu lẻ − 5) + (25 − tổng câu chẵn)], thang 0–100.
Điểm khoảng 68 được coi là trung bình.

## B. NASA-TLX (raw) — mỗi thang 0 … 100, bước 5

| Thang | 0 nghĩa là | 100 nghĩa là | Điểm |
|---|---|---|---|
| Tải tinh thần | rất thấp | rất cao | |
| Tải thể chất | rất thấp | rất cao | |
| Áp lực thời gian | rất thấp | rất cao | |
| Hiệu quả tự đánh giá | rất tốt | rất kém | |
| Mức nỗ lực | rất thấp | rất cao | |
| Mức bực bội, căng thẳng | rất thấp | rất cao | |

Raw TLX = trung bình 6 thang (càng thấp càng ít tải).

## C. Người quan sát ghi

Số va chạm: ____  Số lần suýt va (phải can thiệp): ____  Thời gian hoàn thành (s): ____

Ghi chú (âm báo khó hiểu, cảnh báo thừa, lỗi hệ thống): ______________________________

---

Chuyển dữ liệu sang `bang_ket_qua_mau.csv` (một dòng = một người × một điều kiện) rồi chạy
`scripts/tinh_sus_nasa.py`.

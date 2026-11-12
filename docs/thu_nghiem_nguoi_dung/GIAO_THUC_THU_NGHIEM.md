# Giao thức thử nghiệm người dùng (P3)

Mục tiêu: trả lời câu hỏi mà số liệu trên bộ dữ liệu công khai không trả lời được — **hệ thống có
giúp người dùng đi an toàn hơn không, và bản cải tiến có tốt hơn bản gốc không**. Tài liệu này chỉ là
kế hoạch; nhóm thực hiện và ghi kết quả thật vào `bang_ket_qua_mau.csv`.

## 1. Người tham gia và đạo đức

- **8–12 người sáng mắt bịt mắt** (dễ tuyển, đo được xu hướng) cùng **2–3 người khiếm thị** đã quen
  dùng gậy (đánh giá tính phù hợp thực tế), nếu liên hệ được qua hội người mù địa phương.
- Có phiếu đồng ý tham gia (mục đích, rủi ro, quyền dừng bất kỳ lúc nào, không quay mặt nếu không đồng ý).
- **Luôn có người giám sát an toàn** đi cạnh, đủ gần để chặn va chạm. Vật cản dùng loại mềm hoặc nhẹ
  (thùng carton, ghế nhựa, chậu cây giả). "Bậc xuống" mô phỏng bằng bục thấp 10–15 cm có đệm, không
  dùng cầu thang thật.
- Hỏi ý kiến giảng viên hướng dẫn về thủ tục đạo đức của trường trước khi làm.

## 2. Đường thử

Hành lang hoặc phòng học dài 15–25 m, gồm 6–8 "sự kiện" đặt theo 3 bố trí khác nhau (A, B, C), mỗi bố trí
có cùng số loại sự kiện:

| Loại | Ví dụ | Số lượng mỗi bố trí |
|---|---|---|
| Vật cản đứng yên giữa lối | ghế, thùng | 3 |
| Vật cản thấp | va li, chậu cây | 1 |
| Bậc xuống mô phỏng | bục thấp có đệm | 1 |
| Người đi ngược chiều | trợ lý đi tới | 1 |
| Vật bên lề (không chắn lối) | ghế sát tường | 1–2 (đo cảnh báo thừa) |

## 3. Điều kiện (thiết kế trong người tham gia, thứ tự cân bằng Latin)

1. **G**: chỉ dùng gậy.
2. **G + bản gốc**: gậy + hệ thống chạy với `configs/thu_nghiem/nhu_ban_goc.yaml`.
3. **G + bản cải tiến**: gậy + hệ thống với cấu hình mặc định.

Mỗi người đi 3 lượt, mỗi lượt một điều kiện với một bố trí. Ghép điều kiện–bố trí và thứ tự theo
hình vuông Latin để tránh hiệu ứng học đường. Trước khi đo, cho mỗi người làm quen 5 phút với hệ thống
(nghe từng loại âm báo, thử phím S/H).

Camera đeo ngang ngực đúng độ cao `camera.height_m`; đo và ghi lại độ cao thật cho từng người. Bật
nhật ký: `--event-log ket_qua/<ma_nguoi>_<dieu_kien>.jsonl`. Nếu người tham gia đồng ý, quay video
góc thứ ba để chấm lại.

## 4. Chỉ số

**Khách quan** (do người quan sát ghi, đối chiếu video/nhật ký):

- Số **va chạm** (chạm vật bằng thân người, không tính gậy) và số lần **suýt va** (người giám sát phải can thiệp).
- **Thời gian** hoàn thành đường thử.
- Từ nhật ký: số cảnh báo/phút, cảnh báo thừa (vật bên lề), độ trễ từ lúc vật vào hành lang đến lúc có âm thanh.

**Chủ quan** (sau mỗi lượt): phiếu SUS 10 câu và NASA-TLX 6 thang (`PHIEU_SUS_NASA_TLX.md`), cùng một
câu hỏi mở: "Âm báo nào khó hiểu hoặc gây phiền?".

## 5. Phân tích

```bash
python scripts/tinh_sus_nasa.py docs/thu_nghiem_nguoi_dung/bang_ket_qua.csv --out ket_qua/sus_tlx.json
```

- Báo cáo trung bình ± độ lệch chuẩn theo điều kiện. Kiểm định Wilcoxon theo cặp (mẫu nhỏ, không giả
  định phân phối chuẩn), báo cáo cả kích thước hiệu ứng.
- Với n ≈ 10, kết quả chỉ mang tính **thăm dò**. Ghi rõ giới hạn này trong khoá luận và bài báo, không
  khẳng định "an toàn".

## 6. Hai người chấm độc lập

Nhãn sự kiện trên video (`scripts/gan_nhan_su_kien.py --annotator ...`) và đếm va chạm nên do **hai
người** làm độc lập. Báo cáo Cohen's kappa cho nhãn "có va chạm/không" và độ lệch trung vị của thời điểm
`contact`.

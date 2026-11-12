# Project context

> BẢN CẢI TIẾN (thư mục Do-An-Tot-Nghiep-CaiTien) - so với bản gốc xem BAO_CAO_CAI_TIEN.md.
> Một lõi chung cho hai máy: ban_mac/ (Apple Silicon) và ban_windows/ (laptop CPU).

Hệ thống hỗ trợ người khiếm thị, chạy offline trên macOS Apple Silicon M1 16GB.

## Architecture invariants

- Detection, depth và đánh giá nguy cơ phải tiếp tục hoạt động trong mọi chế độ.
- Lời cảnh báo vật cản chỉ phát trong chế độ Cảnh báo. [Thay đổi có chủ đích của bản cải tiến] Ở OCR/VQA,
  nguy cơ HIGH vẫn được báo bằng tiếng bíp ngắn theo hướng (safety.on_demand_high_alert: earcon), không
  chen lời nói; đặt "off" để trở lại hành vi bản gốc (tắt hết cho đến khi bấm C).
- Dữ liệu cũ/suy giảm, ảnh quá tối hoặc ống kính bị che không bao giờ được diễn giải là "an toàn".
- VQA và OCR chỉ chạy on-demand.
- Không bao giờ dùng VLM để khẳng định đường đi an toàn.
- Không thay đổi Detection, Depth hoặc Risk contracts nếu chưa cập nhật consumer.
- Bảo toàn drop-oldest camera buffering.
- Không thêm cloud API hoặc telemetry (nhật ký --event-log chỉ ghi ra tệp cục bộ).
- Mọi cải tiến phải tắt được bằng cấu hình; configs/thu_nghiem/nhu_ban_goc.yaml tái tạo hành vi bản gốc.
- Khác biệt Mac/Windows chỉ nằm ở ban_mac/cau_hinh_mac.yaml và ban_windows/ - không rẽ nhánh lõi theo nền tảng.

## Hardware constraints

- Target device: Apple M1, 16GB unified memory.
- Ưu tiên Core ML, MLX hoặc MPS.
- Không đưa dependency chỉ hoạt động với CUDA vào runtime chính.
- Model lớn phải lazy-load và có fallback.

## Verification

- Windows: ban_windows\chay_kiem_thu_windows.bat ; Mac: bash ban_mac/chay_kiem_thu_mac.sh
- Run: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  /opt/anaconda3/envs/ai-macbook/bin/python \
  -m unittest discover -s tests -p "test_*.py" -v
- Với thay đổi runtime, kiểm tra deadlock, shutdown và thread safety.
- Với thay đổi safety, thêm test cho lỗi và dữ liệu stale.

## Working rules

- Giữ thay đổi nhỏ và tập trung.
- Không sửa file ngoài phạm vi nếu không cần thiết.
- Không ghi đè thay đổi chưa commit của người dùng.
- Báo cáo test đã chạy và giới hạn còn lại.

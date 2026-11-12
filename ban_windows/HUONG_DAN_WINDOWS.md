# BẢN WINDOWS — hướng dẫn chạy bản cải tiến

> Thư mục này chỉ dùng cho **máy Windows** (laptop Intel, chạy CPU, không GPU). Thành viên dùng máy Mac
> xem [`../ban_mac/HUONG_DAN_MAC.md`](../ban_mac/HUONG_DAN_MAC.md). Hai bản dùng **chung một lõi thuật toán**
> (`src/`, `configs/`). Chỉ phần nền tảng khác nhau, nằm trong [`cau_hinh_windows.yaml`](cau_hinh_windows.yaml)
> và [`windows_tts.py`](windows_tts.py).

## Khác gì bản Mac

| Thành phần | Bản Windows | Bản Mac |
|---|---|---|
| Thiết bị tính toán | CPU, `torch_threads: 4` | MPS (GPU Apple) |
| Giọng đọc | SAPI, giọng OneCore "Microsoft An" (`windows_tts.py` + `sapi_tts.vbs`); âm báo phát bằng `winsound` | `say` giọng Linh + `afplay` |
| Đọc chữ (OCR) | EasyOCR | Apple Vision, dự phòng EasyOCR |
| Hỏi đáp (VQA) | BLIP + MarianMT (`legacy_caption`); chưa có model thì dùng ngữ cảnh detection | Qwen2.5-VL-3B 4-bit (MLX) |
| YOLO | YOLOv8n, ảnh 480 | YOLOv8s, ảnh 640 |
| Độ sâu | 182 px (thực tế 182×238) | 364 px (thực tế 364×490) |
| Camera mặc định | `1` (webcam tích hợp; `0` là camera ảo Camo) | `0` |

Độ phân giải độ sâu trên Windows thấp hơn có chủ đích. Ở 224 px, độ sâu trên CPU mất khoảng 420 ms
khi chạy trong app, làm khoảng 30% cặp detection–depth lệch quá 600 ms (DEGRADED, không cảnh báo).
Ở 182 px, kết quả đo được là 278/279 cặp VALID.

## Chạy

Môi trường conda `second-eye` đã được tạo sẵn trên máy này (Python 3.10, PyTorch CPU).

```bat
ban_windows\chay_app_windows.bat
```

```bat
ban_windows\chay_app_windows.bat --source 0
```

```bat
ban_windows\chay_nhu_ban_goc_windows.bat
```

```bat
ban_windows\chay_kiem_thu_windows.bat
```

- Lệnh thứ hai đổi sang camera khác.
- Lệnh thứ ba chạy cùng camera nhưng **tắt mọi cải tiến**, để so sánh A/B với bản gốc.
- Lệnh cuối chạy toàn bộ kiểm thử.

Mọi tham số của `app.py` đều truyền được qua các tệp `.bat`, ví dụ `--no-gui --duration 60` hoặc
`--event-log ket_qua\su_kien.jsonl`.

### Phím

| Phím | Chức năng |
|---|---|
| Space | đọc chữ |
| Q | hỏi "Phía trước có gì?" |
| V | hỏi bằng giọng nói |
| C | về chế độ cảnh báo |
| S | dừng đọc và tạm im tiếng bíp |
| F | tìm đồ (bấm tiếp để đổi món: cốc, chai, ghế, người, bàn…) |
| B / Enter / nút Bluetooth | **nút một phím**: 1 lần = hỏi, 2 lần = đọc chữ, 3 lần = tìm đồ, giữ lâu = im lặng/về cảnh báo |
| H | nghe hướng dẫn |
| I | nghe trạng thái hệ thống |
| Esc | thoát |

Khi chạy `--no-gui`, bấm phím ngay trong cửa sổ dòng lệnh (X để thoát).

## Tính năng tuỳ chọn (cần mạng, cài một lần)

```bat
ban_windows\cai_dat_tuy_chon_windows.bat
```

Script cài `sounddevice`, `faster-whisper`, `pynput` (xem `requirements-tuy-chon.txt`) và tải model
Whisper "small" cho phím V. Chưa cài thì khi bấm V hệ thống nói rõ là chưa cài và nhắc dùng phím Q.
Muốn bấm phím được kể cả khi cửa sổ không được chọn, đặt `ui.global_hotkeys: true` trong `configs/app_config.yaml`.

## Lưu ý khi dùng số liệu

- Windows là **máy thử nghiệm**. Không dùng FPS hay độ trễ đo trên máy này làm số liệu hiệu năng trên
  thiết bị mục tiêu (M1).
- Đánh giá theo sự kiện bằng `scripts/danh_gia_su_kien.py` chạy offline và độc lập với tốc độ máy.
  Muốn so sánh với kết quả của máy Mac thì dùng `--platform none` ở cả hai máy, để cùng kích thước suy luận.
- Test `test_runtime_modes...test_return_requires_new_sources_and_valid_quality` đôi khi lỗi trên
  Windows vì đồng hồ `time.monotonic()` chỉ có độ phân giải khoảng 15,6 ms. Đây là lỗi có sẵn ở bản gốc.

## Dành cho người khiếm thị: một nút bấm

Người dùng không nhìn thấy bàn phím. Dùng **nút chụp ảnh Bluetooth** (khoảng 50–100 nghìn đồng,
loại "AB Shutter") ghép đôi với máy, cầm một tay, tay kia cầm gậy:

- Bấm 1 lần: hỏi "Phía trước có gì?"
- Bấm 2 lần: đọc chữ
- Bấm 3 lần: tìm đồ (mỗi lần bấm 3 lần đổi sang món tiếp theo)
- Giữ lâu: im lặng; nếu đang im thì về chế độ cảnh báo

Nút Bluetooth gửi phím "tăng âm lượng", nên cần `ui.global_hotkeys: true` và thư viện `pynput`
(cài bằng script cài đặt tuỳ chọn). Thử trên bàn phím bằng phím **B**.

Kiểm thử tự động từng phím, kèm ảnh minh chứng:

```
python scripts/kiem_thu_phim.py
```

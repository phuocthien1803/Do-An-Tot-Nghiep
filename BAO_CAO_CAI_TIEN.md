# Báo cáo bản cải tiến — Hệ thống AI hỗ trợ người khiếm thị

Thư mục `Do-An-Tot-Nghiep-CaiTien` được dựng từ đúng mã nguồn GitHub của nhóm
(`lqmnhat13/Do-An-Tot-Nghiep`, commit `b020e38`). Ảnh chụp nguyên trạng được lưu thành commit đầu tiên,
gắn tag **`ban-goc`**. Thư mục mã cũ `Do-An-Tot-Nghiep` không bị sửa.

## 1. Cấu trúc: hai bản triển khai, một lõi chung

```
Do-An-Tot-Nghiep-CaiTien/
├── ban_windows/   BẢN WINDOWS: chay_app_windows.bat, cau_hinh_windows.yaml, windows_tts.py (SAPI), HUONG_DAN_WINDOWS.md
├── ban_mac/       BẢN MAC:     chay_app_mac.sh,     cau_hinh_mac.yaml,                         HUONG_DAN_MAC.md
├── configs/       cấu hình lõi (đã bật mọi cải tiến) + thu_nghiem/nhu_ban_goc.yaml (tắt hết cải tiến)
├── src/           lõi thuật toán dùng chung cho cả hai máy
├── scripts/       công cụ thực nghiệm mới (đo độ trễ, quay/gán nhãn video, chấm theo sự kiện, SUS/TLX)
└── docs/thu_nghiem_nguoi_dung/   giao thức thử nghiệm người dùng + phiếu SUS/NASA-TLX
```

Hai máy dùng chung một lõi, không tách hai bản mã. Lý do: nếu mỗi máy có một bản mã riêng, sửa lỗi ở
bản này rất dễ quên sửa ở bản kia, và khi đó kết quả Mac với Windows không còn so sánh được. Phần khác
nhau giữa hai máy (giọng đọc, OCR, VQA, thiết bị, kích thước suy luận) nằm hết trong hai lớp phủ
`ban_*/cau_hinh_*.yaml`. Test `test_platform_overlays` khẳng định các tham số thuật toán cảnh báo của
hai bản giống hệt nhau.

## 2. Ba cách so sánh với bản gốc

1. **Cùng máy, cùng camera (A/B)**: chạy `ban_windows\chay_nhu_ban_goc_windows.bat` hoặc
   `bash ban_mac/chay_nhu_ban_goc_mac.sh`. Lớp phủ này tắt toàn bộ cải tiến ảnh hưởng tới quyết định
   cảnh báo, cho cùng hành vi cảnh báo như bản gốc.
2. **Trên dữ liệu**: mọi benchmark và `scripts/danh_gia_su_kien.py` nhận
   `--overlay configs/thu_nghiem/nhu_ban_goc.yaml`; `danh_gia_su_kien.py --so-sanh` chạy cả hai biến thể.
3. **Theo mã nguồn**: `git diff ban-goc --stat` trong thư mục này (cần commit các thay đổi trước), hoặc
   so hai thư mục bằng WinMerge hay Meld.

## 3. Đính chính một nhận xét trong buổi phản biện trước

Mình từng nói ảnh vào mô hình độ sâu bị ép vuông 256×256 làm méo tỉ lệ. **Nhận xét này sai.** Bộ tiền
xử lý `DPTImageProcessor` của Depth Anything V2 có `keep_aspect_ratio=True` và `ensure_multiple_of=14`.
Với ảnh 640×480, cấu hình 256×256 thực chất cho tensor **252×336**, giữ đúng tỉ lệ. Bản cải tiến ghi
kích thước thật vào `preprocess_transform["model_input_hw"]` và chú thích lại trong `model_config.yaml`.

## 4. Đã làm gì (P0–P3)

Mọi cải tiến bật bằng cấu hình. Giá trị mặc định trong mã giữ hành vi gốc, nhờ đó 181 test gốc vẫn
kiểm tra hành vi gốc, còn 66 test mới trong `tests/test_cai_tien_*.py` kiểm tra phần cải tiến.

### P0 — An toàn và tương tác

| Vấn đề ở bản gốc | Đã sửa | Tệp chính | Khoá cấu hình |
|---|---|---|---|
| Ở OCR/VQA, mọi cảnh báo bị tắt, kể cả HIGH, tới khi bấm C | Kênh an toàn: ở OCR/VQA, nguy cơ HIGH mới có âm báo kép theo hướng, sau đó là tiếng tích song song với giọng đọc, không chen lời | `src/audio/safety_channel.py`, `system_coordinator.py` | `safety.on_demand_high_alert: earcon` (off/earcon/speech) |
| Lời cảnh báo dài 1,5–2 s ("Chú ý, có ghế phía trước, ở rất gần") | Âm báo lập thể < 100 ms phát trước, lệch trái/phải theo hướng; câu ngắn "Ghế phía trước", "Người đang lại gần, bên trái", "Cẩn thận, bậc xuống hoặc hố phía trước" | `src/audio/earcons.py`, `alert_aggregator.py` | `audio.earcons`, `alert_aggregator.message_style: ngan` |
| Chưa có nhịp bíp theo mức khẩn cấp | Tiếng tích nhanh dần theo TTC (khoảng cách giữa hai tiếng = 0,35×TTC, trong khoảng 0,15–0,9 s); dừng ngay khi dữ liệu cũ; phím S tạm im nguy cơ hiện tại | `safety_channel.py` | `safety.beeper` |
| Khoảng nghỉ toàn cục 3,5 s chặn cả vật mới; điều kiện `is_escalation` không bao giờ đúng | Vượt khoảng nghỉ (cách lần trước ≥ 1 s) khi có bậc xuống/hố mới, vật mới chắn giữa, hoặc vật khác có TTC ≤ 1,5 s | `alert_aggregator.py` | `alert_aggregator.bypass_interval` |
| Phím bị từ chối mà không có tiếng | Âm "nhận" hoặc "từ chối" cho mọi phím, kèm câu "Đang xử lý yêu cầu trước…" | `system_coordinator.py` | `audio.earcons` |
| Không có trợ giúp hay trạng thái bằng giọng nói | Phím **H** đọc danh sách phím, phím **I** đọc chế độ, tình trạng camera/độ sâu và trạng thái tự hiệu chỉnh | `system_coordinator.py`, `app_runner.py` | – |
| Quên bấm C thì mất cảnh báo bằng lời mãi | Tự quay về chế độ Cảnh báo sau khi OCR/VQA xong và loa im 10 s | `system_coordinator.py` | `ui.auto_return_warning_sec` |
| HUD ghi UNDETERMINED/NO_ALERT thành "NGUY CO THAP" | Nhãn riêng "CHUA XAC DINH" (xám) và "KHONG CANH BAO"; hiện TTC; vẽ vùng vật cản chung, bậc xuống, vật thấp; huy hiệu "BIP AN TOAN" và hệ số sàn | `hud_renderer.py` | – |
| Báo "dữ liệu độ sâu gián đoạn" giả lúc khởi động trên máy chậm | Khởi động thích ứng: chờ detection, depth và frame camera đầu tiên (tối đa 30 s; Windows 45 s). Mất tín hiệu sau đó vẫn báo ngay | `system_coordinator.py` | `runtime.startup_grace_max_sec` |
| **(Phát hiện khi chạy thử)** Camera bị che hoặc phòng tối, mô hình độ sâu vẫn cho số và hệ thống báo "vật cản" giả | Ảnh có độ sáng trung bình < 15 thì dữ liệu chuyển DEGRADED; sau 2 s đọc "Ảnh camera quá tối hoặc ống kính bị che…" | `system_coordinator.py` | `runtime.min_frame_brightness` |
| Chưa đo độ trễ người dùng cảm nhận | Đo độ trễ "frame camera → bắt đầu phát" trong app (`alert_latency_ms`); nhật ký JSONL; script kích thích trên màn hình | `event_log.py`, `scripts/do_tre_kich_thich.py`, `scripts/phan_tich_do_tre.py` | `--event-log` |

### P1 — Đóng góp khoa học chính: TTC và tự hiệu chỉnh độ sâu

| Cải tiến | Nội dung | Tệp | Khoá |
|---|---|---|---|
| **ByteTrack + Kalman** | Thay bộ ghép IoU tham lam một tầng (docstring gốc ghi nhầm "hai tầng"). Ghép Hungary hai tầng; detection tin cậy thấp 0,10–0,25 chỉ dùng để nối track (giữ mã track khi ảnh nhoè); track mất dấu được giữ 1 s | `src/tracking/bytetrack.py` | `tracking.type: bytetrack` |
| **TTC kết hợp hai nguồn** | (a) Theil–Sen trên chuỗi khoảng cách, cho TTC = d/v; (b) tốc độ phình bbox d(ln h)/dt = 1/TTC, không cần mét tuyệt đối. Hai nguồn lệch nhau ≤ 2 lần thì lấy trung bình nhân, lệch hơn thì lấy giá trị nhỏ | `src/fusion/ttc.py` | `ttc.*` |
| **Luật nguy cơ TTC** | HIGH khi ≤ 1,0 m **hoặc** TTC ≤ 2,5 s (trong 5 m). Vật đã biết là đứng yên tương đối, nằm trong 1,5 m, hạ xuống MEDIUM để bớt nhắc thừa. Track mới chưa đủ mẫu dùng nguyên luật 1,5 m của bản gốc | `risk_fsm.py` | `risk_fsm.risk_model: ttc` |
| **Tự hiệu chỉnh hệ số mét theo mặt sàn** | Sàn phẳng cho 1/Z tuyến tính theo hàng ảnh. Khớp RANSAC trên độ sâu thô, suy ra cùng lúc góc nghiêng camera và hệ số k = A·fy·h/cos θ. Chỉ cần độ cao camera `camera.height_m` và trường nhìn `camera.hfov_deg`. Làm mượt bằng EMA; không thấy sàn thì giữ hệ số cũ hoặc 0,60 | `src/depth/floor_model.py` | `depth.auto_scale` |
| Độ phân giải độ sâu theo máy | Mac 364 px (thực tế 364×490); Windows 182 px (182×238), đủ để độ lệch thời gian luôn < 600 ms | `ban_*/cau_hinh_*.yaml` | `depth.input_*` |
| YOLOv8s cho Mac | mAP50 0,672 so với 0,596 của bản n; thiếu tệp thì tự dùng bản n | `cau_hinh_mac.yaml`, `factory.py` | `detection.weights_path` |
| Danh sách lớp | Thêm ghế dài, chậu cây, va li, xe đạp, xe máy, ba lô; bỏ laptop và điện thoại khỏi danh sách cảnh báo | `model_config.yaml` | – |

### P2 — Mở rộng phạm vi an toàn và dữ liệu

| Cải tiến | Nội dung | Tệp |
|---|---|---|
| **Bậc xuống/hố** | Pixel trong hành lang xa hơn mặt sàn kỳ vọng (nghịch đảo độ sâu < 70%) nghĩa là sàn bị hụt | `src/fusion/floor_hazards.py` |
| **Vật cản thấp** | Pixel gần hơn sàn kỳ vọng > 35%, trong 1,5 m, ngoài các bbox YOLO; xét cả dải sàn sát chân mà bản gốc bỏ qua | `floor_hazards.py` |
| Xác nhận theo thời gian | Vật cản chung, bậc xuống, vật thấp cần 2 trên 3 frame độ sâu (bản gốc: vật cản chung chỉ cần 1 frame) | `risk_fsm.py`, `hazard_confirmation` |
| Bộ dữ liệu riêng | Quay video kèm nhật ký frame; công cụ gán nhãn sự kiện (start/contact/end, loại); chấm recall, tỉ lệ báo kịp thời, thời gian báo trước, cảnh báo thừa/phút | `scripts/ghi_du_lieu.py`, `gan_nhan_su_kien.py`, `danh_gia_su_kien.py` |
| Benchmark có biến thể | TUM/NYU nhận `--overlay`, `--tag`; TUM ghi thêm sai số hệ số tự hiệu chỉnh và hệ số cố định so với hệ số tối ưu theo ground truth | `evaluation/benchmarks/*` |

### P3 — Tương tác và đánh giá hướng người dùng

| Cải tiến | Nội dung | Tệp |
|---|---|---|
| **Hỏi bằng giọng nói (V)** | Ghi âm 5 s rồi nhận dạng bằng faster-whisper (offline). Hiểu các lệnh "đọc chữ…", "trợ giúp", "trạng thái", "về chế độ cảnh báo"; câu khác chuyển sang VQA. Thiếu thư viện thì báo rõ và nhắc dùng phím Q | `src/asr/voice_input.py` |
| Guardrail mở rộng | Bắt thêm câu "lối đi trống/thông thoáng", "có gì cản/chắn/đụng", "cứ đi… là được đúng không", tiếng Anh "path clear/in my way", câu gõ không dấu; câu trả lời "an toàn thực phẩm" không còn bị chặn nhầm | `src/vqa/safety.py` |
| Câu trả lời VQA | Bỏ tiền tố "Khung cảnh:" ở mọi câu; không đọc "(chưa rõ)"; phần cảm biến ghi nhận được tách riêng | `src/vqa/formatting.py` |
| Lọc chữ rác OCR | Bỏ chuỗi toàn ký hiệu và ký tự lẻ có độ tin cậy thấp | `src/ocr/ocr_service.py` |
| `--no-gui` vẫn điều khiển được | Bấm phím ngay trong console; tuỳ chọn phím tắt toàn hệ thống (pynput) | `src/ui/app_runner.py` |
| Thử nghiệm người dùng | Giao thức (G / G+bản gốc / G+bản cải tiến, hình vuông Latin, an toàn, đạo đức); phiếu SUS và NASA-TLX tiếng Việt; script tính điểm và kiểm định Wilcoxon | `docs/thu_nghiem_nguoi_dung/`, `scripts/tinh_sus_nasa.py` |

Các thay đổi kỹ thuật khác:
- Worker lấy frame **mới nhất**; bản gốc lấy FIFO.
- Từ điển chống lặp câu được dọn định kỳ; bản gốc để phình mãi.
- `src/runtime/config.py` và `factory.py` gom việc dựng pipeline về một chỗ.
- Giọng đọc chọn được qua cấu hình: `say`, `sapi`, hoặc `console` cho CI.
- `scripts/download_models.py` có thêm `--yolov8s` và `--asr`.
- Sơ đồ kiến trúc không còn ghi "Độ gần tương đối".

## 5. Kết quả đã kiểm chứng trên máy Windows của bạn (12/11/2026)

Máy Windows là máy thử nghiệm (i5-1135G7, CPU). Số tốc độ ở đây **không** đại diện cho M1.

**Kiểm thử tự động**: 247 test (181 gốc và 66 mới). 245 qua, 1 bỏ qua (cần macOS). 1 lỗi là test chập
chờn có sẵn từ bản gốc (`test_return_requires_new_sources_and_valid_quality`, do đồng hồ Windows phân
giải 15,6 ms). flake8 theo cấu hình CI báo 0 lỗi.

**Guardrail** (`evaluation/benchmarks/eval_guardrail_mo_rong.py`, kết quả trong `evaluation/results/guardrail_mo_rong.json`):

| Bộ câu hỏi | Bản gốc (regex) | Bản cải tiến (mo_rong) |
|---|---|---|
| Bộ gốc: câu phải chặn | 22/30 | **30/30** |
| Bộ gốc: câu không được chặn | 20/20 | 20/20 |
| Bộ giữ riêng (viết trước khi thử luật): câu phải chặn | 12/30 | **29/30** |
| Bộ giữ riêng: câu không được chặn | 25/25 | 25/25 |
| Câu trả lời của VLM | 10/11 | **11/11** |

Câu duy nhất còn lọt là "Phía trước có nguy cơ gì không?". Mình **không** thêm từ "nguy cơ" vào luật
sau khi đã thấy kết quả, để con số trên bộ giữ riêng không bị "học tủ". Cả luật và bộ câu đều do nhóm
viết, nên vẫn cần câu hỏi thật của người dùng; nhật ký `--event-log` có ghi lại câu hỏi bằng giọng nói
cho việc này.

**Chạy trực tiếp webcam, bản Windows, không giao diện, 30–40 s** (camera lúc đó gần như đen, độ sáng trung bình 4/255):

| Chỉ số | Như bản gốc (độ sâu 224 px) | Cải tiến (224 px) | Cải tiến (182 px) | Cải tiến (182 px + chặn ảnh tối) |
|---|---|---|---|---|
| Cặp dữ liệu VALID | 70% (171/243) | 70% | **99,6%** (278/279) | 0% (DEGRADED có chủ đích) |
| Báo lỗi giả lúc khởi động | có ("độ sâu gián đoạn") | không về độ sâu, nhưng có báo "mất camera" giả (đã sửa) | không | không |
| Cảnh báo "vật cản" trên ảnh đen | 5 lần / 40 s | 0 | 7 lần / 40 s | **0, thay bằng "Ảnh camera quá tối hoặc ống kính bị che…"** |
| Độ trễ frame → bắt đầu phát cảnh báo (p50 / p95) | 593 / 763 ms | – | **515 / 625 ms** | – |

Bảng này cho thấy hai lỗ hổng của bản gốc, đều có thật trên máy chậm:
1. Khoảng 30% thời gian dữ liệu DEGRADED nên hệ thống im lặng.
2. Khi camera bị che, mô hình độ sâu vẫn "nhìn thấy" vật cản.
Bản gốc trên M1 ít gặp lỗi (1) hơn, nhưng lỗi (2) xảy ra ở mọi máy.

**Video tổng hợp "người tiến lại gần"** (phóng to dần ảnh có người trong 4 s, sự kiện chạm ở giây thứ 4,
chạy bằng `scripts/danh_gia_su_kien.py --so-sanh`):

| | Như bản gốc | Bản cải tiến |
|---|---|---|
| Cảnh báo đầu tiên | 1,07 s: "Chú ý, có người phía trước, ở rất gần" | **0,37 s: "Người đang lại gần, phía trước"** |
| Thời gian báo trước lúc chạm | 2,93 s | **3,63 s** |

Đây chỉ là video tổng hợp để kiểm tra công cụ và luật TTC, **chưa phải bằng chứng thực tế**.

## 6. Việc còn phải làm để có số liệu cho khoá luận và bài báo

1. **Trên Mac** (có bộ TUM/NYU/COCO): chạy lại TUM và NYU cho hai biến thể; lệnh có sẵn trong
   `ban_mac/HUONG_DAN_MAC.md`, mục 4. Số cần nhất cho bài báo:
   - recall của `walking_xyz` (bản gốc 0,31);
   - sai số hệ số tự hiệu chỉnh so với hệ số cố định;
   - số cảnh báo/phút.
2. **Bộ video riêng**: 15–20 video ở hành lang, phòng học, cầu thang (có bậc xuống thật, quay an toàn).
   Hai người gán nhãn, chạy `danh_gia_su_kien.py --so-sanh`.
3. **Đo độ trễ đầu–cuối** bằng `do_tre_kich_thich.py` trên cả hai máy.
4. **Thử nghiệm người dùng** theo `docs/thu_nghiem_nguoi_dung/GIAO_THUC_THU_NGHIEM.md`.
5. **Tinh chỉnh tham số** trên video thật, tách tập chỉnh với tập đánh giá: `ttc_high_sec`,
   `critical_distance_m`, `drop_ratio`, `rise_ratio`, `min_frame_brightness`. Các giá trị hiện tại là
   giá trị khởi đầu hợp lý, **chưa được tối ưu**.

## 6b. Bổ sung ngày 13/11/2026

| Hạng mục | Nội dung | Tệp |
|---|---|---|
| **Sửa: che camera không báo** | Che tay lên webcam không làm ảnh tối hẳn (camera tự tăng sáng), nên ngưỡng độ sáng không bắt được. Nay xét thêm "ảnh mất chi tiết": hầu hết ô lưới 4×4 gần như phẳng và độ nét Laplacian thấp, đo trên ảnh đã thu nhỏ để khử nhiễu hạt. Đúng trên 8/8 loại ảnh thử, gồm webcam thật, phòng tối vừa, tay che sáng/tối | `system_coordinator.py` (`is_blind_frame`) |
| **Nút bấm một phím** | Cho người khiếm thị: nút chụp ảnh Bluetooth hoặc một phím. 1 lần = hỏi, 2 lần = đọc chữ, 3 lần = tìm đồ, giữ lâu = im lặng/về cảnh báo. Nhận cả sự kiện nhả phím (pynput) và chuỗi nhấn lặp khi giữ phím (OpenCV) | `src/ui/one_button.py`, `app_runner.py` |
| **Chế độ Tìm đồ** | Phím F, bấm nút 3 lần, hoặc nói "tìm cái cốc". Âm dẫn hướng lập thể 7 vị trí trái→phải; tiếng cao khi vật ở ngay giữa; nhanh dần khi lại gần; lời nhắc ngắn ("Thấy cốc bên trái, còn xa", "Cốc trong tầm tay", "Chưa thấy cốc, hãy xoay người chậm"). Lời cảnh báo vật cản khác vẫn bật; riêng vật đang tìm không bị coi là vật cản | `src/fusion/object_finder.py` |
| **Kiểm thử từng phím có ảnh** | 16 kịch bản trên hệ thống thật (YOLO, độ sâu, OCR, VQA), camera kịch bản, ghi lời nói: **16/16 đạt** | `scripts/kiem_thu_phim.py`, `evaluation/results/kiem_thu_phim/` |

Phát hiện khi kiểm thử: trên Windows, hỏi đáp dùng BLIP chỉ **mô tả ảnh, bỏ qua nội dung câu hỏi**. Hỏi đáp thật sự chỉ có ở bản Mac (Qwen2.5-VL). Giới hạn của Tìm đồ: chỉ tìm được lớp YOLO/COCO đã bật; chưa có "cửa", "chìa khoá", "cầu thang". Hướng mở rộng là bộ phát hiện từ vựng mở (YOLO-World) hoặc tinh chỉnh YOLO.

## 7. Giới hạn và những gì chưa làm (nói thẳng)

- **Chưa chạy được trên Mac.** Các đường chạy riêng của Mac là MPS, `afplay` phát âm báo song song với
  `say`, Apple Vision, MLX-VLM và YOLOv8s. Chúng chỉ được kiểm tra bằng test giả lập; thành viên dùng Mac
  cần chạy `bash ban_mac/chay_kiem_thu_mac.sh` và chạy thử app.
- **Hỏi bằng giọng nói chưa chạy đầu–cuối.** `sounddevice` và `faster-whisper` chưa được cài trên máy
  này. Phần định tuyến lệnh và xử lý thiếu thư viện đã có test. Muốn dùng, chạy
  `ban_windows\cai_dat_tuy_chon_windows.bat` (cần mạng, tải khoảng 500 MB).
- **Mô hình sàn có giả định**: sàn phẳng, nhìn thấy được, camera không nghiêng ngang nhiều, biết độ cao
  camera. Camera đặt trên bàn hoặc sát tường thì không khớp được sàn, hệ thống dùng hệ số dự phòng.
  Phát hiện bậc xuống phụ thuộc việc Depth Anything có tạo bước nhảy độ sâu ở mép bậc hay không. Mới
  kiểm tra trên dữ liệu tổng hợp, **phải đánh giá trên video thật**.
- **TTC từ độ phình bbox** nhạy với dao động bbox của YOLO và với vật bị cắt ở mép ảnh (trường hợp này
  đã loại). Khi người dùng quay người, TTC có thể dao động.
- **Chưa làm**:
  - chuyển sang iPhone (là dự án riêng: Swift, Core ML, LiDAR);
  - YOLO-World hoặc tinh chỉnh YOLO, vì cần tải model hoặc dữ liệu, GPU huấn luyện, và không kiểm chứng
    được offline ở đây;
  - bộ phân loại ý định học máy cho guardrail. Với vài chục câu tự viết, một bộ phân loại học máy không
    đáng tin hơn luật; nên thu câu hỏi thật trước rồi mới huấn luyện.
- **Đổi một quy tắc của AGENTS.md có chủ đích.** Ở OCR/VQA, bản gốc tắt toàn bộ cảnh báo; bản cải tiến
  vẫn phát tiếng bíp an toàn. `AGENTS.md` của thư mục này đã ghi rõ thay đổi, và đặt
  `safety.on_demand_high_alert: off` để quay lại hành vi gốc. Nhóm và giảng viên nên thống nhất chọn
  hành vi nào trước khi bảo vệ.
- Hệ thống **không** đảm bảo an toàn và không thay thế gậy dò đường. Mọi thông báo vẫn nhắc dùng gậy
  khi dữ liệu không tin cậy.

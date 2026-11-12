# BẢN MAC — hướng dẫn chạy bản cải tiến (Apple Silicon M1 16 GB)

> Thư mục này chỉ dùng cho **máy Mac** của thành viên nhóm. Bản Windows xem
> [`../ban_windows/HUONG_DAN_WINDOWS.md`](../ban_windows/HUONG_DAN_WINDOWS.md). Hai bản dùng **chung một lõi
> thuật toán** (`src/`, `configs/`). Phần riêng của Mac chỉ là lớp phủ [`cau_hinh_mac.yaml`](cau_hinh_mac.yaml):
> MPS, giọng Linh, Apple Vision, Qwen2.5-VL (MLX), YOLOv8s và độ sâu 364 px.

## 1. Lấy mã

Chép nguyên thư mục `Do-An-Tot-Nghiep-CaiTien` sang Mac (USB, Drive hoặc `git clone` nếu đã đẩy lên
GitHub). Không cần chép `models/weights/` nếu Mac đã có sẵn từ bản gốc. Có thể tạo liên kết tới
thư mục weights cũ:

```bash
ln -s /duong/dan/Do-An-Tot-Nghiep/models/weights/* models/weights/
```

## 2. Môi trường

Dùng lại env `ai-macbook` của bản gốc (Python 3.10, scipy < 1.15, `mlx-vlm` đã cài riêng). Bản cải
tiến **không thêm thư viện bắt buộc nào**. Tuỳ chọn, cần mạng:

```bash
bash ban_mac/cai_dat_tuy_chon_mac.sh
```

Script cài `sounddevice`, `faster-whisper`, `pynput` và tải YOLOv8s cùng Whisper "small". Thiếu YOLOv8s
thì app tự dùng `yolov8n.pt`. Thiếu Whisper thì phím V báo chưa cài.

macOS sẽ hỏi quyền **Micro** khi dùng phím V lần đầu. Phím tắt toàn hệ thống (`ui.global_hotkeys: true`)
cần cấp quyền **Trợ năng (Accessibility)** cho Terminal.

## 3. Chạy

```bash
bash ban_mac/chay_app_mac.sh
```

```bash
bash ban_mac/chay_app_mac.sh --source 0 --event-log ket_qua/su_kien.jsonl
```

```bash
bash ban_mac/chay_nhu_ban_goc_mac.sh
```

```bash
bash ban_mac/chay_kiem_thu_mac.sh
```

- Lệnh đầu chạy bản cải tiến.
- Lệnh thứ hai chạy kèm nhật ký sự kiện để đo độ trễ.
- Lệnh thứ ba tắt mọi cải tiến để so sánh A/B.
- Lệnh cuối chạy toàn bộ kiểm thử.

Đường dẫn Python mặc định là `/opt/anaconda3/envs/ai-macbook/bin/python`. Dùng Python khác thì đặt `PY`:

```bash
PY=python3 bash ban_mac/chay_app_mac.sh
```

### Phím

| Phím | Chức năng |
|---|---|
| Space | đọc chữ |
| Q | hỏi "Phía trước có gì?" |
| V | hỏi bằng giọng nói |
| C | về chế độ cảnh báo |
| S | dừng đọc và tạm im tiếng bíp |
| H | nghe hướng dẫn |
| I | nghe trạng thái |
| Esc | thoát |

## 4. Việc cần làm trên Mac để có số liệu cho bài báo

Bộ dữ liệu TUM, NYU, COCO (~4,6 GB) chỉ có trên Mac. Chạy lại cùng benchmark cho **cả hai biến thể**:

```bash
cd evaluation/benchmarks
```

Mỗi chuỗi TUM chạy hai lần, bản gốc và bản cải tiến:

```bash
python eval_warning_tum.py --seq-dir $DATA/rgbd_dataset_freiburg3_walking_xyz --overlay configs/thu_nghiem/nhu_ban_goc.yaml --tag _goc
```

```bash
python eval_warning_tum.py --seq-dir $DATA/rgbd_dataset_freiburg3_walking_xyz --tag _caitien
```

Lặp lại cho `freiburg1_desk` và `freiburg2_desk` (hoặc `long_office`).

NYU là ảnh đơn lẻ nên cần thêm lớp phủ `anh_don_le.yaml`, không làm mượt hệ số giữa các ảnh:

```bash
python eval_depth_warning_nyu.py --nyu-dir $DATA/nyu --overlay configs/thu_nghiem/nhu_ban_goc.yaml --tag _goc
```

```bash
python eval_depth_warning_nyu.py --nyu-dir $DATA/nyu --overlay configs/thu_nghiem/anh_don_le.yaml --tag _caitien
```

Guardrail không cần dữ liệu:

```bash
python eval_guardrail_mo_rong.py --out ../results/guardrail_mo_rong.json
```

Kết quả nằm ở `evaluation/results/real_data/`. Tệp TUM có thêm mục `depth_scale_vs_oracle`, cho biết
sai số giữa hệ số độ sâu tự hiệu chỉnh và hệ số cố định 0,60 so với hệ số tối ưu theo ground truth.
TUM là camera cầm tay chưa biết độ cao, nên nên thử `--camera-height-m 1.2`, `1.4` và báo cáo độ nhạy.

Tiếp theo, quay video thật và chấm theo sự kiện (xem `BAO_CAO_CAI_TIEN.md`, mục 5). Nếu thay đổi
`depth.input_height` hoặc tắt `auto_scale`, cần hiệu chỉnh lại `metric_scale`:

```bash
python scripts/calibrate_depth.py --distance 1.0
```

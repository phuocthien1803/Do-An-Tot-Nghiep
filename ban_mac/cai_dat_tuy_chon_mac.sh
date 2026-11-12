#!/usr/bin/env bash
# ======================================================================
#  BẢN MAC (Apple Silicon) - cài thư viện TUỲ CHỌN + tải YOLOv8s, Whisper (cần mạng)
#  Chỉ dùng trên macOS. Bản Windows: xem ban_windows/.
# ======================================================================
set -e
cd "$(dirname "$0")/.."
PY="${PY:-/opt/anaconda3/envs/ai-macbook/bin/python}"
[ -x "$PY" ] || PY=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
"$PY" -m pip install -r requirements-tuy-chon.txt
"$PY" scripts/download_models.py --yolov8s
"$PY" scripts/download_models.py --asr small

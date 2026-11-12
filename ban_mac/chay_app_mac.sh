#!/usr/bin/env bash
# ======================================================================
#  BẢN MAC (Apple Silicon) - chạy ứng dụng bản cải tiến
#  Chỉ dùng trên macOS. Bản Windows: xem ban_windows/.
# ======================================================================
set -e
cd "$(dirname "$0")/.."
PY="${PY:-/opt/anaconda3/envs/ai-macbook/bin/python}"
[ -x "$PY" ] || PY=python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$PY" app.py --platform mac "$@"

#!/usr/bin/env python3
"""Đo độ trễ đầu-cuối "vật xuất hiện -> hệ thống phát âm thanh" bằng kích thích trên màn hình.

Bản gốc chỉ đo độ trễ từng mô-đun (detection p95 69 ms...), chưa có con số người dùng cảm nhận.
Cách làm (một máy, hai tiến trình dùng chung đồng hồ monotonic của hệ điều hành):
  1. Cửa sổ 1: python app.py --event-log ket_qua/su_kien.jsonl     (camera hướng vào màn hình,
     cách 0,6-0,9 m để vật trên màn hình nằm trong hành lang và "rất gần")
  2. Cửa sổ 2: python scripts/do_tre_kich_thich.py --out ket_qua/kich_thich.jsonl --lan 20
     Màn hình đen; ngẫu nhiên 4-7 s lại hiện ảnh người cỡ lớn trong 2,5 s (ghi thời điểm hiện).
  3. python scripts/phan_tich_do_tre.py ket_qua/kich_thich.jsonl ket_qua/su_kien.jsonl
Lưu ý: số đo là tới lúc phần mềm RA LỆNH phát âm thanh; độ trễ loa/driver (~20-100 ms) chưa tính.
Màn hình cũng có độ trễ hiển thị (~1-2 frame) - nên dùng màn hình riêng hoặc điện thoại kê trước camera.
"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np


def default_image() -> np.ndarray:
    try:
        import ultralytics
        path = os.path.join(os.path.dirname(ultralytics.__file__), "assets", "zidane.jpg")
        img = cv2.imread(path)
        if img is not None:
            return img
    except Exception:
        pass
    raise SystemExit("Không tìm thấy ảnh mẫu; dùng --anh <ảnh có người hoặc ghế cỡ lớn>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anh", default=None, help="Ảnh kích thích (mặc định ảnh người của ultralytics)")
    ap.add_argument("--lan", type=int, default=20, help="Số lần kích thích")
    ap.add_argument("--hien", type=float, default=2.5, help="Thời gian hiện ảnh (giây)")
    ap.add_argument("--nghi-min", type=float, default=4.0)
    ap.add_argument("--nghi-max", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed)
    img = cv2.imread(args.anh) if args.anh else default_image()
    black = np.zeros_like(img)
    name = "kich_thich"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as log:
        for i in range(args.lan):
            cv2.imshow(name, black)
            wait_until = time.monotonic() + random.uniform(args.nghi_min, args.nghi_max)
            while time.monotonic() < wait_until:
                if (cv2.waitKey(10) & 0xFF) == 27:
                    return
            cv2.imshow(name, img)
            cv2.waitKey(1)  # đẩy khung hình ra màn hình rồi mới ghi thời điểm
            onset = time.monotonic()
            log.write(json.dumps({"lan": i, "onset": onset, "duration": args.hien}) + "\n")
            log.flush()
            print(f"Kích thích {i + 1}/{args.lan}")
            while time.monotonic() < onset + args.hien:
                if (cv2.waitKey(10) & 0xFF) == 27:
                    return
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

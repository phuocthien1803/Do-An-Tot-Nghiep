#!/usr/bin/env python3
"""Quay video thực nghiệm (bộ dữ liệu riêng tại Việt Nam) để đánh giá theo sự kiện.

Gắn camera ngang ngực (đúng độ cao camera.height_m), đi chậm qua hành lang/phòng học/nhà ở có vật
cản đặt sẵn, bậc thang, vật thấp. Mỗi tệp nên 1-3 phút. Không quay mặt người khác khi chưa xin phép.

    python scripts/ghi_du_lieu.py --source 0 --out data/su_kien/hanh_lang_01.mp4 --duration 120

Phím: Esc/Q dừng. Ghi kèm <ten>.frames.csv (chỉ số frame, thời điểm monotonic, thời gian thực).
"""
import argparse
import csv
import os
import time

import cv2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=0.0, help="Giây; 0 = đến khi bấm Esc")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được camera {args.source}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (640, 480))
    log_path = os.path.splitext(args.out)[0] + ".frames.csv"
    t0 = time.monotonic()
    n = 0
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        log = csv.writer(fh)
        log.writerow(["frame", "t_mono", "t_wall"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (640, 480))
            writer.write(frame)
            log.writerow([n, f"{time.monotonic():.4f}", f"{time.time():.4f}"])
            n += 1
            if not args.no_preview:
                cv2.putText(frame, f"REC {time.monotonic() - t0:5.1f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("ghi_du_lieu", frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
            if args.duration and time.monotonic() - t0 >= args.duration:
                break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Đã ghi {n} frame ({time.monotonic() - t0:.1f} s) -> {args.out}; nhật ký frame: {log_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Gán nhãn sự kiện vật cản trên video để chấm điểm theo sự kiện (scripts/danh_gia_su_kien.py).

    python scripts/gan_nhan_su_kien.py data/su_kien/hanh_lang_01.mp4

Phím:
  Space     dừng / chạy          A / D   lùi / tiến 1 giây     , / .  lùi / tiến 1 frame
  1 2 3 4   chọn loại: 1 vật cản, 2 bậc xuống/hố, 3 vật cản thấp, 4 người đi tới
  S         đánh dấu START  (vật bắt đầu nằm trên đường đi)
  C         đánh dấu CONTACT (lúc chạm/đến sát vật - nếu dừng kịp thì lúc dừng lại)
  E         đánh dấu END    (vật ra khỏi đường đi) -> lưu sự kiện
  U         xoá sự kiện cuối    W  ghi tệp     Esc/Q  ghi và thoát
Nên có HAI người gán nhãn độc lập (rồi tính Cohen's kappa / độ lệch thời gian) để tránh thiên lệch.
"""
import argparse
import json
import os

import cv2

TYPES = {ord("1"): "obstacle", ord("2"): "drop_off", ord("3"): "low_obstacle", ord("4"): "person"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--annotator", default="nguoi_gan_nhan_1")
    args = ap.parse_args()
    out = os.path.splitext(args.video)[0] + ".events.json"
    data = {"video": os.path.basename(args.video), "annotator": args.annotator, "events": []}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pos, playing, kind, pending = 0, False, "obstacle", {}

    def save():
        data["fps"] = fps
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print("Đã ghi", out, len(data["events"]), "sự kiện")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            pos = max(0, total - 1)
            playing = False
            continue
        t = pos / fps
        info = f"t={t:6.2f}s  loai={kind}  {'>' if playing else '||'}  su_kien={len(data['events'])}  " \
               f"dang_do={','.join(pending) or '-'}"
        cv2.putText(frame, info, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow("gan_nhan_su_kien", frame)
        key = cv2.waitKey(int(1000 / fps) if playing else 0) & 0xFF
        if key in (27, ord("q")):
            save()
            break
        if key == ord(" "):
            playing = not playing
        elif key == ord("a"):
            pos = max(0, pos - int(fps))
        elif key == ord("d"):
            pos = min(total - 1, pos + int(fps))
        elif key == ord(","):
            pos = max(0, pos - 1)
        elif key == ord("."):
            pos = min(total - 1, pos + 1)
        elif key in TYPES:
            kind = TYPES[key]
        elif key == ord("s"):
            pending = {"start": round(t, 3), "type": kind}
        elif key == ord("c") and pending:
            pending["contact"] = round(t, 3)
        elif key == ord("e") and pending:
            pending.setdefault("contact", round(t, 3))
            pending["end"] = round(t, 3)
            data["events"].append(pending)
            print("+", pending)
            pending = {}
        elif key == ord("u") and data["events"]:
            print("- xoá", data["events"].pop())
        elif key == ord("w"):
            save()
        if playing:
            pos += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

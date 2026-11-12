#!/usr/bin/env python3
"""Đánh giá THEO SỰ KIỆN trên video tự quay: bản gốc và bản cải tiến chạy trên CÙNG video.

Bản gốc chỉ có số liệu theo khung hình. Với người dùng, câu hỏi đúng là "mỗi lần tiến gần vật cản,
hệ thống có cảnh báo KỊP không" và "mỗi phút có bao nhiêu cảnh báo thừa".

Quy trình:
  1. Quay video: python scripts/ghi_du_lieu.py --out data/su_kien/hanh_lang_01.mp4
  2. Gán nhãn:   python scripts/gan_nhan_su_kien.py data/su_kien/hanh_lang_01.mp4
     -> hanh_lang_01.events.json: [{"start", "contact", "end", "type"}] (giây)
  3. Chấm:       python scripts/danh_gia_su_kien.py data/su_kien/hanh_lang_01.mp4 --so-sanh

Pipeline chạy OFFLINE, xử lý mọi frame, độ sâu mỗi `--depth-every` frame, thời gian lấy theo
chỉ số frame / fps -> kết quả không phụ thuộc tốc độ máy (Mac hay Windows cho cùng kết quả,
trừ sai khác số học nhỏ giữa MPS và CPU).

Chỉ số (mỗi sự kiện: start = vật bắt đầu chắn đường, contact = lúc chạm/đến sát vật):
  - recall: có cảnh báo trong [start - 0,5 s, contact]
  - timely: cảnh báo đầu tiên sớm hơn contact ít nhất --min-lead giây
  - lead_time: contact - thời điểm cảnh báo đầu tiên
  - false_alarms_per_min: cảnh báo không rơi vào cửa sổ [start - 0,5 s, end + 1 s] của sự kiện nào
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.contracts.frame_packet import FramePacket  # noqa: E402
from src.runtime import factory  # noqa: E402
from src.runtime.config import load_configs  # noqa: E402

BASELINE_OVERLAY = "configs/thu_nghiem/nhu_ban_goc.yaml"


def run_pipeline(video: str, overlays, platform: str, device: str, depth_every: int, max_sec: float):
    cfg = load_configs(PROJECT_ROOT, platform=platform, overlays=overlays)
    stack = factory.build_risk_stack(cfg)
    detector = factory.build_detector(cfg, device, stack["class_filter"])
    depth = factory.build_depth(cfg, device)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    alerts, high_frames, frames = [], 0, 0
    idx = 0
    t_base = 1000.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if max_sec and t > max_sec:
            break
        frame = cv2.resize(frame, (640, 480)) if frame.shape[:2] != (480, 640) else frame
        packet = FramePacket(idx, "video", frame, (640, 480), t_base + t)
        if idx % depth_every == 0:
            stack["synchronizer"].register_depth(depth.estimate(packet))
        tracked = stack["tracker"].update(detector.detect(packet))
        pair = stack["synchronizer"].synchronize(tracked, current_mono=t_base + t)
        assessments = stack["risk_fsm"].update(pair, current_mono=t_base + t)
        task = stack["aggregator"].aggregate(assessments, current_mono=t_base + t)
        if task is not None:
            top = stack["aggregator"].last_reason
            alerts.append({"t": round(t, 3), "text": task.text, "reason": top})
        high_frames += any(a.risk_level.value == "HIGH" for a in assessments)
        frames += 1
        idx += 1
    cap.release()
    return {"fps": fps, "frames": frames, "duration_sec": frames / fps, "alerts": alerts,
            "high_frame_ratio": high_frames / frames if frames else None,
            "overlays": [os.path.basename(p) for p in cfg["_meta"]["overlays"]]}


def score(run, events, min_lead: float):
    alert_times = [a["t"] for a in run["alerts"]]
    rows = []
    for ev in events:
        window = [t for t in alert_times if ev["start"] - 0.5 <= t <= ev["contact"]]
        first = min(window) if window else None
        lead = ev["contact"] - first if first is not None else None
        rows.append({"type": ev.get("type", "obstacle"), "start": ev["start"], "contact": ev["contact"],
                     "first_alert": first, "lead_time_sec": lead,
                     "detected": first is not None, "timely": lead is not None and lead >= min_lead})
    def in_event(t):
        return any(ev["start"] - 0.5 <= t <= ev.get("end", ev["contact"]) + 1.0 for ev in events)
    false_alarms = [t for t in alert_times if not in_event(t)]
    minutes = run["duration_sec"] / 60.0 if run["duration_sec"] else None
    leads = [r["lead_time_sec"] for r in rows if r["lead_time_sec"] is not None]
    by_type = {}
    for r in rows:
        b = by_type.setdefault(r["type"], {"n": 0, "detected": 0, "timely": 0})
        b["n"] += 1
        b["detected"] += r["detected"]
        b["timely"] += r["timely"]
    n = len(rows)
    return {
        "events": n,
        "recall": sum(r["detected"] for r in rows) / n if n else None,
        "timely_rate": sum(r["timely"] for r in rows) / n if n else None,
        "lead_time_median_sec": float(np.median(leads)) if leads else None,
        "alerts_per_min": len(alert_times) / minutes if minutes else None,
        "false_alarms": len(false_alarms),
        "false_alarms_per_min": len(false_alarms) / minutes if minutes else None,
        "by_type": by_type,
        "per_event": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--events", default=None, help="Tệp nhãn (mặc định <video>.events.json)")
    ap.add_argument("--overlay", action="append", default=[], help="Lớp phủ cho biến thể chạy đơn")
    ap.add_argument("--so-sanh", action="store_true", help="Chạy cả bản gốc (nhu_ban_goc.yaml) và bản cải tiến")
    ap.add_argument("--platform", default="none", choices=["none", "mac", "windows", "auto"],
                    help="Giữ 'none' khi so sánh giữa hai máy để cùng kích thước suy luận")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--depth-every", type=int, default=4, help="30 fps / 4 ~ 7,5 Hz như depth worker")
    ap.add_argument("--min-lead", type=float, default=1.0)
    ap.add_argument("--max-sec", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    events_path = args.events or os.path.splitext(args.video)[0] + ".events.json"
    events = []
    if os.path.exists(events_path):
        with open(events_path, encoding="utf-8") as f:
            events = json.load(f).get("events", [])
    else:
        print(f"[!] Chưa có tệp nhãn {events_path}: chỉ in dòng thời gian cảnh báo.")

    variants = ({"ban_goc": [BASELINE_OVERLAY], "ban_cai_tien": []} if args.so_sanh
                else {"bien_the": args.overlay})
    report = {"video": args.video, "events_file": events_path if events else None, "variants": {}}
    for name, overlays in variants.items():
        print(f"== {name}: {overlays or ['(cấu hình bản cải tiến)']}")
        run = run_pipeline(args.video, overlays, args.platform, args.device, args.depth_every, args.max_sec)
        entry = {"run": run}
        if events:
            entry["score"] = score(run, events, args.min_lead)
        report["variants"][name] = entry
        print(f"   {len(run['alerts'])} cảnh báo / {run['duration_sec']:.1f} s")
        if events:
            s = entry["score"]
            print(f"   recall={s['recall']:.2f} kịp thời={s['timely_rate']:.2f} "
                  f"lead={s['lead_time_median_sec']} cảnh báo thừa/phút={s['false_alarms_per_min']:.2f}")
    out = args.out or os.path.splitext(args.video)[0] + ".danh_gia.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("Đã lưu", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Đánh giá chế độ Cảnh báo theo thời gian trên chuỗi video thật TUM RGB-D.

Nguồn: TUM RGB-D benchmark (Sturm et al., IROS 2012), cvg.cit.tum.de/data/datasets/rgbd-dataset.
Ảnh 640x480 @30Hz kèm độ sâu Kinect (png 16-bit, chia 5000 = mét).

Pipeline chạy theo đúng nhịp runtime: detection + ByteTrack mỗi frame, depth ~8 Hz
(mỗi 4 frame), Synchronizer kiểm tra lệch/tuổi dữ liệu theo timestamp thật,
RiskFSM + AlertAggregator quyết định câu cảnh báo được phát.

Ground truth (độc lập với model):
- Mức vật: khoảng cách thật (phân vị 20 GT trong ROI) của mỗi vật alert_classes trong hành lang.
- Mức frame: "vật cản gần" = phân vị 10 GT trong phần trên 75% của hành lang < ngưỡng
  (bỏ 25% đáy để không tính sàn nhà), bất kể vật có thuộc lớp COCO nào.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from common import (add_config_args, build_depth, load_configs, make_packet, save_json, summarize)


def read_list(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            t, p = line.split()[:2]
            rows.append((float(t), p))
    return rows


def associate(rgb, depth, max_dt=0.02):
    dts = np.array([t for t, _ in depth])
    out = []
    for t, p in rgb:
        j = int(np.argmin(np.abs(dts - t)))
        if abs(dts[j] - t) <= max_dt:
            out.append((t, p, depth[j][1]))
    return out


def corridor_mask(corridor, w, h, skip_bottom=0.25):
    poly = np.array(corridor.polygon(w, h), dtype=np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [poly], 1)
    m[int(h * (1 - skip_bottom)):, :] = 0
    return m.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True, help="Thư mục rgbd_dataset_freiburg*_*")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--depth-every", type=int, default=4, help="30fps/4 ≈ 7.5 Hz như depth worker")
    ap.add_argument("--near-m", type=float, default=1.0)
    ap.add_argument("--camera-height-m", type=float, default=None,
                    help="Ghi đè độ cao camera cho tự hiệu chỉnh độ sâu (TUM: camera cầm tay)")
    add_config_args(ap)
    args = ap.parse_args()

    from src.contracts.risk import RiskLevel
    from src.runtime import factory

    cfg = load_configs(args.overlay, args.platform)
    if args.camera_height_m is not None:
        cfg["app"].setdefault("camera", {})["height_m"] = args.camera_height_m
    # TUM fr1/fr2/fr3 đều 640x480; trường nhìn ngang Kinect ~ 62 độ (fx ~ 517-535 px)
    cfg["app"].setdefault("camera", {})["hfov_deg"] = 62.5  # fx 517-535 px ở 640 px ngang
    stack = factory.build_risk_stack(cfg)
    detector = factory.build_detector(cfg, args.device, stack["class_filter"])
    depth_est = build_depth(cfg, device=args.device)
    tracker, sync = stack["tracker"], stack["synchronizer"]
    fsm, corridor, aggregator = stack["risk_fsm"], stack["corridor"], stack["aggregator"]
    erosion = float(cfg["model"]["depth"].get("roi_erosion_ratio", 0.15))
    scale_rows = []

    frames = associate(read_list(os.path.join(args.seq_dir, "rgb.txt")),
                       read_list(os.path.join(args.seq_dir, "depth.txt")))
    t_base = frames[0][0]
    obj_rows, frame_rows, spoken = [], [], []
    lat_det, lat_depth = [], []
    cmask = None

    for idx, (t, rgb_p, d_p) in enumerate(frames):
        ts = 1000.0 + (t - t_base)  # timestamp giả lập kiểu monotonic
        bgr = cv2.imread(os.path.join(args.seq_dir, rgb_p))
        gt = cv2.imread(os.path.join(args.seq_dir, d_p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000.0
        h, w = gt.shape
        if cmask is None:
            cmask = corridor_mask(corridor, w, h)
        packet = make_packet(bgr, idx, timestamp=ts)

        if idx % args.depth_every == 0:
            t0 = time.monotonic()
            dm = depth_est.estimate(packet)
            lat_depth.append((time.monotonic() - t0) * 1000.0)
            sync.register_depth(dm)
            # Hệ số "tối ưu" của riêng frame này theo GT, để so với hệ số cố định và tự hiệu chỉnh.
            applied = float(dm.preprocess_transform.get("metric_scale") or 1.0)
            raw = dm.values / max(applied, 1e-6)
            ok = (gt > 0.1) & dm.valid_mask & (raw > 0.05)
            if ok.sum() > 1000:
                floor = dm.preprocess_transform.get("floor") or {}
                scale_rows.append({"oracle": float(np.median(gt[ok] / raw[ok])), "applied": applied,
                                   "floor_valid": bool(floor.get("valid", False))})
        det = detector.detect(packet)
        lat_det.append(det.latency_ms)
        tracked = tracker.update(det)
        pair = sync.synchronize(tracked, current_mono=ts)
        assessments = fsm.update(pair, current_mono=ts)
        task = aggregator.aggregate(assessments, current_mono=ts)
        if task is not None:
            spoken.append({"t": t - t_base, "text": task.text})

        any_high = any(a.risk_level == RiskLevel.HIGH for a in assessments)
        g = gt[cmask & (gt > 0.1)]
        near_any = bool(g.size > 500 and np.percentile(g, 10) < args.near_m)
        frame_rows.append({"t": t - t_base, "near_any_gt": near_any, "system_high": any_high,
                           "generic_high": any(a.track_id == -100 for a in assessments),
                           "quality": pair.data_quality.value})

        for d, a in zip(tracked.detections, assessments):
            if not fsm.class_filter.is_alert_candidate(d.class_name):
                continue
            ov = corridor.overlap_ratio(d.bbox, w, h)
            if ov is None or ov < corridor.min_overlap_ratio:
                continue
            b = d.bbox
            x1, y1 = int(b.xmin + b.width * erosion), int(b.ymin + b.height * erosion)
            x2, y2 = int(b.xmax - b.width * erosion), int(b.ymax - b.height * erosion)
            roi_gt = gt[y1:y2, x1:x2]
            roi_gt = roi_gt[roi_gt > 0.1]
            if roi_gt.size < 15:
                continue
            obj_rows.append({"t": t - t_base, "class": d.class_name, "track": d.track_id,
                             "risk": a.risk_level.value, "prox": a.relative_proximity,
                             "gt_dist_m": float(np.percentile(roi_gt, 20))})

    def cls(rows, limit):
        valid = [r for r in rows if r["risk"] not in ("UNDETERMINED",)]
        tp = sum(1 for r in valid if r["risk"] == "HIGH" and r["gt_dist_m"] <= limit)
        fp = sum(1 for r in valid if r["risk"] == "HIGH" and r["gt_dist_m"] > limit)
        fn = sum(1 for r in valid if r["risk"] != "HIGH" and r["gt_dist_m"] <= limit)
        return {"limit_m": limit, "tp": tp, "fp": fp, "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None}

    near_frames = [f for f in frame_rows if f["near_any_gt"]]
    duration = frame_rows[-1]["t"] if frame_rows else 0
    from collections import Counter
    report = {
        "sequence": os.path.basename(args.seq_dir.rstrip("/")),
        "dataset": "TUM RGB-D (cvg.cit.tum.de)",
        "frames": len(frame_rows), "duration_sec": duration,
        "data_quality_counts": dict(Counter(f["quality"] for f in frame_rows)),
        "frame_level": {
            "frames_with_near_obstacle_gt(<%.1fm, any object)" % args.near_m: len(near_frames),
            "of_which_system_HIGH": sum(f["system_high"] for f in near_frames),
            "frames_without_near_obstacle": len(frame_rows) - len(near_frames),
            "of_which_system_HIGH(false_alarm_frames)": sum(f["system_high"] for f in frame_rows if not f["near_any_gt"]),
            "generic_obstacle_frames": sum(f["generic_high"] for f in frame_rows),
            "generic_obstacle_frames_with_near_gt": sum(f["generic_high"] for f in near_frames),
        },
        "object_level": {f"<= {L}m": cls(obj_rows, L) for L in (1.0, 1.5)},
        "object_gt_distance_by_risk": {
            r: summarize([o["gt_dist_m"] for o in obj_rows if o["risk"] == r])
            for r in sorted({o["risk"] for o in obj_rows})
        },
        "alert_classes_seen": dict(Counter(o["class"] for o in obj_rows)),
        "config": {"overlays": cfg["_meta"]["overlays"], "risk_model": fsm.risk_model,
                   "tracker": type(tracker).__name__,
                   "auto_scale": bool(cfg["model"]["depth"].get("auto_scale", {}).get("enabled"))},
        "depth_scale_vs_oracle": {
            "applied_abs_rel_err": summarize([abs(r["applied"] - r["oracle"]) / r["oracle"] for r in scale_rows]),
            "fixed_cfg_abs_rel_err": summarize([
                abs(float(cfg["model"]["depth"].get("metric_scale", 1.0)) - r["oracle"]) / r["oracle"]
                for r in scale_rows]),
            "floor_valid_ratio": (sum(r["floor_valid"] for r in scale_rows) / len(scale_rows)) if scale_rows else None,
        },
        "spoken_alerts": {"count": len(spoken),
                          "per_minute": len(spoken) / (duration / 60) if duration else None,
                          "samples": spoken[:15]},
        "latency_ms": {"detect": summarize(lat_det), "depth": summarize(lat_depth)},
    }
    path = save_json(f"warning_tum_{report['sequence']}{args.tag}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:3500])
    print("saved", path)


if __name__ == "__main__":
    main()

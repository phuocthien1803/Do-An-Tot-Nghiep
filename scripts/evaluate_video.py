#!/usr/bin/env python3
"""
Đánh giá pipeline trên video thực nghiệm hoặc video clip kiểm thử (chạy THỜI GIAN THỰC qua
SystemCoordinator, đúng như app). Bản cải tiến dựng thành phần bằng src/runtime/factory.py và nhận
--overlay để so sánh biến thể; giọng đọc mặc định "console" (in ra màn hình) để chạy được trên CI.
Đánh giá theo sự kiện trên video có nhãn: dùng scripts/danh_gia_su_kien.py.
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.audio.audio_coordinator import AudioCoordinator  # noqa: E402
from src.audio.tts_factory import create_tts_engine  # noqa: E402
from src.camera.camera_manager import CameraManager  # noqa: E402
from src.runtime import factory  # noqa: E402
from src.runtime.config import load_configs  # noqa: E402
from src.runtime.system_coordinator import SystemCoordinator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Đánh giá pipeline trên video")
    parser.add_argument("--video", type=str, default="dummy", help="Đường dẫn tệp video hoặc 'dummy'")
    parser.add_argument("--duration", type=float, default=5.0, help="Thời gian đánh giá (giây)")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--out", type=str, default="evaluation/results/eval_report.json")
    parser.add_argument("--weights", type=str, default=None, help="Ghi đè detection.weights_path")
    parser.add_argument("--overlay", action="append", default=[], help="Lớp phủ cấu hình (lặp lại được)")
    parser.add_argument("--platform", default="none", choices=["none", "mac", "windows", "auto"])
    parser.add_argument("--tts", default="console", help="console | say | sapi")
    args = parser.parse_args()

    print(f"[EVALUATION] Bắt đầu chạy đánh giá trên nguồn '{args.video}' trong {args.duration}s...")
    os.chdir(PROJECT_ROOT)
    cfg = load_configs(PROJECT_ROOT, platform=args.platform, overlays=args.overlay)
    camera = CameraManager(source=args.video, target_fps=30.0)
    stack = factory.build_risk_stack(cfg)
    detector = factory.build_detector(cfg, args.device, stack["class_filter"], args.weights)
    depth_estimator = factory.build_depth(cfg, args.device)
    audio_cfg = dict(cfg["app"].get("audio", {}), engine=args.tts)
    audio_coordinator = AudioCoordinator(tts_engine=create_tts_engine(audio_cfg))

    coordinator = SystemCoordinator(
        camera_manager=camera,
        detector=detector,
        tracker=stack["tracker"],
        depth_estimator=depth_estimator,
        synchronizer=stack["synchronizer"],
        risk_fsm=stack["risk_fsm"],
        alert_aggregator=stack["aggregator"],
        audio_coordinator=audio_coordinator,
        startup_grace_max_sec=cfg["app"].get("runtime", {}).get("startup_grace_max_sec"),
    )

    coordinator.start()
    t_start = time.monotonic()
    total_frames_processed = 0

    try:
        while (time.monotonic() - t_start) < args.duration:
            time.sleep(0.1)
            snap = coordinator.get_runtime_snapshot()
            if snap.get("packet"):
                total_frames_processed = snap["packet"].frame_id
    finally:
        coordinator.stop()

    report = {
        "video_source": args.video,
        "evaluation_duration_sec": args.duration,
        "overlays": [os.path.basename(p) for p in cfg["_meta"]["overlays"]],
        "total_camera_frames": camera.frame_count,
        "total_frames_processed": total_frames_processed,
        "performance_metrics": coordinator.metrics.get_summary()
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[EVALUATION] Hoàn tất đánh giá. Báo cáo đã lưu tại: {args.out}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

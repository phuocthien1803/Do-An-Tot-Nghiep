#!/usr/bin/env python3
"""
Entry point chính của Hệ Thống AI Đa Phương Thức Hỗ Trợ Người Khiếm Thị - BẢN CẢI TIẾN.

Một lõi chung cho hai máy; khác biệt nền tảng nằm trong lớp phủ cấu hình:
    python app.py --platform mac      (ban_mac/cau_hinh_mac.yaml)
    python app.py --platform windows  (ban_windows/cau_hinh_windows.yaml)
    python app.py                     (tự nhận nền tảng đang chạy)
So sánh với hành vi bản gốc trên CÙNG máy:
    python app.py --overlay configs/thu_nghiem/nhu_ban_goc.yaml
"""

import argparse
import os
import sys

from src.camera.camera_manager import CameraManager
from src.audio.tts_factory import create_tts_engine
from src.audio.audio_coordinator import AudioCoordinator
from src.audio.safety_channel import SafetySoundChannel
from src.ocr.ocr_service import OCRService
from src.ocr.image_quality import ImageQualityChecker
from src.vqa.vqa_service import VQAService
from src.runtime.config import PROJECT_ROOT, load_configs, load_yaml  # noqa: F401 (load_yaml giữ tương thích)
from src.runtime.event_log import EventLog
from src.runtime import factory
from src.runtime.system_coordinator import SystemCoordinator
from src.ui.hud_renderer import HUDRenderer
from src.ui.app_runner import AppRunner


def build_ocr_service(model_cfg: dict, device: str) -> OCRService:
    ocr_cfg = model_cfg.get("ocr", {})
    return OCRService(
        languages=ocr_cfg.get("languages", ["vi", "en"]),
        engine=ocr_cfg.get("engine", "easyocr"),
        use_gpu=(device == "mps"),
        min_confidence=float(ocr_cfg.get("min_confidence", 0.35)),
        read_timeout_sec=float(ocr_cfg.get("read_timeout_sec", 10.0)),
        quality_checker=ImageQualityChecker(
            blur_threshold=float(ocr_cfg.get("blur_laplacian_threshold", 50.0)),
            min_brightness=float(ocr_cfg.get("min_brightness", 40.0)),
            max_brightness=float(ocr_cfg.get("max_brightness", 235.0)),
        ),
        filter_garbage=bool(ocr_cfg.get("filter_garbage", False)),
    )


def build_vqa_service(model_cfg: dict, device: str) -> VQAService:
    vqa_cfg = model_cfg.get("vqa", {})
    guard_cfg = vqa_cfg.get("guardrail", {}) or {}
    return VQAService(
        device=device,
        use_vlm=vqa_cfg.get("use_vlm", True),
        backend_name=vqa_cfg.get("backend", "legacy_caption"),
        caption_model_name=vqa_cfg.get("caption_model_name", "Salesforce/blip-image-captioning-base"),
        translation_model_name=vqa_cfg.get("translation_model_name", "Helsinki-NLP/opus-mt-en-vi"),
        lazy_load=True,
        mlx_vlm_config=vqa_cfg.get("mlx_vlm", {}),
        guardrail_mode=guard_cfg.get("mode", "regex"),
        answer_style=vqa_cfg.get("answer_style", "goc"),
    )


def build_voice_input(app_cfg: dict):
    asr = app_cfg.get("asr", {}) or {}
    if not asr.get("enabled", False):
        return None
    from src.asr.voice_input import VoiceInput
    return VoiceInput(
        model_path=asr.get("model_path", "models/weights/faster_whisper_small_model"),
        model_size=asr.get("model_size", "small"),
        language=asr.get("language", "vi"),
        record_sec=float(asr.get("record_sec", 5.0)),
        device=asr.get("device", "cpu"),
        compute_type=asr.get("compute_type", "int8"),
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Hệ thống AI đa phương thức hỗ trợ người khiếm thị (bản cải tiến)")
    parser.add_argument("--source", type=str, default=None, help="Nguồn camera: 0 (webcam), đường dẫn tệp video, hoặc 'dummy'")
    parser.add_argument("--weights", type=str, default=None, help="Đường dẫn trọng số YOLO")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị tính toán: 'mps' hoặc 'cpu'")
    parser.add_argument("--platform", type=str, default="auto", choices=["auto", "mac", "windows", "none"],
                        help="Lớp phủ cấu hình nền tảng (mặc định tự nhận)")
    parser.add_argument("--overlay", action="append", default=[],
                        help="Tệp lớp phủ cấu hình thêm (lặp lại được), ví dụ configs/thu_nghiem/nhu_ban_goc.yaml")
    parser.add_argument("--event-log", type=str, default=None, help="Ghi nhật ký sự kiện JSONL (đo độ trễ)")
    parser.add_argument("--no-gui", action="store_true", help="Chạy ở chế độ không mở cửa sổ giao diện")
    parser.add_argument("--duration", type=float, default=None, help="Thời lượng chạy tối đa (giây)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.chdir(PROJECT_ROOT)
    cfg = load_configs(PROJECT_ROOT, platform=args.platform, overlays=args.overlay)
    app_cfg, model_cfg = cfg["app"], cfg["model"]

    cam_source = args.source if args.source is not None else app_cfg.get("camera", {}).get("source", 0)
    device = args.device if args.device is not None else app_cfg.get("app", {}).get("device", "mps")

    print("=" * 60)
    print("KHỞI ĐỘNG HỆ THỐNG AI HỖ TRỢ NGƯỜI KHIẾM THỊ - BẢN CẢI TIẾN")
    print(f"- Nền tảng / lớp phủ: {cfg['_meta']['platform']} / "
          f"{[os.path.basename(p) for p in cfg['_meta']['overlays']]}")
    print(f"- Nguồn camera: {cam_source}")
    print(f"- Thiết bị tính toán: {device}")
    print(f"- Giao diện GUI: {'Tắt (--no-gui)' if args.no_gui else 'Bật'}")
    print("=" * 60)

    camera_cfg = app_cfg.get("camera", {})
    camera = CameraManager(
        source=cam_source,
        width=camera_cfg.get("width", 640),
        height=camera_cfg.get("height", 480),
        target_fps=camera_cfg.get("fps", 30.0),
        reconnect_delay_sec=float(camera_cfg.get("reconnect_delay_sec", 2.0)),
        max_reconnect_attempts=int(camera_cfg.get("max_reconnect_attempts", 5))
    )

    earcons = factory.build_earcons(cfg)
    stack = factory.build_risk_stack(cfg, earcons)
    detector = factory.build_detector(cfg, device, stack["class_filter"], args.weights)
    depth_estimator = factory.build_depth(cfg, device)

    audio_cfg = app_cfg.get("audio", {})
    audio_coordinator = AudioCoordinator(tts_engine=create_tts_engine(audio_cfg))
    safety_cfg = app_cfg.get("safety", {}) or {}
    safety_channel = None
    if earcons is not None and safety_cfg.get("beeper", False):
        safety_channel = SafetySoundChannel(
            audio_coordinator, earcons,
            ticks_in_warning=bool(safety_cfg.get("ticks_in_warning", True)),
            ticks_in_on_demand=bool(safety_cfg.get("ticks_in_on_demand", True)),
        )

    event_log = EventLog(args.event_log)
    runtime_cfg = app_cfg.get("runtime", {})
    coordinator = SystemCoordinator(
        camera_manager=camera,
        detector=detector,
        tracker=stack["tracker"],
        depth_estimator=depth_estimator,
        synchronizer=stack["synchronizer"],
        risk_fsm=stack["risk_fsm"],
        alert_aggregator=stack["aggregator"],
        audio_coordinator=audio_coordinator,
        ocr_service=build_ocr_service(model_cfg, device),
        vqa_service=build_vqa_service(model_cfg, device),
        watchdog_timeout_sec=float(runtime_cfg.get("watchdog_timeout_sec", 3.0)),
        safety_channel=safety_channel,
        on_demand_high_alert=safety_cfg.get("on_demand_high_alert", "off"),
        earcons=earcons,
        event_log=event_log,
        voice_input=build_voice_input(app_cfg),
        auto_return_warning_sec=float(app_cfg.get("ui", {}).get("auto_return_warning_sec", 0.0)),
        startup_grace_max_sec=runtime_cfg.get("startup_grace_max_sec"),
        min_frame_brightness=float(runtime_cfg.get("min_frame_brightness", 0.0) or 0.0),
    )

    hud_renderer = HUDRenderer(
        target_width=runtime_cfg.get("preview_width", 800),
        target_height=runtime_cfg.get("preview_height", 600),
        config=app_cfg.get("ui", {}),
        walking_corridor=stack["corridor"]
    )
    ui_cfg = app_cfg.get("ui", {})
    runner = AppRunner(
        coordinator=coordinator,
        hud_renderer=hud_renderer,
        enable_gui=(not args.no_gui),
        duration_sec=args.duration,
        headless_keyboard=bool(ui_cfg.get("headless_keyboard", False)),
        global_hotkeys=bool(ui_cfg.get("global_hotkeys", False)),
    )
    try:
        runner.run()
    finally:
        event_log.close()


if __name__ == "__main__":
    sys.exit(main())

"""Tiện ích dùng chung cho bộ đánh giá trên dữ liệu thật.

Các thành phần được dựng giống app.py (cùng file cấu hình, cùng tham số) để
kết quả phản ánh đúng hệ thống đang chạy, không phải một bản mô phỏng.
"""

import json
import os
import sys
import time
import unicodedata
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.contracts.frame_packet import FramePacket  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "evaluation", "results", "real_data")


def load_configs(overlays=None, platform: str = "none") -> Dict[str, dict]:
    """Cấu hình như app; bản cải tiến cho phép ghép lớp phủ (ví dụ configs/thu_nghiem/nhu_ban_goc.yaml)."""
    from src.runtime.config import load_configs as _load_all
    return _load_all(PROJECT_ROOT, platform=platform, overlays=overlays or ())


def add_config_args(parser) -> None:
    """Tham số chung chọn biến thể cấu hình cho mọi benchmark."""
    parser.add_argument("--overlay", action="append", default=[],
                        help="Lớp phủ cấu hình (lặp lại được), ví dụ configs/thu_nghiem/nhu_ban_goc.yaml")
    parser.add_argument("--platform", default="none", choices=["none", "mac", "windows", "auto"])
    parser.add_argument("--tag", default="", help="Hậu tố tên tệp kết quả, ví dụ _goc hoặc _caitien")


def make_packet(image_bgr: np.ndarray, frame_id: int, timestamp: Optional[float] = None) -> FramePacket:
    h, w = image_bgr.shape[:2]
    return FramePacket(
        frame_id=frame_id,
        source_id="eval",
        timestamp_mono=time.monotonic() if timestamp is None else timestamp,
        image=image_bgr,
        original_size=(w, h),
    )


def build_class_filter(cfg):
    from src.detection.class_filter import ClassFilter
    det = cfg["model"].get("detection", {})
    return ClassFilter(
        enabled_classes=det.get("enabled_classes"),
        alert_classes=det.get("alert_classes"),
        custom_vi_names=cfg["fusion"].get("labels_vi"),
    )


def build_detector(cfg, device="mps", confidence=None, class_filter=None):
    from src.detection.yolo_detector import YoloDetector
    det = cfg["model"].get("detection", {})
    return YoloDetector(
        weights_path=det.get("weights_path", "models/weights/yolov8n.pt"),
        class_filter=class_filter or build_class_filter(cfg),
        confidence_threshold=det.get("confidence_threshold", 0.45) if confidence is None else confidence,
        device=device,
    )


def build_depth(cfg, device="mps", model_name=None, representation=None, metric_scale=None):
    """Mặc định dựng đúng như app (src.depth.factory); tham số ghi đè để so sánh biến thể."""
    from src.depth.factory import build_depth_estimator
    dep = dict(cfg["model"].get("depth", {}))
    if model_name is not None:
        dep["hf_repo_id"] = model_name
    if representation is not None:
        dep["representation"] = representation
    if metric_scale is not None:
        dep["metric_scale"] = metric_scale
    return build_depth_estimator({"depth": dep}, device, app_cfg=cfg.get("app"))


def build_risk_stack(cfg):
    """RiskFSM + các phụ thuộc, dựng bằng đúng src/runtime/factory.py như app.py (bản cải tiến)."""
    from src.runtime import factory
    stack = factory.build_risk_stack(cfg)
    return stack["risk_fsm"], stack["roi"], stack["corridor"]


def patch_vision_options() -> None:
    """Vá tạm lỗi Apple Vision trên macOS 27 + pyobjc 12: dict Python rỗng `{}`
    truyền vào initWithData_options_ gây NSInvalidArgumentException. Truyền
    NSDictionary rỗng thì chạy đúng. Chỉ dùng cho đánh giá; mã nguồn chưa sửa."""
    import Foundation
    import Vision
    from src.ocr import vision_engine

    class _VisionProxy:
        def __getattr__(self, name):
            return getattr(Vision, name)

        class VNImageRequestHandler:
            @staticmethod
            def alloc():
                return _HandlerFactory()

    class _HandlerFactory:
        def initWithData_options_(self, data, options):
            return Vision.VNImageRequestHandler.alloc().initWithData_options_(
                data, options if options else Foundation.NSDictionary.dictionary())

    original_init = vision_engine.VisionOCRReader.__init__

    def patched_init(self, languages):
        original_init(self, languages)
        self._vision = _VisionProxy()

    vision_engine.VisionOCRReader.__init__ = patched_init


# ----------------------------- Text metrics -----------------------------

def normalize_text(text: str, strip_accents: bool = False) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    if strip_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        text = text.replace("đ", "d")
    out = []
    for ch in text:
        out.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    return " ".join("".join(out).split())


def levenshtein(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def word_prf(pred_words: Iterable[str], gt_words: Iterable[str]) -> Dict[str, float]:
    from collections import Counter
    p, g = Counter(pred_words), Counter(gt_words)
    tp = sum((p & g).values())
    np_, ng = sum(p.values()), sum(g.values())
    prec = tp / np_ if np_ else (1.0 if ng == 0 else 0.0)
    rec = tp / ng if ng else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "n_pred": np_, "n_gt": ng, "precision": prec, "recall": rec, "f1": f1}


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def save_json(name: str, payload: Dict[str, Any]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
    return path

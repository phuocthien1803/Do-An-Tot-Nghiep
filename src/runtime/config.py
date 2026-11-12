"""Nạp cấu hình YAML, chọn nền tảng (Mac/Windows) và ghép các lớp phủ cấu hình.

Bản cải tiến dùng CHUNG một lõi cho cả hai máy để kết quả so sánh công bằng; phần khác nhau
giữa Mac và Windows (giọng đọc, OCR, VQA, thiết bị tính toán, kích thước suy luận) nằm trong
lớp phủ `ban_mac/cau_hinh_mac.yaml` và `ban_windows/cau_hinh_windows.yaml`.

Định dạng một tệp lớp phủ:

    app_config:   {...}   # ghép vào configs/app_config.yaml
    model_config: {...}   # ghép vào configs/model_config.yaml
    fusion_rules: {...}   # ghép vào configs/fusion_rules.yaml

Từ điển được ghép đệ quy; danh sách và giá trị đơn được thay thế toàn bộ.
"""
import copy
import os
import sys
from typing import Any, Dict, Iterable, Optional

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_FILES = {
    "app": "app_config.yaml",
    "model": "model_config.yaml",
    "fusion": "fusion_rules.yaml",
}
OVERLAY_KEYS = {"app_config": "app", "model_config": "model", "fusion_rules": "fusion"}
PLATFORM_OVERLAYS = {
    "mac": os.path.join("ban_mac", "cau_hinh_mac.yaml"),
    "windows": os.path.join("ban_windows", "cau_hinh_windows.yaml"),
}


def load_yaml(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Trả về bản sao của base đã ghép overlay (không sửa đối số)."""
    result = copy.deepcopy(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def detect_platform() -> str:
    if sys.platform == "darwin":
        return "mac"
    if sys.platform.startswith("win"):
        return "windows"
    return "linux"


def resolve_platform(name: Optional[str]) -> Optional[str]:
    """'auto' -> nền tảng đang chạy; 'none' -> không dùng lớp phủ nền tảng."""
    if name is None or name == "auto":
        detected = detect_platform()
        return detected if detected in PLATFORM_OVERLAYS else None
    if name == "none":
        return None
    if name not in PLATFORM_OVERLAYS:
        raise ValueError(f"Nền tảng không hỗ trợ: {name} (mac | windows | auto | none)")
    return name


def apply_overlay(configs: Dict[str, dict], overlay: Dict[str, Any]) -> Dict[str, dict]:
    merged = dict(configs)
    for key, value in (overlay or {}).items():
        target = OVERLAY_KEYS.get(key)
        if target is None:
            if key in ("mo_ta", "description"):
                continue
            raise ValueError(f"Khoá lớp phủ không hợp lệ: {key} (dùng {', '.join(OVERLAY_KEYS)})")
        merged[target] = deep_merge(merged.get(target, {}), value or {})
    return merged


def load_configs(
    root: str = PROJECT_ROOT,
    platform: Optional[str] = "none",
    overlays: Iterable[str] = (),
) -> Dict[str, dict]:
    """Nạp 3 tệp cấu hình gốc, rồi ghép lớp phủ nền tảng và các lớp phủ thí nghiệm theo thứ tự."""
    configs = {name: load_yaml(os.path.join(root, "configs", fname))
               for name, fname in CONFIG_FILES.items()}
    applied = []
    resolved = resolve_platform(platform)
    if resolved is not None:
        path = os.path.join(root, PLATFORM_OVERLAYS[resolved])
        if os.path.exists(path):
            configs = apply_overlay(configs, load_yaml(path))
            applied.append(path)
    for overlay_path in overlays or ():
        path = overlay_path if os.path.isabs(overlay_path) else os.path.join(root, overlay_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy tệp lớp phủ cấu hình: {overlay_path}")
        configs = apply_overlay(configs, load_yaml(path))
        applied.append(path)
    configs["_meta"] = {"platform": resolved or "none", "overlays": applied}
    return configs

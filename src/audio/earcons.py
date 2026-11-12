"""Âm báo ngắn (earcon) lập thể sinh bằng numpy, không cần tệp âm thanh có bản quyền.

Âm báo phát ngay (< 100 ms sau khi có quyết định) TRƯỚC lời nói 1,5-2 s, và được đặt lệch
trái/phải theo hướng vật để người dùng định hướng ngay cả khi chưa nghe hết câu.

- alert_<hướng>.wav : hai tiếng bíp cao 1 kHz (vật cản trong hành lang).
- drop_<hướng>.wav  : tiếng trượt giảm 900 -> 450 Hz (sàn hụt: bậc xuống/hố), dễ phân biệt.
- tick_<hướng>.wav  : tiếng tích 40 ms cho nhịp bíp kiểu cảm biến lùi xe (nhanh dần theo TTC).
- ack.wav / reject.wav : phản hồi phím được nhận / bị từ chối.
"""
import math
import os
import tempfile
import wave
from typing import Dict, Optional

import numpy as np

from src.contracts.risk import Direction

SAMPLE_RATE = 22050
GUIDE_POSITIONS = 7
PAN = {  # (trái, phải)
    Direction.LEFT: (1.0, 0.25),
    Direction.CENTER: (0.9, 0.9),
    Direction.RIGHT: (0.25, 1.0),
}


def _tone(freq_start: float, duration: float, freq_end: Optional[float] = None,
          volume: float = 0.6) -> np.ndarray:
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n) / SAMPLE_RATE
    f1 = freq_end if freq_end is not None else freq_start
    phase = 2 * math.pi * (freq_start * t + (f1 - freq_start) * t * t / (2 * duration))
    env = np.ones(n)
    ramp = max(1, int(0.005 * SAMPLE_RATE))  # 5 ms tránh tiếng "bụp"
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return volume * np.sin(phase) * env


def _silence(duration: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * duration))


def _write(path: str, mono: np.ndarray, pan=(0.9, 0.9)) -> None:
    stereo = np.stack([mono * pan[0], mono * pan[1]], axis=1)
    data = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data.tobytes())


class EarconLibrary:
    """Sinh (một lần) và trả đường dẫn các tệp earcon trong thư mục cache."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "second_eye_earcons")
        self._paths: Dict[str, str] = {}

    def ensure(self) -> "EarconLibrary":
        os.makedirs(self.cache_dir, exist_ok=True)
        beep = np.concatenate([_tone(1000, 0.06), _silence(0.04), _tone(1000, 0.06)])
        sweep = _tone(900, 0.22, freq_end=450, volume=0.7)
        tick = _tone(1400, 0.04, volume=0.5)
        for direction, pan in PAN.items():
            key = direction.value.lower()
            for name, sound in (("alert", beep), ("drop", sweep), ("tick", tick)):
                path = os.path.join(self.cache_dir, f"{name}_{key}.wav")
                if not os.path.exists(path):
                    _write(path, sound, pan)
                self._paths[f"{name}_{key}"] = path
        # Âm dẫn hướng cho chế độ Tìm đồ: 7 vị trí trái->phải (cân bằng công suất), tiếng
        # trầm khi lệch hướng, tiếng cao khi vật ở ngay phía trước. Âm sắc khác tiếng tích an toàn.
        for i in range(GUIDE_POSITIONS):
            p = -1.0 + 2.0 * i / (GUIDE_POSITIONS - 1)
            pan = (math.cos((p + 1) * math.pi / 4), math.sin((p + 1) * math.pi / 4))
            for pitch, freq in (("lo", 520), ("hi", 1250)):
                path = os.path.join(self.cache_dir, f"guide_{i}_{pitch}.wav")
                if not os.path.exists(path):
                    _write(path, _tone(freq, 0.07, volume=0.55), pan)
                self._paths[f"guide_{i}_{pitch}"] = path
        for name, sound in (("ack", np.concatenate([_tone(660, 0.05), _tone(990, 0.06)])),
                            ("reject", _tone(220, 0.18, volume=0.7))):
            path = os.path.join(self.cache_dir, f"{name}.wav")
            if not os.path.exists(path):
                _write(path, sound)
            self._paths[name] = path
        return self

    def path(self, name: str) -> Optional[str]:
        if not self._paths:
            self.ensure()
        return self._paths.get(name)

    def for_alert(self, direction: Direction, hazard_type: str = "object") -> Optional[str]:
        kind = "drop" if hazard_type == "drop_off" else "alert"
        return self.path(f"{kind}_{direction.value.lower()}")

    def tick(self, direction: Direction) -> Optional[str]:
        return self.path(f"tick_{direction.value.lower()}")

    def guide(self, offset: float, on_target: bool) -> Optional[str]:
        """offset trong [-1, 1]: -1 mép trái ảnh, 0 chính giữa, 1 mép phải."""
        i = int(round((max(-1.0, min(1.0, offset)) + 1.0) / 2.0 * (GUIDE_POSITIONS - 1)))
        return self.path(f"guide_{i}_{'hi' if on_target else 'lo'}")

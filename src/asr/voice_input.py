"""Hỏi bằng giọng nói, chạy offline: ghi âm micro (sounddevice) + nhận dạng (faster-whisper).

Bản gốc chỉ hỏi được một câu cố định ("Phía trước có gì?") bằng phím Q. Mô-đun này cho phép
bấm V rồi nói câu hỏi tự do, hoặc ra lệnh "đọc chữ", "trợ giúp", "trạng thái", "về chế độ cảnh báo".
Cả hai thư viện là phụ thuộc TUỲ CHỌN và được nạp lười; thiếu thư viện/model thì hệ thống nói rõ
lý do và người dùng vẫn dùng phím Q/Space như cũ. Không gửi âm thanh ra ngoài máy.

Cài đặt (cần mạng một lần):
    python -m pip install sounddevice faster-whisper
    python scripts/download_models.py --asr
"""
import os
import re
import threading
import unicodedata
from typing import Optional, Tuple

import numpy as np

from src.runtime.model_loading import is_offline_mode


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def route_voice_command(text: str) -> Tuple[str, str]:
    """Trả về (lệnh, câu hỏi): lệnh thuộc ocr | help | status | warning | vqa | empty."""
    t = _norm(text).strip(" .?!,")
    if not t:
        return "empty", ""
    if re.search(r"\b(?:trợ giúp|hướng dẫn|giúp tôi dùng)\b", t):
        return "help", t
    if re.search(r"\btrạng thái\b", t):
        return "status", t
    if re.search(r"\b(?:chế độ cảnh báo|quay lại cảnh báo|về cảnh báo|bật cảnh báo)\b", t):
        return "warning", t
    if re.search(r"\bđọc\b", t) and re.search(
            r"\b(?:chữ|biển|bảng|giấy|nhãn|văn bản|thông báo|hoá đơn|hóa đơn|thực đơn|tài liệu)\b", t):
        return "ocr", t
    question = t[0].upper() + t[1:]
    if not question.endswith("?"):
        question += "?"
    return "vqa", question


class VoiceInput:
    def __init__(
        self,
        model_path: str = "models/weights/faster_whisper_small_model",
        model_size: str = "small",
        language: str = "vi",
        record_sec: float = 5.0,
        sample_rate: int = 16000,
        device: str = "cpu",
        compute_type: str = "int8",
        min_rms: float = 0.004,
    ):
        self.model_path = model_path
        self.model_size = model_size
        self.language = language
        self.record_sec = record_sec
        self.sample_rate = sample_rate
        self.device = device
        self.compute_type = compute_type
        self.min_rms = min_rms
        self._model = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    @staticmethod
    def dependency_error() -> Optional[str]:
        missing = []
        for module in ("sounddevice", "faster_whisper"):
            try:
                __import__(module)
            except Exception:
                missing.append(module.replace("_", "-"))
        if missing:
            return ("Chưa cài nhận dạng giọng nói (thiếu " + ", ".join(missing) +
                    "). Hãy dùng phím Q để hỏi.")
        return None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _ensure_model(self) -> bool:
        with self._lock:
            if self._model is not None:
                return True
            error = self.dependency_error()
            if error:
                self._load_error = error
                return False
            try:
                from faster_whisper import WhisperModel
                source = self.model_path if os.path.isdir(self.model_path) else self.model_size
                self._model = WhisperModel(source, device=self.device, compute_type=self.compute_type,
                                           local_files_only=is_offline_mode())
                self._load_error = None
                return True
            except Exception as exc:
                self._load_error = ("Chưa có model nhận dạng giọng nói trên máy. "
                                    f"Hãy chạy scripts/download_models.py --asr. ({exc})")
                return False

    def record(self, seconds: Optional[float] = None) -> np.ndarray:
        import sounddevice as sd
        duration = seconds or self.record_sec
        audio = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate,
                       channels=1, dtype="float32")
        sd.wait()
        return audio.reshape(-1)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0 or float(np.sqrt(np.mean(np.square(audio)))) < self.min_rms:
            return ""  # quá nhỏ: coi như không nói gì, tránh Whisper "bịa" chữ trên khoảng lặng
        segments, _ = self._model.transcribe(audio, language=self.language, beam_size=1,
                                             vad_filter=True, condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()

    def listen(self) -> Tuple[Optional[str], Optional[str]]:
        """Ghi âm rồi nhận dạng. Trả về (văn bản, lỗi)."""
        if not self._ensure_model():
            return None, self._load_error
        try:
            return self.transcribe(self.record()), None
        except Exception as exc:
            return None, f"Không ghi âm được từ micro ({exc})."

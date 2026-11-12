"""Chọn bộ đọc giọng nói theo `audio.engine`: say (macOS) | sapi (Windows) | console (in ra màn hình)."""
import threading
import time
from typing import Optional

from src.audio.tts_engine import TTSEngine


class ConsoleTTSEngine(TTSEngine):
    """In lời nói ra console và giả lập thời lượng đọc; dùng cho máy không có giọng đọc/CI."""

    WORDS_PER_SEC = 3.0

    def __init__(self, voice: str = "console", speech_rate_wpm: int = 200, quiet: bool = False):
        super().__init__(voice=voice, speech_rate_wpm=speech_rate_wpm)
        self.quiet = quiet
        self._until = 0.0
        self._lock = threading.Lock()
        self.spoken = []

    def speak(self, text: str) -> bool:
        with self._lock:
            self.spoken.append(text)
            self._until = time.monotonic() + max(0.3, len(text.split()) / self.WORDS_PER_SEC)
        if not self.quiet:
            print(f"[TIẾNG NÓI] {text}")
        return True

    def play_sound(self, sound_file: str) -> bool:
        return True

    def play_earcon(self, sound_file: str) -> bool:
        if not self.quiet:
            print(f"[ÂM BÁO] {sound_file}")
        return True

    def stop_earcon(self) -> None:
        pass

    def stop(self) -> None:
        with self._lock:
            self._until = 0.0

    def is_speaking(self) -> bool:
        with self._lock:
            return time.monotonic() < self._until

    def wait_until_done(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.is_speaking():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.02)
        return True


def create_tts_engine(audio_cfg: dict) -> TTSEngine:
    engine = str(audio_cfg.get("engine", "say")).lower()
    voice = audio_cfg.get("voice", "Linh")
    rate = int(audio_cfg.get("speech_rate_wpm", 200))
    if engine == "sapi":
        from ban_windows.windows_tts import WindowsTTSEngine
        return WindowsTTSEngine(voice=voice, speech_rate_wpm=rate)
    if engine in ("console", "dummy"):
        return ConsoleTTSEngine(speech_rate_wpm=rate)
    return TTSEngine(voice=voice, speech_rate_wpm=rate)

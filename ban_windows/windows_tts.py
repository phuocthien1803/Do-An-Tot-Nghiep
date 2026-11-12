"""[BẢN WINDOWS] Bộ đọc giọng nói thay cho 'say'/'afplay' của macOS.

Kế thừa src.audio.tts_engine.TTSEngine và chỉ thay cách tạo âm thanh, nên AudioCoordinator,
cơ chế ngắt tiếng và chờ phát xong giữ nguyên như trên macOS.
- Lời nói: SAPI qua cscript + sapi_tts.vbs, tự chọn giọng tiếng Việt OneCore ("Microsoft An").
- Tệp WAV (âm báo trước lời cảnh báo, tiếng bíp an toàn): winsound trong tiến trình Python,
  độ trễ thấp hơn nhiều so với khởi động cscript (~150 ms).
Chọn bằng `audio.engine: sapi` trong ban_windows/cau_hinh_windows.yaml.
"""
import os
import subprocess
import threading
from typing import List, Optional

from src.audio.tts_engine import TTSEngine

try:
    import winsound
except ImportError:  # chạy test trên Linux/macOS: winsound không có, mock trong test
    winsound = None

SAPI_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sapi_tts.vbs")
# Giọng macOS mặc định; trên Windows tên này không tồn tại nên script tự chọn giọng viVN.
MACOS_DEFAULT_VOICE = "Linh"


class WindowsTTSEngine(TTSEngine):
    def __init__(self, voice: str = "Linh", speech_rate_wpm: int = 200):
        super().__init__(voice=voice, speech_rate_wpm=speech_rate_wpm)
        self._sound_thread: Optional[threading.Thread] = None

    @staticmethod
    def sapi_rate(speech_rate_wpm: int) -> int:
        """Quy đổi từ/phút (macOS) sang thang SAPI -10..10 (0 ≈ 180 từ/phút)."""
        return max(-10, min(10, int(round((speech_rate_wpm - 180) / 20.0))))

    def speak_command(self, text: str) -> List[str]:
        hint = "" if self.voice == MACOS_DEFAULT_VOICE else self.voice
        return ["cscript", "//nologo", SAPI_SCRIPT, "say", hint,
                str(self.sapi_rate(self.speech_rate_wpm)), text]

    def speak(self, text: str) -> bool:
        self.stop()
        try:
            self._current_process = subprocess.Popen(
                self.speak_command(text), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            print(f"[WindowsTTSEngine] Lỗi khởi chạy cscript: {e}")
            return False

    def play_sound(self, sound_file: str) -> bool:
        if winsound is None or not os.path.exists(sound_file):
            return False
        self.stop()
        path = os.path.abspath(sound_file)
        flags = winsound.SND_FILENAME | winsound.SND_NODEFAULT
        self._sound_thread = threading.Thread(target=winsound.PlaySound, args=(path, flags),
                                              name="WinSound", daemon=True)
        self._sound_thread.start()
        return True

    def play_earcon(self, sound_file: str) -> bool:
        if winsound is None or not sound_file or not os.path.exists(sound_file):
            return False
        try:
            winsound.PlaySound(os.path.abspath(sound_file),
                               winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            return True
        except Exception as e:
            print(f"[WindowsTTSEngine] Lỗi phát âm báo: {e}")
            return False

    def stop_earcon(self) -> None:
        if winsound is not None and (self._sound_thread is None or not self._sound_thread.is_alive()):
            try:
                winsound.PlaySound(None, 0)
            except Exception:
                pass

    def stop(self) -> None:
        if self._sound_thread is not None and self._sound_thread.is_alive() and winsound is not None:
            try:
                winsound.PlaySound(None, 0)
            except Exception:
                pass
        self._sound_thread = None
        super().stop()

    def is_speaking(self) -> bool:
        thread_alive = self._sound_thread is not None and self._sound_thread.is_alive()
        return thread_alive or super().is_speaking()

    def wait_until_done(self, timeout: Optional[float] = None) -> bool:
        thread = self._sound_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
            return not thread.is_alive()
        return super().wait_until_done(timeout)

import subprocess
import os
import signal
import time
from typing import Optional

class TTSEngine:
    """
    Trình phát âm thanh và tổng hợp giọng nói bản địa trên macOS.
    Tuân thủ mục 6 của Kế hoạch:
    - Sử dụng lệnh 'say' với giọng tiếng Việt 'Linh' offline, độ trễ thấp.
    - Phát các tệp WAV cảnh báo cố định không độ trễ bằng 'afplay'.
    - Cho phép ngắt tiếng tức thì (< 50ms) bằng cách gửi tín hiệu SIGTERM/SIGKILL tới tiến trình đang phát.
    """

    def __init__(self, voice: str = "Linh", speech_rate_wpm: int = 200):
        self.voice = voice
        self.speech_rate_wpm = speech_rate_wpm
        self._current_process: Optional[subprocess.Popen] = None
        self._earcon_process: Optional[subprocess.Popen] = None

    def speak(self, text: str) -> bool:
        """Phát âm một chuỗi văn bản bằng giọng tiếng Việt macOS."""
        self.stop() # Ngắt âm thanh trước đó nếu đang nói

        cmd = ["say", "-v", self.voice, "-r", str(self.speech_rate_wpm), text]
        try:
            self._current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except Exception as e:
            print(f"[TTSEngine] Lỗi khởi chạy lệnh say: {e}")
            return False

    def play_sound(self, sound_file: str) -> bool:
        """Phát một tệp âm thanh WAV bằng 'afplay'."""
        if not os.path.exists(sound_file):
            return False

        self.stop()
        try:
            self._current_process = subprocess.Popen(
                ["afplay", sound_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return True
        except Exception as e:
            print(f"[TTSEngine] Lỗi khởi chạy afplay: {e}")
            return False

    def play_earcon(self, sound_file: str) -> bool:
        """Bản cải tiến: phát âm báo ngắn KHÔNG chặn và KHÔNG ngắt lời nói đang phát.

        Dùng tiến trình afplay riêng nên chạy song song với 'say' (ví dụ tiếng bíp an toàn
        trong lúc đang đọc kết quả OCR/VQA). Âm báo trước đó (nếu còn) bị thay thế.
        """
        if not sound_file or not os.path.exists(sound_file):
            return False
        self.stop_earcon()
        try:
            self._earcon_process = subprocess.Popen(
                ["afplay", sound_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            print(f"[TTSEngine] Lỗi phát âm báo: {e}")
            return False

    def stop_earcon(self) -> None:
        proc = self._earcon_process
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        self._earcon_process = None

    def stop(self) -> None:
        """Ngắt ngay lập tức âm thanh đang phát."""
        proc = self._current_process
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=0.04)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._current_process = None

    def is_speaking(self) -> bool:
        """Kiểm tra xem âm thanh có đang được phát hay không."""
        return self._current_process is not None and self._current_process.poll() is None

    def wait_until_done(self, timeout: Optional[float] = None) -> bool:
        """Đợi cho đến khi phát âm thanh xong."""
        proc = self._current_process
        if proc is None:
            return True
        try:
            proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False

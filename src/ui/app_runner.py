import queue
import sys
import threading
import time
from typing import Callable, Dict, Optional

import cv2

from src.runtime.system_coordinator import SystemCoordinator
from src.ui.hud_renderer import HUDRenderer

DEFAULT_VQA_QUESTION = "Phía trước có gì?"

# Lệnh chung cho mọi nguồn phím (cửa sổ OpenCV, bàn phím console khi --no-gui, phím tắt toàn hệ thống).
KEY_COMMANDS: Dict[str, str] = {
    " ": "ocr", "o": "ocr",
    "q": "vqa",
    "v": "voice",
    "c": "warning",
    "s": "stop",
    "h": "help", "?": "help",
    "i": "status",
    "\x1b": "quit", "x": "quit",
}


class AppRunner:
    """
    Vòng lặp tương tác giao diện và bắt sự kiện phím bấm của ứng dụng.

    Bản cải tiến:
    - Thêm phím H (trợ giúp bằng giọng nói), I (trạng thái), V (hỏi bằng giọng nói).
    - headless_keyboard: khi chạy --no-gui vẫn điều khiển được bằng bàn phím console
      (bản gốc --no-gui không có cách nào gọi OCR/VQA).
    - global_hotkeys: phím tắt toàn hệ thống qua pynput (tuỳ chọn) - bấm được kể cả khi cửa sổ
      không được chọn; người khiếm thị không thể biết cửa sổ nào đang có focus.
    """

    def __init__(
        self,
        coordinator: SystemCoordinator,
        hud_renderer: Optional[HUDRenderer] = None,
        enable_gui: bool = True,
        duration_sec: Optional[float] = None,
        headless_keyboard: bool = False,
        global_hotkeys: bool = False,
    ):
        self.coordinator = coordinator
        self.hud_renderer = hud_renderer or HUDRenderer()
        self.enable_gui = enable_gui
        self.duration_sec = duration_sec
        self.headless_keyboard = headless_keyboard
        self.global_hotkeys = global_hotkeys
        self._commands: "queue.Queue[str]" = queue.Queue()
        self._hotkey_listener = None

    # ---------- Lệnh ----------

    def dispatch(self, command: str) -> bool:
        """Thực thi một lệnh; trả về False nếu là lệnh thoát."""
        actions: Dict[str, Callable[[], object]] = {
            "ocr": self.coordinator.trigger_ocr,
            # "Mô tả khung cảnh phía trước" khiến Qwen2.5-VL-3B-4bit sinh câu ảo giác cố định
            # trên ảnh phòng khách (NYU); câu dưới đây là câu đã được đánh giá.
            "vqa": lambda: self.coordinator.trigger_vqa(DEFAULT_VQA_QUESTION),
            "voice": getattr(self.coordinator, "trigger_voice_question", lambda: None),
            "warning": self.coordinator.enter_warning_mode,
            "stop": self.coordinator.stop_speech,
            "help": getattr(self.coordinator, "speak_help", lambda: None),
            "status": getattr(self.coordinator, "speak_status", lambda: None),
        }
        if command == "quit":
            print("[AppRunner] Thoát theo yêu cầu người dùng...")
            return False
        action = actions.get(command)
        if action is not None:
            print(f"[AppRunner] Lệnh: {command}")
            action()
        return True

    @staticmethod
    def command_for_key(key: int) -> Optional[str]:
        if key < 0 or key == 255:
            return None
        return KEY_COMMANDS.get(chr(key).lower()) if key < 256 else None

    # ---------- Nguồn phím phụ ----------

    def _console_reader(self) -> None:
        """Đọc phím từ console khi chạy --no-gui (Windows: từng phím; nơi khác: theo dòng)."""
        try:
            import msvcrt  # Windows
            while True:
                ch = msvcrt.getwch()
                cmd = KEY_COMMANDS.get(ch.lower())
                if cmd:
                    self._commands.put(cmd)
                    if cmd == "quit":
                        return
        except ImportError:
            for line in sys.stdin:
                text = line.rstrip("\n")
                key = text[:1].lower() if text else " "
                cmd = KEY_COMMANDS.get(key) or ("quit" if text.lower() in ("esc", "quit", "thoat") else None)
                if cmd:
                    self._commands.put(cmd)
                    if cmd == "quit":
                        return

    def _start_global_hotkeys(self) -> None:
        try:
            from pynput import keyboard
        except Exception:
            print("[AppRunner] Chưa cài pynput: phím tắt toàn hệ thống tắt (python -m pip install pynput).")
            return

        def on_press(key):
            try:
                ch = key.char
            except AttributeError:
                ch = " " if key == keyboard.Key.space else ("\x1b" if key == keyboard.Key.esc else None)
            if ch:
                cmd = KEY_COMMANDS.get(ch.lower())
                if cmd:
                    self._commands.put(cmd)

        self._hotkey_listener = keyboard.Listener(on_press=on_press)
        self._hotkey_listener.daemon = True
        self._hotkey_listener.start()
        print("[AppRunner] Đã bật phím tắt toàn hệ thống (không cần chọn cửa sổ).")

    def _drain_commands(self) -> bool:
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return True
            if not self.dispatch(cmd):
                return False

    # ---------- Vòng lặp ----------

    def run(self) -> None:
        print("[AppRunner] Khởi động hệ thống...")
        self.coordinator.start()
        t_start = time.monotonic()

        window_name = "Do-An-Tot-Nghiep - Multimodal Assistive System"
        if self.enable_gui:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        elif self.headless_keyboard:
            print("[AppRunner] Không có cửa sổ: bấm phím trong console "
                  "(cách/O=đọc chữ, Q=hỏi, V=giọng nói, C=cảnh báo, S=dừng, H=trợ giúp, I=trạng thái, X=thoát).")
            threading.Thread(target=self._console_reader, name="ConsoleKeys", daemon=True).start()
        if self.global_hotkeys:
            self._start_global_hotkeys()

        try:
            while True:
                # Kiểm tra giới hạn thời gian chạy nếu có
                if self.duration_sec and (time.monotonic() - t_start) >= self.duration_sec:
                    print(f"[AppRunner] Đã đạt thời lượng chỉ định ({self.duration_sec}s), dừng ứng dụng.")
                    break
                if not self._drain_commands():
                    break

                if self.enable_gui:
                    snapshot = self.coordinator.get_runtime_snapshot()
                    rendered_frame = self.hud_renderer.render(snapshot)
                    cv2.imshow(window_name, rendered_frame)
                    command = self.command_for_key(cv2.waitKey(1) & 0xFF)
                    if command and not self.dispatch(command):
                        break
                else:
                    # Chế độ headless (không có GUI)
                    time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n[AppRunner] Nhận tín hiệu KeyboardInterrupt, đang dừng...")
        finally:
            if self._hotkey_listener is not None:
                try:
                    self._hotkey_listener.stop()
                except Exception:
                    pass
            self.coordinator.stop()
            if self.enable_gui:
                cv2.destroyAllWindows()
            print("[AppRunner] Đã dừng toàn bộ hệ thống.")

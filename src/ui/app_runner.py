import queue
import sys
import threading
import time
from typing import Callable, Dict, Optional

import cv2

from src.runtime.system_coordinator import SystemCoordinator
from src.ui.hud_renderer import HUDRenderer
from src.ui.one_button import DEFAULT_ACTIONS, ButtonGestureDecoder, gesture_to_command

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
    "f": "find",
    "\x1b": "quit", "x": "quit",
}
# Phím của nút bấm cầm tay (nút chụp ảnh Bluetooth thường gửi "tăng âm lượng" hoặc Enter).
DEFAULT_BUTTON_KEYS = ("b", "enter", "media_volume_up", "media_play_pause")
_OPENCV_KEY_NAMES = {13: "enter", 10: "enter"}


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
        one_button: Optional[dict] = None,
    ):
        self.coordinator = coordinator
        self.hud_renderer = hud_renderer or HUDRenderer()
        self.enable_gui = enable_gui
        self.duration_sec = duration_sec
        self.headless_keyboard = headless_keyboard
        self.global_hotkeys = global_hotkeys
        self._commands: "queue.Queue[str]" = queue.Queue()
        self._hotkey_listener = None
        cfg = one_button or {}
        self.one_button_enabled = bool(cfg.get("enabled", False))
        self.button_keys = {str(k).lower() for k in cfg.get("keys", DEFAULT_BUTTON_KEYS)}
        self.button_actions = dict(DEFAULT_ACTIONS, **(cfg.get("actions") or {}))
        self.decoder = ButtonGestureDecoder(
            multi_click_sec=float(cfg.get("multi_click_sec", 0.45)),
            long_press_sec=float(cfg.get("long_press_sec", 0.7)))
        self._button_events: "queue.Queue[tuple]" = queue.Queue()

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
            "find": getattr(self.coordinator, "find_next", lambda: None),
            "stop_or_warning": getattr(self.coordinator, "stop_or_warning", lambda: None),
        }
        if command == "quit":
            print("[AppRunner] Thoát theo yêu cầu người dùng...")
            return False
        action = actions.get(command)
        if action is not None:
            print(f"[AppRunner] Lệnh: {command}")
            action()
        return True

    # ---------- Nút bấm một phím ----------

    def is_button_key(self, name: Optional[str]) -> bool:
        return self.one_button_enabled and name is not None and name.lower() in self.button_keys

    def button_event(self, kind: str, t: Optional[float] = None) -> None:
        """kind: 'press' | 'release'; an toàn khi gọi từ luồng khác (pynput)."""
        self._button_events.put((kind, time.monotonic() if t is None else t))

    def _process_button(self, now: float) -> bool:
        gestures = []
        while True:
            try:
                kind, t = self._button_events.get_nowait()
            except queue.Empty:
                break
            g = self.decoder.press(t) if kind == "press" else self.decoder.release(t)
            if g:
                gestures.append(g)
        g = self.decoder.poll(now)
        if g:
            gestures.append(g)
        for gesture in gestures:
            command = gesture_to_command(gesture, self.button_actions)
            print(f"[AppRunner] Nút bấm: {gesture} -> {command}")
            if command and not self.dispatch(command):
                return False
        return True

    @staticmethod
    def key_name(key: int) -> Optional[str]:
        if key < 0 or key == 255:
            return None
        return _OPENCV_KEY_NAMES.get(key) or (chr(key).lower() if key < 256 else None)

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
                if self.is_button_key("enter" if ch == "\r" else ch.lower()):
                    self.button_event("press")
                    continue
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

        def name_of(key):
            try:
                return key.char.lower() if key.char else None
            except AttributeError:
                return {keyboard.Key.space: " ", keyboard.Key.esc: "\x1b"}.get(key, getattr(key, "name", None))

        def on_press(key):
            name = name_of(key)
            if self.is_button_key(name):
                self.button_event("press")
            elif name:
                cmd = KEY_COMMANDS.get(name)
                if cmd:
                    self._commands.put(cmd)

        def on_release(key):
            if self.is_button_key(name_of(key)):
                self.button_event("release")

        self._hotkey_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
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
                if self.one_button_enabled and not self._process_button(time.monotonic()):
                    break

                if self.enable_gui:
                    snapshot = self.coordinator.get_runtime_snapshot()
                    rendered_frame = self.hud_renderer.render(snapshot)
                    cv2.imshow(window_name, rendered_frame)
                    raw = cv2.waitKey(1) & 0xFF
                    if self.is_button_key(self.key_name(raw)):
                        self.button_event("press")  # OpenCV không có sự kiện nhả phím
                        continue
                    command = self.command_for_key(raw)
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

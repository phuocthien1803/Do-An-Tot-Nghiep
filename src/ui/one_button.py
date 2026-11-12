"""Điều khiển MỘT NÚT cho người khiếm thị (nút chụp ảnh Bluetooth, remote nhạc, hoặc một phím bất kỳ).

Người mù không nhìn thấy bàn phím/màn hình; một nút cầm tay (vẫn cầm gậy bằng tay kia) với cử chỉ
quen thuộc như tai nghe không dây là đủ:
    bấm 1 lần  -> hỏi "Phía trước có gì?"
    bấm 2 lần  -> đọc chữ
    bấm 3 lần  -> tìm đồ (mỗi lần bấm 3 lần chuyển sang món tiếp theo)
    giữ ≥ 0,7 s -> im lặng; nếu đang im thì về chế độ cảnh báo
(ánh xạ đổi được trong cấu hình ui.one_button.actions).

Hai nguồn sự kiện:
- Có cả nhả phím (pynput, nút Bluetooth): giữ lâu đo bằng thời gian nhấn-nhả.
- Chỉ có nhấn (cửa sổ OpenCV): giữ phím sinh chuỗi nhấn lặp lại cách nhau < repeat_gap_sec
  (auto-repeat của bàn phím) -> coi là đang giữ.
"""
from typing import Dict, Optional

DEFAULT_ACTIONS = {"single": "vqa", "double": "ocr", "triple": "find", "long": "stop_or_warning"}


class ButtonGestureDecoder:
    def __init__(self, multi_click_sec: float = 0.45, long_press_sec: float = 0.7,
                 repeat_gap_sec: float = 0.12):
        self.multi_click_sec = multi_click_sec
        self.long_press_sec = long_press_sec
        self.repeat_gap_sec = repeat_gap_sec
        self._clicks = 0
        self._last_event = None
        self._down_at: Optional[float] = None
        self._hold_start: Optional[float] = None
        self._long_fired = False

    def press(self, t: float) -> Optional[str]:
        """Nhấn (hoặc nhấn lặp do giữ phím). Trả về 'long' ngay khi đủ thời gian giữ."""
        if self._hold_start is not None and self._last_event is not None \
                and t - self._last_event <= self.repeat_gap_sec:
            self._last_event = t  # đang giữ phím (auto-repeat)
            if not self._long_fired and t - self._hold_start >= self.long_press_sec:
                self._long_fired = True
                self._clicks = 0
                return "long"
            return None
        if self._down_at is None:
            self._down_at = t
        self._hold_start = t
        self._long_fired = False
        self._clicks += 1
        self._last_event = t
        return None

    def release(self, t: float) -> Optional[str]:
        down = self._down_at
        self._down_at = None
        self._hold_start = None
        if self._long_fired:
            self._long_fired = False
            self._clicks = 0
            return None
        if down is not None and t - down >= self.long_press_sec:
            self._clicks = 0
            return "long"
        self._last_event = t
        return None

    def poll(self, t: float) -> Optional[str]:
        """Gọi đều đặn; trả về single/double/triple khi đã hết cửa sổ chờ bấm tiếp."""
        if self._clicks and self._last_event is not None and not self._long_fired \
                and t - self._last_event > self.multi_click_sec:
            n, self._clicks = self._clicks, 0
            self._down_at = None
            self._hold_start = None
            return {1: "single", 2: "double"}.get(n, "triple")
        return None


def gesture_to_command(gesture: str, actions: Optional[Dict[str, str]] = None) -> Optional[str]:
    return (actions or DEFAULT_ACTIONS).get(gesture)

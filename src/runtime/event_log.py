"""Nhật ký sự kiện JSONL (cục bộ, không gửi đi đâu) phục vụ đo độ trễ và đánh giá theo sự kiện.

Mỗi dòng: {"t": time.monotonic(), "wall": thời gian thực, "type": ..., ...}. Đồng hồ monotonic
là chung cho mọi tiến trình trên cùng máy, nên có thể ghép với nhật ký của
scripts/do_tre_kich_thich.py để đo độ trễ từ lúc vật xuất hiện trên màn hình tới lúc phát âm.
"""
import json
import threading
import time
from typing import Any, Optional, TextIO


class EventLog:
    def __init__(self, path: Optional[str]):
        self.path = path
        self._lock = threading.Lock()
        self._fh: Optional[TextIO] = open(path, "a", encoding="utf-8") if path else None

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def write(self, event_type: str, t: Optional[float] = None, **fields: Any) -> None:
        if self._fh is None:
            return
        record = {"t": time.monotonic() if t is None else t, "wall": time.time(), "type": event_type}
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            if self._fh is not None:
                self._fh.write(line + "\n")
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


NULL_LOG = EventLog(None)

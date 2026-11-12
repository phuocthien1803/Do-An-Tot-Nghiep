"""Kênh âm thanh an toàn: tiếng bíp nhịp nhanh dần theo TTC/khoảng cách, KHÔNG bị tắt theo chế độ.

Bản gốc tắt toàn bộ cảnh báo (kể cả HIGH) khi người dùng ở chế độ OCR/VQA cho tới khi bấm C.
Kênh này giữ một tín hiệu an toàn tối thiểu ở mọi chế độ mà không chen lời nói:
- Chế độ Cảnh báo: lời cảnh báo ngắn vẫn do AlertAggregator phát; kênh này thêm tiếng tích
  nhịp theo mức khẩn cấp trong lúc nguy cơ còn tồn tại (tạm dừng khi đang đọc lời cảnh báo).
- Chế độ OCR/VQA: phát âm báo kép khi có nguy cơ HIGH MỚI, rồi tiếng tích song song với giọng
  đọc kết quả (cấu hình safety.on_demand_high_alert: earcon).
Nguy cơ chỉ được tính khi assessment HIGH, dữ liệu VALID và còn mới (< stale_after_sec);
dữ liệu cũ dừng tiếng bíp ngay - không bao giờ diễn giải dữ liệu cũ là "an toàn" hay "nguy hiểm".
Phím S tạm im tiếng bíp cho các nguy cơ đang có (snooze) cho đến khi chúng biến mất.
"""
import threading
import time
from typing import Iterable, Optional, Set

from src.audio.earcons import EarconLibrary
from src.contracts.audio import AudioPriority
from src.contracts.risk import DataQuality, Direction, RiskAssessment, RiskLevel


def pick_urgent(assessments: Iterable[RiskAssessment], now: float,
                max_age_sec: float = 1.0) -> Optional[RiskAssessment]:
    """Chọn nguy cơ HIGH khẩn cấp nhất còn hợp lệ (cùng thứ tự ưu tiên với AlertAggregator)."""
    best, best_key = None, None
    for a in assessments:
        if (a.risk_level != RiskLevel.HIGH or a.data_quality != DataQuality.VALID
                or a.is_expired(now) or now - a.source_timestamp > max_age_sec):
            continue
        ttc = getattr(a, "ttc_sec", None)
        key = (1 if getattr(a, "hazard_type", "object") == "drop_off" else 0,
               2 if a.direction == Direction.CENTER else 1,
               1.0 / ttc if ttc else 0.0, a.relative_proximity)
        if best_key is None or key > best_key:
            best, best_key = a, key
    return best


class SafetySoundChannel:
    def __init__(
        self,
        audio_coordinator,
        earcons: EarconLibrary,
        ticks_in_warning: bool = True,
        ticks_in_on_demand: bool = True,
        alert_earcon_in_on_demand: bool = True,
        stale_after_sec: float = 0.6,
        min_interval_sec: float = 0.15,
        max_interval_sec: float = 0.9,
        ttc_factor: float = 0.35,
        distance_factor: float = 0.45,
    ):
        self.audio = audio_coordinator
        self.earcons = earcons
        self.ticks_in_warning = ticks_in_warning
        self.ticks_in_on_demand = ticks_in_on_demand
        self.alert_earcon_in_on_demand = alert_earcon_in_on_demand
        self.stale_after_sec = stale_after_sec
        self.min_interval_sec = min_interval_sec
        self.max_interval_sec = max_interval_sec
        self.ttc_factor = ttc_factor
        self.distance_factor = distance_factor
        self._lock = threading.Lock()
        self._hazard: Optional[RiskAssessment] = None
        self._hazard_time = float("-inf")
        self._on_demand = False
        self._enabled = True
        self._known_ids: Set[int] = set()
        self._snoozed: Set[int] = set()
        self._next_tick = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.tick_count = 0
        self.alert_earcon_count = 0

    def interval_for(self, hazard: RiskAssessment) -> float:
        ttc = getattr(hazard, "ttc_sec", None)
        dist = getattr(hazard, "distance_m", None)
        if ttc is not None:
            value = self.ttc_factor * ttc
        elif dist is not None:
            value = self.distance_factor * dist
        else:
            value = self.max_interval_sec
        return max(self.min_interval_sec, min(self.max_interval_sec, value))

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="SafetySoundChannel", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._hazard = None
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled
            if not enabled:
                self._hazard = None

    def snooze_current(self) -> None:
        """S: im tiếng bíp của các nguy cơ đang có; nguy cơ mới vẫn được báo."""
        with self._lock:
            self._snoozed |= set(self._known_ids)
            self._hazard = None

    def update(self, assessments: Iterable[RiskAssessment], now: float, on_demand: bool) -> Optional[RiskAssessment]:
        assessments = list(assessments)
        current_ids = {a.track_id for a in assessments
                       if a.risk_level == RiskLevel.HIGH and a.track_id is not None}
        urgent = pick_urgent(assessments, now, self.stale_after_sec)
        play_alert = None
        with self._lock:
            if not self._enabled:
                return None
            self._snoozed &= current_ids  # nguy cơ đã biến mất thì bỏ trạng thái im
            if urgent is not None and urgent.track_id in self._snoozed:
                urgent = None
            new_hazard = urgent is not None and urgent.track_id not in self._known_ids
            self._known_ids = current_ids
            self._hazard = urgent
            self._hazard_time = now
            self._on_demand = on_demand
            if new_hazard and on_demand and self.alert_earcon_in_on_demand:
                play_alert = self.earcons.for_alert(urgent.direction, getattr(urgent, "hazard_type", "object"))
                self.alert_earcon_count += 1
                self._next_tick = now + 0.35  # để âm báo kép phát xong trước tiếng tích
        if play_alert:
            self.audio.play_earcon(play_alert)
        return urgent

    def _loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    return
                now = time.monotonic()
                hazard = self._hazard
                fresh = hazard is not None and now - self._hazard_time <= self.stale_after_sec
                allowed = self.ticks_in_on_demand if self._on_demand else self.ticks_in_warning
                due = fresh and allowed and self._enabled and now >= self._next_tick
                on_demand = self._on_demand
                if due:
                    self._next_tick = now + self.interval_for(hazard)
                    direction = hazard.direction
            if due:
                # Ở chế độ Cảnh báo, không chèn tiếng tích lên lời cảnh báo HIGH đang đọc.
                if on_demand or self.audio.current_priority != AudioPriority.HIGH_RISK:
                    if self.audio.play_earcon(self.earcons.tick(direction)):
                        self.tick_count += 1
            time.sleep(0.02)

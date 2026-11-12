"""Chế độ TÌM ĐỒ: dẫn người khiếm thị tới một vật cụ thể bằng âm thanh lập thể.

Nhu cầu hằng ngày của người mù thường là "cái cốc của tôi đâu", "ghế trống ở đâu" chứ không chỉ né
vật cản. Khi bật chế độ (phím F, bấm nút 3 lần, hoặc nói "tìm cái cốc"):
- Âm dẫn hướng phát liên tục, vị trí trái/phải theo vị trí vật trong ảnh (7 mức), tiếng CAO khi vật
  ở ngay phía trước, nhịp nhanh dần khi lại gần (theo khoảng cách độ sâu, không có thì theo cỡ bbox).
- Lời nhắc ngắn khi mới thấy vật, khi hướng thay đổi, khi vật "trong tầm tay", và khi mất dấu
  ("Chưa thấy cốc, hãy xoay người chậm"). Không bao giờ đọc số mét.
Giới hạn: chỉ tìm được lớp YOLO đã bật (COCO không có "cửa", "chìa khoá"...); dữ liệu độ sâu cũ thì
chỉ dẫn hướng trái/phải, không nói gần/xa.
"""
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.contracts.audio import AudioPriority, AudioTask
from src.contracts.detection import Detection
from src.contracts.risk import DataQuality

# (tên tiếng Việt đọc ra loa, lớp COCO, các cách gọi)
FIND_TARGETS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("cốc", "cup", ("cốc", "ly", "tách", "cái ly", "cái cốc")),
    ("chai", "bottle", ("chai", "chai nước", "bình nước")),
    ("ghế", "chair", ("ghế", "cái ghế", "chỗ ngồi", "ghế trống")),
    ("người", "person", ("người", "ai", "người nào")),
    ("bàn", "dining table", ("bàn", "cái bàn", "bàn ăn")),
    ("điện thoại", "cell phone", ("điện thoại", "đt", "di động")),
    ("sách", "book", ("sách", "quyển sách", "cuốn sách", "vở")),
    ("ba lô", "backpack", ("ba lô", "balo", "cặp")),
    ("bát", "bowl", ("bát", "chén", "tô")),
    ("máy tính", "laptop", ("máy tính", "laptop", "máy tính xách tay")),
    ("bồn rửa", "sink", ("bồn rửa", "lavabo", "chỗ rửa tay")),
    ("bồn cầu", "toilet", ("bồn cầu", "nhà vệ sinh", "toilet")),
    ("ti vi", "tv", ("ti vi", "tivi", "tv")),
]


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def resolve_target(text: str) -> Optional[Tuple[str, str]]:
    """'tìm cái cốc giúp tôi' -> ('cốc', 'cup'); không nhận ra -> None."""
    t = _norm(text)
    best = None
    for vi, coco, names in FIND_TARGETS:
        for name in names:
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", t) and (best is None or len(name) > best[0]):
                best = (len(name), vi, coco)
    return (best[1], best[2]) if best else None


@dataclass
class FinderState:
    active: bool = False
    target_vi: str = ""
    target_class: str = ""
    seen: bool = False
    offset: float = 0.0
    distance_m: Optional[float] = None
    box: Optional[Tuple[float, float, float, float]] = None
    last_seen: float = float("-inf")


def direction_words(offset: float) -> str:
    if offset < -0.45:
        return "bên trái"
    if offset < -0.15:
        return "hơi lệch trái"
    if offset <= 0.15:
        return "ngay phía trước"
    if offset <= 0.45:
        return "hơi lệch phải"
    return "bên phải"


def distance_words(distance_m: Optional[float]) -> str:
    if distance_m is None:
        return ""
    if distance_m < 0.6:
        return "trong tầm tay"
    if distance_m < 1.5:
        return "cách vài bước"
    return "còn xa"


class ObjectFinder:
    def __init__(self, audio_coordinator, earcons, roi_extractor=None, enabled_classes=None,
                 hint_gap_sec: float = 2.5, lost_after_sec: float = 1.5, lost_hint_every_sec: float = 6.0,
                 on_target_offset: float = 0.15):
        self.audio = audio_coordinator
        self.earcons = earcons
        self.roi_extractor = roi_extractor
        self.enabled_classes = set(enabled_classes) if enabled_classes else None
        self.hint_gap_sec = hint_gap_sec
        self.lost_after_sec = lost_after_sec
        self.lost_hint_every_sec = lost_hint_every_sec
        self.on_target_offset = on_target_offset
        self._lock = threading.Lock()
        self._state = FinderState()
        self._started_at = 0.0
        self._last_hint_at = float("-inf")
        self._last_dir = ""
        self._reach_said = False
        self._lost_said_at = float("-inf")
        self._cycle_index = -1
        self._next_beep = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.beep_count = 0

    # ---------- điều khiển ----------

    def available_targets(self) -> List[Tuple[str, str]]:
        return [(vi, coco) for vi, coco, _ in FIND_TARGETS
                if self.enabled_classes is None or coco in self.enabled_classes]

    def next_target(self) -> Tuple[str, str]:
        targets = self.available_targets()
        self._cycle_index = (self._cycle_index + 1) % len(targets)
        return targets[self._cycle_index]

    def start(self, target_vi: str, target_class: str) -> None:
        with self._lock:
            self._state = FinderState(active=True, target_vi=target_vi, target_class=target_class)
            self._started_at = time.monotonic()
            self._last_hint_at = float("-inf")
            self._lost_said_at = time.monotonic()  # chờ một lúc rồi mới nhắc "chưa thấy"
            self._last_dir = ""
            self._reach_said = False
            if not self._running:
                self._running = True
                self._thread = threading.Thread(target=self._loop, name="ObjectFinder", daemon=True)
                self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._state = FinderState()

    def shutdown(self) -> None:
        with self._lock:
            self._running = False
            self._state = FinderState()
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)

    @property
    def active(self) -> bool:
        with self._lock:
            return self._state.active

    def snapshot(self) -> Dict:
        with self._lock:
            s = self._state
            return {"active": s.active, "target": s.target_vi, "seen": s.seen and
                    time.monotonic() - s.last_seen <= self.lost_after_sec,
                    "offset": s.offset, "distance_m": s.distance_m, "box": s.box}

    # ---------- cập nhật từ luồng detection ----------

    def _pick(self, detections: List[Detection], cls: str) -> Optional[Detection]:
        cands = [d for d in detections if d.class_name == cls]
        return max(cands, key=lambda d: d.bbox.area * d.confidence) if cands else None

    def update(self, sync_pair, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            if not self._state.active:
                return
            cls = self._state.target_class
        det_result = sync_pair.detection_result
        width = det_result.preprocess_transform.get("original_size", (640, 480))[0]
        det = self._pick(det_result.detections, cls)
        hint = None
        with self._lock:
            s = self._state
            if not s.active or s.target_class != cls:
                return
            if det is None:
                if now - s.last_seen > self.lost_after_sec:
                    s.seen = False
                    if now - self._lost_said_at >= self.lost_hint_every_sec:
                        hint = f"Chưa thấy {s.target_vi}, hãy xoay người chậm."
                        self._lost_said_at = now
                        self._last_dir = ""
            else:
                distance = None
                depth = sync_pair.depth_map
                if (self.roi_extractor is not None and depth is not None
                        and sync_pair.data_quality == DataQuality.VALID):
                    try:
                        _, _, _, distance = self.roi_extractor.extract_with_distance(det.bbox, depth)
                    except Exception:
                        distance = None
                first = not s.seen
                s.seen, s.last_seen = True, now
                s.offset = (det.bbox.center_x - width / 2.0) / (width / 2.0)
                s.distance_m = distance
                s.box = det.bbox.as_tuple()
                self._lost_said_at = now
                d_words = direction_words(s.offset)
                reach = distance is not None and distance < 0.6
                if reach and not self._reach_said:
                    hint = f"{s.target_vi.capitalize()} trong tầm tay, {d_words}."
                    self._reach_said = True
                elif first:
                    extra = distance_words(distance)
                    hint = f"Thấy {s.target_vi} {d_words}" + (f", {extra}." if extra else ".")
                elif d_words != self._last_dir and now - self._last_hint_at >= self.hint_gap_sec:
                    hint = f"{s.target_vi.capitalize()} {d_words}."
                if distance is not None and distance > 0.9:
                    self._reach_said = False  # đã lùi ra xa: lần tới gần lại sẽ nhắc "trong tầm tay"
                self._last_dir = d_words
            if hint:
                self._last_hint_at = now
        if hint:
            self.audio.post_task(AudioTask(priority=AudioPriority.ON_DEMAND, text=hint,
                                           expires_at=time.monotonic() + 3.0, interruptible=True,
                                           category="find"))

    # ---------- âm dẫn hướng ----------

    def interval_for(self, distance_m: Optional[float], box) -> float:
        if distance_m is not None:
            return max(0.15, min(1.0, 0.15 + 0.3 * distance_m))
        if box:
            area = (box[2] - box[0]) * (box[3] - box[1]) / (640.0 * 480.0)
            return max(0.15, min(1.0, 0.9 - 1.5 * area))
        return 0.8

    def _loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    return
                s = self._state
                now = time.monotonic()
                due = (s.active and s.seen and now - s.last_seen <= self.lost_after_sec
                       and now >= self._next_beep)
                if due:
                    self._next_beep = now + self.interval_for(s.distance_m, s.box)
                    offset = s.offset
            if due and self.audio.current_priority != AudioPriority.HIGH_RISK:
                if self.audio.play_earcon(self.earcons.guide(offset, abs(offset) <= self.on_target_offset)):
                    self.beep_count += 1
            time.sleep(0.02)

from typing import Callable, List, Optional
import time

from src.contracts.risk import RiskAssessment, RiskLevel, Direction, DataQuality
from src.contracts.audio import AudioTask, AudioPriority
from src.fusion.risk_fsm import RiskFSM

DIRECTION_TEXT_VI = {
    Direction.LEFT: "bên trái",
    Direction.CENTER: "phía trước",
    Direction.RIGHT: "bên phải"
}

class AlertAggregator:
    """
    Bộ tổng hợp và lựa chọn cảnh báo nguy cơ quan trọng nhất.
    Tuân thủ mục 4.3 của Kế hoạch:
    - Chỉ tổng hợp cảnh báo HIGH; MEDIUM/LOW không kích hoạt âm thanh hoặc cooldown.
    - Báo hướng và mức gần bằng lời (proximity_desc), không đọc khoảng cách mét.
    - Lọc bỏ dữ liệu stale/degraded trước khi xếp hạng hoặc kích hoạt cooldown.
    - Cảnh báo có thời hạn hiệu lực rõ ràng (expires_at) ràng buộc với tuổi dữ liệu nguồn.

    Bản cải tiến (bật qua cấu hình; giá trị mặc định của hàm khởi tạo giữ hành vi bản gốc):
    - message_style="ngan": câu cực ngắn "Ghế phía trước" (bản gốc ~1,5-2 s lời nói) để
      âm báo + lời nói kết thúc trước khi người dùng đi hết quãng TTC.
    - bypass_interval=True: vượt khoảng nghỉ toàn cục khi có (a) bậc xuống/hố mới,
      (b) vật MỚI chắn giữa hành lang chưa từng được báo, (c) vật khác có TTC <= critical_ttc_sec.
      Bản gốc có điều kiện "is_escalation" nhưng nó không bao giờ đúng (chỉ HIGH được phát nên
      cảnh báo trước luôn là HIGH) -> khoảng nghỉ 3,5 s chặn cả vật mới xuất hiện.
    - earcon_provider(direction, hazard_type) -> đường dẫn WAV âm báo lập thể theo hướng.
    """

    def __init__(
        self,
        risk_fsm: RiskFSM,
        alert_lifetime_sec: float = 3.0,
        global_alert_interval_sec: float = 0.0,
        max_source_age_sec: Optional[float] = None,
        max_detection_age_ms: Optional[float] = None,
        max_depth_age_ms: Optional[float] = None,
        max_alert_age_ms: Optional[float] = None,
        message_style: str = "day_du",
        bypass_interval: bool = False,
        min_bypass_gap_sec: float = 1.0,
        critical_ttc_sec: float = 1.5,
        earcon_provider: Optional[Callable[[Direction, str], Optional[str]]] = None,
    ):
        if message_style not in ("ngan", "day_du"):
            raise ValueError("message_style phải là 'ngan' hoặc 'day_du'")
        self.message_style = message_style
        self.bypass_interval = bypass_interval
        self.min_bypass_gap_sec = min_bypass_gap_sec
        self.critical_ttc_sec = critical_ttc_sec
        self.earcon_provider = earcon_provider
        self._last_track_id: Optional[int] = None
        self._last_hazard_type: Optional[str] = None
        self.risk_fsm = risk_fsm
        self.alert_lifetime_sec = alert_lifetime_sec
        self.global_alert_interval_sec = global_alert_interval_sec
        self._last_global_alert_time: float = 0.0
        self._last_alert_risk_level: Optional[RiskLevel] = None

        if max_source_age_sec is not None:
            self.max_source_age_sec = max_source_age_sec
        else:
            ages = [v for v in [max_detection_age_ms, max_depth_age_ms, max_alert_age_ms] if v is not None]
            if ages:
                self.max_source_age_sec = min(ages) / 1000.0
            else:
                # Mặc định theo configs/fusion_rules.yaml: min(1000, 1200, 1500) ms = 1.0s
                self.max_source_age_sec = 1.0

        self._last_selected_reason: str = ""

    def aggregate(self, assessments: List[RiskAssessment], current_mono: Optional[float] = None) -> Optional[AudioTask]:
        now = current_mono if current_mono is not None else time.monotonic()

        # 1. Lọc các đối tượng:
        # - data_quality phải là VALID (không dùng dữ liệu DEGRADED, STALE hoặc UNAVAILABLE)
        # - risk_level phải là HIGH
        # - Chưa hết hạn assessment (a.is_expired(now))
        # - Chưa quá hạn dữ liệu nguồn (now - a.source_timestamp <= max_source_age_sec)
        # - Được phép phát theo quy tắc cooldown của FSM
        active_candidates: List[RiskAssessment] = []
        for a in assessments:
            if a.data_quality != DataQuality.VALID:
                continue
            if a.risk_level != RiskLevel.HIGH:
                continue
            if a.is_expired(now):
                continue
            if (now - a.source_timestamp) > self.max_source_age_sec:
                continue

            t_id = a.track_id if a.track_id is not None else -1
            if self.risk_fsm.can_trigger_alert(t_id, a.risk_level, now):
                active_candidates.append(a)

        if not active_candidates:
            return None

        # 2. Xếp hạng độ ưu tiên khẩn cấp
        # Thứ tự: HIGH ở CENTER > HIGH ở LEFT/RIGHT, sau đó theo độ gần.
        def sort_key(a: RiskAssessment):
            dir_score = 2 if a.direction == Direction.CENTER else 1
            if self.message_style == "day_du" and not self.bypass_interval:
                return (dir_score, a.relative_proximity)  # thứ tự của bản gốc
            ttc = getattr(a, "ttc_sec", None)
            urgency = 1.0 / ttc if ttc else 0.0
            drop = 1 if getattr(a, "hazard_type", "object") == "drop_off" else 0
            return (drop, dir_score, urgency, a.relative_proximity)

        active_candidates.sort(key=sort_key, reverse=True)
        top_risk = active_candidates[0]

        # 3. Kiểm tra khoảng nghỉ âm thanh toàn cục (Global Alert Interval)
        if self.global_alert_interval_sec > 0.0:
            time_since_global = now - self._last_global_alert_time
            if time_since_global < self.global_alert_interval_sec:
                if not self._may_bypass_interval(top_risk, time_since_global):
                    return None

        # 4. Tạo câu nói tiếng Việt ngắn gọn, không khẳng định khoảng cách mét
        alert_text = self._compose_text(top_risk)
        t_id = top_risk.track_id if top_risk.track_id is not None else -1
        self._last_selected_reason = (
            f"Chọn track {t_id} ({top_risk.class_name}) - "
            f"Mức: {top_risk.risk_level.value}, Hướng: {top_risk.direction.value}, "
            f"Độ gần tương đối: {top_risk.relative_proximity:.2f} ({top_risk.proximity_desc})"
        )

        # 5. Đánh dấu đã phát cảnh báo cho track này và toàn cục để kích hoạt cooldown
        self.risk_fsm.mark_alert_triggered(t_id, now, risk_level=top_risk.risk_level)
        self._last_global_alert_time = now
        self._last_alert_risk_level = top_risk.risk_level
        self._last_track_id = t_id
        self._last_hazard_type = getattr(top_risk, "hazard_type", "object")

        priority = AudioPriority.HIGH_RISK

        # 6. Hạn của AudioTask là thời điểm sớm nhất giữa hạn assessment, hạn nguồn và task lifetime
        task_lifetime_limit = now + self.alert_lifetime_sec
        source_age_limit = top_risk.source_timestamp + self.max_source_age_sec
        assessment_limit = top_risk.expires_at

        task_expires_at = min(task_lifetime_limit, source_age_limit, assessment_limit)

        sound_file = "assets/audio/alert_high.wav"
        if self.earcon_provider is not None:
            sound_file = self.earcon_provider(top_risk.direction,
                                              getattr(top_risk, "hazard_type", "object")) or sound_file
        return AudioTask(
            priority=priority,
            text=alert_text,
            sound_file=sound_file,
            created_at=now,
            expires_at=task_expires_at,
            interruptible=True,
            task_id=f"alert_track_{t_id}_{int(now*1000)}",
            source_timestamp=top_risk.source_timestamp,
            category="alert",
        )

    def _may_bypass_interval(self, top: RiskAssessment, since_last: float) -> bool:
        if not self.bypass_interval or since_last < self.min_bypass_gap_sec:
            return False
        t_id = top.track_id if top.track_id is not None else -1
        hazard = getattr(top, "hazard_type", "object")
        if hazard == "drop_off" and self._last_hazard_type != "drop_off":
            return True
        if t_id == self._last_track_id:
            return False
        if top.direction == Direction.CENTER and not self.risk_fsm.has_alerted(t_id):
            return True
        ttc = getattr(top, "ttc_sec", None)
        return ttc is not None and ttc <= self.critical_ttc_sec

    def _compose_text(self, top: RiskAssessment) -> str:
        dir_text = DIRECTION_TEXT_VI.get(top.direction, "phía trước")
        hazard = getattr(top, "hazard_type", "object")
        if self.message_style == "ngan":
            if hazard == "drop_off":
                return f"Cẩn thận, {top.class_name} {dir_text}"
            if getattr(top, "ttc_sec", None) is not None and hazard == "object":
                return f"{top.class_name.capitalize()} đang lại gần, {dir_text}"
            return f"{top.class_name.capitalize()} {dir_text}"
        # Câu đầy đủ của bản gốc.
        prox_text = top.proximity_desc
        prefix = "Chú ý, có"
        if prox_text and prox_text != "chưa rõ":
            return f"{prefix} {top.class_name} {dir_text}, {prox_text}"
        return f"{prefix} {top.class_name} {dir_text}"

    @property
    def last_reason(self) -> str:
        return self._last_selected_reason

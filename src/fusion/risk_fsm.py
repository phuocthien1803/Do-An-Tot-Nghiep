from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import time
import math

from src.contracts.detection import BoundingBox, Detection
from src.contracts.risk import DataQuality, RiskLevel, Direction, RiskAssessment
from src.detection.class_filter import ClassFilter
from src.depth.roi_extractor import ROIExtractor
from src.fusion.walking_corridor import WalkingCorridor
from src.fusion.generic_obstacle import GenericObstacleDetector
from src.fusion.floor_hazards import DROP_OFF, LOW_OBSTACLE, FloorHazardDetector, TemporalVote
from src.fusion.ttc import TTCEstimator, TTCEstimate, NO_ESTIMATE
from src.fusion.spatial_zones import SpatialZones
from src.fusion.synchronizer import SynchronizedPair

# Track id dành riêng cho nguy cơ không thuộc lớp COCO; ByteTrack chỉ cấp id dương.
GENERIC_OBSTACLE_TRACK_ID = -100
GENERIC_OBSTACLE_NAME = "vật cản"
DROP_OFF_TRACK_ID = -101
DROP_OFF_NAME = "bậc xuống hoặc hố"
LOW_OBSTACLE_TRACK_ID = -102
LOW_OBSTACLE_NAME = "vật cản thấp"

RISK_MODELS = ("distance", "ttc")


@dataclass
class TrackState:
    track_id: int
    current_risk: RiskLevel = RiskLevel.UNDETERMINED
    consecutive_high_count: int = 0
    consecutive_safe_count: int = 0
    last_alert_time: float = 0.0
    last_alert_risk: Optional[RiskLevel] = None
    last_depth_frame_id: Optional[int] = None
    last_assessment: Optional[RiskAssessment] = None


class RiskFSM:
    """
    Finite State Machine (FSM) đánh giá nguy cơ va chạm cho từng đối tượng được theo vết.
    Tuân thủ mục 4.3 của Kế hoạch:
    - Xác nhận tăng mức; hạ mức ngay khi điều kiện cảnh báo không còn.
    - Không đếm lặp cùng một kết quả depth.
    - Chỉ trích ROI và đánh giá nguy cơ khi data_quality == VALID.
    - Kích hoạt nhanh HIGH trong hành lang (tên tùy chọn cũ: instant_high_risk_in_center).
    - Cooldown chống phát lặp, nhưng cho phép vượt cooldown khi mức nguy cơ tăng cấp.

    Bản cải tiến:
    - risk_model="ttc": HIGH khi vật trong hành lang ở rất gần (<= critical_distance_m) HOẶC
      thời gian đến va chạm <= ttc_high_sec. Khi đã đủ mẫu để biết vật đứng yên tương đối
      với người dùng, vật trong 1,5 m nhưng không tiến lại gần chỉ là MEDIUM (giảm nhắc thừa).
      Track mới chưa đủ mẫu -> dùng nguyên luật khoảng cách của bản gốc (không giảm độ nhạy).
    - Vật cản chung, bậc xuống/hố, vật cản thấp phải được xác nhận k/n mẫu độ sâu
      (bản gốc kích hoạt vật cản chung chỉ từ một frame).
    """

    def __init__(
        self,
        class_filter: ClassFilter,
        spatial_zones: SpatialZones,
        roi_extractor: ROIExtractor,
        depth_threshold_high: float = 0.75,
        depth_threshold_medium: float = 0.45,
        depth_threshold_low: float = 0.25,
        confirmations_to_escalate: int = 1,
        confirmations_to_deescalate: int = 3,
        instant_high_risk_in_center: bool = True,
        cooldown_sec: float = 4.0,
        allow_escalation_override: bool = True,
        walking_corridor: Optional[WalkingCorridor] = None,
        generic_obstacle_detector: Optional[GenericObstacleDetector] = None,
        risk_model: str = "distance",
        ttc_estimator: Optional[TTCEstimator] = None,
        critical_distance_m: float = 1.0,
        ttc_high_sec: float = 2.5,
        ttc_max_distance_m: float = 5.0,
        floor_hazard_detector: Optional[FloorHazardDetector] = None,
        hazard_confirm_required: int = 1,
        hazard_confirm_window: int = 1,
    ):
        if risk_model not in RISK_MODELS:
            raise ValueError(f"risk_model phải thuộc {RISK_MODELS}")
        self.generic_obstacle_detector = generic_obstacle_detector
        self.floor_hazard_detector = floor_hazard_detector
        self.class_filter = class_filter
        self.spatial_zones = spatial_zones
        self.walking_corridor = walking_corridor or WalkingCorridor()
        self.roi_extractor = roi_extractor
        self.depth_threshold_high = depth_threshold_high
        self.depth_threshold_medium = depth_threshold_medium
        self.depth_threshold_low = depth_threshold_low
        self.confirmations_to_escalate = confirmations_to_escalate
        self.confirmations_to_deescalate = confirmations_to_deescalate
        self.instant_high_risk_in_center = instant_high_risk_in_center
        self.cooldown_sec = cooldown_sec
        self.allow_escalation_override = allow_escalation_override
        self.risk_model = risk_model
        self.ttc_estimator = ttc_estimator if ttc_estimator is not None else (
            TTCEstimator() if risk_model == "ttc" else None)
        self.critical_distance_m = critical_distance_m
        self.ttc_high_sec = ttc_high_sec
        self.ttc_max_distance_m = ttc_max_distance_m
        self._votes = {kind: TemporalVote(hazard_confirm_required, hazard_confirm_window)
                       for kind in ("generic", DROP_OFF, LOW_OBSTACLE)}

        self._tracks: Dict[int, TrackState] = {}

    def _roi(self, bbox: BoundingBox, depth_map) -> Tuple[float, str, str, Optional[float]]:
        fn = getattr(self.roi_extractor, "extract_with_distance", None)
        if fn is not None:
            result = fn(bbox, depth_map)
            if isinstance(result, tuple) and len(result) == 4:
                return result
        # Bộ trích ROI cũ/giả lập chỉ có extract_proximity -> không có khoảng cách mét.
        score, desc, reason = self.roi_extractor.extract_proximity(bbox, depth_map)
        return score, desc, reason, None

    def update(self, sync_pair: SynchronizedPair, current_mono: Optional[float] = None) -> List[RiskAssessment]:
        """
        Đánh giá nguy cơ cho toàn bộ đối tượng trong SynchronizedPair.
        Trả về danh sách RiskAssessment của các đối tượng trong khung hình.
        Chỉ tính nguy cơ khi data_quality == VALID; nếu không trả UNDETERMINED và reset bộ đếm.
        """
        now = current_mono if current_mono is not None else time.monotonic()
        det_result = sync_pair.detection_result
        depth_map = sync_pair.depth_map
        data_quality = sync_pair.data_quality
        fallback_size = (depth_map.width, depth_map.height) if depth_map is not None else (640, 480)
        orig_w, orig_h = det_result.preprocess_transform.get("original_size", fallback_size)

        assessments: List[RiskAssessment] = []
        current_frame_track_ids = set()

        for det in det_result.detections:
            t_id = det.track_id if det.track_id is not None else -1
            current_frame_track_ids.add(t_id)

            if t_id not in self._tracks:
                self._tracks[t_id] = TrackState(track_id=t_id)
            state = self._tracks[t_id]

            # 1. Xác định hướng không gian
            overlap = self.walking_corridor.overlap_ratio(det.bbox, orig_w, orig_h)
            direction = (self.spatial_zones.determine_direction(det.bbox, orig_w)
                         if overlap is not None else Direction.CENTER)
            distance_m: Optional[float] = None
            estimate: TTCEstimate = NO_ESTIMATE

            # 2. Xử lý theo chất lượng dữ liệu: chỉ trích ROI khi VALID
            if data_quality != DataQuality.VALID or depth_map is None:
                proximity = 0.0
                prox_desc = "chưa rõ"
                if data_quality == DataQuality.UNAVAILABLE or depth_map is None:
                    reason = "Chưa có dữ liệu độ sâu"
                elif data_quality == DataQuality.STALE:
                    reason = "Dữ liệu độ sâu hoặc nhận diện đã cũ"
                elif data_quality == DataQuality.DEGRADED:
                    reason = "Dữ liệu độ sâu bị suy giảm hoặc lệch thời gian"
                else:
                    reason = "Chất lượng dữ liệu không hợp lệ"

                # Reset bộ đếm xác nhận và trạng thái phụ thuộc depth cho track này
                state.consecutive_high_count = 0
                state.consecutive_safe_count = 0
                state.current_risk = RiskLevel.UNDETERMINED
                state.last_depth_frame_id = None
                if self.ttc_estimator is not None:
                    # Không nối chuỗi khoảng cách qua đoạn dữ liệu cũ/suy giảm.
                    self.ttc_estimator.reset_track(t_id)
                final_risk = RiskLevel.UNDETERMINED

            else:
                is_new_depth_sample = False
                if state.last_depth_frame_id != depth_map.frame_id:
                    is_new_depth_sample = True
                    state.last_depth_frame_id = depth_map.frame_id

                proximity, prox_desc = 0.0, "chưa rõ"
                if overlap is None:
                    reason = "BBox hoặc kích thước ảnh không hợp lệ"
                    candidate_risk = RiskLevel.UNDETERMINED
                elif depth_map.shape != (orig_h, orig_w):
                    reason = "Kích thước depth không khớp ảnh gốc"
                    candidate_risk = RiskLevel.UNDETERMINED
                elif overlap < self.walking_corridor.min_overlap_ratio:
                    reason = f"Vật ngoài hành lang đi thẳng (giao {overlap:.1%})"
                    candidate_risk = RiskLevel.NO_ALERT
                    self._observe_scale(t_id, det, sync_pair, orig_h)
                else:
                    proximity, prox_desc, reason, distance_m = self._roi(
                        det.bbox.clamp(orig_w, orig_h), depth_map)
                    if prox_desc == "chưa rõ" or not math.isfinite(proximity):
                        if not math.isfinite(proximity):
                            reason = "Độ gần ROI không hợp lệ"
                        proximity, prox_desc, distance_m = 0.0, "chưa rõ", None
                        candidate_risk = RiskLevel.UNDETERMINED
                        self._observe_scale(t_id, det, sync_pair, orig_h)
                    else:
                        estimate = self._observe_and_estimate(t_id, det, sync_pair, orig_h, distance_m)
                        candidate_risk, rule = self._evaluate_candidate_risk(
                            det=det, direction=direction, proximity=proximity,
                            data_quality=data_quality, distance_m=distance_m, estimate=estimate)
                        reason = f"Trong hành lang (giao {overlap:.1%}); {reason}{rule}"

                final_risk = self._apply_hysteresis(
                    state, candidate_risk, direction, proximity, is_new_depth_sample)
                state.current_risk = final_risk

            # 5. Đóng gói kết quả RiskAssessment
            vi_name = self.class_filter.get_vietnamese_name(det.class_name)
            assessment = RiskAssessment(
                track_id=det.track_id,
                class_name=vi_name,
                direction=direction,
                risk_level=final_risk,
                data_quality=data_quality,
                relative_proximity=proximity,
                proximity_desc=prox_desc,
                reason=reason,
                source_timestamp=sync_pair.effective_timestamp,
                expires_at=now + 1.0,  # Hết hạn sau 1.0s nếu không được refresh
                distance_m=distance_m,
                ttc_sec=estimate.ttc_sec,
                closing_speed_mps=estimate.closing_speed_mps,
                hazard_type="object",
            )
            state.last_assessment = assessment
            assessments.append(assessment)

        generic = self._assess_generic_obstacle(sync_pair, orig_w, orig_h, now)
        if generic is not None:
            current_frame_track_ids.add(GENERIC_OBSTACLE_TRACK_ID)
            assessments.append(generic)
        for hazard in self._assess_floor_hazards(sync_pair, orig_w, orig_h, now,
                                                 skip_low=generic is not None):
            current_frame_track_ids.add(hazard.track_id)
            assessments.append(hazard)

        if self.ttc_estimator is not None:
            self.ttc_estimator.forget_missing(t for t in current_frame_track_ids if t >= 0)

        # Dọn dẹp tracks cũ không còn xuất hiện
        if len(self._tracks) > 50:
            active_ids = {t for t in self._tracks if t in current_frame_track_ids}
            if len(active_ids) < len(self._tracks):
                self._tracks = {t: s for t, s in self._tracks.items() if t in current_frame_track_ids}

        return assessments

    # ---------------- TTC ----------------

    def _observe_scale(self, t_id: int, det: Detection, sync_pair: SynchronizedPair, orig_h: float) -> None:
        if self.ttc_estimator is not None and t_id >= 0:
            self.ttc_estimator.observe(t_id, sync_pair.detection_result.timestamp_mono, det.bbox, orig_h)

    def _observe_and_estimate(self, t_id: int, det: Detection, sync_pair: SynchronizedPair,
                              orig_h: float, distance_m: Optional[float]) -> TTCEstimate:
        if self.ttc_estimator is None or t_id < 0:
            return NO_ESTIMATE
        depth_map = sync_pair.depth_map
        self.ttc_estimator.observe(
            t_id, sync_pair.detection_result.timestamp_mono, det.bbox, orig_h,
            distance_m=distance_m, depth_time=depth_map.capture_timestamp,
            depth_frame_id=depth_map.frame_id)
        return self.ttc_estimator.estimate(t_id)

    # ---------------- Nguy cơ không có bbox YOLO ----------------

    def _hazard_assessment(self, track_id: int, name: str, hazard_type: str, bbox: BoundingBox,
                           distance_m: float, reason: str, sync_pair: SynchronizedPair,
                           orig_w: float, now: float) -> RiskAssessment:
        state = self._tracks.setdefault(track_id, TrackState(track_id=track_id))
        proximity = max(self.roi_extractor.score_from_distance(distance_m), self.depth_threshold_high)
        state.current_risk = RiskLevel.HIGH
        assessment = RiskAssessment(
            track_id=track_id,
            class_name=name,
            direction=self.spatial_zones.determine_direction(bbox, orig_w),
            risk_level=RiskLevel.HIGH,
            data_quality=sync_pair.data_quality,
            relative_proximity=proximity,
            proximity_desc=self.roi_extractor.describe(proximity, metric=True),
            reason=reason,
            source_timestamp=sync_pair.effective_timestamp,
            expires_at=now + 1.0,
            distance_m=distance_m,
            hazard_type=hazard_type,
            region=bbox,
        )
        state.last_assessment = assessment
        return assessment

    def _hazard_input_ok(self, sync_pair: SynchronizedPair, orig_w: float, orig_h: float) -> bool:
        depth_map = sync_pair.depth_map
        return (depth_map is not None and sync_pair.data_quality == DataQuality.VALID
                and depth_map.shape == (int(orig_h), int(orig_w)))

    def _assess_generic_obstacle(self, sync_pair: SynchronizedPair, orig_w: float, orig_h: float,
                                 now: float) -> Optional[RiskAssessment]:
        """HIGH cho vùng gần trong hành lang mà không vật được nhận diện nào phủ (chỉ depth metric)."""
        if self.generic_obstacle_detector is None:
            return None
        if not self._hazard_input_ok(sync_pair, orig_w, orig_h):
            self._votes["generic"].reset()
            return None
        depth_map = sync_pair.depth_map
        found = self.generic_obstacle_detector.find(
            depth_map, [d.bbox for d in sync_pair.detection_result.detections])
        if not self._votes["generic"].vote(depth_map.frame_id, found is not None) or found is None:
            return None
        return self._hazard_assessment(
            GENERIC_OBSTACLE_TRACK_ID, GENERIC_OBSTACLE_NAME, "generic", found.bbox, found.distance_m,
            (f"Vật cản chưa nhận diện trong hành lang: {found.area_ratio:.0%} vùng "
             f"< {self.generic_obstacle_detector.near_distance_m:.1f} m, "
             f"ước lượng {found.distance_m:.1f} m"),
            sync_pair, orig_w, now)

    def _assess_floor_hazards(self, sync_pair: SynchronizedPair, orig_w: float, orig_h: float,
                              now: float, skip_low: bool) -> List[RiskAssessment]:
        if self.floor_hazard_detector is None:
            return []
        if not self._hazard_input_ok(sync_pair, orig_w, orig_h):
            self._votes[DROP_OFF].reset()
            self._votes[LOW_OBSTACLE].reset()
            return []
        depth_map = sync_pair.depth_map
        found = {h.kind: h for h in self.floor_hazard_detector.find(
            depth_map, [d.bbox for d in sync_pair.detection_result.detections])}
        result = []
        specs = ((DROP_OFF, DROP_OFF_TRACK_ID, DROP_OFF_NAME, "Sàn phía trước bị hụt (bậc xuống/hố)"),
                 (LOW_OBSTACLE, LOW_OBSTACLE_TRACK_ID, LOW_OBSTACLE_NAME, "Có vật nhô lên khỏi mặt sàn"))
        for kind, track_id, name, text in specs:
            hazard = found.get(kind)
            confirmed = self._votes[kind].vote(depth_map.frame_id, hazard is not None)
            if hazard is None or not confirmed or (kind == LOW_OBSTACLE and skip_low):
                continue
            result.append(self._hazard_assessment(
                track_id, name, kind, hazard.bbox, hazard.distance_m,
                f"{text}: {hazard.area_ratio:.0%} vùng sàn hành lang, ước lượng {hazard.distance_m:.1f} m",
                sync_pair, orig_w, now))
        return result

    # ---------------- Luật nguy cơ ----------------

    def _evaluate_candidate_risk(
        self,
        det: Detection,
        direction: Direction,
        proximity: float,
        data_quality: DataQuality,
        distance_m: Optional[float] = None,
        estimate: TTCEstimate = NO_ESTIMATE,
    ) -> Tuple[RiskLevel, str]:
        # Nếu hoàn toàn không có dữ liệu độ sâu -> UNDETERMINED
        if data_quality != DataQuality.VALID or not math.isfinite(proximity) or proximity < 0.0:
            return RiskLevel.UNDETERMINED, ""

        # Nếu không nằm trong nhóm alert_classes -> NO_ALERT
        if not self.class_filter.is_alert_candidate(det.class_name):
            return RiskLevel.NO_ALERT, ""

        by_score = self._level_from_score(proximity)
        if self.risk_model != "ttc" or distance_m is None:
            # Hướng chỉ mô tả vị trí; điều kiện chắn đường đã được xét bằng hành lang.
            return by_score, ""

        if distance_m <= self.critical_distance_m:
            return RiskLevel.HIGH, "; rất gần"
        if (estimate.ttc_sec is not None and estimate.ttc_sec <= self.ttc_high_sec
                and distance_m <= self.ttc_max_distance_m):
            return RiskLevel.HIGH, f"; TTC {estimate.ttc_sec:.1f} s ({estimate.source})"
        if not estimate.kinematics_known:
            return by_score, "; chưa đủ mẫu TTC, dùng luật khoảng cách"
        # Đã biết chuyển động tương đối: HIGH chỉ qua hai điều kiện trên.
        capped = RiskLevel.MEDIUM if by_score == RiskLevel.HIGH else by_score
        if estimate.ttc_sec is not None:
            return capped, f"; đang lại gần, TTC {estimate.ttc_sec:.1f} s"
        return capped, "; đứng yên tương đối"

    def _level_from_score(self, proximity: float) -> RiskLevel:
        if proximity >= self.depth_threshold_high:
            return RiskLevel.HIGH
        elif proximity >= self.depth_threshold_medium:
            return RiskLevel.MEDIUM
        elif proximity >= self.depth_threshold_low:
            return RiskLevel.LOW
        return RiskLevel.NO_ALERT

    def _apply_hysteresis(
        self,
        state: TrackState,
        candidate: RiskLevel,
        direction: Direction,
        proximity: float,
        is_new_depth_sample: bool
    ) -> RiskLevel:
        # Tên tùy chọn tương thích cũ: kích hoạt nhanh cho HIGH trong hành lang, bất kể hướng.
        if (
            self.instant_high_risk_in_center
            and candidate == RiskLevel.HIGH
        ):
            state.consecutive_high_count = self.confirmations_to_escalate
            state.consecutive_safe_count = 0
            return RiskLevel.HIGH

        if candidate == RiskLevel.HIGH:
            if is_new_depth_sample:
                state.consecutive_high_count += 1
            state.consecutive_safe_count = 0
            if state.consecutive_high_count >= self.confirmations_to_escalate:
                return RiskLevel.HIGH
            return RiskLevel.MEDIUM if state.current_risk != RiskLevel.HIGH else RiskLevel.HIGH

        elif candidate == RiskLevel.MEDIUM:
            state.consecutive_high_count = 0
            state.consecutive_safe_count = 0
            return RiskLevel.MEDIUM

        # Không giữ cảnh báo cũ khi vật ra khỏi hành lang, ở xa hoặc ROI lỗi.
        state.consecutive_high_count = 0
        state.consecutive_safe_count = 0
        return candidate

    def can_trigger_alert(self, track_id: int, target_risk: RiskLevel, current_mono: float) -> bool:
        """
        Kiểm tra xem track_id có được phép phát cảnh báo âm thanh không.
        Xử lý cooldown và ghi đè (override) khi nguy cơ tăng cấp.
        """
        if track_id not in self._tracks:
            return True

        state = self._tracks[track_id]
        time_since_alert = current_mono - state.last_alert_time

        # Nếu đang trong cooldown
        if time_since_alert < self.cooldown_sec:
            # Cho phép ghi đè nếu nguy cơ tăng lên HIGH mà trước đó chưa cảnh báo HIGH
            if self.allow_escalation_override and target_risk == RiskLevel.HIGH and state.last_alert_risk != RiskLevel.HIGH:
                return True
            return False

        return True

    def has_alerted(self, track_id: int) -> bool:
        state = self._tracks.get(track_id)
        return state is not None and state.last_alert_risk is not None

    def mark_alert_triggered(self, track_id: int, current_mono: float, risk_level: Optional[RiskLevel] = None) -> None:
        """Đánh dấu thời điểm đã phát cảnh báo cho track_id."""
        if track_id in self._tracks:
            self._tracks[track_id].last_alert_time = current_mono
            self._tracks[track_id].last_alert_risk = risk_level

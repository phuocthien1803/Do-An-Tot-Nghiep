"""Dựng toàn bộ pipeline từ cấu hình đã ghép (dùng chung cho app.py, đánh giá video và benchmark).

Bản gốc dựng từng thành phần trực tiếp trong app.py và lặp lại trong evaluation/benchmarks/common.py;
tách ra đây để mọi biến thể thí nghiệm (bản cải tiến / như bản gốc / bật tắt từng phần) dùng đúng
một cách dựng, kết quả so sánh mới công bằng.
"""
import os
from typing import Any, Dict, Optional

from src.audio.earcons import EarconLibrary
from src.detection.class_filter import ClassFilter
from src.fusion.alert_aggregator import AlertAggregator
from src.fusion.floor_hazards import FloorHazardDetector
from src.fusion.generic_obstacle import build_generic_obstacle_detector
from src.fusion.risk_fsm import RiskFSM
from src.fusion.spatial_zones import SpatialZones
from src.fusion.synchronizer import Synchronizer
from src.fusion.ttc import TTCEstimator
from src.fusion.walking_corridor import WalkingCorridor


def build_class_filter(cfg: Dict[str, dict]) -> ClassFilter:
    det = cfg["model"].get("detection", {})
    return ClassFilter(
        enabled_classes=det.get("enabled_classes"),
        alert_classes=det.get("alert_classes"),
        custom_vi_names=cfg["fusion"].get("labels_vi"),
    )


def tracking_cfg(cfg: Dict[str, dict]) -> Dict[str, Any]:
    return cfg["model"].get("tracking", {}) or {}


def detector_confidence(cfg: Dict[str, dict]) -> float:
    """Ngưỡng truyền cho YOLO: ByteTrack cần cả detection tin cậy thấp để nối track."""
    det = cfg["model"].get("detection", {})
    conf = float(det.get("confidence_threshold", 0.25))
    trk = tracking_cfg(cfg)
    if trk.get("type", "iou") == "bytetrack":
        return min(conf, float(trk.get("low_threshold", 0.1)))
    return conf


def resolve_weights(cfg: Dict[str, dict], override: Optional[str] = None) -> str:
    det = cfg["model"].get("detection", {})
    path = override or det.get("weights_path", "models/weights/yolov8n.pt")
    fallback = det.get("fallback_weights_path")
    if not os.path.exists(path) and fallback and os.path.exists(fallback):
        print(f"[Factory] Không thấy {path}; dùng trọng số dự phòng {fallback}.")
        return fallback
    return path


def build_detector(cfg: Dict[str, dict], device: str, class_filter: Optional[ClassFilter] = None,
                   weights: Optional[str] = None):
    from src.detection.yolo_detector import YoloDetector
    det = cfg["model"].get("detection", {})
    runtime = cfg["app"].get("runtime", {})
    if runtime.get("torch_threads"):
        import torch
        torch.set_num_threads(int(runtime["torch_threads"]))
    return YoloDetector(
        weights_path=resolve_weights(cfg, weights),
        class_filter=class_filter or build_class_filter(cfg),
        confidence_threshold=detector_confidence(cfg),
        iou_threshold=float(det.get("iou_threshold", 0.45)),
        device=device,
        input_size=int(det.get("input_size", 640)),
    )


def build_tracker(cfg: Dict[str, dict]):
    trk = tracking_cfg(cfg)
    conf = float(cfg["model"].get("detection", {}).get("confidence_threshold", 0.25))
    if trk.get("type", "iou") == "bytetrack":
        from src.tracking.bytetrack import ByteTracker
        return ByteTracker(
            high_threshold=conf,
            low_threshold=float(trk.get("low_threshold", 0.1)),
            new_track_threshold=float(trk.get("new_track_threshold", conf)),
            first_match_iou=float(trk.get("first_match_iou", 0.2)),
            second_match_iou=float(trk.get("second_match_iou", 0.5)),
            max_lost_sec=float(trk.get("max_lost_sec", 1.0)),
        )
    from src.tracking.byte_tracker import ByteTrackerAdapter
    return ByteTrackerAdapter(iou_threshold=float(trk.get("iou_threshold", 0.3)))


def build_synchronizer(cfg: Dict[str, dict]) -> Synchronizer:
    sc = cfg["fusion"].get("synchronizer", {})
    return Synchronizer(
        max_pair_skew_ms=float(sc.get("max_pair_skew_ms", 600.0)),
        max_detection_age_ms=float(sc.get("max_detection_age_ms", 1000.0)),
        max_depth_age_ms=float(sc.get("max_depth_age_ms", 1200.0)),
        max_alert_age_ms=float(sc.get("max_alert_age_ms", 1500.0)),
    )


def build_corridor(cfg: Dict[str, dict]) -> WalkingCorridor:
    return WalkingCorridor(**cfg["fusion"].get("walking_corridor", {}))


def build_risk_fsm(cfg: Dict[str, dict], roi_extractor, corridor: Optional[WalkingCorridor] = None,
                   class_filter: Optional[ClassFilter] = None) -> RiskFSM:
    fusion = cfg["fusion"]
    fsm_cfg = fusion.get("risk_fsm", {})
    spatial_cfg = fusion.get("spatial", {})
    corridor = corridor or build_corridor(cfg)
    ttc_cfg = fusion.get("ttc", {}) or {}
    risk_model = fsm_cfg.get("risk_model", "distance")
    ttc = None
    if risk_model == "ttc":
        ttc = TTCEstimator(
            window_sec=float(ttc_cfg.get("window_sec", 1.5)),
            min_samples=int(ttc_cfg.get("min_samples", 4)),
            min_span_sec=float(ttc_cfg.get("min_span_sec", 0.35)),
            min_closing_speed_mps=float(ttc_cfg.get("min_closing_speed_mps", 0.25)),
            min_expansion_rate=float(ttc_cfg.get("min_expansion_rate", 0.12)),
            max_ttc_sec=float(ttc_cfg.get("max_ttc_sec", 10.0)),
            agree_ratio=float(ttc_cfg.get("agree_ratio", 2.0)),
        )
    floor_cfg = fusion.get("floor_hazards", {}) or {}
    floor_detector = None
    if floor_cfg.get("enabled", False):
        floor_detector = FloorHazardDetector(
            corridor=corridor,
            drop_ratio=float(floor_cfg.get("drop_ratio", 0.70)),
            rise_ratio=float(floor_cfg.get("rise_ratio", 1.35)),
            min_area_ratio=float(floor_cfg.get("min_area_ratio", 0.06)),
            max_drop_distance_m=float(floor_cfg.get("max_drop_distance_m", 3.0)),
            max_low_obstacle_distance_m=float(floor_cfg.get("max_low_obstacle_distance_m", 1.5)),
        )
    confirm = fusion.get("hazard_confirmation", {}) or {}
    return RiskFSM(
        class_filter=class_filter or build_class_filter(cfg),
        spatial_zones=SpatialZones(
            left_ratio=spatial_cfg.get("left_ratio", 0.35),
            center_ratio=spatial_cfg.get("center_ratio", 0.30),
            right_ratio=spatial_cfg.get("right_ratio", 0.35),
        ),
        walking_corridor=corridor,
        roi_extractor=roi_extractor,
        depth_threshold_high=float(fsm_cfg.get("depth_threshold_high", 0.75)),
        depth_threshold_medium=float(fsm_cfg.get("depth_threshold_medium", 0.45)),
        depth_threshold_low=float(fsm_cfg.get("depth_threshold_low", 0.25)),
        confirmations_to_escalate=int(fsm_cfg.get("confirmations_to_escalate", 1)),
        confirmations_to_deescalate=int(fsm_cfg.get("confirmations_to_deescalate", 3)),
        instant_high_risk_in_center=bool(fsm_cfg.get("instant_high_risk_in_center", True)),
        cooldown_sec=float(fsm_cfg.get("track_alert_cooldown_sec", 4.0)),
        allow_escalation_override=bool(fsm_cfg.get("allow_escalation_override", True)),
        generic_obstacle_detector=build_generic_obstacle_detector(fusion, corridor),
        risk_model=risk_model,
        ttc_estimator=ttc,
        critical_distance_m=float(fsm_cfg.get("critical_distance_m", 1.0)),
        ttc_high_sec=float(fsm_cfg.get("ttc_high_sec", 2.5)),
        ttc_max_distance_m=float(fsm_cfg.get("ttc_max_distance_m", 5.0)),
        floor_hazard_detector=floor_detector,
        hazard_confirm_required=int(confirm.get("required", 1)),
        hazard_confirm_window=int(confirm.get("window", 1)),
    )


def build_alert_aggregator(cfg: Dict[str, dict], risk_fsm: RiskFSM,
                           earcons: Optional[EarconLibrary] = None) -> AlertAggregator:
    fusion = cfg["fusion"]
    agg_cfg = fusion.get("alert_aggregator", {})
    fsm_cfg = fusion.get("risk_fsm", {})
    sc = fusion.get("synchronizer", {})
    return AlertAggregator(
        risk_fsm=risk_fsm,
        alert_lifetime_sec=float(agg_cfg.get("alert_lifetime_sec", fsm_cfg.get("alert_lifetime_sec", 3.0))),
        global_alert_interval_sec=float(agg_cfg.get("global_alert_interval_sec", 3.5)),
        max_detection_age_ms=float(sc.get("max_detection_age_ms", 1000.0)),
        max_depth_age_ms=float(sc.get("max_depth_age_ms", 1200.0)),
        max_alert_age_ms=float(sc.get("max_alert_age_ms", 1500.0)),
        message_style=agg_cfg.get("message_style", "day_du"),
        bypass_interval=bool(agg_cfg.get("bypass_interval", False)),
        min_bypass_gap_sec=float(agg_cfg.get("min_bypass_gap_sec", 1.0)),
        critical_ttc_sec=float(agg_cfg.get("critical_ttc_sec", 1.5)),
        earcon_provider=earcons.for_alert if earcons is not None else None,
    )


def build_depth(cfg: Dict[str, dict], device: str):
    from src.depth.factory import build_depth_estimator
    return build_depth_estimator(cfg["model"], device, app_cfg=cfg["app"])


def build_roi(cfg: Dict[str, dict]):
    from src.depth.factory import build_roi_extractor
    return build_roi_extractor(cfg["model"], cfg["fusion"])


def build_earcons(cfg: Dict[str, dict]) -> Optional[EarconLibrary]:
    audio = cfg["app"].get("audio", {})
    if not audio.get("earcons", False):
        return None
    return EarconLibrary(audio.get("earcon_cache_dir")).ensure()


def build_risk_stack(cfg: Dict[str, dict], earcons: Optional[EarconLibrary] = None):
    """(class_filter, corridor, roi, risk_fsm, aggregator, synchronizer, tracker) cho app & benchmark."""
    class_filter = build_class_filter(cfg)
    corridor = build_corridor(cfg)
    roi = build_roi(cfg)
    fsm = build_risk_fsm(cfg, roi, corridor, class_filter)
    aggregator = build_alert_aggregator(cfg, fsm, earcons)
    return {
        "class_filter": class_filter,
        "corridor": corridor,
        "roi": roi,
        "risk_fsm": fsm,
        "aggregator": aggregator,
        "synchronizer": build_synchronizer(cfg),
        "tracker": build_tracker(cfg),
    }

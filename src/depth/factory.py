from typing import Any, Dict, Optional

from src.depth.depth_estimator import DepthEstimator
from src.depth.floor_model import FloorPlaneScaler
from src.depth.roi_extractor import ROIExtractor, metric_breakpoints

DEFAULT_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def build_floor_scaler(model_cfg: Dict[str, Any], app_cfg: Optional[Dict[str, Any]] = None
                       ) -> Optional[FloorPlaneScaler]:
    """Tự hiệu chỉnh hệ số mét theo mặt sàn (depth.auto_scale); cần độ cao và trường nhìn camera."""
    depth_cfg = model_cfg.get("depth", {})
    auto = depth_cfg.get("auto_scale", {}) or {}
    if not auto.get("enabled", False) or depth_cfg.get("representation") != "metric_depth":
        return None
    camera = (app_cfg or {}).get("camera", {})
    return FloorPlaneScaler(
        camera_height_m=float(auto.get("camera_height_m", camera.get("height_m", 1.3))),
        hfov_deg=float(auto.get("hfov_deg", camera.get("hfov_deg", 70.0))),
        fallback_scale=float(depth_cfg.get("metric_scale", 1.0)),
        min_inlier_ratio=float(auto.get("min_inlier_ratio", 0.45)),
        ema_alpha=float(auto.get("ema_alpha", 0.25)),
    )


def build_depth_estimator(model_cfg: Dict[str, Any], device: str,
                          app_cfg: Optional[Dict[str, Any]] = None) -> DepthEstimator:
    """Dựng DepthEstimator từ khối `depth` của configs/model_config.yaml."""
    depth_cfg = model_cfg.get("depth", {})
    runtime = (app_cfg or {}).get("runtime", {})
    return DepthEstimator(
        model_name=depth_cfg.get("hf_repo_id", DEFAULT_DEPTH_MODEL),
        device=device,
        input_size=(int(depth_cfg.get("input_height", 256)), int(depth_cfg.get("input_width", 256))),
        representation=depth_cfg.get("representation", "relative_depth"),
        metric_scale=float(depth_cfg.get("metric_scale", 1.0)),
        floor_scaler=build_floor_scaler(model_cfg, app_cfg),
        torch_threads=runtime.get("torch_threads"),
    )


def build_roi_extractor(model_cfg: Dict[str, Any], fusion_cfg: Dict[str, Any]) -> ROIExtractor:
    """Dựng ROIExtractor; ngưỡng mét được neo vào ngưỡng điểm của RiskFSM."""
    depth_cfg = model_cfg.get("depth", {})
    fsm_cfg = fusion_cfg.get("risk_fsm", {})
    breakpoints = metric_breakpoints(
        high_m=float(fsm_cfg.get("distance_threshold_high_m", 1.5)),
        medium_m=float(fsm_cfg.get("distance_threshold_medium_m", 2.5)),
        low_m=float(fsm_cfg.get("distance_threshold_low_m", 3.5)),
        score_high=float(fsm_cfg.get("depth_threshold_high", 0.75)),
        score_medium=float(fsm_cfg.get("depth_threshold_medium", 0.45)),
        score_low=float(fsm_cfg.get("depth_threshold_low", 0.25)),
    )
    return ROIExtractor(
        erosion_ratio=float(depth_cfg.get("roi_erosion_ratio", 0.15)),
        percentile=float(depth_cfg.get("percentile_roi", 80.0)),
        metric_breakpoints=breakpoints,
    )

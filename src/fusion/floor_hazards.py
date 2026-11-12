"""Nguy cơ trên mặt sàn trong hành lang đi: bậc xuống/hố và vật cản thấp.

Dùng đường sàn kỳ vọng do FloorPlaneScaler khớp (DepthMap.preprocess_transform["floor"]):
với pixel sàn ở hàng v, nghịch đảo độ sâu kỳ vọng e(v) = a v + b. So sánh nghịch đảo độ sâu đo
được m = 1/Z với e(v):
- m / e < drop_ratio : xa hơn mặt sàn đáng lẽ phải có -> sàn bị hụt (bậc xuống, hố, mép cầu thang).
- m / e > rise_ratio : gần hơn mặt sàn -> có vật nhô lên khỏi sàn (vali, bậc lên, chậu cây...).
Vùng đã có bbox YOLO được loại (RiskFSM đánh giá theo lớp). Chỉ chạy khi frame này khớp sàn hợp lệ;
nếu không, hệ thống KHÔNG suy ra "sàn bằng phẳng" mà chỉ đơn giản là không có cảnh báo loại này.
"""
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from src.contracts.depth_map import DepthMap, DepthRepresentation
from src.contracts.detection import BoundingBox
from src.fusion.walking_corridor import WalkingCorridor

DROP_OFF = "drop_off"
LOW_OBSTACLE = "low_obstacle"


@dataclass(frozen=True)
class FloorHazard:
    kind: str
    bbox: BoundingBox
    distance_m: float
    area_ratio: float


class TemporalVote:
    """Xác nhận k trên n mẫu độ sâu gần nhất (mỗi depth frame chỉ bỏ phiếu một lần)."""

    def __init__(self, required: int = 2, window: int = 3):
        if not 1 <= required <= window:
            raise ValueError("Cần 1 <= required <= window")
        self.required = required
        self._votes: Deque[Tuple[int, bool]] = deque(maxlen=window)

    def vote(self, depth_frame_id: int, positive: bool) -> bool:
        if self._votes and self._votes[-1][0] == depth_frame_id:
            self._votes[-1] = (depth_frame_id, positive)
        else:
            self._votes.append((depth_frame_id, positive))
        return sum(1 for _, p in self._votes if p) >= self.required

    def reset(self) -> None:
        self._votes.clear()


class FloorHazardDetector:
    def __init__(
        self,
        corridor: WalkingCorridor,
        drop_ratio: float = 0.70,
        rise_ratio: float = 1.35,
        min_area_ratio: float = 0.06,
        min_pixels: int = 60,
        max_drop_distance_m: float = 3.0,
        max_low_obstacle_distance_m: float = 1.5,
        horizon_margin_px: float = 25.0,
        step_px: int = 2,
    ):
        if not 0 < drop_ratio < 1 < rise_ratio:
            raise ValueError("Cần 0 < drop_ratio < 1 < rise_ratio")
        self.corridor = corridor
        self.drop_ratio = drop_ratio
        self.rise_ratio = rise_ratio
        self.min_area_ratio = min_area_ratio
        self.min_pixels = min_pixels
        self.max_drop_distance_m = max_drop_distance_m
        self.max_low_obstacle_distance_m = max_low_obstacle_distance_m
        self.horizon_margin_px = horizon_margin_px
        self.step_px = step_px

    @staticmethod
    def floor_line(depth_map: DepthMap) -> Optional[Tuple[float, float, float]]:
        info = (depth_map.preprocess_transform or {}).get("floor") or {}
        if not info.get("valid") or "inv_depth_slope" not in info:
            return None
        return float(info["inv_depth_slope"]), float(info["inv_depth_intercept"]), float(info["horizon_row"])

    def find(self, depth_map: Optional[DepthMap], detected_boxes: Iterable[BoundingBox]) -> List[FloorHazard]:
        if depth_map is None or depth_map.representation != DepthRepresentation.METRIC_DEPTH:
            return []
        line = self.floor_line(depth_map)
        if line is None:
            return []
        a, b, horizon = line
        h, w = depth_map.shape
        s = self.step_px
        values = depth_map.values[::s, ::s]
        valid = depth_map.valid_mask[::s, ::s].copy()
        hs, ws = values.shape
        mask = np.zeros((hs, ws), np.uint8)
        poly = self.corridor.polygon(w, h)
        if not poly:
            return []
        cv2.fillPoly(mask, [np.round(np.array(poly) / s).astype(np.int32)], 1)
        region = mask.astype(bool) & valid
        top = int(max(0.0, horizon + self.horizon_margin_px) / s)
        region[:top, :] = False
        for box in detected_boxes:
            bb = box.clamp(w, h)
            region[int(bb.ymin / s):int(np.ceil(bb.ymax / s)) + 1,
                   int(bb.xmin / s):int(np.ceil(bb.xmax / s)) + 1] = False
        total = int(region.sum())
        if total < self.min_pixels:
            return []
        rows = (np.arange(hs) * s).astype(np.float64)[:, None]
        expected_inv = a * rows + b
        region &= expected_inv > 1e-6
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (1.0 / np.maximum(values, 1e-6)) / np.maximum(expected_inv, 1e-6)
            expected_dist = 1.0 / np.maximum(expected_inv, 1e-6)
        hazards = []
        drop = region & (ratio < self.drop_ratio) & (expected_dist <= self.max_drop_distance_m)
        found = self._largest(drop, total, s)
        if found is not None:
            comp, bbox, area_ratio = found
            nearest_row = int(np.nonzero(comp.any(axis=1))[0].max())
            distance = float(1.0 / max(a * nearest_row * s + b, 1e-6))
            hazards.append(FloorHazard(DROP_OFF, bbox, distance, area_ratio))
        rise = region & (ratio > self.rise_ratio) & (values <= self.max_low_obstacle_distance_m)
        found = self._largest(rise, total, s)
        if found is not None:
            comp, bbox, area_ratio = found
            hazards.append(FloorHazard(LOW_OBSTACLE, bbox, float(np.percentile(values[comp], 20)), area_ratio))
        return hazards

    def _largest(self, mask: np.ndarray, total: int, s: int):
        if int(mask.sum()) < self.min_pixels:
            return None
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if n <= 1:
            return None
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < self.min_pixels or area / total < self.min_area_ratio:
            return None
        x, y, bw, bh = (int(v) for v in stats[idx, :4])
        bbox = BoundingBox(float(x * s), float(y * s), float((x + bw) * s), float((y + bh) * s))
        return labels == idx, bbox, area / total

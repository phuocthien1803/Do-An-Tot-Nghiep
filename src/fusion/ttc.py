"""Ước lượng thời gian đến va chạm (TTC) theo từng track, kết hợp hai nguồn độc lập.

1. TTC theo độ sâu: hồi quy Theil-Sen (trung vị các độ dốc cặp, bền với nhiễu) của khoảng cách
   theo thời gian trong cửa sổ ngắn -> tốc độ tiến lại gần v; TTC = d / v.
2. TTC theo độ phình bbox ("looming"): với camera lỗ kim, chiều cao ảnh h ~ 1/Z nên
   d(ln h)/dt = -Z'/Z = 1/TTC. Nguồn này KHÔNG cần mét tuyệt đối nên không bị sai số hệ số
   tỉ lệ độ sâu; chỉ dùng khi bbox không bị cắt ở mép trên/dưới ảnh.

Kết hợp: hai nguồn cùng có và lệch nhau không quá `agree_ratio` lần -> trung bình nhân;
lệch nhiều -> lấy giá trị nhỏ hơn (thiên về an toàn); chỉ một nguồn -> dùng nguồn đó.
TTC chỉ phản ánh chuyển động tương đối (người dùng bước tới cũng làm TTC giảm), đúng với
mục tiêu cảnh báo va chạm.
"""
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.contracts.detection import BoundingBox


@dataclass(frozen=True)
class TTCEstimate:
    ttc_sec: Optional[float]
    closing_speed_mps: Optional[float]
    source: str  # none | depth | scale | fused
    depth_samples: int
    kinematics_known: bool  # đủ mẫu độ sâu để kết luận "đang lại gần" hay "đứng yên tương đối"

    @property
    def approaching(self) -> bool:
        return self.ttc_sec is not None


NO_ESTIMATE = TTCEstimate(None, None, "none", 0, False)


def theil_sen(ts: List[float], ys: List[float], min_dt: float = 0.05) -> Optional[Tuple[float, float]]:
    """Trả về (độ dốc, giá trị khớp tại thời điểm cuối) hoặc None nếu không đủ cặp."""
    slopes = [(ys[j] - ys[i]) / (ts[j] - ts[i])
              for i in range(len(ts)) for j in range(i + 1, len(ts))
              if ts[j] - ts[i] >= min_dt]
    if not slopes:
        return None
    slope = float(np.median(slopes))
    intercept = float(np.median([y - slope * t for t, y in zip(ts, ys)]))
    return slope, intercept + slope * ts[-1]


class TTCEstimator:
    def __init__(
        self,
        window_sec: float = 1.5,
        min_samples: int = 4,
        min_span_sec: float = 0.35,
        min_closing_speed_mps: float = 0.25,
        min_expansion_rate: float = 0.12,
        max_ttc_sec: float = 10.0,
        border_px: float = 3.0,
        min_box_height_px: float = 24.0,
        agree_ratio: float = 2.0,
    ):
        self.window_sec = window_sec
        self.min_samples = min_samples
        self.min_span_sec = min_span_sec
        self.min_closing_speed_mps = min_closing_speed_mps
        self.min_expansion_rate = min_expansion_rate
        self.max_ttc_sec = max_ttc_sec
        self.border_px = border_px
        self.min_box_height_px = min_box_height_px
        self.agree_ratio = agree_ratio
        self._depth: Dict[int, Deque[Tuple[float, float, int]]] = {}
        self._scale: Dict[int, Deque[Tuple[float, float]]] = {}

    def _trim(self, samples: Deque, now: float) -> None:
        while samples and now - samples[0][0] > self.window_sec:
            samples.popleft()

    def observe(self, track_id: int, detection_time: float, bbox: BoundingBox, frame_height: float,
                distance_m: Optional[float] = None, depth_time: Optional[float] = None,
                depth_frame_id: Optional[int] = None) -> None:
        """Ghi một quan sát. Mẫu độ sâu được khử trùng lặp theo depth_frame_id."""
        if distance_m is not None and depth_time is not None and math.isfinite(distance_m) and distance_m > 0:
            dq = self._depth.setdefault(track_id, deque())
            if not dq or dq[-1][2] != depth_frame_id:
                if not dq or depth_time > dq[-1][0]:
                    dq.append((depth_time, float(distance_m), depth_frame_id))
            self._trim(dq, depth_time)
        cut = bbox.ymin <= self.border_px or bbox.ymax >= frame_height - self.border_px
        sq = self._scale.setdefault(track_id, deque())
        if not cut and bbox.height >= self.min_box_height_px:
            if not sq or detection_time > sq[-1][0]:
                sq.append((detection_time, math.log(bbox.height)))
        else:
            sq.clear()  # bbox bị cắt: chuỗi chiều cao không còn đo khoảng cách
        self._trim(sq, detection_time)

    def _span_ok(self, samples) -> bool:
        return len(samples) >= self.min_samples and samples[-1][0] - samples[0][0] >= self.min_span_sec

    def estimate(self, track_id: int) -> TTCEstimate:
        depth = self._depth.get(track_id) or ()
        scale = self._scale.get(track_id) or ()
        ttc_depth = closing = None
        known = self._span_ok(depth)
        if known:
            fit = theil_sen([s[0] for s in depth], [s[1] for s in depth])
            if fit is not None:
                slope, d_now = fit
                closing = -slope
                if closing >= self.min_closing_speed_mps and d_now > 0:
                    ttc_depth = d_now / closing
            else:
                known = False
        ttc_scale = None
        if self._span_ok(scale):
            fit = theil_sen([s[0] for s in scale], [s[1] for s in scale])
            if fit is not None and fit[0] >= self.min_expansion_rate:
                ttc_scale = 1.0 / fit[0]
        values = [v for v in (ttc_depth, ttc_scale) if v is not None and v <= self.max_ttc_sec]
        if not values:
            return TTCEstimate(None, closing, "none", len(depth), known)
        if len(values) == 2:
            lo, hi = min(values), max(values)
            ttc = math.sqrt(lo * hi) if hi / lo <= self.agree_ratio else lo
            source = "fused"
        else:
            ttc = values[0]
            source = "depth" if ttc_depth is not None and ttc_depth <= self.max_ttc_sec else "scale"
        return TTCEstimate(float(ttc), closing, source, len(depth), known)

    def forget_missing(self, active_ids: Iterable[int]) -> None:
        keep = set(active_ids)
        for store in (self._depth, self._scale):
            for tid in [t for t in store if t not in keep]:
                del store[tid]

    def reset_track(self, track_id: int) -> None:
        self._depth.pop(track_id, None)
        self._scale.pop(track_id, None)

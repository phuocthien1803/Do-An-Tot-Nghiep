"""Mô hình mặt sàn từ độ sâu đơn mắt: tự hiệu chỉnh hệ số tỉ lệ mét và tìm bậc/hố/vật thấp.

Hình học: camera cao h (m) so với sàn phẳng, nghiêng xuống góc θ, roll ~ 0, tiêu cự fy (pixel).
Với điểm sàn ở hàng ảnh v, độ sâu theo trục quang Z thoả

    1 / Z = (cos θ / (fy h)) · v + (sin θ - cos θ · cy / fy) / h      (tuyến tính theo v)

Model cho độ sâu thô D = Z / k (k là hệ số tỉ lệ cần tìm). Khớp đường thẳng bền vững
1/D = A v + B trên các pixel phần dưới ảnh (RANSAC loại vật cản), suy ra
    hàng chân trời  v_h = -B / A,   góc nghiêng  θ = atan((cy - v_h) / fy),
    hệ số tỉ lệ     k = A · fy · h / cos θ.
Chỉ cần biết độ cao camera h và trường nhìn (để có fy); không cần biết góc nghiêng.

Giả định có thể sai (khi đó bộ khớp bị loại và hệ thống giữ hệ số cũ/hệ số cấu hình):
sàn không nhìn thấy (camera đặt trên bàn, sát tường), sàn không phẳng, camera nghiêng ngang.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FloorFit:
    valid: bool
    reason: str
    slope: float = 0.0          # A (theo 1/mét thô mỗi pixel)
    intercept: float = 0.0      # B
    horizon_row: float = 0.0    # v_h
    pitch_deg: float = 0.0      # θ
    scale: float = 0.0          # k của frame này
    inlier_ratio: float = 0.0
    inliers: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {"valid": self.valid, "reason": self.reason, "slope": self.slope,
                "intercept": self.intercept, "horizon_row": self.horizon_row,
                "pitch_deg": self.pitch_deg, "scale": self.scale,
                "inlier_ratio": self.inlier_ratio, "inliers": self.inliers}


def focal_from_hfov(width: int, hfov_deg: float) -> float:
    return (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


@dataclass
class FloorPlaneScaler:
    """Khớp mặt sàn mỗi frame độ sâu và làm mượt hệ số tỉ lệ theo thời gian (EMA trong log)."""

    camera_height_m: float = 1.3
    hfov_deg: float = 70.0
    fallback_scale: float = 0.60
    row_start_ratio: float = 0.55
    row_end_ratio: float = 0.97
    col_margin_ratio: float = 0.20
    step_px: int = 4
    ransac_iterations: int = 80
    residual_ratio: float = 0.06
    min_inlier_ratio: float = 0.45
    min_row_span_ratio: float = 0.18
    pitch_range_deg: tuple = (-10.0, 50.0)
    scale_range: tuple = (0.25, 1.6)
    ema_alpha: float = 0.25
    seed: int = 7
    _log_scale: Optional[float] = field(default=None, init=False, repr=False)
    _rng: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.camera_height_m <= 0 or not 20.0 <= self.hfov_deg <= 170.0:
            raise ValueError("camera_height_m phải > 0 và hfov_deg trong [20, 170]")
        self._rng = np.random.default_rng(self.seed)

    @property
    def current_scale(self) -> float:
        return math.exp(self._log_scale) if self._log_scale is not None else self.fallback_scale

    @property
    def calibrated(self) -> bool:
        return self._log_scale is not None

    def fit(self, raw_depth: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> FloorFit:
        """Khớp mặt sàn trên độ sâu THÔ của model (chưa nhân hệ số)."""
        h, w = raw_depth.shape
        r0, r1 = int(h * self.row_start_ratio), int(h * self.row_end_ratio)
        c0, c1 = int(w * self.col_margin_ratio), int(w * (1.0 - self.col_margin_ratio))
        rows = np.arange(r0, r1, self.step_px)
        cols = np.arange(c0, c1, self.step_px)
        if len(rows) < 4 or len(cols) < 2:
            return FloorFit(False, "vùng sàn quá nhỏ")
        grid = raw_depth[np.ix_(rows, cols)].astype(np.float64)
        ok = np.isfinite(grid) & (grid > 0.05)
        if valid_mask is not None:
            ok &= valid_mask[np.ix_(rows, cols)]
        vv = np.repeat(rows[:, None], len(cols), axis=1)[ok].astype(np.float64)
        inv = (1.0 / grid[ok])
        n = inv.size
        if n < 40:
            return FloorFit(False, "không đủ điểm độ sâu hợp lệ")
        best_inliers, best_count = None, 0
        for _ in range(self.ransac_iterations):
            i, j = self._rng.choice(n, 2, replace=False)
            if abs(vv[i] - vv[j]) < self.step_px * 2:
                continue
            a = (inv[j] - inv[i]) / (vv[j] - vv[i])
            if a <= 0:
                continue
            b = inv[i] - a * vv[i]
            pred = a * vv + b
            inl = np.abs(inv - pred) <= self.residual_ratio * np.maximum(pred, 1e-6)
            count = int(inl.sum())
            if count > best_count:
                best_count, best_inliers = count, inl
        if best_inliers is None:
            return FloorFit(False, "không khớp được đường sàn (độ sâu không tăng dần theo hàng)")
        ratio = best_count / n
        a, b = np.polyfit(vv[best_inliers], inv[best_inliers], 1)
        span = (vv[best_inliers].max() - vv[best_inliers].min()) / h
        if a <= 0:
            return FloorFit(False, "độ dốc sàn không hợp lệ", inlier_ratio=ratio, inliers=best_count)
        horizon = -b / a
        fy = focal_from_hfov(w, self.hfov_deg)  # điểm ảnh vuông: fy = fx
        cy = h / 2.0
        pitch = math.degrees(math.atan2(cy - horizon, fy))
        scale = a * fy * self.camera_height_m / math.cos(math.radians(pitch))
        base = dict(slope=float(a), intercept=float(b), horizon_row=float(horizon),
                    pitch_deg=float(pitch), scale=float(scale), inlier_ratio=float(ratio),
                    inliers=best_count)
        if ratio < self.min_inlier_ratio:
            return FloorFit(False, f"sàn bị che nhiều ({ratio:.0%} điểm khớp)", **base)
        if span < self.min_row_span_ratio:
            return FloorFit(False, "điểm sàn trải quá ít hàng ảnh", **base)
        if not self.pitch_range_deg[0] <= pitch <= self.pitch_range_deg[1]:
            return FloorFit(False, f"góc nghiêng {pitch:.0f} độ ngoài khoảng hợp lý", **base)
        if not self.scale_range[0] <= scale <= self.scale_range[1]:
            return FloorFit(False, f"hệ số {scale:.2f} ngoài khoảng hợp lý", **base)
        return FloorFit(True, "ok", **base)

    def update(self, raw_depth: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> FloorFit:
        fit = self.fit(raw_depth, valid_mask)
        if fit.valid:
            log_k = math.log(fit.scale)
            if self._log_scale is None:
                self._log_scale = log_k
            else:
                self._log_scale += self.ema_alpha * (log_k - self._log_scale)
        return fit

    def apply(self, raw_depth: np.ndarray, valid_mask: Optional[np.ndarray] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Khớp sàn, cập nhật hệ số và trả về (độ sâu mét đã nhân hệ số, thông tin sàn).

        Thông tin sàn kèm đường sàn kỳ vọng trong đơn vị của độ sâu đầu ra:
        1/Z(v) = inv_depth_slope * v + inv_depth_intercept (dùng cho floor_hazards).
        """
        fit = self.update(raw_depth, valid_mask)
        scale = self.current_scale
        info = fit.as_dict()
        info["applied_scale"] = scale
        info["calibrated"] = self.calibrated
        if fit.valid:
            info["inv_depth_slope"] = fit.slope / scale
            info["inv_depth_intercept"] = fit.intercept / scale
        return raw_depth * scale, info

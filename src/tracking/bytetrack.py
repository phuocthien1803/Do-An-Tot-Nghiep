"""ByteTrack rút gọn có bộ lọc Kalman (Zhang và cộng sự, ECCV 2022), không phụ thuộc `lap`.

Khác bộ ghép IoU tham lam của bản gốc ở ba điểm, đều phục vụ ước lượng TTC theo track:
1. Bộ lọc Kalman vận tốc không đổi trên (cx, cy, tỉ lệ, chiều cao) dự đoán vị trí bbox ở frame
   kế tiếp, nên vẫn ghép đúng khi người dùng/vật di chuyển nhanh.
2. Ghép hai tầng: detection độ tin cậy cao trước, rồi detection độ tin cậy THẤP (0.1-0.25)
   chỉ được dùng để nối tiếp track đang có. Ảnh nhoè chuyển động làm độ tin cậy tụt dưới
   ngưỡng nhưng track vẫn giữ được mã, không sinh vật "mới" giả.
3. Ghép tối ưu toàn cục bằng thuật toán Hungary (scipy) thay vì tham lam theo thứ tự.

Track mất dấu được giữ `max_lost_sec` giây để nối lại khi vật bị che khuất ngắn.
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from src.contracts.detection import BoundingBox, Detection, DetectionResult

try:  # scipy là phụ thuộc sẵn có (EasyOCR/skimage); vẫn có fallback tham lam nếu thiếu.
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - chỉ chạy khi môi trường thiếu scipy
    linear_sum_assignment = None


class KalmanBoxFilter:
    """Kalman vận tốc không đổi trên trạng thái (cx, cy, a, h, vcx, vcy, va, vh); a = w / h."""

    _std_pos = 1.0 / 20
    _std_vel = 1.0 / 160

    def __init__(self):
        self._motion = np.eye(8)
        for i in range(4):
            self._motion[i, i + 4] = 1.0
        self._update = np.eye(4, 8)

    def initiate(self, xyah: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.r_[xyah, np.zeros(4)]
        h = xyah[3]
        std = [2 * self._std_pos * h, 2 * self._std_pos * h, 1e-2, 2 * self._std_pos * h,
               10 * self._std_vel * h, 10 * self._std_vel * h, 1e-5, 10 * self._std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h,
               self._std_vel * h, self._std_vel * h, 1e-5, self._std_vel * h]
        mean = self._motion @ mean
        cov = self._motion @ cov @ self._motion.T + np.diag(np.square(std))
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray, xyah: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h]
        proj_mean = self._update @ mean
        proj_cov = self._update @ cov @ self._update.T + np.diag(np.square(std))
        gain = np.linalg.solve(proj_cov, (cov @ self._update.T).T).T
        mean = mean + gain @ (xyah - proj_mean)
        cov = cov - gain @ proj_cov @ gain.T
        return mean, cov


def bbox_to_xyah(b: BoundingBox) -> np.ndarray:
    h = max(b.height, 1e-3)
    return np.array([b.center_x, b.center_y, b.width / h, h], dtype=np.float64)


def xyah_to_tlbr(x: np.ndarray) -> np.ndarray:
    cx, cy, a, h = x[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float64)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU giữa hai tập bbox tlbr (N x 4) và (M x 4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def assign(cost: np.ndarray, max_cost: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Ghép tối ưu; chỉ nhận cặp có chi phí <= max_cost."""
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    if linear_sum_assignment is not None:
        r_idx, c_idx = linear_sum_assignment(cost)
        pairs = [(r, c) for r, c in zip(r_idx, c_idx) if cost[r, c] <= max_cost]
    else:  # pragma: no cover
        pairs, used_r, used_c = [], set(), set()
        for flat in np.argsort(cost, axis=None):
            r, c = divmod(int(flat), cols)
            if r in used_r or c in used_c or cost[r, c] > max_cost:
                continue
            pairs.append((r, c))
            used_r.add(r)
            used_c.add(c)
    matched_r = {r for r, _ in pairs}
    matched_c = {c for _, c in pairs}
    return (pairs, [r for r in range(rows) if r not in matched_r],
            [c for c in range(cols) if c not in matched_c])


@dataclass
class _Track:
    track_id: int
    class_name: str
    mean: np.ndarray
    cov: np.ndarray
    score: float
    last_seen: float
    hits: int = 1
    lost: bool = False

    def predicted_tlbr(self) -> np.ndarray:
        return xyah_to_tlbr(self.mean)


class ByteTracker:
    """Bộ theo vết ByteTrack; giao diện `update(DetectionResult) -> DetectionResult` như bản cũ."""

    def __init__(
        self,
        high_threshold: float = 0.25,
        low_threshold: float = 0.10,
        new_track_threshold: float = 0.25,
        first_match_iou: float = 0.2,
        second_match_iou: float = 0.5,
        max_lost_sec: float = 1.0,
        class_aware: bool = True,
    ):
        if not 0.0 <= low_threshold <= high_threshold <= 1.0:
            raise ValueError("Cần 0 <= low_threshold <= high_threshold <= 1")
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.new_track_threshold = new_track_threshold
        self.first_match_iou = first_match_iou
        self.second_match_iou = second_match_iou
        self.max_lost_sec = max_lost_sec
        self.class_aware = class_aware
        self._kf = KalmanBoxFilter()
        self._tracks: List[_Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    @property
    def active_track_count(self) -> int:
        return sum(1 for t in self._tracks if not t.lost)

    def _cost(self, tracks: List[_Track], dets: List[Detection]) -> np.ndarray:
        t_boxes = np.array([t.predicted_tlbr() for t in tracks]).reshape(-1, 4)
        d_boxes = np.array([d.bbox.as_tuple() for d in dets], dtype=np.float64).reshape(-1, 4)
        cost = 1.0 - iou_matrix(t_boxes, d_boxes)
        if self.class_aware and len(tracks) and len(dets):
            same = np.array([[t.class_name == d.class_name for d in dets] for t in tracks])
            cost = np.where(same, cost, 1.0 + 1e-6)
        return cost

    def _apply(self, track: _Track, det: Detection, timestamp: float) -> Detection:
        track.mean, track.cov = self._kf.update(track.mean, track.cov, bbox_to_xyah(det.bbox))
        track.score = det.confidence
        track.last_seen = timestamp
        track.hits += 1
        track.lost = False
        # Trả về bbox đo được (không phải bbox Kalman) để ROI độ sâu khớp đúng pixel của vật.
        return Detection(class_name=det.class_name, confidence=det.confidence,
                         bbox=det.bbox, track_id=track.track_id)

    def update(self, det_result: DetectionResult) -> DetectionResult:
        ts = det_result.timestamp_mono
        dets = [d for d in det_result.detections if d.confidence >= self.low_threshold]
        high = [d for d in dets if d.confidence >= self.high_threshold]
        low = [d for d in dets if d.confidence < self.high_threshold]

        # Bỏ track mất dấu quá lâu TRƯỚC khi ghép, để vật xuất hiện lại sau đó nhận mã mới.
        self._tracks = [t for t in self._tracks
                        if not (t.lost and ts - t.last_seen > self.max_lost_sec)]
        for t in self._tracks:
            t.mean, t.cov = self._kf.predict(t.mean, t.cov)
            if t.mean[3] <= 0:
                t.mean[3] = 1e-3

        output: List[Detection] = []
        # Tầng 1: detection tin cậy cao với mọi track (kể cả track đang mất dấu).
        pool = list(self._tracks)
        pairs, rest_t, rest_d = assign(self._cost(pool, high), 1.0 - self.first_match_iou)
        for ti, di in pairs:
            output.append(self._apply(pool[ti], high[di], ts))
        # Tầng 2: detection tin cậy thấp chỉ nối tiếp track đang theo vết (không mất dấu).
        remaining = [pool[i] for i in rest_t if not pool[i].lost]
        pairs2, rest_t2, _ = assign(self._cost(remaining, low), 1.0 - self.second_match_iou)
        for ti, di in pairs2:
            output.append(self._apply(remaining[ti], low[di], ts))
        matched_ids = {d.track_id for d in output}
        for t in self._tracks:
            if t.track_id not in matched_ids:
                t.lost = True
        # Track mới chỉ sinh từ detection tin cậy cao chưa được ghép.
        for di in rest_d:
            det = high[di]
            if det.confidence < self.new_track_threshold:
                continue
            mean, cov = self._kf.initiate(bbox_to_xyah(det.bbox))
            track = _Track(track_id=self._next_id, class_name=det.class_name, mean=mean, cov=cov,
                           score=det.confidence, last_seen=ts)
            self._next_id += 1
            self._tracks.append(track)
            output.append(Detection(class_name=det.class_name, confidence=det.confidence,
                                    bbox=det.bbox, track_id=track.track_id))
        self._tracks = [t for t in self._tracks
                        if not (t.lost and ts - t.last_seen > self.max_lost_sec)]
        return DetectionResult(
            detections=output,
            frame_id=det_result.frame_id,
            timestamp_mono=det_result.timestamp_mono,
            latency_ms=det_result.latency_ms,
            preprocess_transform=det_result.preprocess_transform,
        )

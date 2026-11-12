"""Kiểm thử các cải tiến nhận thức: ByteTrack, TTC, mô hình mặt sàn, nguy cơ trên sàn, luật TTC."""
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.contracts.depth_map import DepthMap, DepthRepresentation
from src.contracts.detection import BoundingBox, Detection, DetectionResult
from src.contracts.frame_packet import FramePacket
from src.contracts.risk import DataQuality, RiskLevel
from src.depth.depth_estimator import DepthEstimator
from src.depth.floor_model import FloorPlaneScaler, focal_from_hfov
from src.depth.roi_extractor import ROIExtractor
from src.detection.class_filter import ClassFilter
from src.fusion.floor_hazards import DROP_OFF, LOW_OBSTACLE, FloorHazardDetector, TemporalVote
from src.fusion.generic_obstacle import GenericObstacleDetector
from src.fusion.risk_fsm import GENERIC_OBSTACLE_TRACK_ID, RiskFSM
from src.fusion.spatial_zones import SpatialZones
from src.fusion.synchronizer import SynchronizedPair
from src.fusion.ttc import TTCEstimator
from src.fusion.walking_corridor import WalkingCorridor
from src.tracking.bytetrack import ByteTracker

W, H = 640, 480


def dets(frame_id, t, items):
    return DetectionResult([Detection(c, s, BoundingBox(*b)) for c, s, b in items], frame_id, t, 1.0,
                           {"original_size": (W, H)})


def synth_floor(k=0.6, h=1.3, pitch=20.0, hfov=70.0):
    """Độ sâu thô (Z / k) của sàn phẳng; phía trên đường sàn 8 m là tường."""
    fy = focal_from_hfov(W, hfov)
    th = math.radians(pitch)
    v = np.arange(H, dtype=np.float64)[:, None]
    inv = (math.cos(th) / (fy * h)) * v + (math.sin(th) - math.cos(th) * (H / 2) / fy) / h
    z = np.where(inv > 1 / 8.0, 1.0 / np.maximum(inv, 1e-6), 8.0)
    z = np.repeat(z, W, axis=1)
    return (z / k).astype(np.float32), z


def metric_map(values, frame_id=1, t=10.0, floor=None):
    return DepthMap(DepthRepresentation.METRIC_DEPTH, "meter", False, values.astype(np.float32),
                    np.ones(values.shape, bool), frame_id, t, {"floor": floor} if floor else {})


class TestByteTracker(unittest.TestCase):
    def test_id_persists_while_object_moves(self):
        tracker = ByteTracker()
        ids = []
        for i in range(10):
            x = 100 + i * 15
            out = tracker.update(dets(i, i * 0.1, [("person", 0.8, (x, 100, x + 80, 300))]))
            ids.append(out.detections[0].track_id)
        self.assertEqual(len(set(ids)), 1)

    def test_low_confidence_continues_but_never_starts_a_track(self):
        tracker = ByteTracker(high_threshold=0.25, low_threshold=0.1)
        self.assertEqual(tracker.update(dets(0, 0.0, [("chair", 0.15, (100, 100, 200, 200))])).detections, [])
        tid = tracker.update(dets(1, 0.1, [("chair", 0.6, (100, 100, 200, 200))])).detections[0].track_id
        out = tracker.update(dets(2, 0.2, [("chair", 0.15, (104, 100, 204, 200))]))  # ảnh nhoè
        self.assertEqual([d.track_id for d in out.detections], [tid])

    def test_lost_track_recovers_within_buffer_and_expires_after(self):
        tracker = ByteTracker(max_lost_sec=1.0)
        box = ("person", 0.9, (100, 100, 180, 300))
        tid = tracker.update(dets(0, 0.0, [box])).detections[0].track_id
        tracker.update(dets(1, 0.3, []))
        self.assertEqual(tracker.update(dets(2, 0.6, [box])).detections[0].track_id, tid)
        tracker.update(dets(3, 0.7, []))
        self.assertNotEqual(tracker.update(dets(4, 2.5, [box])).detections[0].track_id, tid)

    def test_class_aware_matching(self):
        tracker = ByteTracker()
        a = tracker.update(dets(0, 0.0, [("person", 0.9, (100, 100, 200, 300))])).detections[0].track_id
        b = tracker.update(dets(1, 0.1, [("chair", 0.9, (100, 100, 200, 300))])).detections[0].track_id
        self.assertNotEqual(a, b)


class TestTTC(unittest.TestCase):
    def test_depth_ttc_for_approaching_object(self):
        est = TTCEstimator()
        for i in range(8):
            t = i * 0.125
            # bbox chạm mép trên/dưới -> không dùng nguồn phình bbox
            est.observe(1, t, BoundingBox(300, 0, 340, H), H, distance_m=3.0 - t, depth_time=t, depth_frame_id=i)
        e = est.estimate(1)
        self.assertEqual(e.source, "depth")
        self.assertAlmostEqual(e.closing_speed_mps, 1.0, places=2)
        self.assertAlmostEqual(e.ttc_sec, 3.0 - 0.875, places=1)

    def test_static_object_has_known_kinematics_but_no_ttc(self):
        est = TTCEstimator()
        for i in range(8):
            est.observe(1, i * 0.125, BoundingBox(300, 0, 340, H), H, distance_m=1.3, depth_time=i * 0.125,
                        depth_frame_id=i)
        e = est.estimate(1)
        self.assertTrue(e.kinematics_known)
        self.assertIsNone(e.ttc_sec)

    def test_repeated_depth_frame_is_not_counted_twice(self):
        est = TTCEstimator()
        for i in range(8):
            est.observe(1, i * 0.06, BoundingBox(300, 0, 340, H), H, distance_m=2.0, depth_time=0.0,
                        depth_frame_id=0)
        self.assertEqual(est.estimate(1).depth_samples, 1)

    def test_scale_ttc_from_looming_bbox(self):
        est = TTCEstimator()
        for i in range(8):
            t = i * 0.125
            z = 4.0 - t
            hh = 400.0 / z
            est.observe(2, t, BoundingBox(300, 240 - hh / 2, 340, 240 + hh / 2), H)
        e = est.estimate(2)
        self.assertEqual(e.source, "scale")
        self.assertAlmostEqual(e.ttc_sec, 3.125, delta=0.5)

    def test_outlier_does_not_break_depth_estimate(self):
        est = TTCEstimator()
        for i in range(9):
            t = i * 0.125
            d = 3.0 - t if i != 4 else 6.0  # một mẫu độ sâu lỗi
            est.observe(1, t, BoundingBox(300, 0, 340, H), H, distance_m=d, depth_time=t, depth_frame_id=i)
        self.assertAlmostEqual(est.estimate(1).closing_speed_mps, 1.0, delta=0.15)

    def test_disagreeing_sources_take_the_smaller_ttc(self):
        est = TTCEstimator()
        for i in range(8):
            t = i * 0.125
            hh = 400.0 / (4.0 - t)  # nguồn phình bbox ~3 s
            est.observe(3, t, BoundingBox(300, 240 - hh / 2, 340, 240 + hh / 2), H,
                        distance_m=1.2 - 0.5 * t, depth_time=t, depth_frame_id=i)  # nguồn độ sâu ~1,5 s
        e = est.estimate(3)
        self.assertEqual(e.source, "fused")
        self.assertLess(e.ttc_sec, 2.0)


class TestFloorModel(unittest.TestCase):
    def test_recovers_scale_and_pitch_from_flat_floor(self):
        raw, _ = synth_floor(k=0.6, pitch=20.0)
        scaler = FloorPlaneScaler(camera_height_m=1.3, hfov_deg=70.0, fallback_scale=1.0)
        fit = scaler.update(raw)
        self.assertTrue(fit.valid, fit.reason)
        self.assertAlmostEqual(fit.scale, 0.6, delta=0.02)
        self.assertAlmostEqual(fit.pitch_deg, 20.0, delta=1.0)
        self.assertAlmostEqual(scaler.current_scale, 0.6, delta=0.02)

    def test_no_floor_keeps_fallback_scale(self):
        scaler = FloorPlaneScaler(fallback_scale=0.6)
        fit = scaler.update(np.full((H, W), 1.5, np.float32))  # tường sát camera
        self.assertFalse(fit.valid)
        self.assertEqual(scaler.current_scale, 0.6)
        self.assertFalse(scaler.calibrated)

    def test_scale_is_smoothed_over_time(self):
        scaler = FloorPlaneScaler(ema_alpha=0.25)
        scaler.update(synth_floor(k=0.6)[0])
        scaler.update(synth_floor(k=0.8)[0])
        self.assertTrue(0.6 < scaler.current_scale < 0.7)

    def test_apply_outputs_metric_depth_and_floor_line(self):
        raw, z = synth_floor(k=0.6)
        scaled, info = FloorPlaneScaler().apply(raw)
        self.assertTrue(info["valid"])
        self.assertLess(float(np.median(np.abs(scaled[300:] - z[300:]) / z[300:])), 0.03)
        v = 400
        self.assertAlmostEqual(1.0 / (info["inv_depth_slope"] * v + info["inv_depth_intercept"]),
                               float(z[v, 320]), delta=0.05)

    def test_estimator_applies_auto_scale(self):
        raw, z = synth_floor(k=0.6)
        est = DepthEstimator.__new__(DepthEstimator)
        est.model_name, est.input_size, est.device = "fake", (256, 256), "cpu"
        est.representation = DepthRepresentation.METRIC_DEPTH
        est.metric_scale = 1.0
        est.floor_scaler = FloorPlaneScaler()
        est._load_error = None
        est._processor = lambda images, return_tensors, size: {"pixel_values": torch.zeros(1, 3, 252, 336)}
        est._model = lambda **kw: SimpleNamespace(predicted_depth=torch.from_numpy(raw)[None])
        packet = FramePacket(1, "t", np.zeros((H, W, 3), np.uint8), (W, H), 5.0)
        dm = est.estimate(packet)
        self.assertEqual(dm.preprocess_transform["model_input_hw"], (252, 336))
        self.assertTrue(dm.preprocess_transform["floor"]["valid"])
        self.assertAlmostEqual(dm.preprocess_transform["metric_scale"], 0.6, delta=0.02)
        self.assertAlmostEqual(float(dm.values[400, 320]), float(z[400, 320]), delta=0.08)


class TestFloorHazards(unittest.TestCase):
    def build(self, modify):
        raw, z = synth_floor(k=0.6)
        z = z.copy()
        modify(z)
        scaled, info = FloorPlaneScaler().apply((z / 0.6).astype(np.float32))
        return metric_map(scaled, floor=info)

    def test_drop_off_is_detected(self):
        def drop(z):
            z[290:380, 240:400] *= 2.2
        hazards = {h.kind for h in FloorHazardDetector(WalkingCorridor()).find(self.build(drop), [])}
        self.assertIn(DROP_OFF, hazards)

    def test_low_obstacle_is_detected_and_detection_boxes_are_excluded(self):
        def box(z):
            z[360:450, 260:380] = 0.9
        dm = self.build(box)
        detector = FloorHazardDetector(WalkingCorridor())
        self.assertIn(LOW_OBSTACLE, {h.kind for h in detector.find(dm, [])})
        covered = detector.find(dm, [BoundingBox(250, 350, 390, 460)])
        self.assertNotIn(LOW_OBSTACLE, {h.kind for h in covered})

    def test_flat_floor_or_missing_floor_line_gives_nothing(self):
        detector = FloorHazardDetector(WalkingCorridor())
        self.assertEqual(detector.find(self.build(lambda z: None), []), [])
        self.assertEqual(detector.find(metric_map(synth_floor()[1]), []), [])

    def test_temporal_vote_requires_k_of_n_depth_frames(self):
        vote = TemporalVote(2, 3)
        self.assertFalse(vote.vote(1, True))
        self.assertFalse(vote.vote(1, True))  # cùng depth frame không bỏ phiếu hai lần
        self.assertTrue(vote.vote(2, True))
        self.assertTrue(vote.vote(3, False))
        self.assertFalse(vote.vote(4, False))


def object_pair(bbox, distance, frame_id, t, quality=DataQuality.VALID, track_id=1, cls="chair"):
    values = np.full((H, W), 8.0, np.float32)
    x1, y1, x2, y2 = (int(v) for v in bbox)
    values[y1:y2, x1:x2] = distance
    det = DetectionResult([Detection(cls, 0.9, BoundingBox(*bbox), track_id)], frame_id, t, 1.0,
                          {"original_size": (W, H)})
    return SynchronizedPair(det, metric_map(values, frame_id, t), quality, 0.0, t)


class TestTTCRiskModel(unittest.TestCase):
    BOX = (260, 150, 380, 400)

    def fsm(self, model="ttc"):
        return RiskFSM(ClassFilter(), SpatialZones(), ROIExtractor(), risk_model=model)

    def run_sequence(self, fsm, distances, dt=0.125):
        out = None
        for i, d in enumerate(distances):
            out = fsm.update(object_pair(self.BOX, d, i, 100 + i * dt), current_mono=100 + i * dt)[0]
        return out

    def test_new_track_uses_original_distance_rule(self):
        self.assertEqual(self.run_sequence(self.fsm(), [1.3]).risk_level, RiskLevel.HIGH)

    def test_static_object_within_1_5_m_becomes_medium_once_motion_is_known(self):
        a = self.run_sequence(self.fsm(), [1.3] * 8)
        self.assertEqual(a.risk_level, RiskLevel.MEDIUM)
        self.assertIn("đứng yên", a.reason)

    def test_approaching_object_is_high_before_1_5_m(self):
        distances = [4.0 - 1.2 * i * 0.125 for i in range(8)]  # 4,0 -> 2,95 m
        a = self.run_sequence(self.fsm(), distances)
        self.assertEqual(a.risk_level, RiskLevel.HIGH)
        self.assertGreater(a.distance_m, 1.5)
        self.assertIsNotNone(a.ttc_sec)
        baseline = self.run_sequence(self.fsm("distance"), distances)
        self.assertNotEqual(baseline.risk_level, RiskLevel.HIGH)

    def test_critical_distance_is_always_high(self):
        self.assertEqual(self.run_sequence(self.fsm(), [0.8] * 8).risk_level, RiskLevel.HIGH)

    def test_stale_data_is_undetermined_and_clears_motion_history(self):
        fsm = self.fsm()
        self.run_sequence(fsm, [3.0 - 0.15 * i for i in range(8)])
        a = fsm.update(object_pair(self.BOX, 2.0, 99, 102.0, DataQuality.STALE), current_mono=102.0)[0]
        self.assertEqual(a.risk_level, RiskLevel.UNDETERMINED)
        self.assertEqual(fsm.ttc_estimator.estimate(1).depth_samples, 0)

    def test_generic_obstacle_needs_two_of_three_depth_frames(self):
        corridor = WalkingCorridor()
        fsm = RiskFSM(ClassFilter(), SpatialZones(), ROIExtractor(), walking_corridor=corridor,
                      generic_obstacle_detector=GenericObstacleDetector(corridor),
                      hazard_confirm_required=2, hazard_confirm_window=3)
        near = np.full((H, W), 0.6, np.float32)

        def generic(frame_id):
            empty = DetectionResult([], frame_id, 100.0 + frame_id, 1.0, {"original_size": (W, H)})
            pair = SynchronizedPair(empty, metric_map(near, frame_id, 100.0 + frame_id), DataQuality.VALID,
                                    0.0, 100.0 + frame_id)
            return [a for a in fsm.update(pair, current_mono=100.0 + frame_id)
                    if a.track_id == GENERIC_OBSTACLE_TRACK_ID]

        self.assertEqual(generic(1), [])
        hazard = generic(2)
        self.assertEqual(len(hazard), 1)
        self.assertEqual(hazard[0].hazard_type, "generic")
        self.assertIsNotNone(hazard[0].region)


if __name__ == "__main__":
    unittest.main()

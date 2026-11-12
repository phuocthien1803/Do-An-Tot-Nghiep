"""Kiểm thử nút bấm một phím và chế độ Tìm đồ."""
import tempfile
import time
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.asr.voice_input import route_voice_command
from src.audio.audio_coordinator import AudioCoordinator
from src.audio.earcons import EarconLibrary
from src.audio.tts_factory import ConsoleTTSEngine
from src.contracts.depth_map import DepthMap, DepthRepresentation
from src.contracts.detection import BoundingBox, Detection, DetectionResult
from src.contracts.risk import DataQuality, Direction, RiskAssessment, RiskLevel
from src.depth.roi_extractor import ROIExtractor
from src.fusion.object_finder import ObjectFinder, direction_words, distance_words, resolve_target
from src.fusion.synchronizer import SynchronizedPair
from src.runtime.system_coordinator import SystemCoordinator
from src.ui.app_runner import AppRunner
from src.ui.one_button import ButtonGestureDecoder


class TestGestureDecoder(unittest.TestCase):
    def clicks(self, times, dec=None):
        dec = dec or ButtonGestureDecoder()
        for t in times:
            dec.press(t)
            dec.release(t + 0.05)
        return dec.poll(times[-1] + 1.0)

    def test_single_double_triple(self):
        self.assertEqual(self.clicks([0.0]), "single")
        self.assertEqual(self.clicks([0.0, 0.25]), "double")
        self.assertEqual(self.clicks([0.0, 0.25, 0.5]), "triple")
        self.assertEqual(self.clicks([0.0, 0.25, 0.5, 0.75]), "triple")

    def test_no_gesture_before_window_ends(self):
        dec = ButtonGestureDecoder()
        dec.press(0.0)
        dec.release(0.05)
        self.assertIsNone(dec.poll(0.3))
        self.assertEqual(dec.poll(0.6), "single")

    def test_long_press_with_release_event(self):
        dec = ButtonGestureDecoder()
        dec.press(0.0)
        self.assertEqual(dec.release(0.9), "long")
        self.assertIsNone(dec.poll(2.0))

    def test_long_press_from_keyboard_autorepeat(self):
        dec = ButtonGestureDecoder()
        got = [dec.press(t) for t in np.arange(0.0, 1.0, 0.03)]  # giữ phím: nhấn lặp liên tục
        self.assertIn("long", got)
        self.assertIsNone(dec.poll(3.0))


class FakeAudio:
    def __init__(self):
        self.tasks, self.earcons = [], []
        self.current_priority = None

    def post_task(self, task):
        self.tasks.append(task.text)
        return True

    def play_earcon(self, path):
        self.earcons.append(path)
        return True


def pair_with(cx, distance=None, cls="cup"):
    box = BoundingBox(cx - 40, 200, cx + 40, 300)
    det = DetectionResult([Detection(cls, 0.8, box, 5)], 1, time.monotonic(), 1.0, {"original_size": (640, 480)})
    depth = None
    if distance is not None:
        vals = np.full((480, 640), 5.0, np.float32)
        vals[200:300, int(cx - 40):int(cx + 40)] = distance
        depth = DepthMap(DepthRepresentation.METRIC_DEPTH, "meter", False, vals, np.ones((480, 640), bool),
                         1, time.monotonic())
    return SynchronizedPair(det, depth, DataQuality.VALID if depth is not None else DataQuality.UNAVAILABLE,
                            0.0, time.monotonic())


class TestObjectFinder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio = FakeAudio()
        self.finder = ObjectFinder(self.audio, EarconLibrary(self.tmp.name).ensure(), ROIExtractor(),
                                   lost_after_sec=0.3, lost_hint_every_sec=0.5)

    def tearDown(self):
        self.finder.shutdown()
        self.tmp.cleanup()

    def test_resolve_target_and_words(self):
        self.assertEqual(resolve_target("tìm cái ly giúp tôi"), ("cốc", "cup"))
        self.assertEqual(resolve_target("tìm ghế trống"), ("ghế", "chair"))
        self.assertIsNone(resolve_target("tìm chìa khoá"))
        self.assertEqual(direction_words(-0.8), "bên trái")
        self.assertEqual(direction_words(0.05), "ngay phía trước")
        self.assertEqual(distance_words(0.4), "trong tầm tay")

    def test_guidance_hints(self):
        self.finder.start("cốc", "cup")
        self.finder.update(pair_with(100, 2.0))
        self.assertEqual(self.audio.tasks[-1], "Thấy cốc bên trái, còn xa.")
        self.finder.update(pair_with(320, 0.4))
        self.assertEqual(self.audio.tasks[-1], "Cốc trong tầm tay, ngay phía trước.")
        snap = self.finder.snapshot()
        self.assertTrue(snap["seen"])
        self.assertAlmostEqual(snap["offset"], 0.0)

    def test_lost_target_hint(self):
        self.finder.start("cốc", "cup")
        time.sleep(0.6)
        self.finder.update(pair_with(320, cls="person"))
        self.assertIn("Chưa thấy cốc", self.audio.tasks[-1])

    def test_guide_beeps_are_panned_and_high_when_centered(self):
        self.finder.start("cốc", "cup")
        self.finder.update(pair_with(600, 1.0))
        time.sleep(0.25)
        self.assertTrue(any("guide_6_lo" in p for p in self.audio.earcons))
        self.audio.earcons.clear()
        self.finder.update(pair_with(320, 1.0))
        time.sleep(0.6)
        self.assertTrue(any("guide_3_hi" in p for p in self.audio.earcons))
        self.assertLess(self.finder.interval_for(0.3, None), self.finder.interval_for(2.5, None))


class TestFindInCoordinator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio = AudioCoordinator(ConsoleTTSEngine(quiet=True))
        self.aggregator = MagicMock()
        self.aggregator.aggregate.return_value = None
        self.finder = ObjectFinder(self.audio, EarconLibrary(self.tmp.name).ensure())
        self.channel = MagicMock()
        self.system = SystemCoordinator(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                                        MagicMock(), self.aggregator, self.audio, MagicMock(), MagicMock(),
                                        object_finder=self.finder, safety_channel=self.channel)

    def tearDown(self):
        self.system.stop()
        self.tmp.cleanup()

    def test_find_mode_keeps_spoken_alerts_but_not_for_the_target(self):
        self.assertTrue(self.system.start_find("ghế", "chair"))
        snap = self.system.get_runtime_snapshot()
        self.assertEqual(snap["selected_mode"], "FIND")
        self.assertTrue(snap["safety_alerts_enabled"])
        t = time.monotonic() + 1
        chair = RiskAssessment(1, "ghế", Direction.CENTER, RiskLevel.HIGH, DataQuality.VALID, .8, "x", "x", t, t + 1)
        person = RiskAssessment(2, "người", Direction.LEFT, RiskLevel.HIGH, DataQuality.VALID, .8, "x", "x", t, t + 1)
        self.system._post_safety_alert([chair, person])
        self.assertEqual([a.class_name for a in self.aggregator.aggregate.call_args[0][0]], ["người"])
        self.assertEqual(self.channel.update.call_args[0][0], [])  # âm dẫn hướng thay tiếng tích

    def test_find_next_cycles_and_c_exits(self):
        self.system.find_next()
        first = self.finder.snapshot()["target"]
        self.system.find_next()
        self.assertNotEqual(self.finder.snapshot()["target"], first)
        self.system.enter_warning_mode()
        self.assertFalse(self.finder.active)
        self.assertEqual(self.system.get_runtime_snapshot()["selected_mode"], "WARNING")

    def test_voice_find_command(self):
        self.assertEqual(route_voice_command("tìm cái cốc")[0], "find")


class TestRunnerOneButton(unittest.TestCase):
    def test_button_gestures_dispatch_commands(self):
        coordinator = MagicMock()
        runner = AppRunner(coordinator, MagicMock(), enable_gui=False, one_button={"enabled": True})
        self.assertTrue(runner.is_button_key("b"))
        self.assertTrue(runner.is_button_key("media_volume_up"))
        for n, method in ((1, "trigger_vqa"), (2, "trigger_ocr"), (3, "find_next")):
            t0 = time.monotonic()
            for i in range(n):
                runner.button_event("press", t0 + i * 0.2)
                runner.button_event("release", t0 + i * 0.2 + 0.05)
            runner._process_button(t0 + 5)
            self.assertTrue(getattr(coordinator, method).called, method)
        runner.button_event("press", 100.0)
        runner.button_event("release", 101.0)
        runner._process_button(101.1)
        coordinator.stop_or_warning.assert_called_once()
        self.assertEqual(AppRunner.command_for_key(ord("f")), "find")


if __name__ == "__main__":
    unittest.main()

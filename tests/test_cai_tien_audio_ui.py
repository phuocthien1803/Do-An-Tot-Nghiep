"""Kiểm thử cải tiến âm thanh/tương tác: earcon, kênh bíp an toàn, cảnh báo ngắn, chế độ, phím, Windows TTS."""
import os
import tempfile
import time
import unittest
import wave
from unittest.mock import MagicMock, patch

import numpy as np

from src.audio.audio_coordinator import AudioCoordinator
from src.audio.earcons import EarconLibrary
from src.audio.safety_channel import SafetySoundChannel, pick_urgent
from src.audio.tts_engine import TTSEngine
from src.audio.tts_factory import ConsoleTTSEngine, create_tts_engine
from src.camera.bounded_buffer import BoundedBuffer
from src.contracts.audio import AudioPriority, AudioTask
from src.contracts.frame_packet import FramePacket
from src.contracts.risk import DataQuality, Direction, RiskAssessment, RiskLevel
from src.fusion.alert_aggregator import AlertAggregator
from src.runtime.system_coordinator import SystemCoordinator
from src.ui.app_runner import AppRunner


def high(track_id=1, direction=Direction.CENTER, t=None, ttc=None, distance=1.2, quality=DataQuality.VALID,
         hazard="object", name="ghế", level=RiskLevel.HIGH):
    t = time.monotonic() if t is None else t
    return RiskAssessment(track_id, name, direction, level, quality, 0.8, "ở rất gần", "test", t, t + 1.0,
                          distance_m=distance, ttc_sec=ttc, hazard_type=hazard)


class FakeAudio:
    def __init__(self):
        self.earcons = []
        self.current_priority = None

    def play_earcon(self, path):
        self.earcons.append(path)
        return True


class TestEarcons(unittest.TestCase):
    def test_files_are_generated_and_panned(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = EarconLibrary(tmp).ensure()
            for name in ("alert_left", "drop_center", "tick_right", "ack", "reject"):
                self.assertTrue(os.path.exists(lib.path(name)), name)
            with wave.open(lib.for_alert(Direction.LEFT), "rb") as wf:
                data = np.frombuffer(wf.readframes(wf.getnframes()), "<i2").reshape(-1, 2).astype(float)
            self.assertGreater(np.abs(data[:, 0]).mean(), 2 * np.abs(data[:, 1]).mean())
            self.assertIn("drop_", lib.for_alert(Direction.CENTER, "drop_off"))


class TestSafetyChannel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lib = EarconLibrary(self.tmp.name).ensure()
        self.audio = FakeAudio()
        self.channel = SafetySoundChannel(self.audio, self.lib)

    def tearDown(self):
        self.channel.stop()
        self.tmp.cleanup()

    def test_pick_urgent_ignores_stale_degraded_and_non_high(self):
        now = 100.0
        cands = [high(1, t=now - 5), high(2, t=now, quality=DataQuality.DEGRADED),
                 high(3, t=now, level=RiskLevel.MEDIUM)]
        self.assertIsNone(pick_urgent(cands, now))
        self.assertEqual(pick_urgent(cands + [high(4, t=now)], now).track_id, 4)

    def test_interval_follows_ttc_then_distance(self):
        self.assertAlmostEqual(self.channel.interval_for(high(ttc=2.0)), 0.7)
        self.assertEqual(self.channel.interval_for(high(ttc=0.1)), self.channel.min_interval_sec)
        self.assertAlmostEqual(self.channel.interval_for(high(distance=1.0)), 0.45)

    def test_new_hazard_in_on_demand_plays_alert_earcon_once(self):
        now = time.monotonic()
        self.channel.update([high(7, t=now)], now, on_demand=True)
        self.channel.update([high(7, t=now)], now, on_demand=True)
        self.assertEqual(sum("alert_" in p for p in self.audio.earcons), 1)

    def test_ticks_while_fresh_and_stop_when_stale(self):
        self.channel.start()
        now = time.monotonic()
        self.channel.update([high(1, t=now, ttc=0.5)], now, on_demand=False)
        time.sleep(0.4)
        self.assertGreaterEqual(self.channel.tick_count, 1)
        time.sleep(0.7)  # không còn cập nhật -> dữ liệu cũ -> dừng bíp
        count = self.channel.tick_count
        time.sleep(0.4)
        self.assertEqual(self.channel.tick_count, count)

    def test_snooze_silences_current_hazard_but_not_new_ones(self):
        now = time.monotonic()
        self.channel.update([high(1, t=now)], now, on_demand=True)
        self.channel.snooze_current()
        self.assertIsNone(self.channel.update([high(1, t=now)], now, on_demand=True))
        self.assertEqual(self.channel.update([high(2, t=now)], now, on_demand=True).track_id, 2)


class TestAudioCoordinatorImprovements(unittest.TestCase):
    def test_earcon_does_not_interrupt_speech(self):
        engine = MagicMock(spec=TTSEngine)
        engine.play_earcon.return_value = True
        audio = AudioCoordinator(engine)
        audio.start()
        try:
            self.assertTrue(audio.play_earcon("tick.wav"))
            engine.play_earcon.assert_called_once_with("tick.wav")
            engine.stop.assert_not_called()
        finally:
            audio.stop()

    def test_playback_listener_receives_source_timestamp(self):
        engine = ConsoleTTSEngine(quiet=True)
        audio = AudioCoordinator(engine)
        seen = []
        audio.playback_listener = lambda task, t: seen.append((task.category, t - task.source_timestamp))
        audio.start()
        try:
            audio.post_task(AudioTask(AudioPriority.HIGH_RISK, "Ghế phía trước",
                                      source_timestamp=time.monotonic() - 0.2, category="alert"))
            deadline = time.monotonic() + 2
            while not seen and time.monotonic() < deadline:
                time.sleep(0.02)
        finally:
            audio.stop()
        self.assertEqual(seen[0][0], "alert")
        self.assertGreaterEqual(seen[0][1], 0.19)  # đồng hồ Windows phân giải ~16 ms

    def test_recent_text_cache_is_pruned(self):
        audio = AudioCoordinator(ConsoleTTSEngine(quiet=True), dedup_window_sec=0.0)
        for i in range(100):
            audio.post_task(AudioTask(AudioPriority.INFO, f"câu {i}"))
        self.assertLessEqual(len(audio._recent_texts), 65)


class TestAlertAggregatorImprovements(unittest.TestCase):
    def make(self, **kw):
        fsm = MagicMock()
        fsm.can_trigger_alert.return_value = True
        fsm.has_alerted.return_value = False
        return AlertAggregator(fsm, global_alert_interval_sec=3.5, message_style="ngan",
                               bypass_interval=True, **kw), fsm

    def test_short_messages(self):
        agg, _ = self.make()
        self.assertEqual(agg.aggregate([high(1, t=100)], 100).text, "Ghế phía trước")
        agg, _ = self.make()
        self.assertEqual(agg.aggregate([high(1, Direction.LEFT, t=100, ttc=2.0, name="người")], 100).text,
                         "Người đang lại gần, bên trái")
        agg, _ = self.make()
        task = agg.aggregate([high(-101, t=100, hazard="drop_off", name="bậc xuống hoặc hố")], 100)
        self.assertEqual(task.text, "Cẩn thận, bậc xuống hoặc hố phía trước")

    def test_new_center_object_bypasses_interval_but_same_track_does_not(self):
        agg, fsm = self.make()
        self.assertIsNotNone(agg.aggregate([high(1, t=100)], 100))
        self.assertIsNone(agg.aggregate([high(1, t=101.5)], 101.5))  # cùng track: chờ khoảng nghỉ
        self.assertIsNotNone(agg.aggregate([high(2, t=101.6)], 101.6))  # vật mới chắn giữa
        self.assertIsNone(agg.aggregate([high(3, Direction.LEFT, t=101.8)], 101.8))  # quá gần lần trước

    def test_side_object_bypasses_only_with_critical_ttc(self):
        agg, fsm = self.make()
        agg.aggregate([high(1, t=100)], 100)
        self.assertIsNone(agg.aggregate([high(2, Direction.LEFT, t=101.5, ttc=3.0)], 101.5))
        self.assertIsNotNone(agg.aggregate([high(2, Direction.LEFT, t=101.6, ttc=1.0)], 101.6))

    def test_original_defaults_never_bypass(self):
        fsm = MagicMock()
        fsm.can_trigger_alert.return_value = True
        agg = AlertAggregator(fsm, global_alert_interval_sec=3.5)
        agg.aggregate([high(1, t=100)], 100)
        self.assertIsNone(agg.aggregate([high(2, t=101.5, ttc=0.5)], 101.5))

    def test_earcon_and_source_timestamp(self):
        agg, _ = self.make(earcon_provider=lambda d, h: f"{h}_{d.value}.wav")
        task = agg.aggregate([high(1, Direction.RIGHT, t=99.9)], 100)
        self.assertEqual(task.sound_file, "object_RIGHT.wav")
        self.assertEqual(task.source_timestamp, 99.9)
        self.assertEqual(task.category, "alert")


class TestCoordinatorImprovements(unittest.TestCase):
    def build(self, **kw):
        self.camera = MagicMock()
        self.camera.is_alive.return_value = True
        self.audio = kw.pop("audio", None) or AudioCoordinator(ConsoleTTSEngine(quiet=True))
        self.aggregator = MagicMock()
        self.aggregator.aggregate.return_value = None
        self.channel = MagicMock()
        depth = MagicMock()
        depth.load_error = None
        self.system = SystemCoordinator(self.camera, MagicMock(), MagicMock(), depth, MagicMock(),
                                        MagicMock(), self.aggregator, self.audio, MagicMock(), MagicMock(),
                                        safety_channel=self.channel, **kw)
        return self.system

    def tearDown(self):
        self.system.stop()

    def test_earcon_mode_keeps_safety_beeps_but_no_speech_in_ocr(self):
        system = self.build(on_demand_high_alert="earcon")
        system.set_on_demand_active(True, "OCR")
        a = high()
        system._post_safety_alert([a])
        self.aggregator.aggregate.assert_not_called()
        self.assertEqual(self.channel.update.call_args[0][0], [a])
        self.assertTrue(self.channel.update.call_args[0][2])  # on_demand=True
        self.assertTrue(system.get_runtime_snapshot()["safety_beeps_enabled"])
        self.assertFalse(system.get_runtime_snapshot()["safety_alerts_enabled"])

    def test_off_mode_matches_original_and_silences_channel(self):
        system = self.build(on_demand_high_alert="off")
        system.set_on_demand_active(True, "VQA")
        system._post_safety_alert([high()])
        self.aggregator.aggregate.assert_not_called()
        self.assertEqual(self.channel.update.call_args[0][0], [])

    def test_speech_mode_alerts_in_on_demand(self):
        system = self.build(on_demand_high_alert="speech")
        system.set_on_demand_active(True, "OCR")
        system._post_safety_alert([high()])
        self.aggregator.aggregate.assert_called_once()

    def test_rejected_request_gives_audio_feedback(self):
        earcons = MagicMock()
        earcons.path.side_effect = lambda name: f"{name}.wav"
        audio = MagicMock()
        audio.is_busy = True
        system = self.build(earcons=earcons, audio=audio)
        system.set_on_demand_active(True, "OCR")  # đang có tác vụ khác
        self.assertFalse(system.trigger_vqa())
        audio.play_earcon.assert_called_with("reject.wav")

    def test_auto_return_to_warning_after_idle(self):
        audio = MagicMock()
        audio.is_busy = False
        system = self.build(audio=audio, auto_return_warning_sec=5.0)
        system.set_on_demand_active(True, "OCR")
        system._finish_on_demand(system._active_on_demand_token)
        now = time.monotonic()
        self.assertFalse(system._maybe_auto_return(now))
        self.assertTrue(system._maybe_auto_return(now + 6))
        self.assertEqual(system.get_runtime_snapshot()["selected_mode"], "WARNING")

    def test_adaptive_startup_grace_waits_for_models_but_not_for_camera(self):
        system = self.build(startup_grace_max_sec=30.0)
        t0 = 1000.0
        system._started_at = t0
        self.camera.get_latest_frame.return_value = FramePacket(1, "c", np.zeros((4, 4, 3), np.uint8), (4, 4), t0 + 10)
        self.assertIsNone(system._diagnose(t0 + 10))
        self.camera.get_latest_frame.return_value = FramePacket(1, "c", np.zeros((4, 4, 3), np.uint8), (4, 4), t0 + 31)
        with patch("src.runtime.watchdog.time.monotonic", return_value=t0 + 31):
            self.assertIn(system._diagnose(t0 + 31), ("detection", "depth"))
        self.camera.is_alive.return_value = False
        self.assertEqual(system._diagnose(t0 + 10), "camera")
        # Camera chưa từng cho frame nào (đang mở): chờ trong thời gian khởi động thích ứng.
        self.camera.get_latest_frame.return_value = None
        self.assertIsNone(system._diagnose(t0 + 10))
        self.assertEqual(system._diagnose(t0 + 31), "camera")

    def test_dark_or_covered_camera_degrades_data_and_is_announced(self):
        from src.fusion.synchronizer import SynchronizedPair
        system = self.build(min_frame_brightness=15)
        system._started_at = time.monotonic() - 100
        dark = FramePacket(1, "c", np.full((48, 64, 3), 3, np.uint8), (64, 48), time.monotonic())
        pair = SynchronizedPair(MagicMock(), MagicMock(), DataQuality.VALID, 0.0, time.monotonic())
        system._apply_darkness_gate(dark, pair)
        self.assertEqual(pair.data_quality, DataQuality.DEGRADED)
        self.camera.get_latest_frame.return_value = dark
        system._dark_since -= 3.0
        self.assertEqual(system._diagnose(time.monotonic()), "dark")
        bright = FramePacket(2, "c", np.full((48, 64, 3), 120, np.uint8), (64, 48), time.monotonic())
        pair = SynchronizedPair(MagicMock(), MagicMock(), DataQuality.VALID, 0.0, time.monotonic())
        system._apply_darkness_gate(bright, pair)
        self.assertEqual(pair.data_quality, DataQuality.VALID)
        self.assertIsNone(system._dark_since)

    def test_help_and_status_are_spoken(self):
        system = self.build()
        posted = []
        system.audio_coordinator.post_task = lambda task: posted.append(task.text) or True
        system.speak_help()
        system.speak_status()
        self.assertIn("Phím Q", posted[0])
        self.assertIn("chế độ cảnh báo", posted[1])

    def test_voice_command_routes_to_vqa_or_ocr(self):
        voice = MagicMock()
        voice.dependency_error.return_value = None
        system = self.build(voice_input=voice)
        system._say_on_demand = MagicMock()
        system._vqa_body = MagicMock()
        system._ocr_body = MagicMock()
        for text, target in (("trên bàn có gì", "_vqa_body"), ("đọc chữ trên biển", "_ocr_body")):
            voice.listen.return_value = (text, None)
            self.assertTrue(system.trigger_voice_question())
            system._on_demand_thread.join(timeout=3)
            self.assertTrue(getattr(system, target).called, text)
        self.assertEqual(system._vqa_body.call_args[0][1], "Trên bàn có gì?")


class TestRunnerAndBuffer(unittest.TestCase):
    def test_dispatch_maps_keys_to_actions(self):
        coordinator = MagicMock()
        runner = AppRunner(coordinator, hud_renderer=MagicMock(), enable_gui=False)
        for key, method in (("h", "speak_help"), ("i", "speak_status"), ("v", "trigger_voice_question"),
                            (" ", "trigger_ocr"), ("c", "enter_warning_mode"), ("s", "stop_speech")):
            self.assertTrue(runner.dispatch(AppRunner.command_for_key(ord(key))))
            self.assertTrue(getattr(coordinator, method).called, key)
        self.assertFalse(runner.dispatch(AppRunner.command_for_key(27)))
        self.assertIsNone(AppRunner.command_for_key(255))

    def test_get_newest_drops_older_frames(self):
        buf = BoundedBuffer(maxsize=3)
        for i in range(3):
            buf.put(i)
        self.assertEqual(buf.get_newest(timeout=0.01), 2)
        self.assertTrue(buf.is_empty())
        self.assertEqual(buf.dropped_count, 2)
        self.assertIsNone(buf.get_newest(timeout=0.01))


class TestTTSBackends(unittest.TestCase):
    def test_factory_selects_engine(self):
        self.assertIsInstance(create_tts_engine({"engine": "console"}), ConsoleTTSEngine)
        self.assertIs(type(create_tts_engine({"engine": "say"})), TTSEngine)

    def test_windows_engine_uses_sapi_and_winsound(self):
        import ban_windows.windows_tts as wt
        fake = MagicMock(SND_FILENAME=1, SND_ASYNC=2, SND_NODEFAULT=4)
        with patch.object(wt, "winsound", fake), tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            with patch.object(wt, "winsound", fake):
                engine = wt.WindowsTTSEngine(voice="Linh", speech_rate_wpm=200)
                self.assertEqual(engine.sapi_rate(200), 1)
                cmd = engine.speak_command("xin chào")
                self.assertEqual(cmd[:3], ["cscript", "//nologo", wt.SAPI_SCRIPT])
                self.assertEqual(cmd[4], "")  # "Linh" -> để script tự chọn giọng tiếng Việt
                self.assertTrue(engine.play_earcon(path))
                self.assertEqual(fake.PlaySound.call_args[0][1], 1 | 2 | 4)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()

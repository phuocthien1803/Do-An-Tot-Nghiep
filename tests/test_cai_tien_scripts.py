"""Kiểm thử các hàm chấm điểm của công cụ thực nghiệm (không cần camera/model)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import danh_gia_su_kien  # noqa: E402
import phan_tich_do_tre  # noqa: E402
import tinh_sus_nasa  # noqa: E402


class TestEventScoring(unittest.TestCase):
    def test_recall_lead_time_and_false_alarms(self):
        run = {"duration_sec": 60.0, "alerts": [{"t": 4.0}, {"t": 9.5}, {"t": 30.0}]}
        events = [{"start": 3.0, "contact": 6.0, "end": 6.5, "type": "obstacle"},
                  {"start": 9.0, "contact": 10.0, "end": 10.5, "type": "drop_off"},
                  {"start": 20.0, "contact": 22.0, "end": 22.5, "type": "obstacle"}]
        s = danh_gia_su_kien.score(run, events, min_lead=1.0)
        self.assertAlmostEqual(s["recall"], 2 / 3)
        self.assertAlmostEqual(s["timely_rate"], 1 / 3)  # sự kiện 2 chỉ báo trước 0,5 s
        self.assertEqual(s["per_event"][0]["lead_time_sec"], 2.0)
        self.assertEqual(s["false_alarms"], 1)
        self.assertAlmostEqual(s["false_alarms_per_min"], 1.0)
        self.assertEqual(s["by_type"]["drop_off"], {"n": 1, "detected": 1, "timely": 0})


class TestLatencyAnalysis(unittest.TestCase):
    def test_first_safety_sound_after_each_stimulus(self):
        stimuli = [{"lan": 0, "onset": 100.0}, {"lan": 1, "onset": 110.0}]
        events = [
            {"type": "audio_start", "category": "info", "t": 100.1, "text": "x"},
            {"type": "audio_start", "category": "alert", "t": 100.6, "latency_ms": 420.0, "text": "Người phía trước"},
            {"type": "earcon", "t": 100.9, "sound": "tick_center.wav"},
        ]
        r = phan_tich_do_tre.analyse(stimuli, events, window=3.0)
        self.assertEqual(r["detected"], 1)
        self.assertAlmostEqual(r["per_stimulus"][0]["latency_ms"], 600.0, places=3)
        self.assertEqual(r["per_stimulus"][0]["internal_ms"], 420.0)
        self.assertFalse(r["per_stimulus"][1]["detected"])


class TestQuestionnaires(unittest.TestCase):
    def test_sus_and_tlx(self):
        best = {f"sus{i}": (5 if i % 2 else 1) for i in range(1, 11)}
        worst = {f"sus{i}": (1 if i % 2 else 5) for i in range(1, 11)}
        self.assertEqual(tinh_sus_nasa.sus_score(best), 100.0)
        self.assertEqual(tinh_sus_nasa.sus_score(worst), 0.0)
        row = {k: 30 for k in tinh_sus_nasa.TLX}
        self.assertEqual(tinh_sus_nasa.tlx_score(row), 30)


if __name__ == "__main__":
    unittest.main()

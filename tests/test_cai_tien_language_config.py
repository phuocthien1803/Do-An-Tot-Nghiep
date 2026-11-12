"""Kiểm thử cải tiến ngôn ngữ và cấu hình: guardrail mở rộng, lệnh giọng nói, lọc OCR, câu trả lời VQA,
lớp phủ cấu hình Mac/Windows/"như bản gốc" và factory."""
import sys
import unittest
from unittest.mock import patch

from src.asr.voice_input import VoiceInput, route_voice_command
from src.ocr.ocr_service import OCRService
from src.runtime import factory
from src.runtime.config import PROJECT_ROOT, apply_overlay, deep_merge, load_configs
from src.vqa.formatting import format_vqa_answer
from src.vqa.safety import SafetyGuardrail, strip_diacritics

PREVIOUSLY_MISSED = [
    "Đường trống không?",
    "Phía trước có gì cản đường không?",
    "Phía trước thông thoáng chứ?",
    "toi co the di tiep khong",
    "phia truoc co vat can khong",
    "Cứ đi thẳng là được đúng không?",
    "Is the path clear?",
    "Anything in my way?",
]
MUST_ALLOW = [
    "Phía trước có gì?", "Đây là đâu?", "Có mấy người trong ảnh?",
    "Biển báo ghi là đi thẳng hay rẽ trái?", "Người đàn ông đang đi đâu?",
    "Tờ giấy này viết về an toàn thực phẩm phải không?", "Cửa đang mở hay đóng?",
    "Bảng thông báo nói gì về lối thoát hiểm?", "Có con mèo nào trong ảnh không?",
    "Người phụ nữ đang đi bộ hay đứng yên?", "What is written on this sign?",
]


class TestGuardrail(unittest.TestCase):
    def test_extended_mode_refuses_questions_the_original_missed(self):
        original, extended = SafetyGuardrail(), SafetyGuardrail(mode="mo_rong")
        for question in PREVIOUSLY_MISSED:
            with self.subTest(question=question):
                self.assertIsNone(original.check(question))
                self.assertIsNotNone(extended.check(question))

    def test_extended_mode_keeps_valid_questions(self):
        extended = SafetyGuardrail(mode="mo_rong")
        for question in MUST_ALLOW:
            with self.subTest(question=question):
                self.assertIsNone(extended.check(question))

    def test_answer_side(self):
        original, extended = SafetyGuardrail(), SafetyGuardrail(mode="mo_rong")
        self.assertIsNotNone(original.check_answer("Tờ giấy nói về an toàn thực phẩm."))
        self.assertIsNone(extended.check_answer("Tờ giấy nói về an toàn thực phẩm."))
        for unsafe in ("Lối đi rất thông thoáng.", "Không có gì cản trước mặt.", "Bạn cứ đi thẳng."):
            self.assertIsNotNone(extended.check_answer(unsafe), unsafe)
        self.assertIsNone(extended.check_answer("Có một cái ghế phía trước."))

    def test_strip_diacritics(self):
        self.assertEqual(strip_diacritics("Đường đi"), "Duong di")


class TestVoiceRouting(unittest.TestCase):
    def test_route_commands(self):
        self.assertEqual(route_voice_command("đọc chữ trên biển này")[0], "ocr")
        self.assertEqual(route_voice_command("Trợ giúp")[0], "help")
        self.assertEqual(route_voice_command("trạng thái hệ thống")[0], "status")
        self.assertEqual(route_voice_command("quay lại cảnh báo")[0], "warning")
        self.assertEqual(route_voice_command("  ")[0], "empty")
        self.assertEqual(route_voice_command("trên bàn có gì"), ("vqa", "Trên bàn có gì?"))

    def test_missing_dependencies_are_reported_in_vietnamese(self):
        with patch.dict(sys.modules, {"sounddevice": None, "faster_whisper": None}):
            message = VoiceInput.dependency_error()
        self.assertIn("sounddevice", message)
        self.assertIn("phím Q", message)


class TestOCRAndVQAText(unittest.TestCase):
    def test_garbage_filter(self):
        ok = OCRService.is_plausible_text
        self.assertTrue(ok("Lối thoát hiểm", 0.4))
        self.assertTrue(ok("25.000", 0.3))
        self.assertFalse(ok("|||-_~", 0.9))
        self.assertFalse(ok("i", 0.5))
        self.assertFalse(ok("~ab", 0.3))

    def test_short_answer_style(self):
        context = {"spatial_objects": [{"name": "ghế", "direction": "LEFT", "proximity": "chưa rõ"}]}
        text = format_vqa_answer("Một phòng khách", context, style="ngan")
        self.assertEqual(text, "Một phòng khách. Cảm biến ghi nhận: ghế bên trái.")
        self.assertTrue(format_vqa_answer("Ghế.", style="goc").startswith("Khung cảnh:"))


class TestConfigAndFactory(unittest.TestCase):
    def test_deep_merge_does_not_mutate(self):
        base = {"a": {"b": 1, "c": [1]}}
        merged = deep_merge(base, {"a": {"b": 2, "c": [9]}})
        self.assertEqual(base, {"a": {"b": 1, "c": [1]}})
        self.assertEqual(merged, {"a": {"b": 2, "c": [9]}})
        with self.assertRaises(ValueError):
            apply_overlay({"app": {}}, {"sai_khoa": {}})

    def test_platform_overlays(self):
        win = load_configs(PROJECT_ROOT, platform="windows")
        self.assertEqual(win["app"]["audio"]["engine"], "sapi")
        self.assertEqual(win["app"]["app"]["device"], "cpu")
        self.assertEqual(win["model"]["ocr"]["engine"], "easyocr")
        mac = load_configs(PROJECT_ROOT, platform="mac")
        self.assertEqual(mac["app"]["audio"]["engine"], "say")
        self.assertEqual(mac["model"]["detection"]["weights_path"], "models/weights/yolov8s.pt")
        # Lõi thuật toán giống nhau trên hai máy.
        for key in ("risk_fsm", "ttc", "floor_hazards", "alert_aggregator", "synchronizer"):
            self.assertEqual(win["fusion"][key], mac["fusion"][key], key)

    def test_original_behaviour_overlay_and_factory(self):
        improved = load_configs(PROJECT_ROOT, platform="none")
        stack = factory.build_risk_stack(improved)
        self.assertEqual(type(stack["tracker"]).__name__, "ByteTracker")
        self.assertEqual(stack["risk_fsm"].risk_model, "ttc")
        self.assertIsNotNone(stack["risk_fsm"].floor_hazard_detector)
        self.assertEqual(stack["aggregator"].message_style, "ngan")
        self.assertAlmostEqual(factory.detector_confidence(improved), 0.10)

        original = load_configs(PROJECT_ROOT, platform="none", overlays=["configs/thu_nghiem/nhu_ban_goc.yaml"])
        stack = factory.build_risk_stack(original)
        self.assertEqual(type(stack["tracker"]).__name__, "ByteTrackerAdapter")
        self.assertEqual(stack["risk_fsm"].risk_model, "distance")
        self.assertIsNone(stack["risk_fsm"].floor_hazard_detector)
        self.assertFalse(stack["aggregator"].bypass_interval)
        self.assertAlmostEqual(factory.detector_confidence(original), 0.25)
        self.assertIn("cell phone", original["model"]["detection"]["alert_classes"])
        self.assertFalse(original["app"]["audio"]["earcons"])


if __name__ == "__main__":
    unittest.main()

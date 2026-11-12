#!/usr/bin/env python3
"""Kiểm thử TỪNG PHÍM trên hệ thống thật (YOLO, độ sâu, OCR, VQA chạy thật) và xuất ảnh minh chứng.

Không cần camera/loa: camera được thay bằng "camera kịch bản" phát các cảnh dựng sẵn (đường phố có
người, biển chữ tiếng Việt, ống kính bị che, người tiến lại gần); giọng đọc được ghi lại thay vì phát.
Mỗi phím cho một ảnh: HUD tại thời điểm có kết quả + bảng ghi lời hệ thống nói + ĐẠT/KHÔNG ĐẠT.

    python scripts/kiem_thu_phim.py --platform windows --device cpu
    python scripts/kiem_thu_phim.py --platform mac --device mps
Kết quả: evaluation/results/kiem_thu_phim/<phim>.png, tong_hop.png, ket_qua.json
"""
import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from app import build_object_finder, build_ocr_service, build_vqa_service, build_voice_input  # noqa: E402
from src.audio.audio_coordinator import AudioCoordinator  # noqa: E402
from src.audio.safety_channel import SafetySoundChannel  # noqa: E402
from src.audio.tts_factory import ConsoleTTSEngine  # noqa: E402
from src.contracts.frame_packet import FramePacket  # noqa: E402
from src.runtime import factory  # noqa: E402
from src.runtime.config import load_configs  # noqa: E402
from src.runtime.system_coordinator import SystemCoordinator  # noqa: E402
from src.ui.app_runner import AppRunner  # noqa: E402
from src.ui.hud_renderer import HUDRenderer  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, "evaluation", "results", "kiem_thu_phim")
FONT_CANDIDATES = ["C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
                   "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ------------------------------ Cảnh dựng sẵn ------------------------------

def scene_street():
    import ultralytics
    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    return cv2.resize(img, (640, 480))


def scene_sign():
    img = Image.new("RGB", (640, 480), (235, 235, 230))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 70, 580, 410], fill=(20, 110, 60))
    for (text, y, size) in (("LỐI THOÁT HIỂM", 120, 58), ("PHÒNG HỌC 4A", 220, 50), ("CẤM HÚT THUỐC", 310, 46)):
        w = d.textlength(text, font=font(size))
        d.text(((640 - w) / 2, y), text, fill=(255, 255, 255), font=font(size))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def scene_covered(t):
    rng = np.random.default_rng(int(t * 1000) % 2 ** 31)
    base = np.full((480, 640, 3), (70, 90, 160), np.float64)
    return np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)


def scene_approach(t, start):
    import ultralytics
    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "zidane.jpg"))
    z = 0.35 + 0.9 * min(max(t - start, 0) / 4.0, 1.0)
    s = cv2.resize(img, None, fx=z, fy=z)
    canvas = np.full((480, 640, 3), 90, np.uint8)
    y0, x0 = (480 - s.shape[0]) // 2, (640 - s.shape[1]) // 2
    ys, xs, yd, xd = max(0, -y0), max(0, -x0), max(0, y0), max(0, x0)
    h, w = min(480 - yd, s.shape[0] - ys), min(640 - xd, s.shape[1] - xs)
    canvas[yd:yd + h, xd:xd + w] = s[ys:ys + h, xs:xs + w]
    return canvas


def scene_person(where):
    """Một người cỡ vừa đứng lệch trái hoặc ở giữa khung hình (cho chế độ Tìm đồ)."""
    import ultralytics
    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    person = cv2.resize(img[230:900, 40:300], (150, 380))  # người đàn ông bên trái ảnh bus.jpg
    canvas = np.full((480, 640, 3), 170, np.uint8)
    canvas[300:, :] = 120
    x = 30 if where == "left" else 245
    canvas[60:440, x:x + 150] = person
    return canvas


class ScriptedCamera:
    """Camera giả có cùng giao diện CameraManager; đổi cảnh bằng set_scene()."""

    def __init__(self, fps=15.0):
        self.fps = fps
        self._subs = []
        self._latest = None
        self._lock = threading.Lock()
        self._running = False
        self._scene = ("street", 0.0)
        self._static = {"street": scene_street(), "sign": scene_sign(),
                        "person_left": scene_person("left"), "person_center": scene_person("center")}
        self.frame_count = 0

    def subscribe(self, buf):
        self._subs.append(buf)

    def set_scene(self, name):
        with self._lock:
            self._scene = (name, time.monotonic())

    def _frame(self, now):
        name, since = self._scene
        if name == "covered":
            return scene_covered(now)
        if name == "approach":
            return scene_approach(now, since)
        return self._static[name]

    def _loop(self):
        while self._running:
            now = time.monotonic()
            with self._lock:
                img = self._frame(now)
            self.frame_count += 1
            pkt = FramePacket(self.frame_count, "kich_ban", img, (640, 480), now)
            with self._lock:
                self._latest = pkt
            for b in self._subs:
                b.put(pkt)
            time.sleep(1.0 / self.fps)

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False

    def is_alive(self):
        return self._running

    def get_latest_frame(self):
        with self._lock:
            return self._latest


class RecordingTTS(ConsoleTTSEngine):
    def __init__(self):
        super().__init__(quiet=True)
        self.log = []

    def speak(self, text):
        self.log.append((time.monotonic(), "noi", text))
        print(f"   [NÓI] {text}")
        return super().speak(text)

    def play_earcon(self, sound_file):
        self.log.append((time.monotonic(), "bip", os.path.basename(sound_file)))
        return True


# ------------------------------ Kịch bản kiểm thử ------------------------------

def build(platform, device):
    cfg = load_configs(PROJECT_ROOT, platform=platform)
    app_cfg, model_cfg = cfg["app"], cfg["model"]
    earcons = factory.build_earcons(cfg)
    stack = factory.build_risk_stack(cfg, earcons)
    tts = RecordingTTS()
    audio = AudioCoordinator(tts)
    camera = ScriptedCamera()
    safety_cfg = app_cfg.get("safety", {})
    channel = SafetySoundChannel(audio, earcons) if earcons and safety_cfg.get("beeper") else None
    runtime = app_cfg.get("runtime", {})
    system = SystemCoordinator(
        camera, factory.build_detector(cfg, device, stack["class_filter"]), stack["tracker"],
        factory.build_depth(cfg, device), stack["synchronizer"], stack["risk_fsm"], stack["aggregator"],
        audio, build_ocr_service(model_cfg, device), build_vqa_service(model_cfg, device),
        watchdog_timeout_sec=float(runtime.get("watchdog_timeout_sec", 3.0)), safety_channel=channel,
        on_demand_high_alert=safety_cfg.get("on_demand_high_alert", "off"), earcons=earcons,
        voice_input=build_voice_input(app_cfg),
        auto_return_warning_sec=0.0,  # tắt tự quay về để mỗi phím được kiểm thử độc lập
        startup_grace_max_sec=runtime.get("startup_grace_max_sec"),
        min_frame_brightness=float(runtime.get("min_frame_brightness", 0) or 0),
        object_finder=build_object_finder(app_cfg, earcons, audio, stack, model_cfg))
    hud = HUDRenderer(960, 720, app_cfg.get("ui", {}), stack["corridor"])
    return cfg, system, camera, tts, hud


def wait_for(tts, since, predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        lines = [e for e in tts.log if e[0] >= since]
        if predicate(lines):
            return True, lines
        time.sleep(0.1)
    return False, [e for e in tts.log if e[0] >= since]


def spoken(lines):
    return [e[2] for e in lines if e[1] == "noi"]


def compose(hud_img, key, title, ok, lines, note):
    panel_h = 330
    canvas = Image.new("RGB", (hud_img.shape[1], hud_img.shape[0] + panel_h), (24, 24, 28))
    canvas.paste(Image.fromarray(cv2.cvtColor(hud_img, cv2.COLOR_BGR2RGB)), (0, 0))
    d = ImageDraw.Draw(canvas)
    y = hud_img.shape[0] + 12
    d.text((16, y), f"PHÍM {key}  —  {title}", fill=(120, 220, 255), font=font(30))
    badge = "ĐẠT" if ok else "KHÔNG ĐẠT"
    d.rounded_rectangle([800, y, 940, y + 40], 8, fill=(30, 150, 70) if ok else (190, 50, 50))
    d.text((812 if ok else 806, y + 6), badge, fill=(255, 255, 255), font=font(24 if ok else 18))
    y += 50
    d.text((16, y), note, fill=(200, 200, 200), font=font(18))
    y += 32
    for t, kind, text in lines[:9]:
        label = "Hệ thống nói:" if kind == "noi" else "Âm báo:"
        msg = text if len(text) <= 95 else text[:92] + "..."
        d.text((16, y), f"{label} {msg}", fill=(255, 255, 255) if kind == "noi" else (255, 200, 90),
               font=font(18))
        y += 26
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="auto")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = load_configs(PROJECT_ROOT, platform=args.platform)
    device = args.device or cfg["app"].get("app", {}).get("device", "cpu")
    cfg, system, camera, tts, hud = build(args.platform, device)
    runner = AppRunner(system, hud, enable_gui=False, one_button={"enabled": True})
    system.start()
    print("Khởi động, chờ mô hình chạy ổn định...")
    time.sleep(8)
    results, images = [], []

    def record(key, title, ok, lines, note, snapshot_img):
        img = compose(snapshot_img, key, title, ok, lines, note)
        path = os.path.join(OUT_DIR, f"{len(results) + 1:02d}_phim_{''.join(c if c.isalnum() else '_' for c in key.lower())}.png")
        img.save(path)
        images.append(img)
        results.append({"phim": key, "chuc_nang": title, "dat": ok, "ghi_chu": note,
                        "loi_noi": spoken(lines), "am_bao": [e[2] for e in lines if e[1] == "bip"],
                        "anh": os.path.relpath(path, PROJECT_ROOT)})
        print(f"== {key}: {'ĐẠT' if ok else 'KHÔNG ĐẠT'} - {note}")

    def shot():
        return hud.render(system.get_runtime_snapshot())

    def run_key(key, cmd, title, scene, predicate, timeout, note, settle=2.0, after=None):
        camera.set_scene(scene)
        time.sleep(settle)
        system.audio_coordinator.wait_until_idle(timeout=15)
        t0 = time.monotonic()
        runner.dispatch(cmd)
        ok, lines = wait_for(tts, t0, predicate, timeout)
        img = shot()
        if after:
            ok, note = after(ok, lines, note)
        record(key, title, ok, lines, note, img)
        system.audio_coordinator.wait_until_idle(timeout=30)

    def has(sub):
        return lambda lines: any(sub.lower() in s.lower() for s in spoken(lines))

    run_key("H", "help", "Trợ giúp (đọc danh sách phím)", "street", has("Phím cách"), 10,
            "Đọc to toàn bộ phím cho người không nhìn thấy màn hình.")
    run_key("I", "status", "Trạng thái hệ thống", "street", has("Đang ở"), 10,
            "Đọc chế độ hiện tại, camera/độ sâu, tự hiệu chỉnh khoảng cách.")
    sign_words = ("thoát", "thoat", "4a", "hút", "phòng", "cấm")
    run_key("SPACE", "ocr", "Đọc chữ (OCR)", "sign",
            lambda ls: any(any(w in s.lower() for w in sign_words) for s in spoken(ls)), 60,
            "Cảnh: biển chữ 'LỐI THOÁT HIỂM / PHÒNG HỌC 4A / CẤM HÚT THUỐC'.")
    system_msgs = ("Đã chuyển", "Đang quan sát", "Đang xử lý", "Vật cản", "Hệ thống")
    run_key("Q", "vqa", "Hỏi 'Phía trước có gì?' (VQA)", "street",
            lambda ls: any(not s.startswith(system_msgs) for s in spoken(ls)), 70,
            f"Cảnh: đường phố có xe buýt và người. Backend: {cfg['model']['vqa'].get('backend')}.")

    # Bấm Q hai lần liên tiếp: lần hai phải bị từ chối CÓ âm thanh.
    camera.set_scene("street")
    system.audio_coordinator.wait_until_idle(timeout=15)
    t0 = time.monotonic()
    runner.dispatch("vqa")
    time.sleep(0.3)
    runner.dispatch("vqa")
    ok, lines = wait_for(tts, t0, lambda ls: any(e[2] == "reject.wav" for e in ls), 5)
    record("Q x2", "Bấm khi hệ thống đang bận", ok, lines,
           "Lần bấm thứ hai bị từ chối nhưng có tiếng báo (bản gốc từ chối trong im lặng).", shot())
    system.audio_coordinator.wait_until_idle(timeout=70)
    time.sleep(1)
    while system.is_on_demand_active():
        time.sleep(0.5)
    system.audio_coordinator.wait_until_idle(timeout=30)

    run_key("V", "voice", "Hỏi bằng giọng nói", "street",
            lambda ls: any(("giọng nói" in s or "Mời bạn" in s) for s in spoken(ls)), 15,
            "Máy chưa cài sounddevice/faster-whisper: hệ thống phải nói rõ lý do.")

    # S: đang đọc trợ giúp (dài) thì bấm S -> phải im ngay.
    camera.set_scene("street")
    system.audio_coordinator.wait_until_idle(timeout=15)
    t0 = time.monotonic()
    runner.dispatch("help")
    time.sleep(1.5)
    speaking_before = system.audio_coordinator.is_speaking
    runner.dispatch("stop")
    time.sleep(0.3)
    stopped = not system.audio_coordinator.is_speaking
    lines = [e for e in tts.log if e[0] >= t0]
    record("S", "Dừng đọc / tạm im tiếng bíp", speaking_before and stopped, lines,
           f"Đang đọc trợ giúp: {'có' if speaking_before else 'không'}; sau khi bấm S im ngay: "
           f"{'có' if stopped else 'không'}.", shot())

    run_key("C", "warning", "Về chế độ cảnh báo", "street", has("chế độ cảnh báo"), 8,
            "Bật lại lời cảnh báo vật cản sau khi dùng đọc chữ/hỏi đáp.")

    # Che camera: không bấm phím, hệ thống phải tự nói.
    camera.set_scene("covered")
    t0 = time.monotonic()
    ok, lines = wait_for(tts, t0, has("ống kính bị che"), 12)
    record("(che camera)", "Phát hiện ống kính bị che", ok, lines,
           "Cảnh: tay che ống kính (ảnh nhoè, không tối hẳn vì camera tự tăng sáng).", shot())
    camera.set_scene("street")
    wait_for(tts, time.monotonic(), has("trở lại"), 10)
    system.audio_coordinator.wait_until_idle(timeout=15)

    # Cảnh báo tự động: người tiến lại gần.
    camera.set_scene("approach")
    t0 = time.monotonic()
    ok, lines = wait_for(tts, t0, lambda ls: any("người" in s.lower() for s in spoken(ls)), 12)
    record("(tự động)", "Cảnh báo người tiến lại gần", ok, lines,
           "Cảnh: người to dần trong 4 s. Kỳ vọng câu ngắn + âm báo theo hướng.", shot())

    # ---------- Nút bấm một phím (mô phỏng nút chụp ảnh Bluetooth) ----------
    def press_button(n=1, hold=0.0):
        t = time.monotonic()
        if hold:
            runner.button_event("press", t)
            runner.button_event("release", t + hold)
        for i in range(n):
            runner.button_event("press", t + i * 0.2)
            runner.button_event("release", t + i * 0.2 + 0.05)
        end = time.monotonic() + 1.5 + hold
        while time.monotonic() < end:  # như vòng lặp app: hỏi bộ giải mã liên tục
            runner._process_button(time.monotonic())
            time.sleep(0.03)

    def run_button(label, title, scene, n, hold, predicate, timeout, note):
        camera.set_scene(scene)
        time.sleep(2.0)
        system.audio_coordinator.wait_until_idle(timeout=20)
        while system.is_on_demand_active():
            time.sleep(0.3)
        t0 = time.monotonic()
        press_button(n, hold)
        ok, lines = wait_for(tts, t0, predicate, timeout)
        record(label, title, ok, lines, note, shot())
        system.audio_coordinator.wait_until_idle(timeout=30)

    system.enter_warning_mode()
    run_button("NÚT x1", "Nút cầm tay: bấm 1 lần = hỏi", "street", 1, 0,
               lambda ls: any(not s.startswith(system_msgs) for s in spoken(ls)), 40,
               "Một nút duy nhất, không cần nhìn bàn phím. Bấm 1 lần -> 'Phía trước có gì?'.")
    while system.is_on_demand_active():
        time.sleep(0.3)
    run_button("NÚT x2", "Nút cầm tay: bấm 2 lần = đọc chữ", "sign", 2, 0,
               lambda ls: any(any(w in s.lower() for w in sign_words) for s in spoken(ls)), 40,
               "Bấm nhanh 2 lần -> đọc biển chữ.")
    while system.is_on_demand_active():
        time.sleep(0.3)
    run_button("NÚT giữ", "Nút cầm tay: giữ lâu = về cảnh báo", "street", 0, 0.9,
               has("chế độ cảnh báo"), 8, "Giữ ≥ 0,7 s khi đang im -> về chế độ cảnh báo.")

    # ---------- Tìm đồ ----------
    run_button("NÚT x3 / F", "Tìm đồ: món đầu tiên trong danh sách", "street", 3, 0,
               lambda ls: any("Chưa thấy" in s for s in spoken(ls)), 12,
               "Bấm 3 lần (hoặc F) bắt đầu tìm; bấm tiếp để đổi món. Cảnh không có cốc -> nhắc xoay người.")
    camera.set_scene("person_left")
    time.sleep(1.5)
    system.audio_coordinator.wait_until_idle(timeout=20)
    t0 = time.monotonic()
    system.start_find("người", "person")  # tương đương nói "tìm người" (phím V) hoặc bấm F tới món "người"
    left_ok, _ = wait_for(tts, t0, lambda ls: any(e[2].startswith(("guide_0", "guide_1")) for e in ls), 10)
    camera.set_scene("person_center")  # người dùng xoay về phía vật
    t1 = time.monotonic()
    center_ok, _ = wait_for(tts, t1, lambda ls: any(e[2].endswith("_hi.wav") for e in ls), 10)
    time.sleep(1.0)
    lines = [e for e in tts.log if e[0] >= t0]
    lines = ([e for e in lines if e[1] == "noi"] + [e for e in lines if e[1] == "bip"][:3]
             + [e for e in lines if e[1] == "bip" and e[2].endswith("_hi.wav")][:2])
    ok = left_ok and center_ok and any(s.startswith(("Thấy người", "Người")) for s in spoken(lines))
    record("TÌM NGƯỜI", "Tìm đồ: dẫn hướng bằng âm thanh lập thể", ok, lines,
           "Người lệch trái -> bíp trầm lệch trái (guide_0/1_lo); xoay vào giữa -> bíp cao (guide_3_hi).",
           shot())
    system.enter_warning_mode()
    system.audio_coordinator.wait_until_idle(timeout=15)

    quit_ok = runner.dispatch("quit") is False
    results.append({"phim": "ESC", "chuc_nang": "Thoát", "dat": quit_ok, "ghi_chu": "dispatch trả về False"})
    system.stop()

    # Ảnh tổng hợp 2 cột.
    cols = 2
    w, h = images[0].size
    scale = 0.5
    tw, th = int(w * scale), int(h * scale)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols, th * rows), (10, 10, 10))
    for i, im in enumerate(images):
        sheet.paste(im.resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
    sheet.save(os.path.join(OUT_DIR, "tong_hop.png"))
    with open(os.path.join(OUT_DIR, "ket_qua.json"), "w", encoding="utf-8") as f:
        json.dump({"nen_tang": cfg["_meta"]["platform"], "thiet_bi": device, "ket_qua": results},
                  f, ensure_ascii=False, indent=1)
    passed = sum(r["dat"] for r in results)
    print(f"\nKẾT QUẢ: {passed}/{len(results)} đạt. Ảnh: {OUT_DIR}")


if __name__ == "__main__":
    main()

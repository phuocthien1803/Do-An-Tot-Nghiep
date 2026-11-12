import threading
import time
from typing import Optional, Dict, Any, List
import numpy as np

from src.contracts.frame_packet import FramePacket
from src.contracts.detection import DetectionResult
from src.contracts.depth_map import DepthMap
from src.contracts.risk import RiskAssessment, RiskLevel, DataQuality
from src.contracts.audio import AudioTask, AudioPriority
from src.contracts.request import OCRRequest, OCRResult, VQARequest, VQAResult

from src.camera.bounded_buffer import BoundedBuffer
from src.camera.camera_manager import CameraManager
from src.detection.class_filter import ClassFilter
from src.detection.yolo_detector import YoloDetector
from src.tracking.byte_tracker import ByteTrackerAdapter
from src.depth.depth_estimator import DepthEstimator
from src.depth.roi_extractor import ROIExtractor
from src.fusion.spatial_zones import SpatialZones
from src.fusion.synchronizer import Synchronizer
from src.fusion.risk_fsm import RiskFSM
from src.fusion.alert_aggregator import AlertAggregator
from src.ocr.ocr_service import OCRService
from src.vqa.vqa_service import VQAService
from src.audio.audio_coordinator import AudioCoordinator
from src.audio.earcons import EarconLibrary
from src.audio.safety_channel import SafetySoundChannel
from src.asr.voice_input import route_voice_command
from src.runtime.event_log import EventLog, NULL_LOG
from src.runtime.metrics import PerformanceMetrics
from src.runtime.watchdog import Watchdog

ON_DEMAND_HIGH_ALERT_MODES = ("off", "earcon", "speech")


class SystemCoordinator:
    """
    Điều phối viên tối cao của toàn bộ hệ thống (Runtime Coordinator).
    Tuân thủ mục 3.1 và mục 5.1 của Kế hoạch:
    - Quản lý các worker threads độc lập (Camera, Detection, Depth, Audio).
    - Bộ đệm riêng biệt cho từng luồng (BoundedBuffer).
    - Điều phối chế độ hoạt động (Quan sát liên tục vs On-Demand OCR/VQA).
    - Đo đạc metrics P50/P95 và giám sát watchdog.

    Bản cải tiến (mọi tham số mới mặc định giữ hành vi bản gốc; app.py bật qua cấu hình):
    - on_demand_high_alert: "off" (gốc: OCR/VQA tắt mọi cảnh báo) | "earcon" (vẫn bíp an toàn)
      | "speech" (vẫn đọc lời cảnh báo, ngắt OCR/VQA).
    - safety_channel: tiếng bíp nhịp theo TTC ở mọi chế độ (xem src/audio/safety_channel.py).
    - Phản hồi âm thanh cho MỌI phím (nhận/từ chối), phím trợ giúp H, trạng thái I, hỏi bằng giọng nói V.
    - auto_return_warning_sec: tự quay lại chế độ Cảnh báo sau khi OCR/VQA xong và im lặng N giây.
    - startup_grace_max_sec: thời gian khởi động thích ứng - chờ detection/depth cho ra kết quả đầu
      tiên (máy chậm như laptop Windows chạy CPU) thay vì báo lỗi giả sau 5 s cố định.
    - event_log: nhật ký JSONL để đo độ trễ camera -> âm thanh và đánh giá theo sự kiện.
    """

    MAX_ON_DEMAND_FRAME_AGE_MS = 1000.0
    # Giới hạn thời gian một tác vụ on-demand: Apple Vision từng treo > 40 s trong app
    # (1/5 lần chạy), khoá luôn VQA. Hết hạn thì báo người dùng và giải phóng worker;
    # luồng suy luận bị treo (daemon) không thể bị dừng cưỡng bức.
    OCR_TIMEOUT_SEC = 30.0
    VQA_TIMEOUT_SEC = 60.0  # gồm cả lần nạp model MLX đầu tiên (~11 s)
    ASR_TIMEOUT_SEC = 40.0  # ghi âm 5 s + nạp/nhận dạng Whisper lần đầu

    # Giám sát sức khỏe: người khiếm thị không thấy HUD, nên mất camera/depth phải được nói ra.
    HEALTH_CHECK_INTERVAL_SEC = 0.5
    CAMERA_TIMEOUT_SEC = 2.0
    STATUS_REPEAT_SEC = 10.0
    STARTUP_GRACE_SEC = 5.0
    STATUS_MESSAGES = {
        "camera": "Mất tín hiệu camera. Cảnh báo vật cản đang tạm ngừng, "
                  "hãy dừng lại và dùng gậy dò đường.",
        "depth": "Dữ liệu độ sâu bị gián đoạn. Cảnh báo vật cản đang tạm ngừng, "
                 "hãy dừng lại và dùng gậy dò đường.",
        "detection": "Bộ nhận diện vật cản bị gián đoạn. Cảnh báo vật cản đang tạm ngừng, "
                     "hãy dừng lại và dùng gậy dò đường.",
        "dark": "Ảnh camera quá tối hoặc ống kính bị che. Cảnh báo vật cản đang tạm ngừng, "
                "hãy dừng lại và dùng gậy dò đường.",
    }
    DARK_ANNOUNCE_AFTER_SEC = 2.0
    RECOVERED_WARNING = "Cảnh báo vật cản đã hoạt động trở lại."
    RECOVERED_OTHER = "Hệ thống đã hoạt động trở lại."
    MODE_NAMES_VI = {"WARNING": "chế độ cảnh báo", "OCR": "chế độ đọc chữ", "VQA": "chế độ hỏi đáp"}
    BUSY_TEXT = "Đang xử lý yêu cầu trước, vui lòng chờ."
    HELP_TEXT = (
        "Phím cách: đọc chữ. Phím Q: hỏi phía trước có gì. Phím V: hỏi bằng giọng nói. "
        "Phím C: về chế độ cảnh báo. Phím S: dừng đọc và tạm im tiếng bíp. "
        "Phím I: nghe trạng thái hệ thống. Phím H: nghe lại hướng dẫn. Phím Escape: thoát."
    )
    AUTO_RETURN_TEXT = "Đã tự động quay lại chế độ cảnh báo."

    def __init__(
        self,
        camera_manager: CameraManager,
        detector: YoloDetector,
        tracker: ByteTrackerAdapter,
        depth_estimator: DepthEstimator,
        synchronizer: Synchronizer,
        risk_fsm: RiskFSM,
        alert_aggregator: AlertAggregator,
        audio_coordinator: AudioCoordinator,
        ocr_service: Optional[OCRService] = None,
        vqa_service: Optional[VQAService] = None,
        watchdog_timeout_sec: float = 3.0,
        safety_channel: Optional[SafetySoundChannel] = None,
        on_demand_high_alert: str = "off",
        earcons: Optional[EarconLibrary] = None,
        event_log: Optional[EventLog] = None,
        voice_input=None,
        auto_return_warning_sec: float = 0.0,
        startup_grace_max_sec: Optional[float] = None,
        min_frame_brightness: float = 0.0,
    ):
        if on_demand_high_alert not in ON_DEMAND_HIGH_ALERT_MODES:
            raise ValueError(f"on_demand_high_alert phải thuộc {ON_DEMAND_HIGH_ALERT_MODES}")
        self.camera_manager = camera_manager
        self.detector = detector
        self.tracker = tracker
        self.depth_estimator = depth_estimator
        self.synchronizer = synchronizer
        self.risk_fsm = risk_fsm
        self.alert_aggregator = alert_aggregator
        self.audio_coordinator = audio_coordinator
        self.ocr_service = ocr_service or OCRService()
        self.vqa_service = vqa_service or VQAService()
        self.safety_channel = safety_channel
        self.on_demand_high_alert = on_demand_high_alert
        self.earcons = earcons
        self.event_log = event_log or NULL_LOG
        self.voice_input = voice_input
        self.auto_return_warning_sec = float(auto_return_warning_sec or 0.0)
        self.startup_grace_max_sec = startup_grace_max_sec
        # Bản cải tiến: ảnh quá tối/ống kính bị che -> độ sâu vô nghĩa; bản gốc vẫn coi là VALID
        # và có thể báo "vật cản" giả hoặc im lặng sai. 0 = tắt (như bản gốc).
        self.min_frame_brightness = float(min_frame_brightness or 0.0)
        self._dark_since: Optional[float] = None

        self.metrics = PerformanceMetrics()
        self.watchdog = Watchdog(timeout_sec=watchdog_timeout_sec)

        # Bộ đệm riêng cho detection và depth
        self.buf_detection: BoundedBuffer[FramePacket] = BoundedBuffer(maxsize=2)
        self.buf_depth: BoundedBuffer[FramePacket] = BoundedBuffer(maxsize=2)
        self.camera_manager.subscribe(self.buf_detection)
        self.camera_manager.subscribe(self.buf_depth)

        # Trạng thái runtime
        self._running = False
        self._det_thread: Optional[threading.Thread] = None
        self._depth_thread: Optional[threading.Thread] = None
        self._on_demand_thread: Optional[threading.Thread] = None
        self._on_demand_starting = False
        self._shutting_down = False

        # Lưu trữ trạng thái mới nhất phục vụ hiển thị HUD
        self._latest_packet: Optional[FramePacket] = None
        self._latest_detection: Optional[DetectionResult] = None
        self._latest_depth: Optional[DepthMap] = None
        self._latest_assessments: List[RiskAssessment] = []
        self._current_mode = "OBSERVATION" # "OBSERVATION" | "OCR" | "VQA" | "DEGRADED"
        self._selected_mode = "WARNING"
        self._warning_since = float("-inf")
        self._on_demand_active = False
        self._on_demand_generation = 0
        self._active_on_demand_token: Optional[int] = None
        self._on_demand_finished_at: Optional[float] = None
        self._state_lock = threading.Lock()

        # Giám sát sức khỏe (đọc/ghi dưới _state_lock)
        self._health_thread: Optional[threading.Thread] = None
        self._started_at: Optional[float] = None
        self._last_valid_pair_time: Optional[float] = None
        self._first_detection_time: Optional[float] = None
        self._first_depth_time: Optional[float] = None
        self._health_problem: Optional[str] = None
        self._announced_problem: Optional[str] = None
        self._last_status_time = float("-inf")

        # Đo độ trễ khung hình -> bắt đầu phát âm thanh cảnh báo.
        try:
            self.audio_coordinator.playback_listener = self._on_audio_playback
            self.audio_coordinator.earcon_listener = self._on_earcon
        except Exception:
            pass

    # ================= Âm thanh phản hồi & nhật ký =================

    def _on_audio_playback(self, task: AudioTask, started_at: float) -> None:
        latency_ms = None
        source = getattr(task, "source_timestamp", None)
        if source is not None:
            latency_ms = (started_at - source) * 1000.0
            if getattr(task, "category", "") == "alert":
                self.metrics.record_latency("alert_latency_ms", latency_ms)
        self.event_log.write("audio_start", t=started_at, text=task.text,
                             priority=int(task.priority), category=getattr(task, "category", ""),
                             source_timestamp=source, latency_ms=latency_ms)

    def _on_earcon(self, sound_file: str, t: float) -> None:
        self.event_log.write("earcon", t=t, sound=sound_file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1])

    def _earcon(self, name: str) -> None:
        if self.earcons is not None:
            self.audio_coordinator.play_earcon(self.earcons.path(name))

    def _feedback_ack(self) -> None:
        self._earcon("ack")

    def _feedback_reject(self) -> None:
        """Bản gốc từ chối phím trong im lặng; người khiếm thị không biết phím có được nhận không."""
        self._earcon("reject")
        if not self.audio_coordinator.is_busy:
            self.audio_coordinator.post_task(AudioTask(
                priority=AudioPriority.ON_DEMAND, text=self.BUSY_TEXT,
                expires_at=time.monotonic() + 3.0, interruptible=True, category="feedback"))

    def is_on_demand_active(self) -> bool:
        """Kiểm tra xem hệ thống có đang trong tác vụ On-Demand (VQA/OCR) hay không."""
        with self._state_lock:
            return self._on_demand_active

    def _alerts_spoken_in(self, mode: str) -> bool:
        return mode == "WARNING" or self.on_demand_high_alert == "speech"

    def _select_mode_locked(self, mode: str) -> None:
        changed = self._selected_mode != mode
        self._selected_mode = mode
        self._current_mode = "OBSERVATION" if mode == "WARNING" else mode
        self.audio_coordinator.interrupt()
        self.audio_coordinator.set_safety_alerts_enabled(self._alerts_spoken_in(mode))
        if changed:
            if mode == "WARNING":
                self._warning_since = time.monotonic()
                text = "Đã chuyển sang chế độ cảnh báo."
            elif self.on_demand_high_alert == "earcon":
                text = (f"Đã chuyển sang {self.MODE_NAMES_VI.get(mode, mode)}. "
                        "Vật cản nguy hiểm vẫn được báo bằng tiếng bíp.")
            elif self.on_demand_high_alert == "speech":
                text = f"Đã chuyển sang {self.MODE_NAMES_VI.get(mode, mode)}. Cảnh báo vật cản vẫn bật."
            else:
                text = f"Đã chuyển sang {mode}. Cảnh báo vật cản đã tắt."
            self.audio_coordinator.post_task(AudioTask(
                priority=AudioPriority.ON_DEMAND, text=text, interruptible=True, category="info"))
            self.event_log.write("mode", mode=mode)

    def _cancel_on_demand_locked(self) -> None:
        self._on_demand_generation += 1
        self._on_demand_active = False
        self._active_on_demand_token = None

    def enter_warning_mode(self) -> None:
        """C: cancel on-demand output and explicitly restore fresh safety alerts."""
        with self._state_lock:
            if self._shutting_down:
                return
            self._cancel_on_demand_locked()
            self._select_mode_locked("WARNING")

    def set_on_demand_active(self, active: bool, mode: str = "OBSERVATION") -> None:
        """Compatibility setter: ending a task no longer changes the selected mode."""
        if active and mode not in ("OCR", "VQA"):
            raise ValueError("On-demand mode must be OCR or VQA")
        with self._state_lock:
            if self._shutting_down and active:
                return
            self._cancel_on_demand_locked()
            if active:
                self._select_mode_locked(mode)
                self._on_demand_active = True
                self._active_on_demand_token = self._on_demand_generation
            else:
                self.audio_coordinator.interrupt()

    def _is_current_on_demand(self, token: int) -> bool:
        with self._state_lock:
            return self._on_demand_active and self._active_on_demand_token == token

    def _post_on_demand_audio(self, token: int, task: AudioTask) -> bool:
        """Chỉ token hiện hành mới được tạo side effect âm thanh."""
        with self._state_lock:
            if not self._on_demand_active or self._active_on_demand_token != token:
                return False
            return self.audio_coordinator.post_task(task)

    def _wait_on_demand_audio(self, token: int, timeout: float) -> bool:
        """Bounded polling lets C/S/shutdown release a cancelled worker promptly."""
        deadline = time.monotonic() + timeout
        while self._is_current_on_demand(token):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if self.audio_coordinator.wait_until_idle(timeout=min(0.1, remaining)):
                return True
        return False

    def _publish_on_demand_result(
        self,
        token: int,
        metric_name: str,
        latency: float,
        task: AudioTask
    ) -> bool:
        """Ghi metrics và phát kết quả như một thao tác có kiểm tra generation."""
        with self._state_lock:
            if not self._on_demand_active or self._active_on_demand_token != token:
                return False
            self.metrics.record_latency(metric_name, latency)
            posted = self.audio_coordinator.post_task(task)
        self.event_log.write("on_demand_result", kind=metric_name, latency_sec=latency, text=task.text)
        return posted

    def _finish_on_demand(self, token: int) -> None:
        """Only finish the current task; retain the selected OCR/VQA mode."""
        with self._state_lock:
            if self._active_on_demand_token != token:
                return
            self._on_demand_active = False
            self._active_on_demand_token = None
            self._on_demand_finished_at = time.monotonic()

    def _start_on_demand(
        self,
        mode: str,
        target,
        args: tuple = ()
    ) -> bool:
        """Dành token và khởi động đúng một worker inference on-demand."""
        with self._state_lock:
            shutting_down = self._shutting_down
            if self._shutting_down or self._on_demand_starting or (
                self._on_demand_thread is not None and self._on_demand_thread.is_alive()
            ):
                rejection = "worker on-demand trước chưa kết thúc"
                thread = None
                token = -1
            elif self._on_demand_active:
                rejection = "tác vụ on-demand khác đang hoạt động"
                thread = None
                token = -1
            else:
                rejection = ""
                self._on_demand_generation += 1
                token = self._on_demand_generation
                self._on_demand_active = True
                self._active_on_demand_token = token
                self._select_mode_locked(mode)
                self._on_demand_starting = True
                thread = threading.Thread(
                    target=target,
                    args=(token, *args),
                    name=f"{mode}TaskThread",
                    daemon=True
                )
                self._on_demand_thread = thread

        if thread is None:
            print(f"[SystemCoordinator] Từ chối {mode}: {rejection}.")
            if not shutting_down:
                self._feedback_reject()
            return False

        try:
            # Không start thread trong lúc giữ _state_lock.
            thread.start()
            self._feedback_ack()
            return True
        except Exception as exc:
            with self._state_lock:
                if self._on_demand_thread is thread:
                    self._on_demand_thread = None
                if self._active_on_demand_token == token:
                    self._on_demand_generation += 1
                    self._active_on_demand_token = None
                    self._on_demand_active = False
            print(f"[SystemCoordinator] Không thể khởi động {mode}: {exc}")
            return False
        finally:
            with self._state_lock:
                self._on_demand_starting = False

    def start(self) -> None:
        """Khởi động toàn bộ các luồng của hệ thống."""
        if self._running:
            return
        with self._state_lock:
            self._shutting_down = False
        self._running = True

        self.audio_coordinator.start()
        if self.safety_channel is not None:
            self.safety_channel.start()
        self.camera_manager.start()

        self._det_thread = threading.Thread(target=self._detection_worker, name="DetectionWorker", daemon=True)
        self._depth_thread = threading.Thread(target=self._depth_worker, name="DepthWorker", daemon=True)

        self._det_thread.start()
        self._depth_thread.start()

        with self._state_lock:
            self._started_at = time.monotonic()
        self._health_thread = threading.Thread(target=self._health_worker, name="HealthMonitor", daemon=True)
        self._health_thread.start()
        self.event_log.write("start", on_demand_high_alert=self.on_demand_high_alert)

        # Phát âm thanh chào mừng hệ thống khởi động
        self.audio_coordinator.post_task(AudioTask(
            priority=AudioPriority.INFO,
            text="Hệ thống đã sẵn sàng hỗ trợ bạn." + (" Bấm H để nghe hướng dẫn." if self.earcons else ""),
            sound_file="assets/audio/ready.wav",
            category="info",
        ))

    def stop(self) -> None:
        """Dừng toàn bộ hệ thống an toàn."""
        with self._state_lock:
            self._shutting_down = True
            self._cancel_on_demand_locked()
            self.audio_coordinator.interrupt()
        self._running = False
        if self.safety_channel is not None:
            self.safety_channel.stop()
        self.camera_manager.stop()
        if self._det_thread and self._det_thread.is_alive():
            self._det_thread.join(timeout=1.5)
        if self._depth_thread and self._depth_thread.is_alive():
            self._depth_thread.join(timeout=1.5)
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=1.5)
        if (
            self._on_demand_thread
            and self._on_demand_thread.is_alive()
            and self._on_demand_thread is not threading.current_thread()
        ):
            self._on_demand_thread.join(timeout=1.5)
        self.audio_coordinator.stop()
        self.event_log.write("stop", metrics=self.metrics.get_summary())

    # ================= Giám sát sức khỏe =================

    def _health_worker(self) -> None:
        while self._running:
            time.sleep(self.HEALTH_CHECK_INTERVAL_SEC)
            if not self._running:
                break
            try:
                now = time.monotonic()
                self._check_health(now)
                self._maybe_auto_return(now)
            except Exception as e:
                print(f"[SystemCoordinator] Lỗi giám sát sức khỏe: {e}")

    def _diagnose(self, now: float) -> Optional[str]:
        """Trả về 'camera' | 'detection' | 'depth' khi cảnh báo vật cản không còn tin cậy."""
        with self._state_lock:
            started_at = self._started_at
            last_valid = self._last_valid_pair_time
            first_det = self._first_detection_time
            first_depth = self._first_depth_time
        if started_at is None or now - started_at < self.STARTUP_GRACE_SEC:
            return None
        packet = self.camera_manager.get_latest_frame()
        if (packet is None and self.startup_grace_max_sec
                and now - started_at < self.startup_grace_max_sec):
            # Khởi động thích ứng: camera USB/DirectShow có thể mất vài giây mới cho frame ĐẦU TIÊN.
            # Đã từng có frame thì mất tín hiệu sau đó vẫn được báo ngay như bản gốc.
            return None
        if (not self.camera_manager.is_alive() or packet is None
                or now - packet.timestamp_mono > self.CAMERA_TIMEOUT_SEC):
            return "camera"
        with self._state_lock:
            dark_since = self._dark_since
        if dark_since is not None and now - dark_since >= self.DARK_ANNOUNCE_AFTER_SEC:
            return "dark"
        load_error = getattr(self.depth_estimator, "load_error", None)
        if isinstance(load_error, str) and load_error:
            return "depth"
        baseline = started_at
        if self.startup_grace_max_sec:
            # Khởi động thích ứng: chờ hai mô hình cho ra kết quả đầu tiên (tối đa startup_grace_max_sec).
            if first_det is None or first_depth is None:
                if now - started_at < self.startup_grace_max_sec:
                    return None
            else:
                ready_at = max(first_det, first_depth)
                if now - ready_at < 2.0:
                    return None
                baseline = max(started_at, ready_at)
        health = self.watchdog.check_health()
        if not health.get("detection_worker", False):
            return "detection"
        if not health.get("depth_worker", False):
            return "depth"
        # Depth còn chạy nhưng cặp detection–depth không còn VALID (stale/degraded).
        if now - (last_valid if last_valid is not None else baseline) > self.watchdog.timeout_sec:
            return "depth"
        return None

    def _check_health(self, now: float) -> Optional[str]:
        """Nói ra khi hỏng (nhắc lại mỗi STATUS_REPEAT_SEC) và khi phục hồi.

        Mất camera được báo ở mọi chế độ; lỗi depth/detection chỉ báo ở chế độ Cảnh báo,
        vì ở OCR/VQA cảnh báo vật cản vốn đã tắt.
        """
        problem = self._diagnose(now)
        with self._state_lock:
            if self._shutting_down:
                return problem
            warning = self._selected_mode == "WARNING"
            self._health_problem = problem
            if problem is not None and warning:
                self._current_mode = "DEGRADED"
            audible = problem if (problem in ("camera", "dark") or warning) else None
            text = None
            if audible is not None:
                if (audible != self._announced_problem
                        or now - self._last_status_time >= self.STATUS_REPEAT_SEC):
                    text = self.STATUS_MESSAGES[audible]
                    self._announced_problem = audible
                    self._last_status_time = now
            elif self._announced_problem is not None:
                if problem is None:
                    text = self.RECOVERED_WARNING if warning else self.RECOVERED_OTHER
                self._announced_problem = None
            if text is not None:
                self.audio_coordinator.post_task(AudioTask(
                    priority=AudioPriority.SYSTEM_STATUS, text=text,
                    expires_at=time.monotonic() + 5.0, interruptible=True, category="status"))
        if text is not None:
            self.event_log.write("health", problem=problem, text=text)
        return problem

    def _maybe_auto_return(self, now: float) -> bool:
        """Sau khi OCR/VQA xong và loa im lặng auto_return_warning_sec giây -> về chế độ Cảnh báo."""
        if self.auto_return_warning_sec <= 0:
            return False
        with self._state_lock:
            if (self._shutting_down or self._selected_mode == "WARNING" or self._on_demand_active
                    or self._on_demand_finished_at is None
                    or now - self._on_demand_finished_at < self.auto_return_warning_sec):
                return False
        if self.audio_coordinator.is_busy:
            with self._state_lock:
                self._on_demand_finished_at = now  # đếm lại từ khi loa im
            return False
        self.enter_warning_mode()
        self.audio_coordinator.post_task(AudioTask(
            priority=AudioPriority.ON_DEMAND, text=self.AUTO_RETURN_TEXT, interruptible=True,
            category="info"))
        return True

    def _call_with_deadline(self, fn, timeout_sec: float):
        """Chạy fn ở luồng daemon riêng; TimeoutError nếu quá hạn (luồng treo bị bỏ lại)."""
        box: Dict[str, Any] = {}

        def run():
            try:
                box["result"] = fn()
            except BaseException as exc:  # chuyển lỗi về worker on-demand
                box["error"] = exc

        worker = threading.Thread(target=run, name="OnDemandInference", daemon=True)
        worker.start()
        worker.join(timeout_sec)
        if worker.is_alive():
            raise TimeoutError(f"quá {timeout_sec:.0f} s")
        if "error" in box:
            raise box["error"]
        return box["result"]

    @staticmethod
    def _take_frame(buffer: BoundedBuffer, timeout: float) -> Optional[FramePacket]:
        """Lấy frame MỚI NHẤT (bỏ frame cũ hơn); bản gốc lấy FIFO nên có thể xử lý frame cũ."""
        getter = getattr(buffer, "get_newest", None)
        return getter(timeout=timeout) if getter is not None else buffer.get(timeout=timeout)

    def _depth_worker(self) -> None:
        """Luồng chuyên chạy ước lượng độ sâu (6 - 8 Hz), nhường GPU cho luồng detection."""
        depth_interval = 0.125 # 8 FPS mục tiêu theo kiến trúc
        while self._running:
            packet = self._take_frame(self.buf_depth, 0.2)
            if packet is None:
                continue

            self.watchdog.beat("depth_worker")
            t0 = time.monotonic()
            try:
                depth_map = self.depth_estimator.estimate(packet)
                self.synchronizer.register_depth(depth_map)
                dt_ms = (time.monotonic() - t0) * 1000.0
                self.metrics.record_latency("depth_ms", dt_ms)

                with self._state_lock:
                    self._latest_depth = depth_map
                    if self._first_depth_time is None and bool(np.any(depth_map.valid_mask)):
                        self._first_depth_time = time.monotonic()
            except Exception as e:
                print(f"[DepthWorker] Lỗi suy luận độ sâu: {e}")

            # Điều hòa nhịp độ sâu để tránh nghẽn Metal/GPU trên Apple Silicon
            elapsed = time.monotonic() - t0
            rem = depth_interval - elapsed
            if rem > 0.005:
                time.sleep(rem)

    def _detection_worker(self) -> None:
        """Luồng chuyên chạy Detection, Tracking và Risk Fusion (10 - 20 Hz)."""
        while self._running:
            packet = self._take_frame(self.buf_detection, 0.2)
            if packet is None:
                continue

            self.watchdog.beat("detection_worker")
            self.metrics.record_frame()
            t0 = time.monotonic()

            try:
                # 1. Detection qua YOLO
                raw_det = self.detector.detect(packet)
                self.metrics.record_latency("detection_ms", raw_det.latency_ms)

                # 2. Tracking qua ByteTrack
                tracked_det = self.tracker.update(raw_det)

                # 3. Synchronizer ghép cặp với Depth map tương ứng
                sync_pair = self.synchronizer.synchronize(tracked_det)
                self._apply_darkness_gate(packet, sync_pair)
                self.metrics.record_count(f"quality_{sync_pair.data_quality.value}")

                # 4. Risk Fusion FSM đánh giá rủi ro
                assessments = self.risk_fsm.update(sync_pair)

                # 5. Lời cảnh báo chỉ ở chế độ Cảnh báo; kênh bíp an toàn theo cấu hình.
                self._post_safety_alert(assessments)

                e2e_ms = (time.monotonic() - t0) * 1000.0
                self.metrics.record_latency("end_to_end_ms", e2e_ms)

                with self._state_lock:
                    self._latest_packet = packet
                    self._latest_detection = tracked_det
                    self._latest_assessments = assessments
                    if self._first_detection_time is None:
                        self._first_detection_time = time.monotonic()
                    if sync_pair.data_quality == DataQuality.VALID:
                        self._last_valid_pair_time = time.monotonic()
                    if self._selected_mode == "WARNING":
                        if sync_pair.data_quality in (DataQuality.STALE, DataQuality.DEGRADED):
                            self._current_mode = "DEGRADED"
                        else:
                            self._current_mode = "OBSERVATION"

            except Exception as e:
                print(f"[DetectionWorker] Lỗi suy luận và fusion: {e}")

    def _apply_darkness_gate(self, packet: FramePacket, sync_pair) -> None:
        """Ảnh gần như đen: hạ chất lượng cặp dữ liệu xuống DEGRADED (không cảnh báo theo độ sâu)."""
        if self.min_frame_brightness <= 0:
            return
        brightness = float(packet.image[::8, ::8].mean())
        dark = brightness < self.min_frame_brightness
        with self._state_lock:
            if dark and self._dark_since is None:
                self._dark_since = time.monotonic()
            elif not dark:
                self._dark_since = None
        if dark and sync_pair.data_quality == DataQuality.VALID:
            sync_pair.data_quality = DataQuality.DEGRADED

    def _post_safety_alert(self, assessments: List[RiskAssessment]) -> None:
        """Gate before aggregation so muted modes do not consume cooldown."""
        task = None
        with self._state_lock:
            if self._shutting_down:
                return
            mode = self._selected_mode
            if self._alerts_spoken_in(mode):
                since = self._warning_since if mode == "WARNING" else float("-inf")
                candidates = [a for a in assessments if a.source_timestamp > since]
                if candidates:
                    task = self.alert_aggregator.aggregate(candidates)
                    if task is not None:
                        self.audio_coordinator.post_task(task)
        if task is not None:
            self.event_log.write("alert_posted", text=task.text, source_timestamp=task.source_timestamp,
                                 reason=self.alert_aggregator.last_reason)
        if self.safety_channel is not None:
            on_demand = mode != "WARNING"
            allowed = not on_demand or self.on_demand_high_alert != "off"
            self.safety_channel.update(assessments if allowed else [], time.monotonic(), on_demand)

    # ================= On-Demand OCR & VQA =================

    def trigger_ocr(self) -> bool:
        """Kích hoạt tác vụ đọc chữ theo yêu cầu."""
        return self._start_on_demand("OCR", self._run_ocr_task)

    @classmethod
    def _is_fresh_on_demand_frame(cls, packet: FramePacket, now: float) -> bool:
        age_ms = packet.age_ms(now)
        return 0.0 <= age_ms <= cls.MAX_ON_DEMAND_FRAME_AGE_MS

    def _run_ocr_task(self, token: int) -> None:
        try:
            self._ocr_body(token)
        except Exception as e:
            print(f"[SystemCoordinator] Lỗi xử lý OCR: {e}")
        finally:
            self._finish_on_demand(token)

    def _ocr_body(self, token: int) -> None:
        if not self._is_current_on_demand(token):
            return
        packet = self.get_latest_frame()
        if packet is None or not self._is_fresh_on_demand_frame(
            packet, time.monotonic()
        ):
            if not self._post_on_demand_audio(token, AudioTask(
                priority=AudioPriority.ON_DEMAND,
                text="Chưa có ảnh camera mới để đọc chữ. Vui lòng thử lại.",
                interruptible=True
            )):
                return
            if self._is_current_on_demand(token):
                self._wait_on_demand_audio(token, timeout=4.0)
            return

        if not self._post_on_demand_audio(token, AudioTask(
            priority=AudioPriority.ON_DEMAND,
            text="Đang xử lý đọc chữ...",
            sound_file="assets/audio/chime.wav",
            interruptible=True
        )):
            return

        req = OCRRequest(request_id=f"ocr_{token}_{packet.frame_id}", image=packet.image)
        try:
            res = self._call_with_deadline(lambda: self.ocr_service.process(req),
                                           self.OCR_TIMEOUT_SEC)
        except TimeoutError:
            print(f"[SystemCoordinator] OCR quá {self.OCR_TIMEOUT_SEC:.0f} s, hủy tác vụ.")
            self._post_on_demand_audio(token, AudioTask(
                priority=AudioPriority.ON_DEMAND,
                text="Đọc chữ quá thời gian. Vui lòng thử lại.",
                interruptible=True))
            return

        if not self._publish_on_demand_result(
            token,
            "ocr_sec",
            res.latency_sec,
            AudioTask(
                priority=AudioPriority.ON_DEMAND,
                text=res.text,
                interruptible=True,
                category="on_demand",
            )
        ):
            return

        if self._is_current_on_demand(token):
            self._wait_on_demand_audio(token, timeout=45.0)

    def trigger_vqa(self, question: str = "Phía trước có gì?") -> bool:
        """Kích hoạt tác vụ hỏi đáp thị giác (VQA)."""
        return self._start_on_demand("VQA", self._run_vqa_task, (question,))

    @staticmethod
    def _fresh_vqa_assessments(packet: FramePacket, assessments: List[RiskAssessment],
                               now: float) -> List[RiskAssessment]:
        return [
            assessment for assessment in assessments
            if assessment.data_quality == DataQuality.VALID
            and not assessment.is_expired(now)
            and abs(packet.timestamp_mono - assessment.source_timestamp) <= 0.75
        ]

    def _run_vqa_task(self, token: int, question: str) -> None:
        # Detection/depth và đánh giá nguy cơ vẫn chạy; lời cảnh báo tắt (kênh bíp theo cấu hình).
        try:
            self._vqa_body(token, question)
        except Exception as e:
            print(f"[SystemCoordinator] Lỗi xử lý VQA: {e}")
        finally:
            # Kết thúc tác vụ; tiếp tục giữ chế độ VQA.
            self._finish_on_demand(token)

    def _vqa_body(self, token: int, question: str) -> None:
        if not self._is_current_on_demand(token):
            return
        packet = self.get_latest_frame()
        if packet is None or not self._is_fresh_on_demand_frame(
            packet, time.monotonic()
        ):
            if not self._post_on_demand_audio(token, AudioTask(
                priority=AudioPriority.ON_DEMAND,
                text="Chưa có ảnh camera mới để trả lời. Vui lòng thử lại.",
                interruptible=True
            )):
                return
            if self._is_current_on_demand(token):
                self._wait_on_demand_audio(token, timeout=4.0)
            return

        # Thông báo đang quan sát
        if not self._post_on_demand_audio(token, AudioTask(
            priority=AudioPriority.ON_DEMAND,
            text="Đang quan sát để trả lời...",
            sound_file="assets/audio/chime.wav",
            interruptible=True
        )):
            return

        with self._state_lock:
            fresh_assessments = self._fresh_vqa_assessments(
                packet, self._latest_assessments, time.monotonic()
            )
            detected_names = [a.class_name for a in fresh_assessments]
            spatial_objects = [
                {
                    "name": a.class_name,
                    "direction": a.direction.value,
                    "proximity": a.proximity_desc,
                    "risk_level": a.risk_level.value
                }
                for a in fresh_assessments
            ]

        if not self._is_current_on_demand(token):
            return

        req = VQARequest(request_id=f"vqa_{token}_{packet.frame_id}", image=packet.image, question=question)
        context = {"detected_classes": detected_names, "spatial_objects": spatial_objects}
        try:
            res = self._call_with_deadline(
                lambda: self.vqa_service.answer(req, visual_context=context),
                self.VQA_TIMEOUT_SEC)
        except TimeoutError:
            print(f"[SystemCoordinator] VQA quá {self.VQA_TIMEOUT_SEC:.0f} s, hủy tác vụ.")
            self._post_on_demand_audio(token, AudioTask(
                priority=AudioPriority.ON_DEMAND,
                text="Trả lời quá thời gian. Vui lòng thử lại.",
                interruptible=True))
            return

        if not self._publish_on_demand_result(
            token,
            "vqa_sec",
            res.latency_sec,
            AudioTask(
                priority=AudioPriority.ON_DEMAND,
                text=res.answer,
                interruptible=True,
                category="on_demand",
            )
        ):
            return

        # Đợi câu trả lời phát xong, không đổi chế độ đã chọn.
        if self._is_current_on_demand(token):
            self._wait_on_demand_audio(token, timeout=45.0)

    # ================= Hỏi bằng giọng nói (V) =================

    def trigger_voice_question(self) -> bool:
        return self._start_on_demand("VQA", self._run_voice_task)

    def _say_on_demand(self, token: int, text: str, wait: float = 8.0) -> None:
        if self._post_on_demand_audio(token, AudioTask(
                priority=AudioPriority.ON_DEMAND, text=text, interruptible=True, category="on_demand")):
            self._wait_on_demand_audio(token, timeout=wait)

    def _run_voice_task(self, token: int) -> None:
        try:
            if not self._is_current_on_demand(token):
                return
            voice = self.voice_input
            error = ("Chưa bật nhận dạng giọng nói. Hãy dùng phím Q để hỏi."
                     if voice is None else voice.dependency_error())
            if error:
                self._say_on_demand(token, error)
                return
            self._say_on_demand(token, "Mời bạn nói câu hỏi sau tiếng bíp.", wait=6.0)
            if not self._is_current_on_demand(token):
                return
            self._earcon("ack")
            time.sleep(0.25)
            try:
                text, err = self._call_with_deadline(voice.listen, self.ASR_TIMEOUT_SEC)
            except TimeoutError:
                text, err = None, "Nhận dạng giọng nói quá thời gian. Vui lòng thử lại."
            if err:
                self._say_on_demand(token, err)
                return
            command, question = route_voice_command(text or "")
            self.event_log.write("voice", text=text, command=command)
            print(f"[SystemCoordinator] Giọng nói: '{text}' -> {command}")
            if command == "empty":
                self._say_on_demand(token, "Không nghe rõ câu hỏi. Hãy bấm V và nói lại.")
            elif command == "help":
                self._say_on_demand(token, self.HELP_TEXT, wait=30.0)
            elif command == "status":
                self._say_on_demand(token, self.status_text(), wait=15.0)
            elif command == "warning":
                self.enter_warning_mode()
            elif command == "ocr":
                with self._state_lock:
                    if self._active_on_demand_token == token and self._on_demand_active:
                        self._select_mode_locked("OCR")
                self._ocr_body(token)
            else:
                self._vqa_body(token, question)
        except Exception as e:
            print(f"[SystemCoordinator] Lỗi hỏi bằng giọng nói: {e}")
        finally:
            self._finish_on_demand(token)

    # ================= Trợ giúp, trạng thái, dừng =================

    def speak_help(self) -> None:
        self._feedback_ack()
        self.audio_coordinator.post_task(AudioTask(
            priority=AudioPriority.ON_DEMAND, text=self.HELP_TEXT, interruptible=True, category="info"))

    def status_text(self) -> str:
        with self._state_lock:
            mode = self._selected_mode
            problem = self._health_problem
            depth = self._latest_depth
        parts = [f"Đang ở {self.MODE_NAMES_VI.get(mode, mode)}."]
        if problem == "camera":
            parts.append("Mất tín hiệu camera.")
        elif problem in ("depth", "detection"):
            parts.append("Cảnh báo vật cản đang tạm ngừng do dữ liệu gián đoạn.")
        else:
            parts.append("Camera và cảm biến độ sâu đang hoạt động.")
        floor = ((depth.preprocess_transform or {}).get("floor") if depth is not None else None) or {}
        if floor.get("calibrated"):
            parts.append("Khoảng cách đã được tự hiệu chỉnh theo mặt sàn.")
        elif floor:
            parts.append("Chưa nhìn thấy mặt sàn để tự hiệu chỉnh khoảng cách.")
        if mode != "WARNING":
            if self.on_demand_high_alert == "earcon":
                parts.append("Vật cản nguy hiểm được báo bằng tiếng bíp.")
            elif self.on_demand_high_alert == "off":
                parts.append("Cảnh báo vật cản đang tắt, bấm C để bật lại.")
        return " ".join(parts)

    def speak_status(self) -> None:
        self._feedback_ack()
        self.audio_coordinator.post_task(AudioTask(
            priority=AudioPriority.ON_DEMAND, text=self.status_text(), interruptible=True, category="info"))

    def stop_speech(self) -> None:
        """S: cancel speech/results while keeping the selected mode; tạm im tiếng bíp hiện tại."""
        with self._state_lock:
            self._cancel_on_demand_locked()
            self.audio_coordinator.interrupt()
        if self.safety_channel is not None:
            self.safety_channel.snooze_current()

    # ================= State Getters for HUD =================

    def get_latest_frame(self) -> Optional[FramePacket]:
        live_packet = self.camera_manager.get_latest_frame()
        if live_packet is not None:
            return live_packet
        with self._state_lock:
            return self._latest_packet

    def get_runtime_snapshot(self) -> Dict[str, Any]:
        live_packet = self.camera_manager.get_latest_frame()
        with self._state_lock:
            depth = self._latest_depth
            return {
                "packet": live_packet if live_packet is not None else self._latest_packet,
                "detection": self._latest_detection,
                "depth": depth,
                "assessments": list(self._latest_assessments),
                "mode": self._current_mode,
                "selected_mode": self._selected_mode,
                "on_demand_active": self._on_demand_active,
                "safety_alerts_enabled": (self._alerts_spoken_in(self._selected_mode)
                                          and not self._shutting_down),
                "safety_beeps_enabled": (self.safety_channel is not None and not self._shutting_down
                                         and (self._selected_mode == "WARNING"
                                              or self.on_demand_high_alert != "off")),
                "health_problem": self._health_problem,
                "floor": ((depth.preprocess_transform or {}).get("floor") if depth is not None else None),
                "metrics": self.metrics.get_summary(),
                "is_speaking": self.audio_coordinator.is_speaking
            }

import re
import unicodedata
from typing import Optional


def strip_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt (đ -> d) để so khớp câu gõ không dấu."""
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


class SafetyGuardrail:
    """Guardrail độc lập, luôn chạy trước backend model.

    mode="regex": đúng luật của bản gốc.
    mode="mo_rong" (bản cải tiến): thêm (1) câu hỏi "lối đi có trống/thông thoáng không",
    (2) câu hỏi có vật cản/chắn/vướng/đụng trên đường, (3) câu xác nhận "cứ đi ... là được đúng
    không", (4) tiếng Anh dạng "is the path clear / anything in my way", (5) câu gõ KHÔNG DẤU
    (chỉ áp dụng khi cả câu không có dấu, tránh chặn nhầm câu có dấu), và ở phía câu trả lời
    không chặn nhầm chủ đề "an toàn thực phẩm/giao thông...". Bản gốc lọt 8/30 câu cần chặn.
    """

    REFUSAL = (
        "Hệ thống không thể xác nhận đường đi có an toàn hay không. "
        "Xin hãy cẩn thận dùng gậy dẫn đường và kiểm tra xung quanh."
    )
    # Phân loại theo ý định thay vì khớp từ khoá đơn lẻ, vì từ khoá đơn ("đi thẳng",
    # "qua đường", "an toàn") chặn nhầm câu hỏi đọc biển báo hoặc đếm người
    # (9/3545 câu OpenViVQA bị chặn nhầm) mà lại bỏ lọt "Tôi có thể đi tiếp không?".
    _MOVE = r"(?:đi|bước|qua đường|sang đường|băng qua|di chuyển|tiến lên|tiến tới|rẽ)"
    _ASK = r"(?:có thể|được không|được chứ|được chưa|có được|có sao không|có ổn không|nên|có nên|liệu)"
    MOVE = re.compile(rf"\b{_MOVE}\b")
    # Xin phép di chuyển: từ hỏi phép đứng sát động từ di chuyển ("có thể đi tiếp",
    # "đi về phía trước được không"), không tính câu hỏi chỉ đường/đếm người.
    MOVE_PERMISSION = re.compile(
        rf"\b{_ASK}\s+(?:\w+\s+){{0,2}}{_MOVE}\b|\b{_MOVE}(?:\s+\w+){{0,4}}\s+{_ASK}\b")
    WH_QUESTION = re.compile(
        r"\b(?:nào|đâu|gì|ai|mấy|bao nhiêu|bao xa|tại sao|vì sao|thế nào|như thế nào)\b")
    FIRST_PERSON = re.compile(r"\b(?:tôi|mình|tớ|tao)\b")
    THIRD_PERSON = re.compile(r"\b(?:người|họ|khách|xe)\b")
    HAZARD = re.compile(
        r"\b(?:an toàn|vật cản|chướng ngại|nguy hiểm|va chạm|va vào|đâm vào|vấp ngã|té ngã|"
        r"ngã(?! (?:ba|tư|năm|sáu|bảy)))\b")
    # Hỏi sự có mặt của bậc/hố/cầu thang dạng "có ... không" (không chặn câu mô tả "bậc thang dẫn đến đâu").
    HAZARD_PRESENCE = re.compile(
        r"\bcó\b[^?]*\b(?:bậc|cầu thang|hố|ổ gà)\b[^?]*\bkhông\b"
        r"|\bcó\b[^?]*\bxe\b[^?]*\b(?:tới|đến|chạy|lại gần|đi tới)\b[^?]*\bkhông\b")
    SAFETY_TOPIC = re.compile(r"\ban toàn (?:giao thông|thực phẩm|lao động|vệ sinh|thông tin|điện|cháy nổ)\b")
    READING = re.compile(
        r"\b(?:ghi|viết|đọc|chữ|nội dung|tuyên truyền|khẩu hiệu|biển (?:báo|hiệu)|bảng (?:báo|hiệu)|"
        r"tờ giấy|nhãn|thông báo)\b")
    ENGLISH = re.compile(
        r"\b(?:safe to|is it safe|can i (?:walk|cross|go|move|proceed)|"
        r"(?:any|is there an?) (?:obstacle|hazard|danger))")

    # --- Luật mở rộng (mode="mo_rong") ---
    _PATH = r"(?:đường|lối|lối đi|đường đi|phía trước|trước mặt|hành lang|lối ra|dưới chân)"
    _CLEAR = r"(?:trống|thông|thoáng|thông thoáng|vắng|đi được|qua được|có gì không|có vật gì)"
    _QTAIL = r"(?:không|chứ|chưa|à|hả|phải không|đúng không)"
    PATH_CLEAR = re.compile(rf"\b{_PATH}\b[^?.]*\b{_CLEAR}\b(?:[^?.]*\b{_QTAIL}\b)?")
    OBSTRUCT = re.compile(
        r"\b(?:cản|chắn|vướng|đụng|vấp|tông|húc|đâm|va)\b[^?.]*\b(?:không|chưa|chứ)\b"
        r"|\b(?:có gì|có vật gì|có ai|có cái gì)\b[^?.]*\b(?:cản|chắn|vướng|đụng|đâm|va)\b")
    MOVE_CONFIRM = re.compile(
        r"\b(?:đi|bước|qua|băng qua|tiến|tiếp tục|đi tiếp|xuống|lên)\b[^?.]*"
        r"\b(?:là được|được|ổn|có sao)\b[^?.]*\b(?:không|chứ|chưa|đúng không|phải không)\b")
    FLOOR_HAZARD = re.compile(
        r"\bcó\b[^?.]*\b(?:dốc|mép|vũng nước|trơn|bậc|hố|ổ gà|cầu thang)\b[^?.]*\b(?:không|chưa)\b"
        r"|\b(?:sàn|nền|mặt đường)\b[^?.]*\btrơn\b")
    APPROACH = re.compile(r"\b(?:ai|xe|người|con gì)\b[^?.]*\b(?:về phía|lại gần|tới gần|đến gần)\b[^?.]*\b(?:tôi|mình)\b")
    ENGLISH_EXT = re.compile(
        r"\b(?:(?:path|way|road)(?: ahead)? (?:is |looks )?clear|is the (?:path|way|road)(?: ahead)? clear|"
        r"in my way|bump into|run into|keep (?:walking|going)|can i (?:keep|continue))\b")

    def __init__(self, mode: str = "regex"):
        if mode not in ("regex", "mo_rong"):
            raise ValueError("mode phải là 'regex' hoặc 'mo_rong'")
        self.mode = mode
        self._plain = {}
        if mode == "mo_rong":
            names = ("MOVE", "MOVE_PERMISSION", "WH_QUESTION", "FIRST_PERSON", "THIRD_PERSON", "HAZARD",
                     "HAZARD_PRESENCE", "SAFETY_TOPIC", "READING", "PATH_CLEAR", "OBSTRUCT",
                     "MOVE_CONFIRM", "FLOOR_HAZARD", "APPROACH")
            self._plain = {n: re.compile(strip_diacritics(getattr(self, n).pattern)) for n in names}

    def _rules(self, plain: bool):
        if plain:
            return lambda name: self._plain[name]
        return lambda name: getattr(self, name)

    def _extended_refusal(self, text: str, plain: bool) -> bool:
        r = self._rules(plain)
        reading = bool(r("READING").search(text))
        first_person = bool(r("FIRST_PERSON").search(text))
        wh = bool(r("WH_QUESTION").search(text))
        third = bool(r("THIRD_PERSON").search(text))
        if plain:  # áp dụng lại toàn bộ luật gốc cho câu không dấu
            if r("MOVE").search(text) and first_person:
                return True
            if r("MOVE_PERMISSION").search(text) and not (reading or wh or third):
                return True
            hazard = r("HAZARD").search(text) or r("HAZARD_PRESENCE").search(text)
            if hazard and (first_person or not (reading or r("SAFETY_TOPIC").search(text))):
                return True
        if reading:
            return False
        if r("PATH_CLEAR").search(text) or r("OBSTRUCT").search(text):
            return True
        if r("FLOOR_HAZARD").search(text) or r("APPROACH").search(text):
            return True
        if r("MOVE_CONFIRM").search(text) and not wh and (first_person or not third):
            return True
        return False

    def check(self, question: str) -> Optional[str]:
        normalized = " ".join(unicodedata.normalize("NFC", question).strip().lower().split())
        refusal = self._check_original(normalized)
        if refusal is not None or self.mode != "mo_rong":
            return refusal
        if self.ENGLISH_EXT.search(normalized):
            return self.REFUSAL
        plain = strip_diacritics(normalized) == normalized
        if self._extended_refusal(normalized, plain=False):
            return self.REFUSAL
        if plain and self._extended_refusal(normalized, plain=True):
            return self.REFUSAL
        return None

    def _check_original(self, normalized: str) -> Optional[str]:
        if self.ENGLISH.search(normalized):
            return self.REFUSAL
        reading = bool(self.READING.search(normalized))
        first_person = bool(self.FIRST_PERSON.search(normalized))
        # 1. Người dùng hỏi bản thân có đi/bước/qua đường được không.
        if self.MOVE.search(normalized) and first_person:
            return self.REFUSAL
        if (self.MOVE_PERMISSION.search(normalized) and not reading
                and not self.WH_QUESTION.search(normalized)
                and not self.THIRD_PERSON.search(normalized)):
            return self.REFUSAL
        # 2. Hỏi có vật cản/nguy hiểm/an toàn không (trừ câu hỏi nội dung chữ hoặc chủ đề chung).
        hazard = self.HAZARD.search(normalized) or self.HAZARD_PRESENCE.search(normalized)
        if hazard and (first_person or not (reading or self.SAFETY_TOPIC.search(normalized))):
            return self.REFUSAL
        return None

    def check_answer(self, answer: str) -> Optional[str]:
        normalized = answer.strip().lower()
        for sentence in re.split(r"[.!?]", normalized):
            if re.search(
                r"\b(?:có thể|nên|hãy|cứ|được)\s+"
                r"(?:đi|bước|qua đường|di chuyển|tiếp tục)\b", sentence
            ) or re.search(
                r"\bkhông (?:có|thấy) (?:vật cản|chướng ngại|nguy hiểm)\b",
                sentence
            ) or re.search(
                r"\b(?:safe to (?:walk|cross|proceed)|you can (?:walk|cross|proceed)|"
                r"no obstacles? ahead|(?:path|way|road) (?:is|looks) clear)\b", sentence
            ) or re.search(
                # Khẳng định lối đi trống: chỉ khi chủ ngữ là lối đi/đường/phía trước,
                # để "ghế trống", "chai nước trống" vẫn được phép.
                r"\b(?:lối đi|đường đi|con đường|hành lang|phía trước|trước mặt)\b"
                r"(?:\s+[^\s,.!?]+){0,2}\s*,?\s*"
                r"(?:trống|thông thoáng|thoáng|không có gì|không có vật)", sentence
            ):
                return self.REFUSAL
            if "an toàn" in sentence and not re.search(
                r"\b(?:không|chưa|chẳng)\s+an toàn\b|"
                r"\bkhông thể xác nhận[^.!?]{0,30}\ban toàn\b",
                sentence
            ):
                # Bản cải tiến: "an toàn thực phẩm/giao thông..." là chủ đề nội dung, không phải
                # khẳng định lối đi an toàn (bản gốc chặn nhầm câu đọc tờ giấy).
                if not (self.mode == "mo_rong" and self.SAFETY_TOPIC.search(sentence)):
                    return self.REFUSAL
            if self.mode == "mo_rong" and re.search(
                r"\b(?:không|chẳng) (?:có|thấy) (?:gì|vật gì|ai) (?:cản|chắn|vướng|nguy hiểm)"
                r"|\b(?:cứ|có thể) (?:đi|bước|tiến) (?:thẳng|tiếp|tới)"
                r"|\b(?:lối đi|đường đi|đường|phía trước) (?:rất |khá )?(?:trống|thông thoáng)\b",
                sentence
            ):
                return self.REFUSAL
        return None

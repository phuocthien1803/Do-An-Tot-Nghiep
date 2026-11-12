from typing import Any, Dict, List, Optional


def _spatial_details(visual_context: Optional[Dict[str, Any]], skip_unknown: bool = False) -> List[str]:
    if not visual_context:
        return []
    details = []
    for obj in visual_context.get("spatial_objects", [])[:3]:
        name = obj.get("name", "")
        direction = obj.get("direction", "")
        proximity = obj.get("proximity", "")
        direction_vi = (
            "ở giữa" if direction == "CENTER"
            else "bên trái" if direction == "LEFT"
            else "bên phải"
        )
        if skip_unknown and (not proximity or proximity == "chưa rõ"):
            details.append(f"{name} {direction_vi}")  # không đọc "(chưa rõ)" ra loa
        elif skip_unknown:
            details.append(f"{name} {direction_vi}, {proximity}")
        else:
            details.append(f"{name} {direction_vi} ({proximity})")
    return details


def format_vqa_answer(
    backend_text: str,
    visual_context: Optional[Dict[str, Any]] = None,
    style: str = "goc",
) -> str:
    """Ghép output backend với grounding, hoặc dùng grounding làm fallback.

    style="goc": như bản gốc (mọi câu bắt đầu "Khung cảnh:", kể cả khi người dùng hỏi điều khác).
    style="ngan" (bản cải tiến): trả lời thẳng câu hỏi, bỏ tiền tố, không đọc "(chưa rõ)";
    phần do cảm biến ghi nhận được tách rõ là thông tin bổ sung, không trộn với lời của model.
    """
    if style == "ngan":
        details = _spatial_details(visual_context, skip_unknown=True)
        description = backend_text.strip()
        if description:
            ending = description.rstrip('\"\u201d\u2019\u0027)')
            suffix = "" if ending.endswith((".", "!", "?", "…")) else "."
            text = f"{description}{suffix}"
            if details:
                text += f" Cảm biến ghi nhận: {'; '.join(details)}."
            return text
        if details:
            return f"Cảm biến ghi nhận phía trước: {'; '.join(details)}."
        return VQA_UNAVAILABLE_ANSWER
    details = _spatial_details(visual_context)
    description = backend_text.strip()
    if description:
        # Preserve model punctuation; legacy captions still need a final period.
        ending = description.rstrip('\"\u201d\u2019\u0027)')
        suffix = "" if ending.endswith((".", "!", "?", "…")) else "."
        final_text = f"Khung cảnh: {description}{suffix}"
        if details:
            final_text += f" Cụ thể có: {', '.join(details)}."
        return final_text
    if details:
        return f"Trước mặt bạn phát hiện có: {', '.join(details)}."
    # Không dùng câu kiểu "chưa phát hiện vật thể": người khiếm thị dễ hiểu là phía trước trống.
    return VQA_UNAVAILABLE_ANSWER


VQA_UNAVAILABLE_ANSWER = (
    "Chưa mô tả được khung cảnh lúc này. Vui lòng thử lại và tiếp tục dùng gậy dò đường."
)

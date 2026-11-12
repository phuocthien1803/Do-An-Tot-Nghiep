#!/usr/bin/env python3
"""Ghép nhật ký kích thích (do_tre_kich_thich.py) với nhật ký sự kiện của app (--event-log).

Với mỗi lần ảnh hiện lên, tìm âm thanh an toàn ĐẦU TIÊN sau đó (lời cảnh báo hoặc tiếng bíp) trong
--cua-so giây. Báo cáo: tỉ lệ phát hiện, độ trễ p50/p95 và phần độ trễ bên trong phần mềm
(camera chụp frame -> bắt đầu phát) lấy từ trường latency_ms của sự kiện audio_start.

    python scripts/phan_tich_do_tre.py kich_thich.jsonl su_kien.jsonl [--cua-so 3]
"""
import argparse
import json

import numpy as np


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def analyse(stimuli, events, window: float):
    sounds = sorted(
        [e for e in events if (e.get("type") == "audio_start" and e.get("category") == "alert")
         or (e.get("type") == "earcon" and str(e.get("sound", "")).startswith(("alert_", "drop_", "tick_")))],
        key=lambda e: e["t"])
    rows = []
    for s in stimuli:
        after = [e for e in sounds if s["onset"] <= e["t"] <= s["onset"] + window]
        first = after[0] if after else None
        rows.append({"lan": s.get("lan"), "detected": first is not None,
                     "latency_ms": (first["t"] - s["onset"]) * 1000.0 if first else None,
                     "internal_ms": first.get("latency_ms") if first else None,
                     "sound": first.get("text") or first.get("sound") if first else None})
    lat = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    internal = [r["internal_ms"] for r in rows if r["internal_ms"] is not None]

    def pct(values, q):
        return float(np.percentile(values, q)) if values else None
    return {
        "stimuli": len(rows),
        "detected": sum(r["detected"] for r in rows),
        "detection_rate": sum(r["detected"] for r in rows) / len(rows) if rows else None,
        "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "min": min(lat) if lat else None},
        "internal_frame_to_audio_ms": {"p50": pct(internal, 50), "p95": pct(internal, 95)},
        "per_stimulus": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kich_thich")
    ap.add_argument("su_kien")
    ap.add_argument("--cua-so", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = analyse(read_jsonl(args.kich_thich), read_jsonl(args.su_kien), args.cua_so)
    print(json.dumps({k: v for k, v in report.items() if k != "per_stimulus"}, ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

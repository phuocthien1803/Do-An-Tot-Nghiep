#!/usr/bin/env python3
"""Tính điểm SUS và NASA-TLX (raw) theo điều kiện thử nghiệm người dùng, kèm kiểm định cặp.

Đầu vào: CSV theo mẫu docs/thu_nghiem_nguoi_dung/bang_ket_qua_mau.csv, mỗi dòng = 1 người x 1 điều kiện:
  nguoi,dieu_kien,sus1..sus10 (1-5),tlx_tinh_than,tlx_the_chat,tlx_thoi_gian,tlx_hieu_qua,
  tlx_no_luc,tlx_that_vong (0-100),so_va_cham,thoi_gian_s
SUS = 2,5 x [(tổng câu lẻ - 5) + (25 - tổng câu chẵn)]  (0-100, cao = dễ dùng).
Raw TLX = trung bình 6 thang (0-100, thấp = ít tải). Câu "hiệu quả" dùng thang 0 = rất tốt.

    python scripts/tinh_sus_nasa.py ket_qua_nguoi_dung.csv
"""
import argparse
import csv
import json
from collections import defaultdict
from statistics import mean, stdev

TLX = ["tlx_tinh_than", "tlx_the_chat", "tlx_thoi_gian", "tlx_hieu_qua", "tlx_no_luc", "tlx_that_vong"]


def sus_score(row) -> float:
    items = [float(row[f"sus{i}"]) for i in range(1, 11)]
    odd = sum(items[0::2]) - 5
    even = 25 - sum(items[1::2])
    return 2.5 * (odd + even)


def tlx_score(row) -> float:
    return mean(float(row[k]) for k in TLX)


def summarize(values):
    return {"n": len(values), "mean": mean(values) if values else None,
            "sd": stdev(values) if len(values) > 1 else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    per = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        cond = r["dieu_kien"]
        per[cond][r["nguoi"]] = {"sus": sus_score(r), "tlx": tlx_score(r),
                                 "va_cham": float(r["so_va_cham"]), "thoi_gian": float(r["thoi_gian_s"])}
    report = {"dieu_kien": {}}
    for cond, people in per.items():
        report["dieu_kien"][cond] = {m: summarize([p[m] for p in people.values()])
                                     for m in ("sus", "tlx", "va_cham", "thoi_gian")}
    conds = sorted(per)
    try:
        from scipy.stats import wilcoxon
        tests = {}
        for i, a in enumerate(conds):
            for b in conds[i + 1:]:
                common = sorted(set(per[a]) & set(per[b]))
                for m in ("sus", "tlx", "va_cham", "thoi_gian"):
                    x = [per[a][p][m] for p in common]
                    y = [per[b][p][m] for p in common]
                    if len(common) >= 5 and any(u != v for u, v in zip(x, y)):
                        stat = wilcoxon(x, y)
                        tests[f"{a} vs {b} / {m}"] = {"n": len(common), "W": float(stat.statistic),
                                                     "p": float(stat.pvalue)}
        report["wilcoxon_cap"] = tests
    except Exception as exc:
        report["wilcoxon_cap"] = f"không tính được: {exc}"
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

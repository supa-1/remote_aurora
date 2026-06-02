from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="可视化真假指令结果（HTML/CSV/JSON）")
    p.add_argument("--input_jsonl", required=True, help="一致性样本 JSONL 路径")
    p.add_argument("--output_dir", required=True, help="可视化输出目录")
    p.add_argument("--max_rows", type=int, default=80, help="HTML 展示的最大样例行数")
    return p.parse_args()


def load_rows(input_jsonl: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with Path(input_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "image": str(obj.get("image", "")),
                    "true_instruction": str(obj.get("true_instruction", "")).strip(),
                    "fake_instruction": str(obj.get("fake_instruction", "")).strip(),
                    "action_text": str(obj.get("action_text", "")).strip(),
                    "negative_type": str(obj.get("negative_type", "llm_or_other")).strip() or "llm_or_other",
                }
            )
    return rows


def build_summary(rows: List[Dict[str, str]]) -> Dict[str, object]:
    if not rows:
        return {
            "num_pairs": 0,
            "type_distribution": {},
            "avg_true_len": 0,
            "avg_fake_len": 0,
            "avg_len_delta": 0,
        }

    type_counter = Counter(r["negative_type"] for r in rows)
    true_lens = [len(r["true_instruction"].split()) for r in rows]
    fake_lens = [len(r["fake_instruction"].split()) for r in rows]
    deltas = [f - t for t, f in zip(true_lens, fake_lens)]

    return {
        "num_pairs": len(rows),
        "type_distribution": dict(type_counter),
        "avg_true_len": round(mean(true_lens), 3),
        "avg_fake_len": round(mean(fake_lens), 3),
        "avg_len_delta": round(mean(deltas), 3),
    }


def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image", "negative_type", "true_instruction", "fake_instruction", "action_text"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _bar_html(key: str, value: int, max_count: int) -> str:
    width = 0 if max_count == 0 else int((value / max_count) * 100)
    return (
        "<div class=\"bar-row\">"
        f"<div class=\"bar-key\">{html.escape(key)}</div>"
        "<div class=\"bar-track\">"
        f"<div class=\"bar-fill\" style=\"width:{width}%\"></div>"
        "</div>"
        f"<div class=\"bar-value\">{value}</div>"
        "</div>"
    )


def write_html(rows: List[Dict[str, str]], summary: Dict[str, object], out_html: Path, max_rows: int) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    type_dist = summary.get("type_distribution", {})
    if not isinstance(type_dist, dict):
        type_dist = {}
    max_count = max(type_dist.values()) if type_dist else 0
    bars = "\n".join(_bar_html(str(k), int(v), int(max_count)) for k, v in sorted(type_dist.items()))

    head = """
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AuroraIG 真伪指令可视化</title>
<style>
body { font-family: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif; margin: 24px; color: #1f2937; }
.grid { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
.card { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; }
.card .k { font-size: 12px; color: #6b7280; }
.card .v { font-size: 22px; font-weight: 700; margin-top: 6px; }
.panel { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
.bar-row { display: grid; grid-template-columns: 170px 1fr 60px; align-items: center; gap: 8px; margin: 8px 0; }
.bar-key { font-size: 13px; color: #374151; }
.bar-track { height: 12px; background: #eef2ff; border-radius: 999px; overflow: hidden; }
.bar-fill { height: 100%; background: linear-gradient(90deg, #2563eb, #60a5fa); }
.bar-value { text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #334155; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th, td { border: 1px solid #e5e7eb; padding: 8px; vertical-align: top; }
th { background: #f1f5f9; position: sticky; top: 0; }
.fake { background: #fff7ed; }
.true { background: #ecfeff; }
.small { font-size: 12px; color: #6b7280; }
</style>
</head>
<body>
<h1>AuroraIG 真/假指令可视化</h1>
"""

    cards = (
        "<div class=\"grid\">"
        f"<div class=\"card\"><div class=\"k\">样本对数量</div><div class=\"v\">{summary['num_pairs']}</div></div>"
        f"<div class=\"card\"><div class=\"k\">真指令平均词数</div><div class=\"v\">{summary['avg_true_len']}</div></div>"
        f"<div class=\"card\"><div class=\"k\">假指令平均词数</div><div class=\"v\">{summary['avg_fake_len']}</div></div>"
        f"<div class=\"card\"><div class=\"k\">长度均值差(F-T)</div><div class=\"v\">{summary['avg_len_delta']}</div></div>"
        "</div>"
    )

    dist_panel = (
        "<div class=\"panel\">"
        "<h2>负样本类型分布</h2>"
        + (bars if bars else "<p class=\"small\">无数据</p>")
        + "</div>"
    )

    table_rows: List[str] = []
    for i, r in enumerate(rows[: max(1, max_rows)], start=1):
        table_rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{html.escape(r['negative_type'])}</td>"
            f"<td class=\"true\">{html.escape(r['true_instruction'])}</td>"
            f"<td class=\"fake\">{html.escape(r['fake_instruction'])}</td>"
            f"<td>{html.escape(r['image'])}</td>"
            "</tr>"
        )

    table_panel = (
        "<div class=\"panel\">"
        "<h2>真假指令样例表</h2>"
        f"<p class=\"small\">仅显示前 {min(len(rows), max(1, max_rows))} 条。</p>"
        "<div style=\"max-height: 72vh; overflow: auto;\">"
        "<table>"
        "<thead><tr><th>#</th><th>negative_type</th><th>true_instruction</th><th>fake_instruction</th><th>image</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody>"
        "</table>"
        "</div>"
        "</div>"
    )

    tail = "</body></html>"
    with out_html.open("w", encoding="utf-8") as f:
        f.write(head + cards + dist_panel + table_panel + tail)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_jsonl)
    summary = build_summary(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "summary.json"
    out_csv = out_dir / "pairs.csv"
    out_html = out_dir / "visualization.html"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_csv(rows, out_csv)
    write_html(rows, summary, out_html, args.max_rows)

    print(f"done: summary={out_json} csv={out_csv} html={out_html}")


if __name__ == "__main__":
    main()

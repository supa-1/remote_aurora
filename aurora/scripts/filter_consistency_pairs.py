from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroraig.data.consistency_quality_filter import (
    QualityFilterConfig,
    filter_consistency_rows,
    summarize_filter_result,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter and balance AuroraIG consistency JSONL pairs.")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--dropped_jsonl", default="")
    p.add_argument("--review_jsonl", default="")
    p.add_argument("--review_size", type=int, default=100)
    p.add_argument("--max_action_polarity_flip", type=int, default=1)
    p.add_argument("--max_direction_replacement", type=int, default=1)
    p.add_argument("--max_color_replacement", type=int, default=1)
    p.add_argument("--max_hard_color_negative", type=int, default=1)
    p.add_argument("--max_easy_color_negative", type=int, default=1)
    p.add_argument("--max_neighbor_object_replacement", type=int, default=2)
    p.add_argument("--max_subject_object_swap", type=int, default=1)
    p.add_argument("--max_spatial_replacement", type=int, default=1)
    p.add_argument("--max_other_rewrite", type=int, default=0)
    return p.parse_args()


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: Iterable[Mapping]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    max_per_type = {
        "action_polarity_flip": args.max_action_polarity_flip,
        "direction_replacement": args.max_direction_replacement,
        "color_replacement": args.max_color_replacement,
        "hard_color_negative": args.max_hard_color_negative,
        "easy_color_negative": args.max_easy_color_negative,
        "neighbor_object_replacement": args.max_neighbor_object_replacement,
        "subject_object_swap": args.max_subject_object_swap,
        "spatial_replacement": args.max_spatial_replacement,
        "other_rewrite": args.max_other_rewrite,
    }
    cfg = QualityFilterConfig(max_per_type=max_per_type, review_sample_size=args.review_size)

    rows = _read_jsonl(args.input_jsonl)
    kept, dropped, review = filter_consistency_rows(rows, cfg)
    _write_jsonl(args.output_jsonl, kept)
    if args.dropped_jsonl:
        _write_jsonl(args.dropped_jsonl, dropped)
    if args.review_jsonl:
        _write_jsonl(args.review_jsonl, review)

    print(json.dumps(summarize_filter_result(kept, dropped), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

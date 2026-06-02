#!/usr/bin/env python3
"""校验 Reconvla/AuroraIG 训练 JSON 格式是否满足 train_vla.py 读取要求。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--json_path", required=True)
    p.add_argument("--image_folder", required=True)
    p.add_argument("--target_image_folder", required=True)
    p.add_argument("--expected_obs_dim", type=int, default=15)
    p.add_argument("--expected_action_dim", type=int, default=35)
    p.add_argument("--max_report", type=int, default=20)
    return p.parse_args()


def validate_row(
    idx: int,
    row: Dict,
    image_folder: Path,
    target_folder: Path,
    expected_obs_dim: int,
    expected_action_dim: int,
) -> Tuple[Dict[str, int], List[str]]:
    counters = {
        "missing_keys": 0,
        "bad_conversations": 0,
        "bad_human_obs_dim": 0,
        "bad_action_dim": 0,
        "missing_image": 0,
        "missing_target": 0,
    }
    issues: List[str] = []

    required = {"id", "image", "image_target", "conversations"}
    miss = required - set(row.keys())
    if miss:
        counters["missing_keys"] += 1
        issues.append(f"[{idx}] missing_keys={sorted(miss)}")
        return counters, issues

    conv = row.get("conversations", [])
    if not isinstance(conv, list) or len(conv) < 2:
        counters["bad_conversations"] += 1
        issues.append(f"[{idx}] bad_conversations_len")
        return counters, issues

    if conv[0].get("from") != "human" or conv[1].get("from") != "gpt":
        counters["bad_conversations"] += 1
        issues.append(f"[{idx}] roles={conv[0].get('from')}->{conv[1].get('from')}")

    human_text = str(conv[0].get("value", ""))
    gpt_text = str(conv[1].get("value", ""))

    human_lines = [x.strip() for x in human_text.split("\n") if x.strip()]
    if not human_lines:
        counters["bad_human_obs_dim"] += 1
        issues.append(f"[{idx}] human_empty")
    else:
        obs_dim = len(NUM_RE.findall(human_lines[-1]))
        if obs_dim != expected_obs_dim:
            counters["bad_human_obs_dim"] += 1
            issues.append(f"[{idx}] human_obs_dim={obs_dim} expected={expected_obs_dim}")

    action_dim = len(NUM_RE.findall(gpt_text))
    if action_dim != expected_action_dim:
        counters["bad_action_dim"] += 1
        issues.append(f"[{idx}] action_dim={action_dim} expected={expected_action_dim}")

    image_path = image_folder / str(row["image"])
    target_path = target_folder / str(row["image_target"])
    if not image_path.exists():
        counters["missing_image"] += 1
        issues.append(f"[{idx}] missing_image={image_path}")
    if not target_path.exists():
        counters["missing_target"] += 1
        issues.append(f"[{idx}] missing_target={target_path}")

    return counters, issues


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path)
    image_folder = Path(args.image_folder)
    target_folder = Path(args.target_image_folder)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    summary = {
        "total": len(data),
        "embody_true": 0,
        "missing_keys": 0,
        "bad_conversations": 0,
        "bad_human_obs_dim": 0,
        "bad_action_dim": 0,
        "missing_image": 0,
        "missing_target": 0,
    }
    reports: List[str] = []

    for i, row in enumerate(data):
        if bool(row.get("embody", False)):
            summary["embody_true"] += 1
        counters, issues = validate_row(
            idx=i,
            row=row,
            image_folder=image_folder,
            target_folder=target_folder,
            expected_obs_dim=args.expected_obs_dim,
            expected_action_dim=args.expected_action_dim,
        )
        for k, v in counters.items():
            summary[k] += v
        if issues and len(reports) < args.max_report:
            reports.extend(issues[: max(0, args.max_report - len(reports))])

    print("=== FORMAT CHECK SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("=== SAMPLE ISSUES ===")
    if reports:
        for r in reports:
            print(r)
    else:
        print("none")

    hard_fail = any(
        summary[k] > 0
        for k in [
            "missing_keys",
            "bad_conversations",
            "bad_human_obs_dim",
            "bad_action_dim",
            "missing_image",
            "missing_target",
        ]
    )
    if hard_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroraig.config import RewriterConfig
from auroraig.data.hybrid_rewriter import HybridInstructionRewriter
from auroraig.interfaces.llm_client import resolve_default_llm_client


def _extract_instruction(human_text: str) -> str:
    lines = [x.strip() for x in human_text.split("\n") if x.strip()]
    for line in lines:
        if "<image>" in line:
            continue
        if len(line.split()) >= 3:
            return line
    return ""


def _corrupt_instruction(text: str, ratio: float) -> str:
    words = text.split()
    if len(words) <= 2:
        return text
    ratio = max(0.0, min(0.9, float(ratio)))
    drop_n = max(1, int(len(words) * ratio))
    candidate_indices = list(range(len(words)))
    random.shuffle(candidate_indices)
    drop_set = set(candidate_indices[:drop_n])
    corrupted = [w for idx, w in enumerate(words) if idx not in drop_set]
    return " ".join(corrupted) if corrupted else text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="离线预构建真假指令与文本重建辅助字段")
    p.add_argument("--input_json", required=True, help="原始 Reconvla 训练 JSON")
    p.add_argument("--output_json", required=True, help="输出带辅助字段的训练 JSON")
    p.add_argument("--max_llm_negatives", type=int, default=3 )
    p.add_argument("--consistency_jsonl", default="", help="可选：image-level consistency pairs JSONL；提供后按 true_instruction 聚合成 fake pool")
    p.add_argument("--max_fake_pool", type=int, default=6, help="每条指令保留的假指令池大小")
    p.add_argument("--text_corrupt_ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _instruction_key(text: str) -> str:
    return str(text).strip().lower()


def _load_consistency_pool(path: str, max_fake_pool: int) -> Dict[str, List[Tuple[str, str]]]:
    """Aggregate image-level consistency rows into instruction-level fake pools."""
    pool: Dict[str, OrderedDict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = _instruction_key(row.get("true_instruction", ""))
            fake = str(row.get("fake_instruction", "")).strip()
            if not key or not fake:
                continue
            neg_type = str(row.get("negative_type", "other_rewrite")).strip() or "other_rewrite"
            ordered = pool.setdefault(key, OrderedDict())
            if fake not in ordered and len(ordered) < max(1, int(max_fake_pool)):
                ordered[fake] = neg_type

    out: Dict[str, List[Tuple[str, str]]] = {}
    for k, ordered in pool.items():
        out[k] = [(fake, neg_type) for fake, neg_type in ordered.items()]
    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data: List[Dict] = json.load(f)

    consistency_pool: Dict[str, List[Tuple[str, str]]] = {}
    if args.consistency_jsonl:
        consistency_pool = _load_consistency_pool(args.consistency_jsonl, args.max_fake_pool)

    rewriter = None
    if not consistency_pool:
        cfg = RewriterConfig(
            enable_rule_rewrite=True,
            enable_llm_rewrite=True,
            llm_absolute_lead=True,
            max_rule_negatives=2,
            max_llm_negatives=args.max_llm_negatives,
        )
        rewriter = HybridInstructionRewriter(cfg=cfg, llm_client=resolve_default_llm_client())

    written = 0
    for item in data:
        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            continue

        human_text = conversations[0].get("value", "")
        true_instruction = _extract_instruction(human_text)
        if not true_instruction:
            continue

        pair_pool = consistency_pool.get(_instruction_key(true_instruction), [])
        fake_instruction = ""
        negative_type = "other_rewrite"

        if pair_pool:
            fake_candidates = [x[0] for x in pair_pool]
            type_candidates = [x[1] for x in pair_pool]
            idx = random.randrange(len(fake_candidates))
            fake_instruction = fake_candidates[idx]
            negative_type = type_candidates[idx]
            item["aux_fake_instruction_pool"] = fake_candidates
            item["aux_negative_type_pool"] = type_candidates
        else:
            if rewriter is None:
                cfg = RewriterConfig(
                    enable_rule_rewrite=True,
                    enable_llm_rewrite=True,
                    llm_absolute_lead=True,
                    max_rule_negatives=2,
                    max_llm_negatives=args.max_llm_negatives,
                )
                rewriter = HybridInstructionRewriter(cfg=cfg, llm_client=resolve_default_llm_client())
            rewrite_out = rewriter.rewrite(true_instruction, object_candidates=item.get("object_candidates", []))
            fake_instruction = rewrite_out.negatives[0] if rewrite_out.negatives else ""

        item["aux_true_instruction"] = true_instruction
        item["aux_fake_instruction"] = fake_instruction
        item["aux_negative_type"] = negative_type
        item["aux_text_recon_noisy"] = _corrupt_instruction(true_instruction, args.text_corrupt_ratio)
        item["aux_build_source"] = "consistency_jsonl_precompute" if pair_pool else "offline_llm_precompute"
        written += 1

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"done: wrote {written} aux records to {args.output_json}")


if __name__ == "__main__":
    main()

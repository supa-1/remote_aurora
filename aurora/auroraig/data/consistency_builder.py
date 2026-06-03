from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from auroraig.adapters.reconvla_adapter import ReconvlaJsonAdapter
from auroraig.config import RewriterConfig
from auroraig.data.hybrid_rewriter import HybridInstructionRewriter
from auroraig.data.yolo_neighbor_detector import YOLONeighborDetector
from tqdm import tqdm


def build_consistency_pairs(
    reconvla_json: str,
    output_jsonl: str,
    rewriter: HybridInstructionRewriter,
    cfg: RewriterConfig,
    image_root: Optional[str] = None,
    yolo_detector: Optional[YOLONeighborDetector] = None,
) -> int:
    rows: List[Dict] = []
    negatives_cache: Dict[str, List[Tuple[str, str]]] = {}
    object_candidates_cache: Dict[str, List[str]] = {}
    iterator = ReconvlaJsonAdapter.iter_records(
        reconvla_json,
    )
    for record in tqdm(iterator, desc="building consistency pairs", unit="sample"):
        instruction_key = record.instruction.strip().lower()
        negatives = negatives_cache.get(instruction_key)
        object_candidates = object_candidates_cache.get(instruction_key)
        if negatives is None:
            object_candidates = list(record.object_candidates)
            if yolo_detector is not None and image_root:
                image_abs = Path(image_root) / record.image
                yolo_candidates = yolo_detector.detect_objects(str(image_abs))
                if yolo_candidates:
                    object_candidates = yolo_candidates
            object_candidates_cache[instruction_key] = object_candidates

            rewritten = rewriter.rewrite(
                record.instruction,
                object_candidates=object_candidates,
            )
            negatives = list(rewritten.negatives)
            negative_types = list(getattr(rewritten, "negative_types", []))
            if len(negative_types) != len(negatives):
                negative_types = [
                    _infer_negative_type(
                        record.instruction,
                        fake,
                        cfg,
                        object_candidates=object_candidates,
                    )
                    for fake in negatives
                ]
            negatives = list(zip(negatives, negative_types))
            negatives_cache[instruction_key] = negatives
        elif object_candidates is None:
            object_candidates = list(record.object_candidates)

        for fake_instruction, negative_type in negatives:
            rows.append(
                {
                    "image": record.image,
                    "true_instruction": record.instruction,
                    "fake_instruction": fake_instruction,
                    "action_text": record.action_text,
                    "negative_type": negative_type,
                    "object_candidates": object_candidates,
                }
            )

    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _infer_negative_type(
    true_instruction: str,
    fake_instruction: str,
    cfg: RewriterConfig,
    object_candidates: Optional[List[str]] = None,
) -> str:
    t = true_instruction.strip().lower()
    f = fake_instruction.strip().lower()

    if _has_action_polarity_flip(t, f):
        return "action_polarity_flip"

    if _contains_swap(t, f, cfg.color_swaps):
        return "color_replacement"

    if _contains_swap(t, f, cfg.neighbor_swaps):
        return "spatial_replacement"

    if _contains_swap(t, f, cfg.subject_object_swaps):
        return "subject_object_swap"

    if _has_neighbor_object_replacement(t, f, object_candidates or []):
        return "neighbor_object_replacement"

    if _is_content_simplification(t, f):
        return "content_simplification"

    return "other_rewrite"


def _contains_swap(true_text: str, fake_text: str, swaps: List[str]) -> bool:
    for item in swaps:
        if ":" not in item:
            continue
        src, dst = [x.strip().lower() for x in item.split(":", 1)]
        if src in true_text and dst in fake_text:
            return True
    return False


def _has_action_polarity_flip(true_text: str, fake_text: str) -> bool:
    pairs = [("turn off", "turn on"), ("switch off", "switch on")]
    for a, b in pairs:
        if a in true_text and b in fake_text:
            return True
        if b in true_text and a in fake_text:
            return True
    return False


def _has_neighbor_object_replacement(true_text: str, fake_text: str, object_candidates: List[str]) -> bool:
    def has_phrase(text: str, phrase: str) -> bool:
        token = phrase.strip().lower()
        if not token:
            return False
        pat = r"\b" + r"\s+".join(re.escape(x) for x in token.split()) + r"\b"
        return re.search(pat, text) is not None

    for obj in object_candidates:
        obj = obj.strip().lower()
        if not obj:
            continue
        if has_phrase(fake_text, obj) and not has_phrase(true_text, obj):
            return True
    return False


def _is_content_simplification(true_text: str, fake_text: str) -> bool:
    true_words = [w for w in re.findall(r"[a-z]+", true_text)]
    fake_words = [w for w in re.findall(r"[a-z]+", fake_text)]
    if not true_words or not fake_words:
        return False
    return len(fake_words) + 1 < len(true_words)

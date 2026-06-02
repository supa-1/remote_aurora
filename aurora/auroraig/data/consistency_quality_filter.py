from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_MAX_PER_TYPE = {
    "action_polarity_flip": 1,
    "direction_replacement": 1,
    "hard_color_negative": 1,
    "easy_color_negative": 1,
    "color_replacement": 1,
    "neighbor_object_replacement": 2,
    "subject_object_swap": 1,
    "spatial_replacement": 1,
    "other_rewrite": 0,
}

DEFAULT_CALVIN_OBJECTS = {
    "block",
    "red block",
    "blue block",
    "pink block",
    "drawer",
    "slider",
    "slider inside",
    "sliding cabinet",
    "light switch",
    "switch",
    "button",
    "yellow light",
    "light bulb",
    "table",
}

COLOR_WORDS = {
    "red",
    "blue",
    "green",
    "yellow",
    "pink",
    "purple",
    "orange",
    "black",
    "white",
    "brown",
    "gray",
    "grey",
}

RELATION_WORDS = {
    "left",
    "right",
    "front",
    "behind",
    "under",
    "above",
    "below",
    "near",
    "far",
    "inside",
    "outside",
    "in",
    "on",
}

DIRECTION_WORDS = {
    "left",
    "right",
    "up",
    "down",
}

ACTION_WORDS = {
    "close",
    "grasp",
    "lift",
    "move",
    "open",
    "pick",
    "place",
    "pull",
    "push",
    "put",
    "slide",
    "sweep",
    "switch",
    "turn",
}

PRONOUN_OBJECT_WORDS = {
    "it",
    "its",
    "one",
    "that",
    "them",
    "this",
}


@dataclass
class QualityFilterConfig:
    max_per_type: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_MAX_PER_TYPE))
    allowed_objects: Iterable[str] = field(default_factory=lambda: set(DEFAULT_CALVIN_OBJECTS))
    review_sample_size: int = 100


def filter_consistency_rows(
    rows: Sequence[Mapping],
    cfg: QualityFilterConfig | None = None,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    cfg = cfg or QualityFilterConfig()
    prelim_kept: List[Dict] = []
    dropped: List[Dict] = []

    for idx, row in enumerate(rows):
        item = dict(row)
        item["_source_index"] = idx
        keep, reason = _validate_row(item, cfg)
        if keep:
            item["negative_type"] = _normalize_negative_type(str(item.get("negative_type", "")))
            prelim_kept.append(item)
        else:
            dropped.append(_with_drop_reason(item, reason))

    kept, limit_drops = _dedupe_and_balance(prelim_kept, cfg)
    dropped.extend(limit_drops)
    review = build_review_sample(kept, dropped, cfg.review_sample_size)
    return kept, dropped, review


def build_review_sample(kept: Sequence[Mapping], dropped: Sequence[Mapping], size: int = 100) -> List[Dict]:
    high_risk_types = {"other_rewrite", "neighbor_object_replacement"}
    candidates: List[Dict] = []
    for row in list(kept) + list(dropped):
        item = dict(row)
        ntype = str(item.get("negative_type", "other_rewrite"))
        reason = str(item.get("drop_reason", "kept"))
        if ntype in high_risk_types or reason != "kept":
            item["review_priority"] = _review_priority(ntype, reason)
            item["review_checklist"] = "in_domain;single_atomic_edit;natural_language;actually_false"
            candidates.append(item)

    candidates.sort(key=lambda x: (x.get("review_priority", 99), x.get("_source_index", 0)))
    return candidates[: max(0, int(size))]


def summarize_filter_result(kept: Sequence[Mapping], dropped: Sequence[Mapping]) -> Dict:
    return {
        "kept": len(kept),
        "dropped": len(dropped),
        "kept_by_type": dict(Counter(str(x.get("negative_type", "")) for x in kept)),
        "dropped_by_reason": dict(Counter(str(x.get("drop_reason", "")) for x in dropped)),
    }


def _validate_row(row: Mapping, cfg: QualityFilterConfig) -> Tuple[bool, str]:
    true_text = str(row.get("true_instruction", "")).strip()
    fake_text = str(row.get("fake_instruction", "")).strip()
    if not true_text or not fake_text:
        return False, "missing_instruction"

    if _canonical_text(true_text) == _canonical_text(fake_text):
        return False, "punctuation_or_case_only"

    negative_type = _normalize_negative_type(str(row.get("negative_type", "")))
    if _has_direction_change(true_text, fake_text) and not _has_action_polarity_change(true_text, fake_text):
        negative_type = "direction_replacement"
        if isinstance(row, dict):
            row["negative_type"] = negative_type

    if negative_type == "color_replacement":
        negative_type = _classify_color_negative(true_text, fake_text, row)
        if negative_type == "invalid_color_negative":
            return False, "invalid_color_negative"
        if isinstance(row, dict):
            row["negative_type"] = negative_type

    if negative_type == "other_rewrite" and not bool(row.get("quality_verified", False)):
        return False, "unverified_other_rewrite"

    if _has_invalid_action_object(fake_text):
        return False, "action_object_mismatch"

    if negative_type == "subject_object_swap" and not _subject_object_swap_is_allowed(true_text):
        return False, "unsafe_subject_object_swap"

    if not _has_safe_structure(true_text, fake_text):
        return False, "unsafe_structure"

    if _semantic_edit_count(true_text, fake_text) > 1:
        return False, "multiple_semantic_edits"

    if negative_type == "neighbor_object_replacement" and not _uses_valid_neighbor_candidate(row, cfg):
        return False, "invalid_or_missing_object_candidate"

    return True, "kept"


def _dedupe_and_balance(rows: Sequence[Mapping], cfg: QualityFilterConfig) -> Tuple[List[Dict], List[Dict]]:
    kept: List[Dict] = []
    dropped: List[Dict] = []
    seen_by_true_image: Dict[Tuple[str, str], set] = defaultdict(set)
    counts_by_true_image_type: Dict[Tuple[str, str, str], int] = defaultdict(int)

    for row in rows:
        item = dict(row)
        true_key = _canonical_text(str(item.get("true_instruction", "")))
        image_key = str(item.get("image", "")).strip() or "__no_image__"
        fake_key = _canonical_text(str(item.get("fake_instruction", "")))
        ntype = _normalize_negative_type(str(item.get("negative_type", "")))
        max_n = _max_per_type(ntype, cfg)

        if fake_key in seen_by_true_image[(true_key, image_key)]:
            dropped.append(_with_drop_reason(item, "duplicate_fake_for_true"))
            continue
        if counts_by_true_image_type[(true_key, image_key, ntype)] >= max_n:
            dropped.append(_with_drop_reason(item, "type_limit"))
            continue

        seen_by_true_image[(true_key, image_key)].add(fake_key)
        counts_by_true_image_type[(true_key, image_key, ntype)] += 1
        item["negative_type"] = ntype
        kept.append(item)

    return kept, dropped


def _max_per_type(negative_type: str, cfg: QualityFilterConfig) -> int:
    if negative_type in cfg.max_per_type:
        return int(cfg.max_per_type[negative_type])
    if negative_type in {"hard_color_negative", "easy_color_negative"}:
        return int(cfg.max_per_type.get("color_replacement", cfg.max_per_type.get("other_rewrite", 0)))
    return int(cfg.max_per_type.get("other_rewrite", 0))


def _with_drop_reason(row: Mapping, reason: str) -> Dict:
    out = dict(row)
    out["drop_reason"] = reason
    return out


def _normalize_negative_type(raw: str) -> str:
    text = raw.strip().lower()
    if text.startswith("object_operated") or text.startswith("object_reference"):
        return "neighbor_object_replacement"
    if text == "action_polarity":
        return "action_polarity_flip"
    if text == "color":
        return "color_replacement"
    if text == "subject_object":
        return "subject_object_swap"
    if text == "spatial":
        return "spatial_replacement"
    return text or "other_rewrite"


def _canonical_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]+", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def _semantic_edit_count(true_text: str, fake_text: str) -> int:
    true_tokens = _tokens(true_text)
    fake_tokens = _tokens(fake_text)
    count = 0
    if _has_action_polarity_change(true_text, fake_text):
        count += 1
    object_or_color_changed = (
        _color_signature(true_tokens) != _color_signature(fake_tokens)
        or _content_signature(true_tokens) != _content_signature(fake_tokens)
    )
    if object_or_color_changed:
        count += 1
    if _relation_signature(true_tokens) != _relation_signature(fake_tokens):
        count += 1
    return count


def _changed_spans(true_text: str, fake_text: str) -> List[Tuple[str, int, int, int, int]]:
    import difflib

    true_tokens = _tokens(true_text)
    fake_tokens = _tokens(fake_text)
    matcher = difflib.SequenceMatcher(a=true_tokens, b=fake_tokens)
    return [
        (tag, i1, i2, j1, j2)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]


def _has_safe_structure(true_text: str, fake_text: str) -> bool:
    if _verb_signature(_tokens(true_text)) != _verb_signature(_tokens(fake_text)):
        return False
    return len(_changed_spans(true_text, fake_text)) == 1


def _verb_signature(tokens: Sequence[str]) -> Tuple[str, ...]:
    return tuple(token for token in tokens if token in ACTION_WORDS)


def _subject_object_swap_is_allowed(true_text: str) -> bool:
    tokens = _tokens(true_text)
    if len(tokens) < 6:
        return False
    if any(token in PRONOUN_OBJECT_WORDS for token in tokens):
        return False
    return True


def _classify_color_negative(true_text: str, fake_text: str, row: Mapping) -> str:
    true_tokens = _tokens(true_text)
    fake_tokens = _tokens(fake_text)
    if _color_signature(true_tokens) == _color_signature(fake_tokens):
        return "invalid_color_negative"
    if _content_signature(true_tokens) != _content_signature(fake_tokens):
        return "invalid_color_negative"
    if _relation_signature(true_tokens) != _relation_signature(fake_tokens):
        return "invalid_color_negative"
    if not _has_safe_structure(true_text, fake_text):
        return "invalid_color_negative"

    candidates = [
        str(x).strip().lower()
        for x in row.get("object_candidates", [])
        if str(x).strip()
    ]
    fake_lower = fake_text.lower()
    true_lower = true_text.lower()
    for cand in candidates:
        if not any(color in cand.split() for color in COLOR_WORDS):
            continue
        if _phrase_in_text(cand, fake_lower) and not _phrase_in_text(cand, true_lower):
            return "hard_color_negative"
    return "easy_color_negative"


def _has_action_polarity_change(true_text: str, fake_text: str) -> bool:
    true = _canonical_text(true_text)
    fake = _canonical_text(fake_text)
    pairs = [
        ("turn on", "turn off"),
        ("switch on", "switch off"),
    ]
    return any((a in true and b in fake) or (b in true and a in fake) for a, b in pairs)


def _has_direction_change(true_text: str, fake_text: str) -> bool:
    true_dirs = _direction_signature(_tokens(true_text))
    fake_dirs = _direction_signature(_tokens(fake_text))
    if true_dirs == fake_dirs:
        return False
    pairs = {("left", "right"), ("right", "left"), ("up", "down"), ("down", "up")}
    return len(true_dirs) == len(fake_dirs) == 1 and (true_dirs[0], fake_dirs[0]) in pairs


def _direction_signature(tokens: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for idx, token in enumerate(tokens):
        prev_token = tokens[idx - 1] if idx > 0 else ""
        if token == "up" and prev_token in {"pick"}:
            continue
        if token in DIRECTION_WORDS:
            out.append(token)
    return tuple(out)


def _color_signature(tokens: Sequence[str]) -> Tuple[str, ...]:
    return tuple(x for x in tokens if x in COLOR_WORDS)


def _relation_signature(tokens: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for idx, token in enumerate(tokens):
        prev_token = tokens[idx - 1] if idx > 0 else ""
        if token in {"on", "off"} and prev_token in {"turn", "switch"}:
            continue
        if token in RELATION_WORDS:
            out.append(token)
    return tuple(out)


def _content_signature(tokens: Sequence[str]) -> Tuple[str, ...]:
    stop = {
        "a",
        "an",
        "the",
        "to",
        "from",
        "of",
        "up",
        "down",
        "pick",
        "move",
        "place",
        "put",
        "grasp",
        "sweep",
        "slide",
        "turn",
        "switch",
        "off",
    }
    filtered = [
        token
        for token in tokens
        if token not in stop and token not in COLOR_WORDS and token not in RELATION_WORDS
    ]
    return tuple(filtered)


def _has_invalid_action_object(fake_text: str) -> bool:
    text = _canonical_text(fake_text)
    invalid_pick_objects = {
        "table",
        "floor",
        "slider",
        "switch",
        "light switch",
        "yellow light",
        "light bulb",
        "button",
        "sliding cabinet",
    }
    invalid_turn_objects = {
        "slider",
        "slider inside",
        "drawer",
        "table",
        "block",
        "red block",
        "blue block",
        "pink block",
    }

    for obj in invalid_pick_objects:
        if re.search(rf"\b(pick up|grasp)\s+(the\s+)?{re.escape(obj)}\b", text):
            return True
    for obj in invalid_turn_objects:
        if re.search(rf"\b(turn|switch)\s+(on|off)\s+(the\s+)?{re.escape(obj)}\b", text):
            return True
    return False


def _uses_valid_neighbor_candidate(row: Mapping, cfg: QualityFilterConfig) -> bool:
    true_text = str(row.get("true_instruction", ""))
    fake_text = str(row.get("fake_instruction", ""))
    candidates = [str(x).strip().lower() for x in row.get("object_candidates", []) if str(x).strip()]
    if not candidates:
        candidates = [str(x).strip().lower() for x in cfg.allowed_objects if str(x).strip()]

    allowed = {str(x).strip().lower() for x in cfg.allowed_objects}
    for cand in candidates:
        if _phrase_in_text(cand, fake_text) and not _phrase_in_text(cand, true_text):
            return True
        if not row.get("object_candidates") and cand not in allowed:
            continue
    return False


def _phrase_in_text(phrase: str, text: str) -> bool:
    token = phrase.strip().lower()
    if not token:
        return False
    pat = r"\b" + r"\s+".join(re.escape(x) for x in token.split()) + r"\b"
    return re.search(pat, text.strip().lower()) is not None


def _review_priority(negative_type: str, reason: str) -> int:
    if reason != "kept":
        return 0
    if negative_type == "other_rewrite":
        return 1
    if negative_type == "neighbor_object_replacement":
        return 2
    return 3

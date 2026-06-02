from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Optional, Set, Tuple

from auroraig.config import RewriterConfig
from auroraig.data.schemas import RewriteResult, TypedRewrite
from auroraig.interfaces.llm_client import LLMClient, RewriteRequest


@dataclass
class HybridInstructionRewriter:
    cfg: RewriterConfig
    llm_client: LLMClient

    def rewrite(self, instruction: str, object_candidates: Optional[List[str]] = None) -> RewriteResult:
        llm_candidates: List[TypedRewrite] = []
        rule_candidates: List[TypedRewrite] = []

        if self.cfg.enable_llm_rewrite and self.cfg.max_llm_negatives > 0:
            request = RewriteRequest(
                instruction=instruction,
                n=self.cfg.max_llm_negatives,
                object_candidates=object_candidates or [],
            )
            if hasattr(self.llm_client, "rewrite_instruction_with_types"):
                llm_candidates.extend(self.llm_client.rewrite_instruction_with_types(request))
            else:
                llm_candidates.extend(
                    TypedRewrite(text=x, negative_type="other_rewrite")
                    for x in self.llm_client.rewrite_instruction(request)
                )

        if self.cfg.enable_rule_rewrite and self.cfg.max_rule_negatives > 0:
            rule_candidates.extend(self._rule_rewrite(instruction, object_candidates=object_candidates))

        if self.cfg.llm_absolute_lead:
            candidates = list(llm_candidates)
            if (
                self.cfg.rule_fallback_when_llm_empty
                and not candidates
                and self.cfg.enable_rule_rewrite
            ):
                candidates = list(rule_candidates)
        else:
            candidates = list(rule_candidates) + list(llm_candidates)

        typed_negatives = self._dedupe_and_filter_typed(candidates, instruction)
        negatives = [x.text for x in typed_negatives]
        negative_types = [x.negative_type for x in typed_negatives]
        max_total = self.cfg.max_llm_negatives if self.cfg.llm_absolute_lead else (self.cfg.max_rule_negatives + self.cfg.max_llm_negatives)
        if max_total > 0:
            negatives = negatives[:max_total]
            negative_types = negative_types[:max_total]

        return RewriteResult(instruction=instruction, negatives=negatives, negative_types=negative_types)

    def _rule_rewrite(self, text: str, object_candidates: Optional[List[str]] = None) -> List[TypedRewrite]:
        out: List[TypedRewrite] = []
        # 对象相关替换只使用图像中真实存在的候选对象，避免虚构物体。
        out.extend(
            TypedRewrite(text=x, negative_type="neighbor_object_replacement")
            for x in self._replace_with_neighbor_objects(text, object_candidates or [])
        )
        out.extend(
            TypedRewrite(text=x, negative_type="spatial_replacement")
            for x in self._apply_swaps(text, self._parse_swap_list(self.cfg.neighbor_swaps))
        )
        out.extend(
            TypedRewrite(text=x, negative_type="color_replacement")
            for x in self._apply_swaps(text, self._parse_swap_list(self.cfg.color_swaps))
        )
        return out[: self.cfg.max_rule_negatives]

    def _replace_with_neighbor_objects(self, text: str, object_candidates: List[str]) -> List[str]:
        """将被操作对象替换为图像中邻近实物，生成难负样本。"""
        if not object_candidates:
            return []

        out: List[str] = []
        known_objects = self._collect_known_objects()
        lower_text = text.lower()

        matched_sources = [obj for obj in known_objects if obj.lower() in lower_text]
        if not matched_sources:
            return out

        for src in matched_sources:
            src_phrases = self._source_phrase_candidates(text, src)
            for src_phrase in src_phrases:
                for dst in object_candidates:
                    dst = dst.strip()
                    if not dst:
                        continue
                    if self._is_same_object_label(src_phrase, dst):
                        continue
                    if src_phrase in text:
                        out.append(text.replace(src_phrase, dst, 1))
        return out

    def _source_phrase_candidates(self, text: str, base_obj: str) -> List[str]:
        """为对象词构造可替换短语，优先完整名词短语（如 red block）。"""
        colors = self._collect_color_words()
        base = base_obj.strip()
        if not base:
            return []

        phrases: List[str] = []

        # 1) 英文颜色 + 对象（词边界匹配）
        base_pattern = re.escape(base)
        for color in colors:
            if not re.fullmatch(r"[a-zA-Z]+", color):
                continue
            pat = rf"\b{re.escape(color)}\s+{base_pattern}\b"
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                phrases.append(m.group(0))

        # 2) 兜底：仅当没有匹配到完整短语时，才回退对象本身
        if not phrases:
            if base in text:
                phrases.append(base)
            else:
                # 英文大小写兜底
                for m in re.finditer(rf"\b{base_pattern}\b", text, flags=re.IGNORECASE):
                    phrases.append(m.group(0))

        # 去重并按长度降序，确保优先替换更长短语
        seen = set()
        uniq: List[str] = []
        for p in sorted(phrases, key=len, reverse=True):
            k = p.lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(p)
        return uniq

    def _collect_color_words(self) -> Set[str]:
        words: Set[str] = set()
        for src, dst in self._parse_swap_list(self.cfg.color_swaps):
            words.add(src.strip().lower())
            words.add(dst.strip().lower())
        # 常见补充，避免配置不全时漏检
        words.update({"red", "blue", "green", "yellow", "pink", "黑", "白", "红", "蓝", "绿", "黄", "粉"})
        return words

    @staticmethod
    def _is_same_object_label(src: str, dst: str) -> bool:
        def norm(x: str) -> str:
            aliases = {
                "swich": "switch",
            }
            x = x.strip().lower()
            x = aliases.get(x, x)
            if x.endswith("s") and len(x) > 3:
                x = x[:-1]
            return x

        return norm(src) == norm(dst)

    def _collect_known_objects(self) -> List[str]:
        seen: Set[str] = set()
        objects: List[str] = []
        for src, dst in self._parse_swap_list(self.cfg.subject_object_swaps):
            for token in (src, dst):
                key = token.lower()
                if key not in seen:
                    seen.add(key)
                    objects.append(token)
        return objects

    @staticmethod
    def _parse_swap_list(items: Iterable[str]) -> List[Tuple[str, str]]:
        swaps: List[Tuple[str, str]] = []
        for item in items:
            if ":" not in item:
                continue
            left, right = item.split(":", 1)
            left = left.strip()
            right = right.strip()
            if left and right:
                swaps.append((left, right))
        return swaps

    @staticmethod
    def _apply_swaps(text: str, swaps: List[Tuple[str, str]]) -> List[str]:
        out: List[str] = []
        for src, dst in swaps:
            if src in text:
                out.append(text.replace(src, dst, 1))
        return out

    @staticmethod
    def _dedupe_and_filter(candidates: List[str], origin: str) -> List[str]:
        return [x.text for x in HybridInstructionRewriter._dedupe_and_filter_typed(
            [TypedRewrite(text=cand, negative_type="other_rewrite") for cand in candidates],
            origin,
        )]

    @staticmethod
    def _dedupe_and_filter_typed(candidates: List[TypedRewrite], origin: str) -> List[TypedRewrite]:
        seen: Set[str] = set()
        out: List[TypedRewrite] = []
        for cand in candidates:
            text = cand.text.strip()
            if not text or text == origin:
                continue
            if text in seen:
                continue
            seen.add(text)
            out.append(TypedRewrite(text=text, negative_type=cand.negative_type or "other_rewrite"))
        return out

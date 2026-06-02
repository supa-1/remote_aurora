from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
import re
import sys
from typing import Any, List, Optional, Protocol, Tuple

from auroraig.data.schemas import TypedRewrite


@dataclass
class RewriteRequest:
    instruction: str
    n: int
    object_candidates: Optional[List[str]] = None


class LLMClient(Protocol):
    def rewrite_instruction(self, request: RewriteRequest) -> List[str]:
        ...

    def rewrite_instruction_with_types(self, request: RewriteRequest) -> List[TypedRewrite]:
        ...


class NullLLMClient:
    """空实现：仅在显式禁用 LLM 时使用。"""

    def rewrite_instruction(self, request: RewriteRequest) -> List[str]:
        del request
        return []

    def rewrite_instruction_with_types(self, request: RewriteRequest) -> List[TypedRewrite]:
        del request
        return []


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ReconQwenLLMClient:
    """recon_qwen 风格本地大模型改写器（Transformers generate）。"""

    model_name_or_path: str
    device: str = "auto"
    max_new_tokens: int = 160
    temperature: float = 0.7
    top_p: float = 0.9
    trust_remote_code: bool = True
    model_family: str = "qwen2"
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    offload_buffers: bool = False
    enable_neighbor_rule_filter: bool = False
    enable_non_neighbor_rule_filter: bool = False
    enable_spatial_vocab_filter_only: bool = False
    enable_subject_object_vocab_count_filter_only: bool = False
    log_generation_errors: bool = True
    max_logged_generation_errors: int = 12

    _tokenizer: Optional[Any] = None
    _model: Optional[Any] = None
    _generation_error_count: int = 0

    def enabled(self) -> bool:
        return bool(self.model_name_or_path.strip())

    @classmethod
    def from_env(cls) -> "ReconQwenLLMClient":
        model_name_or_path = os.getenv("AURORAIG_LLM_MODEL_PATH", "").strip()
        if not model_name_or_path:
            model_name_or_path = os.getenv("AURORAIG_LLM_MODEL", "").strip()
        return cls(
            model_name_or_path=model_name_or_path,
            device=os.getenv("AURORAIG_LLM_DEVICE", "auto").strip() or "auto",
            max_new_tokens=int(os.getenv("AURORAIG_LLM_MAX_NEW_TOKENS", "160")),
            temperature=float(os.getenv("AURORAIG_LLM_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("AURORAIG_LLM_TOP_P", "0.9")),
            trust_remote_code=os.getenv("AURORAIG_LLM_TRUST_REMOTE_CODE", "true").lower() != "false",
            model_family=os.getenv("AURORAIG_LLM_FAMILY", "qwen2").strip().lower() or "qwen2",
            load_in_4bit=_env_bool("AURORAIG_LLM_LOAD_IN_4BIT", False),
            bnb_4bit_quant_type=os.getenv("AURORAIG_LLM_BNB_4BIT_QUANT_TYPE", "nf4").strip() or "nf4",
            bnb_4bit_use_double_quant=_env_bool("AURORAIG_LLM_BNB_4BIT_USE_DOUBLE_QUANT", True),
            bnb_4bit_compute_dtype=os.getenv("AURORAIG_LLM_BNB_4BIT_COMPUTE_DTYPE", "float16").strip() or "float16",
            offload_buffers=_env_bool("AURORAIG_LLM_OFFLOAD_BUFFERS", False),
            enable_neighbor_rule_filter=_env_bool("AURORAIG_LLM_ENABLE_NEIGHBOR_RULE_FILTER", False),
            enable_non_neighbor_rule_filter=_env_bool("AURORAIG_LLM_ENABLE_NON_NEIGHBOR_RULE_FILTER", False),
            enable_spatial_vocab_filter_only=_env_bool("AURORAIG_LLM_ENABLE_SPATIAL_VOCAB_FILTER_ONLY", False),
            enable_subject_object_vocab_count_filter_only=_env_bool(
                "AURORAIG_LLM_ENABLE_SUBJECT_OBJECT_VOCAB_COUNT_FILTER_ONLY", False
            ),
        )

    @staticmethod
    def _resolve_torch_dtype(torch_module: Any, dtype_name: str, default: Any) -> Any:
        key = (dtype_name or "").strip().lower()
        mapping = {
            "float16": getattr(torch_module, "float16", default),
            "fp16": getattr(torch_module, "float16", default),
            "bfloat16": getattr(torch_module, "bfloat16", default),
            "bf16": getattr(torch_module, "bfloat16", default),
            "float32": getattr(torch_module, "float32", default),
            "fp32": getattr(torch_module, "float32", default),
        }
        return mapping.get(key, default)

    def _lazy_load(self) -> bool:
        if self._tokenizer is not None and self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._disable_transformers_flash_attention_for_generation()
        except Exception as exc:
            if _env_bool("AURORAIG_LLM_LOG_LOAD_ERRORS", True):
                print(f"[WARN] LLM import failed: {type(exc).__name__}: {exc}")
            return False

        Qwen2ForCausalLM = None
        if self.model_family == "qwen2":
            try:
                from transformers import Qwen2ForCausalLM as _Qwen2ForCausalLM

                Qwen2ForCausalLM = _Qwen2ForCausalLM
            except Exception as exc:
                if _env_bool("AURORAIG_LLM_LOG_LOAD_ERRORS", True):
                    print(f"[WARN] Qwen2ForCausalLM import failed, falling back to AutoModel: {type(exc).__name__}: {exc}")

        # 对齐 recon_qwen.py：优先按 Qwen2ForCausalLM 显式加载。
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )

        model_kwargs = {
            "device_map": self.device,
            "offload_buffers": self.offload_buffers,
            "attn_implementation": os.getenv("AURORAIG_LLM_ATTN_IMPLEMENTATION", "eager").strip() or "eager",
        }

        # 仅在显式开启时启用 4bit；默认路径保持原有行为。
        if self.load_in_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                compute_dtype = self._resolve_torch_dtype(torch, self.bnb_4bit_compute_dtype, torch.float16)
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
            except Exception:
                # 4bit 依赖不可用时，自动回落到常规加载。
                model_kwargs["torch_dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = dtype

        if self.model_family == "qwen2" and Qwen2ForCausalLM is not None:
            try:
                self._model = Qwen2ForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    **model_kwargs,
                )
            except Exception:
                # 当权重或配置不完全匹配时，回退 AutoModel 以保持可用性。
                auto_model_kwargs = dict(model_kwargs)
                auto_model_kwargs["trust_remote_code"] = self.trust_remote_code
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    **auto_model_kwargs,
                )
        else:
            auto_model_kwargs = dict(model_kwargs)
            auto_model_kwargs["trust_remote_code"] = self.trust_remote_code
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                **auto_model_kwargs,
            )
        if self._model is None:
            return False
        self._model.eval()
        return True

    @staticmethod
    def _disable_transformers_flash_attention_for_generation() -> None:
        """Force Qwen generation to avoid broken flash-attn imports.

        The ARM aurora environment can have flash-attn installed without triton.
        Transformers then tries to import Qwen3's flash-attn helpers and fails
        before the model can fall back to eager attention. Generation does not
        need flash-attn, so disable the cached availability flags before
        AutoModel resolves the Qwen3 modeling module.
        """
        try:
            import transformers.utils.import_utils as import_utils
        except Exception:
            return

        for name in (
            "_flash_attn_2_available",
            "_flash_attn_available",
            "_flash_attn_3_available",
        ):
            if hasattr(import_utils, name):
                try:
                    setattr(import_utils, name, False)
                except Exception:
                    pass

        for module_name in list(sys.modules):
            if module_name.startswith("flash_attn"):
                sys.modules.pop(module_name, None)

    def rewrite_instruction(self, request: RewriteRequest) -> List[str]:
        return [x.text for x in self.rewrite_instruction_with_types(request)]

    def rewrite_instruction_with_types(self, request: RewriteRequest) -> List[TypedRewrite]:
        if not self.enabled() or request.n <= 0:
            return []

        if not self._lazy_load():
            return []

        assert self._tokenizer is not None
        assert self._model is not None

        typed_prompts = self._build_rule_prompts_with_types(request)
        filtered_candidates = [
            x.strip().lower()
            for x in (request.object_candidates or [])
            if x.strip() and not re.search(r"\bgripper\b", x, flags=re.IGNORECASE)
        ]

        lines: List[TypedRewrite] = []
        for rule_type, prompt in typed_prompts:
            accepted_for_prompt = False
            try:
                gen_text = self._generate_text(prompt)
            except Exception as e:
                self._generation_error_count += 1
                if self.log_generation_errors and self._generation_error_count <= self.max_logged_generation_errors:
                    print(
                        "[WARN] LLM generation failed "
                        f"(#{self._generation_error_count}, rule={rule_type}): {type(e).__name__}: {e}"
                    )
                gen_text = ""
            for cand in self._extract_rewrite_lines(gen_text):
                skip_markers = {"SKIP", "<SKIP>", "__SKIP__"}
                if cand.strip().upper() in skip_markers:
                    # Prompt-level skip signal: do not generate fallback outside prompt.
                    accepted_for_prompt = True
                    continue
                normalized = self._normalize_rewrite(cand, request.instruction)
                if not normalized:
                    continue
                if not self._is_valid_for_rule(rule_type, request.instruction, normalized, filtered_candidates):
                    continue
                lines.append(
                    TypedRewrite(
                        text=normalized,
                        negative_type=self._negative_type_from_rule_type(rule_type),
                    )
                )
                accepted_for_prompt = True

            if not accepted_for_prompt:
                base_rule, _ = self._split_rule_type(rule_type)
                # For these three categories, rely on prompt-only behavior.
                if base_rule in {"color", "subject_object", "spatial"}:
                    continue
                fallback = self._fallback_rewrite_for_rule(rule_type, request.instruction)
                fallback = self._normalize_rewrite(fallback, request.instruction) if fallback else ""
                if fallback and self._is_valid_for_rule(rule_type, request.instruction, fallback, filtered_candidates):
                    lines.append(
                        TypedRewrite(
                            text=fallback,
                            negative_type=self._negative_type_from_rule_type(rule_type),
                        )
                    )

        return dedupe_typed_rewrites(lines, request.instruction)[: request.n]

    def _build_rule_prompts(self, request: RewriteRequest) -> List[str]:
        return [prompt for _, prompt in self._build_rule_prompts_with_types(request)]

    def _build_rule_prompts_with_types(self, request: RewriteRequest) -> List[Tuple[str, str]]:
        instruction = request.instruction.strip()
        object_candidates = [
            x.strip()
            for x in (request.object_candidates or [])
            if x.strip() and not re.search(r"\bgripper\b", x, flags=re.IGNORECASE)
        ][:5]
        parsed = self._extract_triplet(instruction)
        src_obj1 = parsed[0] if parsed else ""
        src_obj2 = parsed[2] if parsed else ""
        turn_on_objs = self._extract_turn_on_subject_object(instruction)
        source_objects = {
            x.strip().lower()
            for x in (src_obj1, src_obj2, *(turn_on_objs or ("", "")))
            if x and x.strip()
        }

        common_constraints = (
            "Do not add, remove, or explain anything. "
            "Do not output prefixes like Output:, Rewritten:, numbering, or parentheses. "
            "Keep tense and sentence pattern unchanged. "
            "Perform exactly one atomic edit only. "
            "Keep all other tokens in the exact same order and positions. "
            "Do not move words, and do not insert or delete words. "
            "Keep phrasal verb order unchanged (e.g., 'slide up', never 'up slide'). "
            "If these constraints cannot be satisfied exactly, output SKIP."
        )

        prompts: List[Tuple[str, str]] = []

        prompts.append(
            (
                "action_polarity",
                "Task: action polarity rewrite. "
                "If the sentence does not contain turn on, turn off, switch on, or switch off, output exactly SKIP. "
                "Otherwise flip only the action polarity between on and off while keeping every other word unchanged. "
                f"Sentence: {instruction}. {common_constraints} Output exactly one rewritten sentence or SKIP.",
            )
        )

        prompts.append(
            (
                "color",
                "Task: color rewrite. "
                "If the sentence contains no color word, output exactly SKIP. "
                "Example: input 'turn off the light bulb' -> output 'SKIP'. "
                "Otherwise replace only color words and keep all non-color words unchanged in original order. "
                f"Sentence: {instruction}. {common_constraints} Output exactly one rewritten sentence or SKIP.",
            )
        )

        prompts.append(
            (
                "subject_object",
                "Task: subject-object swap. "
                "If the sentence does not contain at least two distinct manipulable objects, output exactly SKIP. "
                "If the sentence is shorter than six words, output exactly SKIP. "
                "If any object is referred to by a pronoun such as it, this, that, or them, output exactly SKIP. "
                "Example: input 'turn off the light bulb' -> output 'SKIP' as the sentence does not contain two distinct objects. "
                "Otherwise swap only the operated object and the reference object phrases. "
                "Treat each object phrase as an atomic unit (keep its modifiers attached), e.g., 'red block' is one unit. "
                "Don't treat verb or verb phrase as an object. Such as slide down, pick up and so on are not objects. "
                "Don't consider spatial relation words or phrases as an object. Such as left, right and so on are not objects. "
                "When swapping, move the full phrase together; never split color/modifier from its noun. "
                "Do not change action verb, tense, relation phrase, or word order outside the two object phrases. "
                f"Sentence: {instruction}. {common_constraints} Output exactly one rewritten sentence or SKIP.",
            )
        )

        prompts.append(
            (
                "spatial",
                "Task: spatial rewrite. "
                "If the sentence has no explicit spatial relation word/phrase (e.g., left/right/front/behind/on/under/in/inside/outside/near/far), output exactly SKIP. "
                "Example: input 'pick up the red block' -> output 'SKIP' as the sentence does not contain an explicit spatial relation. "
                "Otherwise invert only the spatial relation phrase while keeping both objects unchanged and in the same order. "
                "Replace the spatial phrase in-place only; never move it to a new position. "
                "Example: 'sweep the pink block to the right' -> 'sweep the pink block to the left'. "
                "Invalid: 'sweep the right pink block to the left' and 'sweep the right pink block to the right.'"
                f"Sentence: {instruction}. {common_constraints} Output exactly one rewritten sentence or SKIP.",
            )
        )

        if object_candidates:
            for dst in object_candidates:
                dst_norm = dst.strip().lower()
                if not dst_norm:
                    continue
                if dst_norm in source_objects:
                    continue
                if any(self._is_same_object_phrase(dst_norm, src) for src in source_objects):
                    continue
                if self._phrase_in_text(dst, instruction):
                    continue
                prompts.append(
                    (
                        f"object_operated::{dst}",
                        "Replace only the operated object in the sentence. "
                        f"The new operated object must be exactly: {dst}. Sentence: {instruction}. "
                        "Operate on the full noun phrase as one atomic unit (color/modifier + noun stay together), "
                        "for example 'red block' is a single unit and cannot be partially changed. "
                        "If the true object is generic while candidate is a specific variant of the same base object, SKIP. "
                        "If candidate is generic while the true object is a specific variant of the same base object, SKIP. "
                        "Mutual containment counts as same base object and must be skipped (examples: block <-> blue block, slider <-> slider inside). "
                        "Do not perform replacement when either side textually contains the other after removing articles/colors/modifiers. "
                        "The replaced object phrase must be exactly the target candidate text, with no extra adjectives. "
                        "Do not carry over source color/modifier words to the target object unless those words are already in the target candidate. "
                        "Example: 'drawer' is one of the candidates without color/modifiers, 'red block' is the target word with color/modifiers to replace;'red block' -> 'drawer' is valid;'red block' ->'red drawer' is invalid. "
                        "Only replace the operated object without changing the sentence structure. "
                        "Don't add any new words. "
                        f"Keep sentence structure unchanged. {common_constraints} Output one rewritten sentence only or SKIP.",
                    )
                )
            for dst in object_candidates:
                dst_norm = dst.strip().lower()
                if not dst_norm:
                    continue
                if dst_norm in source_objects:
                    continue
                if any(self._is_same_object_phrase(dst_norm, src) for src in source_objects):
                    continue
                if self._phrase_in_text(dst, instruction):
                    continue
                prompts.append(
                    (
                        f"object_reference::{dst}",
                        "Replace only the reference object in the sentence. "
                        f"The new reference object must be exactly: {dst}. Sentence: {instruction}. "
                        "Operate on the full noun phrase as one atomic unit (color/modifier + noun stay together), "
                        "for example 'red block' is a single unit and cannot be partially changed. "
                        "If the true reference object is generic while candidate is a specific variant of the same base object, SKIP. "
                        "If candidate is generic while the true reference object is a specific variant of the same base object, SKIP. "
                        "Mutual containment counts as same base object and must be skipped (examples: block <-> blue block, slider <-> slider inside). "
                        "Do not perform replacement when either side textually contains the other after removing articles/colors/modifiers. "
                        "The replaced object phrase must be exactly the target candidate text, with no extra adjectives. "
                        "Do not carry over source color/modifier words to the target object unless those words are already in the target candidate. "
                        "Example: 'drawer' is one of the candidates without color/modifiers, 'red block' is the target word with color/modifiers to replace;'red block' -> 'drawer' is valid;'red block' ->'red drawer' is invalid. "
                        "Only replace the reference object without changing the sentence structure. "
                        "Don't add any new words. "
                        f"{common_constraints} Output one rewritten sentence only or SKIP.",
                    )
                )
        else:
            # 不提供候选时，不再退化为规则改写，只保留前述通用指令提示。
            pass
        return prompts

    @staticmethod
    def _negative_type_from_rule_type(rule_type: str) -> str:
        base_rule, _ = ReconQwenLLMClient._split_rule_type(rule_type)
        mapping = {
            "action_polarity": "action_polarity_flip",
            "color": "color_replacement",
            "subject_object": "subject_object_swap",
            "spatial": "spatial_replacement",
            "object_operated": "neighbor_object_replacement",
            "object_reference": "neighbor_object_replacement",
        }
        return mapping.get(base_rule, "other_rewrite")

    def _generate_text(self, prompt: str) -> str:
        assert self._tokenizer is not None
        assert self._model is not None

        import torch

        if hasattr(self._tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            encoded = self._tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"]

        input_ids = input_ids.to(self._model.device)
        attention_mask = torch.ones_like(input_ids)
        do_sample = self.temperature > 0
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max(16, self.max_new_tokens),
            "do_sample": do_sample,
            "eos_token_id": getattr(self._tokenizer, "eos_token_id", None),
            "pad_token_id": getattr(self._tokenizer, "pad_token_id", None) or getattr(self._tokenizer, "eos_token_id", None),
        }

        if do_sample:
            gen_kwargs["temperature"] = max(0.0, self.temperature)
            gen_kwargs["top_p"] = min(max(self.top_p, 0.0), 1.0)
        else:
            # Deterministic path: explicitly disable sampling-related fields
            # to avoid warnings from model-level generation config defaults.
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None

        with torch.inference_mode():
            outputs = self._model.generate(**gen_kwargs)
        gen_ids = outputs[:, input_ids.shape[-1]:]
        return self._tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

    @staticmethod
    def _extract_rewrite_lines(gen_text: str) -> List[str]:
        lines = [
            line.strip().lstrip("- ").strip()
            for line in gen_text.splitlines()
            if line.strip()
        ]
        out: List[str] = []
        for line in lines:
            # 兜底清理：去掉常见前缀，尽量保留纯句子。
            line = re.sub(r"^(output\s*:\s*|rewritten\s*:\s*)", "", line, flags=re.IGNORECASE).strip()
            if line:
                out.append(line)
        return out

    @staticmethod
    def _normalize_rewrite(candidate: str, origin: str) -> str:
        line = candidate.strip().strip("\"' ")
        line = re.sub(r"\s*\(replaced with:.*\)\s*$", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"\s+([,.!?;:])", r"\1", line)
        if not line:
            return ""

        # 将首字母大小写对齐到原句风格，避免出现无意义的首字母波动。
        src_alpha = re.search(r"[A-Za-z]", origin)
        dst_alpha = re.search(r"[A-Za-z]", line)
        if src_alpha and dst_alpha:
            src_idx = src_alpha.start()
            dst_idx = dst_alpha.start()
            src_c = origin[src_idx]
            dst_c = line[dst_idx]
            if src_c.islower() and dst_c.isupper():
                line = line[:dst_idx] + dst_c.lower() + line[dst_idx + 1 :]
            elif src_c.isupper() and dst_c.islower():
                line = line[:dst_idx] + dst_c.upper() + line[dst_idx + 1 :]
        return line

    @staticmethod
    def _extract_triplet(text: str) -> Optional[Tuple[str, str, str]]:
        normalized = re.sub(r"\s+", " ", text.strip().lower()).strip(" .!?")
        rels = [
            "in front of",
            "left of",
            "right of",
            "inside",
            "outside",
            "under",
            "above",
            "below",
            "behind",
            "near",
            "over",
            "into",
            "onto",
            " on ",
            " in ",
        ]
        for rel in rels:
            rel_token = rel.strip()
            needle = rel if rel.startswith(" ") else f" {rel_token} "
            if needle not in normalized:
                continue
            left, right = normalized.split(needle, 1)

            # Avoid false spatial parsing on phrasal verbs, e.g. "turn on the light".
            # In such cases, "on" is a verb particle, not a geometric relation.
            if rel_token in {"on", "in"} and re.search(r"\b(turn|switch|power)\s*$", left):
                continue

            left_match = re.search(r"\bthe\s+(.+)$", left)
            right_match = re.match(r"(.+?)$", right)
            if not left_match or not right_match:
                continue
            obj1 = re.sub(r"^(the|a|an)\s+", "", left_match.group(1).strip())
            obj2 = re.sub(r"^(the|a|an)\s+", "", right_match.group(1).strip())
            if obj1 and obj2:
                return obj1, rel_token, obj2
        return None

    def _is_valid_spatial_rewrite(self, origin: str, candidate: str) -> bool:
        src = self._extract_triplet(origin)
        dst = self._extract_triplet(candidate)
        if src is None or dst is None:
            return False
        src_obj1, src_rel, src_obj2 = src
        dst_obj1, dst_rel, dst_obj2 = dst
        return src_obj1 == dst_obj1 and src_obj2 == dst_obj2 and src_rel != dst_rel

    def _is_valid_for_rule(
        self,
        rule_type: str,
        origin: str,
        candidate: str,
        object_candidates: List[str],
    ) -> bool:
        base_rule, rule_arg = self._split_rule_type(rule_type)
        src = self._extract_triplet(origin)
        dst = self._extract_triplet(candidate)

        if base_rule == "spatial":
            # Allow enabling only spatial vocab-set filter while keeping
            # other non-neighbor filters disabled.
            if self.enable_spatial_vocab_filter_only:
                return self._is_valid_spatial_rewrite(origin, candidate)
            if not self.enable_non_neighbor_rule_filter:
                return True
            return self._is_valid_spatial_rewrite(origin, candidate)

        if base_rule in {"object_operated", "object_reference"}:
            if not self.enable_neighbor_rule_filter:
                return True

        if base_rule == "subject_object":
            # Optional statistical filter: subject_object rewrite must preserve
            # token types and token counts exactly.
            if self.enable_subject_object_vocab_count_filter_only:
                origin_counts = Counter(re.findall(r"[a-z]+", origin.lower()))
                cand_counts = Counter(re.findall(r"[a-z]+", candidate.lower()))
                if origin_counts != cand_counts:
                    return False
                # In subject_object-only mode, stop after this statistical check.
                if not self.enable_non_neighbor_rule_filter:
                    return True

            if src is not None and dst is not None:
                src_obj1, src_rel, src_obj2 = src
                dst_obj1, dst_rel, dst_obj2 = dst
                if dst_obj1 == src_obj2 and dst_obj2 == src_obj1 and dst_rel == src_rel:
                    return True

            # Special-case imperative causative pattern:
            # "move the A to turn on the B" -> "move the B to turn on the A"
            src_turn_on = self._extract_turn_on_subject_object(origin)
            dst_turn_on = self._extract_turn_on_subject_object(candidate)
            if src_turn_on is None or dst_turn_on is None:
                return False
            src_obj1, src_obj2 = src_turn_on
            dst_obj1, dst_obj2 = dst_turn_on
            return dst_obj1 == src_obj2 and dst_obj2 == src_obj1

        if base_rule == "color":
            if not self.enable_non_neighbor_rule_filter:
                return True

        if base_rule == "object_operated":
            # User-requested behavior: do not run structural validation for
            # neighbor-object replacement prompts.
            return True

        if base_rule == "object_reference":
            # User-requested behavior: do not run structural validation for
            # neighbor-object replacement prompts.
            return True

        if base_rule == "color":
            if _canonical_text(candidate) == _canonical_text(origin):
                return False
            return self._has_color_change(origin, candidate)

        return True

    @staticmethod
    def _split_rule_type(rule_type: str) -> Tuple[str, str]:
        if "::" not in rule_type:
            return rule_type, ""
        base, arg = rule_type.split("::", 1)
        return base.strip(), arg.strip().lower()

    def _fallback_rewrite_for_rule(self, rule_type: str, origin: str) -> str:
        base_rule, rule_arg = self._split_rule_type(rule_type)
        src = self._extract_triplet(origin)
        if src is None:
            if base_rule == "subject_object":
                return self._fallback_swap_turn_on_subject_object(origin)
            if base_rule == "color":
                return self._fallback_color_rewrite(origin)
            return ""
        src_obj1, src_rel, src_obj2 = src

        if base_rule == "color":
            color_try = self._fallback_color_rewrite(origin)
            if color_try:
                return color_try

        if base_rule == "object_operated" and rule_arg:
            return self._replace_object_token(origin, src_obj1, rule_arg)
        if base_rule == "object_reference" and rule_arg:
            return self._replace_object_token(origin, src_obj2, rule_arg)
        if base_rule == "subject_object":
            return self._fallback_swap_subject_object(origin, src_obj1, src_obj2)
        if base_rule == "spatial":
            return self._fallback_spatial_rewrite(origin, src_rel)
        return ""

    @staticmethod
    def _replace_object_token(text: str, src_obj: str, dst_obj: str) -> str:
        pattern = rf"\b(the\s+)?{re.escape(src_obj)}\b"
        repl = f"the {dst_obj}" if re.search(pattern, text, flags=re.IGNORECASE) else dst_obj
        return re.sub(pattern, repl, text, count=1, flags=re.IGNORECASE)

    @staticmethod
    def _fallback_swap_subject_object(text: str, src_obj1: str, src_obj2: str) -> str:
        tmp = "__AURORAIG_TMP_OBJ__"
        out = ReconQwenLLMClient._replace_object_token(text, src_obj1, tmp)
        out = ReconQwenLLMClient._replace_object_token(out, src_obj2, src_obj1)
        out = re.sub(rf"\b(the\s+)?{re.escape(tmp)}\b", f"the {src_obj2}", out, count=1, flags=re.IGNORECASE)
        return out

    @staticmethod
    def _extract_turn_on_subject_object(text: str) -> Optional[Tuple[str, str]]:
        normalized = re.sub(r"\s+", " ", text.strip().lower()).strip(" .!?")
        m = re.search(r"\bthe\s+(.+?)\s+to\s+turn\s+on\s+the\s+(.+)$", normalized)
        if not m:
            return None
        obj1 = m.group(1).strip()
        obj2 = m.group(2).strip()
        if not obj1 or not obj2 or obj1 == obj2:
            return None
        return obj1, obj2

    @staticmethod
    def _fallback_swap_turn_on_subject_object(text: str) -> str:
        pattern = re.compile(
            r"^(?P<head>.*?\bthe\s+)(?P<obj1>.+?)(?P<link>\s+to\s+turn\s+on\s+the\s+)(?P<obj2>.+?)(?P<tail>\s*[.!?]?\s*)$",
            flags=re.IGNORECASE,
        )
        m = pattern.match(text.strip())
        if not m:
            return ""
        obj1 = m.group("obj1").strip()
        obj2 = m.group("obj2").strip()
        if not obj1 or not obj2 or obj1.lower() == obj2.lower():
            return ""
        return f"{m.group('head')}{obj2}{m.group('link')}{obj1}{m.group('tail')}".strip()

    @staticmethod
    def _phrase_in_text(phrase: str, text: str) -> bool:
        token = phrase.strip().lower()
        if not token:
            return False
        pat = r"\\b" + r"\\s+".join(re.escape(x) for x in token.split()) + r"\\b"
        return re.search(pat, text.strip().lower()) is not None

    @staticmethod
    def _is_same_object_phrase(a: str, b: str) -> bool:
        def norm_tokens(x: str) -> List[str]:
            words = re.findall(r"[a-z]+", x.lower())
            stop = {"the", "a", "an"}
            out: List[str] = []
            for w in words:
                if w in stop:
                    continue
                if w.endswith("s") and len(w) > 3:
                    w = w[:-1]
                out.append(w)
            return out

        ta = norm_tokens(a)
        tb = norm_tokens(b)
        if not ta or not tb:
            return False
        if ta == tb:
            return True

        sa = set(ta)
        sb = set(tb)
        # Treat "light switch" and "switch" as too-similar to replace each other.
        if sa.issubset(sb) or sb.issubset(sa):
            return True
        return False

    @staticmethod
    def _fallback_spatial_rewrite(text: str, src_rel: str) -> str:
        rel_map = {
            "left of": "right of",
            "right of": "left of",
            "in front of": "behind",
            "behind": "in front of",
            "under": "on",
            "on": "under",
            "in": "outside",
            "inside": "outside",
            "outside": "inside",
            "near": "far from",
            "far from": "near",
            "over": "under",
            "below": "above",
            "above": "below",
            "into": "out of",
            "onto": "off",
        }
        dst_rel = rel_map.get(src_rel)
        if not dst_rel:
            return ""
        pattern = rf"\b{re.escape(src_rel)}\b"
        return re.sub(pattern, dst_rel, text, count=1, flags=re.IGNORECASE)

    @staticmethod
    def _fallback_color_rewrite(text: str) -> str:
        color_map = {
            "red": "blue",
            "blue": "red",
            "green": "yellow",
            "yellow": "green",
            "pink": "red",
            "purple": "blue",
            "black": "white",
            "white": "black",
            "brown": "gray",
            "gray": "brown",
            "grey": "brown",
        }
        for src, dst in color_map.items():
            pattern = rf"\b{src}\b"
            if re.search(pattern, text, flags=re.IGNORECASE):
                return re.sub(pattern, dst, text, count=1, flags=re.IGNORECASE)
        return ""

    @staticmethod
    def _contains_color_words(text: str) -> bool:
        color_words = {
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
        words = re.findall(r"[a-z]+", text.lower())
        return any(w in color_words for w in words)

    @staticmethod
    def _has_color_change(origin: str, candidate: str) -> bool:
        color_words = {
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
        origin_colors = [w for w in re.findall(r"[a-z]+", origin.lower()) if w in color_words]
        cand_colors = [w for w in re.findall(r"[a-z]+", candidate.lower()) if w in color_words]
        if not origin_colors and not cand_colors:
            return False
        return origin_colors != cand_colors


def dedupe_rewrites(candidates: List[str], origin: str) -> List[str]:
    seen = set()
    out: List[str] = []
    origin_key = _canonical_text(origin)
    for cand in candidates:
        c = cand.strip()
        if not c:
            continue
        key = _canonical_text(c)
        if not key:
            continue
        if key == origin_key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def dedupe_typed_rewrites(candidates: List[TypedRewrite], origin: str) -> List[TypedRewrite]:
    seen = set()
    out: List[TypedRewrite] = []
    origin_key = _canonical_text(origin)
    for cand in candidates:
        text = cand.text.strip()
        if not text:
            continue
        key = _canonical_text(text)
        if not key:
            continue
        if key == origin_key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(TypedRewrite(text=text, negative_type=cand.negative_type or "other_rewrite"))
    return out


def _canonical_text(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = t.strip(" .!?,;:\"'")
    return t


def resolve_default_llm_client() -> LLMClient:
    """只允许 HuggingFace 大模型客户端；未配置模型路径时直接报错。"""
    provider = os.getenv("AURORAIG_LLM_PROVIDER", "recon_qwen").strip().lower()
    if provider == "null":
        return NullLLMClient()

    recon_client = ReconQwenLLMClient.from_env()
    if recon_client.enabled():
        return recon_client
    raise RuntimeError(
        "LLM-only mode requires HuggingFace model path. "
        "Please set AURORAIG_LLM_MODEL_PATH (or AURORAIG_LLM_MODEL)."
    )

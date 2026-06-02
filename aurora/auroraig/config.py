from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class RewriterConfig:
    enable_rule_rewrite: bool = True
    enable_llm_rewrite: bool = True
    # LLM 绝对主导：默认只取 LLM 结果，规则仅在 LLM 为空时兜底。
    llm_absolute_lead: bool = True
    rule_fallback_when_llm_empty: bool = True
    max_rule_negatives: int = 2
    max_llm_negatives: int = 6

    # 规则重写控制：主语/宾语、邻近对象、颜色替换
    subject_object_swaps: List[str] = field(default_factory=lambda: [
        "apple:banana",
        "cup:bottle",
        "block:cube",
        "mug:bowl",
        "switch:button",
        "苹果:香蕉",
        "杯子:瓶子",
        "方块:积木",
    ])
    neighbor_swaps: List[str] = field(default_factory=lambda: [
        "left:right",
        "front:back",
        "near:far",
        "左:右",
        "前:后",
        "近处:远处",
    ])
    color_swaps: List[str] = field(default_factory=lambda: [
        "red:blue",
        "blue:red",
        "red:pink",
        "pink:red",
        "blue:pink",
        "pink:blue",
        "green:yellow",
        "yellow:green",
        "红:蓝",
        "蓝:红",
        "绿:黄",
        "黄:绿",
    ])


@dataclass
class ContrastiveConfig:
    encoder_name: str = "openai/clip-vit-base-patch32"
    image_size: int = 224
    train_batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    epochs: int = 3
    temperature: float = 0.07
    max_text_length: int = 64


@dataclass
class ConsistencyConfig:
    encoder_name: str = "openai/clip-vit-base-patch32"
    train_batch_size: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    epochs: int = 3
    max_text_length: int = 128
    margin: float = 0.2
    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3


@dataclass
class ProjectConfig:
    seed: int = 42
    rewriter: RewriterConfig = field(default_factory=RewriterConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)
    consistency: ConsistencyConfig = field(default_factory=ConsistencyConfig)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ReconvlaRecord:
    image: str
    image_target: str
    instruction: str
    action_text: str
    object_candidates: List[str]


@dataclass
class ContrastivePair:
    image: str
    instruction: str
    label: int
    source: str


@dataclass
class RewriteResult:
    instruction: str
    negatives: List[str]
    negative_types: List[str] = field(default_factory=list)


@dataclass
class TypedRewrite:
    text: str
    negative_type: str

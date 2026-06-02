from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from auroraig.data.schemas import ReconvlaRecord
from auroraig.data.yolo_neighbor_detector import YOLONeighborDetector


class ReconvlaJsonAdapter:
    """读取 Reconvla 训练 JSON，抽取图像-指令监督样本。"""

    @staticmethod
    def iter_records(
        json_path: str,
        image_root: Optional[str] = None,
        yolo_detector: Optional[YOLONeighborDetector] = None,
    ) -> Iterator[ReconvlaRecord]:
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for row in data:
            image = row.get("image", "")
            image_target = row.get("image_target", "")
            convs = row.get("conversations", [])
            if len(convs) < 2:
                continue

            human = convs[0].get("value", "")
            gpt = convs[1].get("value", "")
            instruction = _extract_instruction(human)
            if not instruction:
                continue

            object_candidates = _extract_object_candidates(row)
            if yolo_detector is not None and image_root:
                image_abs = Path(image_root) / image
                yolo_candidates = yolo_detector.detect_objects(str(image_abs))
                if yolo_candidates:
                    object_candidates = yolo_candidates

            yield ReconvlaRecord(
                image=image,
                image_target=image_target,
                instruction=instruction,
                action_text=gpt,
                object_candidates=object_candidates,
            )


def _extract_instruction(human_text: str) -> str:
    lines = [x.strip() for x in human_text.split("\n") if x.strip()]
    for line in lines:
        if "<image>" in line:
            continue
        return line
    return ""


def _extract_object_candidates(row: Dict[str, Any]) -> List[str]:
    """从样本中提取可用于邻近物体替换的候选对象。"""
    keys = [
        "neighbor_objects",
        "nearby_objects",
        "objects",
        "detected_objects",
        "scene_objects",
        "yolo_objects",
    ]

    candidates: List[str] = []
    for key in keys:
        value = row.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            name = _to_object_name(item)
            if name:
                candidates.append(name)

    seen = set()
    deduped: List[str] = []
    for x in candidates:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(x)
    return deduped


def _to_object_name(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("name", "label", "category", "class"):
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""

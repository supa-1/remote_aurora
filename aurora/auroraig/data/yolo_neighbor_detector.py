from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple


@dataclass
class YOLONeighborDetector:
    model_path: str
    conf: float = 0.25
    device: str = "0"

    _model: Optional[Any] = None

    def enabled(self) -> bool:
        return bool(self.model_path) and Path(self.model_path).exists()

    def _lazy_load(self) -> bool:
        if self._model is not None:
            return True
        if not self.enabled():
            return False
        try:
            from ultralytics import YOLO
        except Exception:
            return False
        self._model = YOLO(self.model_path)
        return True

    def detect_objects(self, image_path: str) -> List[str]:
        if not self._lazy_load():
            return []
        p = Path(image_path)
        if not p.exists():
            return []

        try:
            model = self._model
            if model is None:
                return []
            result = model.predict(
                source=str(p),
                conf=float(self.conf),
                verbose=False,
                device=str(self.device),
            )[0]
        except Exception:
            return []

        if result.boxes is None or len(result.boxes) == 0:
            return []

        names = result.names
        cls_ids = result.boxes.cls.tolist()
        confs = result.boxes.conf.tolist() if hasattr(result.boxes, "conf") else [1.0] * len(cls_ids)
        labeled_scores: List[Tuple[str, float]] = []
        for cls_id, score in zip(cls_ids, confs):
            label = str(names[int(cls_id)]).strip()
            labeled_scores.append((label, float(score)))
        return _normalize_and_filter_with_scores(labeled_scores, min_conf=float(self.conf))


def _normalize_and_filter(labels: List[str]) -> List[str]:
    return _normalize_and_filter_with_scores([(x, 1.0) for x in labels], min_conf=0.0)


def _normalize_and_filter_with_scores(items: List[Tuple[str, float]], min_conf: float) -> List[str]:
    # gripper 不是任务指令的目标对象，直接过滤。
    banned = {"gripper"}
    aliases = {
        "swich": "switch",
    }
    out: List[str] = []
    best_score = {}
    for x, score in items:
        t = x.strip()
        if not t:
            continue
        t = aliases.get(t.lower(), t)
        if float(score) < float(min_conf):
            continue
        key = t.lower()
        if key in banned:
            continue
        if key not in best_score or float(score) > best_score[key][1]:
            best_score[key] = (t, float(score))

    for key in sorted(best_score.keys()):
        out.append(best_score[key][0])
    return out

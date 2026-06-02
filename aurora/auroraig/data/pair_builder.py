from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from auroraig.adapters.reconvla_adapter import ReconvlaJsonAdapter
from auroraig.data.hybrid_rewriter import HybridInstructionRewriter
from auroraig.data.schemas import ContrastivePair
from auroraig.data.yolo_neighbor_detector import YOLONeighborDetector
from tqdm import tqdm


def build_contrastive_pairs(
    reconvla_json: str,
    output_jsonl: str,
    rewriter: HybridInstructionRewriter,
    image_root: Optional[str] = None,
    yolo_detector: Optional[YOLONeighborDetector] = None,
) -> int:
    pairs: List[ContrastivePair] = []

    iterator = ReconvlaJsonAdapter.iter_records(
        reconvla_json,
        image_root=image_root,
        yolo_detector=yolo_detector,
    )
    for record in tqdm(iterator, desc="building contrastive pairs", unit="sample"):
        pairs.append(
            ContrastivePair(
                image=record.image,
                instruction=record.instruction,
                label=1,
                source="positive",
            )
        )

        rewritten = rewriter.rewrite(
            record.instruction,
            object_candidates=record.object_candidates,
        )
        for neg in rewritten.negatives:
            pairs.append(
                ContrastivePair(
                    image=record.image,
                    instruction=neg,
                    label=0,
                    source="hybrid_negative",
                )
            )

    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in pairs:
            f.write(
                json.dumps(
                    {
                        "image": item.image,
                        "instruction": item.instruction,
                        "label": item.label,
                        "source": item.source,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return len(pairs)

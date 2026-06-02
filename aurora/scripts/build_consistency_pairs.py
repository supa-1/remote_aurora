from __future__ import annotations

import argparse
import os

from auroraig.config import RewriterConfig
from auroraig.data.consistency_builder import build_consistency_pairs
from auroraig.data.hybrid_rewriter import HybridInstructionRewriter
from auroraig.data.yolo_neighbor_detector import YOLONeighborDetector
from auroraig.interfaces.llm_client import NullLLMClient, resolve_default_llm_client


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建执行一致性排序训练样本")
    p.add_argument(
        "--reconvla_json",
        default="/home/supa1/myreconvla/AuroraIG/data/processed_json/calvin_debug_dataset/training_r5.json",
        help="输入 Reconvla JSON；默认指向 AuroraIG/data 下的 calvin_debug training_r5.json",
    )
    p.add_argument(
        "--output_jsonl",
        default="/home/supa1/myreconvla/AuroraIG/data/consistency_pairs/calvin_debug_training_llm.jsonl",
        help="输出真假指令 ranking 对（JSONL）",
    )
    p.add_argument("--max_rule_negatives", type=int, default=4)
    p.add_argument("--max_llm_negatives", type=int, default=4)
    p.add_argument("--disable_llm", action="store_true")
    p.add_argument("--disable_llm_absolute_lead", action="store_true", help="关闭 LLM 绝对主导，改为规则+LLM拼接")
    p.add_argument(
        "--disable_rule_fallback_when_llm_empty",
        action="store_true",
        help="当 LLM 无返回时，不再回退规则改写（用于纯 LLM 负样本）",
    )
    p.add_argument(
        "--min_pairs",
        type=int,
        default=1,
        help="最少需要生成的 pair 数；低于该值直接报错，避免空数据继续训练",
    )
    p.add_argument("--image_root", default="", help="图像根目录（用于 YOLO 检测邻近物体）")
    p.add_argument("--enable_yolo_neighbors", action="store_true", help="启用 YOLO 自动提取邻近物体")
    p.add_argument("--yolo_model_path", default="../ReconVLA/reconvla/scripts/helper/best.pt")
    p.add_argument("--yolo_conf", type=float, default=0.25)
    p.add_argument("--yolo_device", default="0")
    p.add_argument("--enable_neighbor_rule_filter", action="store_true", help="开启邻近物体规则过滤（默认关闭）")
    p.add_argument("--enable_non_neighbor_rule_filter", action="store_true", help="开启非邻近规则过滤（默认关闭）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["AURORAIG_LLM_ENABLE_NEIGHBOR_RULE_FILTER"] = "1" if args.enable_neighbor_rule_filter else "0"
    os.environ["AURORAIG_LLM_ENABLE_NON_NEIGHBOR_RULE_FILTER"] = "1" if args.enable_non_neighbor_rule_filter else "0"
    cfg = RewriterConfig(
        enable_rule_rewrite=True,
        enable_llm_rewrite=not args.disable_llm,
        llm_absolute_lead=not args.disable_llm_absolute_lead,
        rule_fallback_when_llm_empty=not args.disable_rule_fallback_when_llm_empty,
        max_rule_negatives=args.max_rule_negatives,
        max_llm_negatives=args.max_llm_negatives,
    )
    llm_client = NullLLMClient() if args.disable_llm else resolve_default_llm_client()
    rewriter = HybridInstructionRewriter(cfg=cfg, llm_client=llm_client)

    yolo_detector = None
    if args.enable_yolo_neighbors:
        yolo_detector = YOLONeighborDetector(
            model_path=args.yolo_model_path,
            conf=args.yolo_conf,
            device=args.yolo_device,
        )

    total = build_consistency_pairs(
        reconvla_json=args.reconvla_json,
        output_jsonl=args.output_jsonl,
        rewriter=rewriter,
        cfg=cfg,
        image_root=args.image_root or None,
        yolo_detector=yolo_detector,
    )
    if total < args.min_pairs:
        raise RuntimeError(
            f"only generated {total} pairs, below --min_pairs={args.min_pairs}. "
            "Please check LLM settings or input JSON."
        )
    print(f"done: wrote {total} ranking pairs to {args.output_jsonl}")


if __name__ == "__main__":
    main()

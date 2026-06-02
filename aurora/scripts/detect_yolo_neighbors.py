from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auroraig.data.yolo_neighbor_detector import YOLONeighborDetector


DEFAULT_MODEL_PATH = "../ReconVLA/reconvla/scripts/helper/best.pt"
DEFAULT_IMAGE_ROOT = "/home/supa1/myreconvla/AuroraIG/data/processed_images/calvin_debug_dataset/vla_processed_r5"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO 识别图片中的邻近物体列表")
    p.add_argument(
        "--image",
        required=True,
        help="图片路径。可传绝对路径，或传相对 image_root 的相对路径。",
    )
    p.add_argument(
        "--image_root",
        default=DEFAULT_IMAGE_ROOT,
        help="当 --image 为相对路径时，使用该目录拼接绝对路径。",
    )
    p.add_argument(
        "--yolo_model_path",
        default=DEFAULT_MODEL_PATH,
        help="YOLO 模型路径（默认使用 best.pt）。",
    )
    p.add_argument("--yolo_conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--yolo_device", default="0", help="推理设备，例如 0/cpu")
    p.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    return p.parse_args()


def resolve_image_path(image_arg: str, image_root: str) -> Path:
    p = Path(image_arg)
    if p.is_absolute():
        return p
    return Path(image_root) / p


def main() -> int:
    args = parse_args()
    image_path = resolve_image_path(args.image, args.image_root)

    if not image_path.exists():
        print(f"image not found: {image_path}")
        return 1

    detector = YOLONeighborDetector(
        model_path=args.yolo_model_path,
        conf=float(args.yolo_conf),
        device=args.yolo_device,
    )

    if not detector.enabled():
        print(f"invalid yolo model path: {args.yolo_model_path}")
        return 2

    objects: List[str] = detector.detect_objects(str(image_path))

    if args.json:
        payload = {
            "image": str(image_path),
            "objects": objects,
            "count": len(objects),
            "model": args.yolo_model_path,
            "conf": float(args.yolo_conf),
            "device": args.yolo_device,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print("=== YOLO Neighbor Objects ===")
    print(f"image: {image_path}")
    print(f"model: {args.yolo_model_path}")
    print(f"conf: {args.yolo_conf}")
    print(f"device: {args.yolo_device}")
    print(f"count: {len(objects)}")
    if not objects:
        print("objects: []")
        return 0

    print("objects:")
    for i, name in enumerate(objects, start=1):
        print(f"{i}. {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

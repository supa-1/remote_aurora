from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

TARGET_IMG_SIZE = 334


@dataclass(frozen=True)
class TaskRange:
    instruction: str
    task: str
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build Reconvla-style JSON + stitched images from raw CALVIN dataset."
    )
    p.add_argument(
        "--calvin_root",
        type=str,
        default="/home/supa1/myreconvla/calvin/dataset/calvin_debug_dataset",
        help="CALVIN dataset root that contains training/ and validation/.",
    )
    p.add_argument(
        "--output_image_root",
        type=str,
        default="/home/supa1/myreconvla/AuroraIG/data/processed_images/calvin_debug_dataset",
        help="Output root for stitched images.",
    )
    p.add_argument(
        "--output_json_root",
        type=str,
        default="/home/supa1/myreconvla/AuroraIG/data/processed_json/calvin_debug_dataset",
        help="Output root for training_r*.json and validation_r*.json.",
    )
    p.add_argument(
        "--future_k",
        type=int,
        default=5,
        help="Use future_k actions, flattened as gpt text (action dim = 7 * future_k).",
    )
    p.add_argument(
        "--target_mode",
        type=str,
        default="same",
        choices=["same", "next_static"],
        help=(
            "How to build image_target: same uses current static+gripper; "
            "next_static uses next step static (or current if at episode end)+current gripper."
        ),
    )
    p.add_argument(
        "--max_samples_per_split",
        type=int,
        default=0,
        help="0 means all samples. >0 means debug cap per split.",
    )
    return p.parse_args()


def load_task_ranges(split_dir: Path) -> List[TaskRange]:
    ann_path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    data = np.load(ann_path, allow_pickle=True).item()

    instructions = data["language"]["ann"]
    tasks = data["language"]["task"]
    ranges = data["info"]["indx"]

    task_ranges: List[TaskRange] = []
    for instruction, task, idx_range in zip(instructions, tasks, ranges):
        start, end = int(idx_range[0]), int(idx_range[1])
        task_ranges.append(
            TaskRange(
                instruction=str(instruction),
                task=str(task),
                start=start,
                end=end,
            )
        )
    return task_ranges


def load_npz(split_dir: Path, frame_id: int) -> Dict[str, np.ndarray]:
    npz_path = split_dir / f"episode_{frame_id:07d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing npz: {npz_path}")
    arr = np.load(npz_path)
    return {k: arr[k] for k in arr.files}


def resize_and_concat(rgb_static: np.ndarray, rgb_gripper: np.ndarray) -> Image.Image:
    h_static = TARGET_IMG_SIZE * 14 // 27
    h_gripper = TARGET_IMG_SIZE - h_static

    img_static = Image.fromarray(rgb_static).resize((TARGET_IMG_SIZE, h_static), Image.Resampling.LANCZOS)
    img_gripper = Image.fromarray(rgb_gripper).resize((TARGET_IMG_SIZE, h_gripper), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    canvas.paste(img_static, (0, 0))
    canvas.paste(img_gripper, (0, h_static))
    return canvas


def collect_future_actions(split_dir: Path, frame_id: int, end_id: int, future_k: int) -> List[np.ndarray]:
    actions: List[np.ndarray] = []
    for delta in range(future_k):
        fid = frame_id + delta
        if fid > end_id:
            break
        arr = load_npz(split_dir, fid)
        actions.append(arr["rel_actions"].reshape(-1))

    if not actions:
        raise ValueError(f"no future actions for frame {frame_id}")

    while len(actions) < future_k:
        actions.append(actions[-1].copy())
    return actions


def format_numbers(values: np.ndarray) -> str:
    return " ".join(map(str, values.reshape(-1)))


def build_item(
    sample_id: str,
    split: str,
    image_rel: str,
    image_target_rel: str,
    instruction: str,
    task: str,
    future_actions: Sequence[np.ndarray],
    robot_obs: np.ndarray,
) -> Dict:
    actions_flat = np.concatenate([x.reshape(-1) for x in future_actions], axis=0)
    actions_text = format_numbers(actions_flat)
    robot_obs_text = format_numbers(robot_obs)

    return {
        "id": sample_id,
        "task": task,
        "image": f"{split}/{image_rel}",
        "image_target": f"{split}/{image_target_rel}",
        "conversations": [
            {
                "from": "human",
                "value": f"{instruction}\n<image>\n{instruction}\n{robot_obs_text}",
            },
            {
                "from": "gpt",
                "value": actions_text,
            },
        ],
        "embody": True,
    }


def build_split(
    split: str,
    calvin_root: Path,
    image_root: Path,
    future_k: int,
    target_mode: str,
    max_samples_per_split: int,
) -> List[Dict]:
    split_dir = calvin_root / split
    task_ranges = load_task_ranges(split_dir)

    split_img_root = image_root / f"vla_processed_r{future_k}" / split
    split_target_root = split_img_root / "target"
    split_img_root.mkdir(parents=True, exist_ok=True)
    split_target_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    total_budget = max_samples_per_split if max_samples_per_split > 0 else None

    pbar = tqdm(task_ranges, desc=f"{split}: tasks")
    for task_idx, task_range in enumerate(pbar):
        for frame_id in range(task_range.start, task_range.end + 1):
            if total_budget is not None and len(rows) >= total_budget:
                return rows

            current = load_npz(split_dir, frame_id)
            rgb_static = current["rgb_static"]
            rgb_gripper = current["rgb_gripper"]
            robot_obs = current["robot_obs"]

            current_img = resize_and_concat(rgb_static, rgb_gripper)

            if target_mode == "same":
                target_img = current_img
            else:
                next_id = min(frame_id + 1, task_range.end)
                next_arr = load_npz(split_dir, next_id)
                target_img = resize_and_concat(next_arr["rgb_static"], rgb_gripper)

            sample_name = f"t{task_idx:03d}_{frame_id:07d}.jpg"
            sample_target_name = f"target/{sample_name}"

            current_img.save(split_img_root / sample_name)
            target_img.save(split_img_root / sample_target_name)

            future_actions = collect_future_actions(
                split_dir=split_dir,
                frame_id=frame_id,
                end_id=task_range.end,
                future_k=future_k,
            )

            item = build_item(
                sample_id=f"{split}_{frame_id:07d}",
                split=split,
                image_rel=sample_name,
                image_target_rel=sample_target_name,
                instruction=task_range.instruction,
                task=task_range.task,
                future_actions=future_actions,
                robot_obs=robot_obs,
            )
            rows.append(item)

    return rows


def main() -> None:
    args = parse_args()
    calvin_root = Path(args.calvin_root)
    output_image_root = Path(args.output_image_root)
    output_json_root = Path(args.output_json_root)

    output_json_root.mkdir(parents=True, exist_ok=True)

    for split in ("training", "validation"):
        rows = build_split(
            split=split,
            calvin_root=calvin_root,
            image_root=output_image_root,
            future_k=args.future_k,
            target_mode=args.target_mode,
            max_samples_per_split=args.max_samples_per_split,
        )

        out_json = output_json_root / f"{split}_r{args.future_k}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        print(f"[{split}] wrote {len(rows)} rows -> {out_json}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, cast

from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor


class ConsistencyJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str, processor: CLIPProcessor, max_text_length: int, image_root: Optional[str] = None):
        self.rows: List[Dict] = []
        self.processor = processor
        self.max_text_length = max_text_length
        self.image_root = Path(image_root) if image_root else None

        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        image = _load_image(row["image"], self.image_root)

        image_encoded = self.processor(images=[image], return_tensors="pt")
        true_encoded = self.processor(
            text=[row["true_instruction"]],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_text_length,
            padding="max_length",
        )
        fake_encoded = self.processor(
            text=[row["fake_instruction"]],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_text_length,
            padding="max_length",
        )
        action_encoded = self.processor(
            text=[row["action_text"]],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_text_length,
            padding="max_length",
        )

        pixel_values = cast("torch.Tensor", image_encoded["pixel_values"])
        true_input_ids = cast("torch.Tensor", true_encoded["input_ids"])
        true_attention_mask = cast("torch.Tensor", true_encoded["attention_mask"])
        fake_input_ids = cast("torch.Tensor", fake_encoded["input_ids"])
        fake_attention_mask = cast("torch.Tensor", fake_encoded["attention_mask"])
        action_input_ids = cast("torch.Tensor", action_encoded["input_ids"])
        action_attention_mask = cast("torch.Tensor", action_encoded["attention_mask"])

        return {
            "pixel_values": pixel_values.squeeze(0),
            "true_input_ids": true_input_ids.squeeze(0),
            "true_attention_mask": true_attention_mask.squeeze(0),
            "fake_input_ids": fake_input_ids.squeeze(0),
            "fake_attention_mask": fake_attention_mask.squeeze(0),
            "action_input_ids": action_input_ids.squeeze(0),
            "action_attention_mask": action_attention_mask.squeeze(0),
        }


def _load_image(image_data: str, image_root: Optional[Path]) -> Image.Image:
    if image_data.startswith("/") or image_data.endswith(".jpg") or image_data.endswith(".png"):
        if image_root is not None:
            image_path = image_root / image_data
        else:
            image_path = Path(image_data)
        return Image.open(image_path).convert("RGB")

    raw = base64.b64decode(image_data)
    return Image.open(io.BytesIO(raw)).convert("RGB")

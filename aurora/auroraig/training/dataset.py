from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Dict, List, cast

from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor


class ContrastiveJsonlDataset(Dataset):
    def __init__(self, jsonl_path: str, processor: CLIPProcessor, max_text_length: int):
        self.rows: List[Dict] = []
        self.processor = processor
        self.max_text_length = max_text_length

        with Path(jsonl_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        image = _decode_base64_image(row["image"])
        instruction = row["instruction"]

        encoded = self.processor(
            text=[instruction],
            images=[image],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_text_length,
            padding="max_length",
        )
        input_ids = cast("torch.Tensor", encoded["input_ids"])
        attention_mask = cast("torch.Tensor", encoded["attention_mask"])
        pixel_values = cast("torch.Tensor", encoded["pixel_values"])
        item = {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "pixel_values": pixel_values.squeeze(0),
        }
        return item


def _decode_base64_image(image_data: str) -> Image.Image:
    raw = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return image

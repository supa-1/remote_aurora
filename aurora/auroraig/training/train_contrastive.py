from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from auroraig.config import ContrastiveConfig
from auroraig.models.contrastive_model import ClipContrastiveModel
from auroraig.training.dataset import ContrastiveJsonlDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--encoder_name", default="openai/clip-vit-base-patch32")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--max_text_length", type=int, default=64)
    return p.parse_args()


def train(cfg: ContrastiveConfig, train_jsonl: str, output_dir: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = cast(CLIPProcessor, CLIPProcessor.from_pretrained(cfg.encoder_name))
    dataset = ContrastiveJsonlDataset(train_jsonl, processor, cfg.max_text_length)
    loader = DataLoader(dataset, batch_size=cfg.train_batch_size, shuffle=True)

    model = ClipContrastiveModel(cfg.encoder_name, cfg.temperature).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            total_loss += float(out.loss.item())

        avg_loss = total_loss / max(1, len(loader))
        print(f"epoch={epoch + 1} loss={avg_loss:.6f}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "auroraig_contrastive.pt")
    processor.save_pretrained(out_dir / "processor")
    with (out_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = parse_args()
    cfg = ContrastiveConfig(
        encoder_name=args.encoder_name,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        max_text_length=args.max_text_length,
    )
    train(cfg=cfg, train_jsonl=args.train_jsonl, output_dir=args.output_dir)

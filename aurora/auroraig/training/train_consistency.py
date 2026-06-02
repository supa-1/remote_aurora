from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from auroraig.models.consistency_model import ConsistencyScorer
from auroraig.training.consistency_dataset import ConsistencyJsonlDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--image_root", default=None)
    p.add_argument("--encoder_name", default="openai/clip-vit-base-patch32")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_text_length", type=int, default=128)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.3)
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = cast(CLIPProcessor, CLIPProcessor.from_pretrained(args.encoder_name))
    dataset = ConsistencyJsonlDataset(
        jsonl_path=args.train_jsonl,
        processor=processor,
        max_text_length=args.max_text_length,
        image_root=args.image_root,
    )
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    model = ConsistencyScorer(
        encoder_name=args.encoder_name,
        margin=args.margin,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_acc = 0.0
        steps = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()

            total_loss += float(out.loss.item())
            total_acc += float((out.true_score > out.fake_score).float().mean().item())
            steps += 1

        avg_loss = total_loss / max(1, steps)
        avg_acc = total_acc / max(1, steps)
        print(f"epoch={epoch + 1} loss={avg_loss:.6f} rank_acc={avg_acc:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "auroraig_consistency.pt")
    processor.save_pretrained(out_dir / "processor")

    meta = {
        "encoder_name": args.encoder_name,
        "margin": args.margin,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "max_text_length": args.max_text_length,
    }
    with (out_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    train(parse_args())

from __future__ import annotations

from auroraig.training.train_contrastive import parse_args, train
from auroraig.config import ContrastiveConfig


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

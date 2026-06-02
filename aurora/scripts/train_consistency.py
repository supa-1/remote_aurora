from __future__ import annotations

from auroraig.training.train_consistency import parse_args, train


if __name__ == "__main__":
    train(parse_args())

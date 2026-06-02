from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from transformers import CLIPModel


@dataclass
class ConsistencyOutput:
    loss: torch.Tensor
    true_score: torch.Tensor
    fake_score: torch.Tensor
    grounding_true: torch.Tensor
    grounding_fake: torch.Tensor
    executability_true: torch.Tensor
    executability_fake: torch.Tensor
    outcome_true: torch.Tensor
    outcome_fake: torch.Tensor


class ConsistencyScorer(nn.Module):
    def __init__(self, encoder_name: str, margin: float = 0.2, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(encoder_name)
        hidden = self.clip.config.projection_dim
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(
            self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask),
            dim=-1,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> ConsistencyOutput:
        image_emb = nn.functional.normalize(self.clip.get_image_features(pixel_values=batch["pixel_values"]), dim=-1)
        true_emb = self._encode_text(batch["true_input_ids"], batch["true_attention_mask"])
        fake_emb = self._encode_text(batch["fake_input_ids"], batch["fake_attention_mask"])
        action_emb = self._encode_text(batch["action_input_ids"], batch["action_attention_mask"])

        grounding_true = (image_emb * true_emb).sum(dim=-1)
        grounding_fake = (image_emb * fake_emb).sum(dim=-1)

        executability_true = (action_emb * true_emb).sum(dim=-1)
        executability_fake = (action_emb * fake_emb).sum(dim=-1)

        outcome_true = self.outcome_head(torch.cat([true_emb, action_emb], dim=-1)).squeeze(-1)
        outcome_fake = self.outcome_head(torch.cat([fake_emb, action_emb], dim=-1)).squeeze(-1)

        true_score = self.alpha * grounding_true + self.beta * executability_true + self.gamma * outcome_true
        fake_score = self.alpha * grounding_fake + self.beta * executability_fake + self.gamma * outcome_fake

        loss = torch.relu(self.margin - true_score + fake_score).mean()

        return ConsistencyOutput(
            loss=loss,
            true_score=true_score,
            fake_score=fake_score,
            grounding_true=grounding_true,
            grounding_fake=grounding_fake,
            executability_true=executability_true,
            executability_fake=executability_fake,
            outcome_true=outcome_true,
            outcome_fake=outcome_fake,
        )

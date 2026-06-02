from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from transformers import CLIPModel


@dataclass
class ContrastiveOutput:
    loss: torch.Tensor
    logits_per_image: torch.Tensor
    logits_per_text: torch.Tensor


class ClipContrastiveModel(nn.Module):
    def __init__(self, encoder_name: str, temperature: float):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(encoder_name)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / max(temperature, 1e-6)).log())

    def forward(self, batch: Dict[str, torch.Tensor]) -> ContrastiveOutput:
        outputs = self.clip(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            return_dict=True,
        )
        image_embeds = nn.functional.normalize(outputs.image_embeds, dim=-1)
        text_embeds = nn.functional.normalize(outputs.text_embeds, dim=-1)

        scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()

        labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
        loss_i = nn.functional.cross_entropy(logits_per_image, labels)
        loss_t = nn.functional.cross_entropy(logits_per_text, labels)
        loss = 0.5 * (loss_i + loss_t)

        return ContrastiveOutput(loss=loss, logits_per_image=logits_per_image, logits_per_text=logits_per_text)

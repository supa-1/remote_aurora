import sys
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import transformers

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
    Qwen2Config, Qwen2Model, Qwen2ForCausalLM

try:
    from transformers import Qwen3Config, Qwen3Model, Qwen3ForCausalLM
except ImportError:
    Qwen3Config = None
    Qwen3Model = None
    Qwen3ForCausalLM = None

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..recon_arch import ReconMetaModel, ReconMetaForCausalLM, CausalLMOutputWithPastWithVM


def _module_device(module: nn.Module, fallback: torch.device) -> torch.device:
    weight = getattr(module, "weight", None)
    if weight is not None:
        return weight.device
    try:
        return next(module.parameters()).device
    except StopIteration:
        return fallback


def _module_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    weight = getattr(module, "weight", None)
    if weight is not None and (weight.is_floating_point() or weight.is_complex()):
        return weight.dtype
    try:
        param = next(module.parameters())
    except StopIteration:
        return fallback
    if param.is_floating_point() or param.is_complex():
        return param.dtype
    return fallback


def _project_logits_and_loss(
    lm_head: nn.Module,
    hidden_states: torch.Tensor,
    labels: Optional[torch.Tensor],
    vocab_size: int,
    return_logits: bool,
    ignore_index: int = -100,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    lm_head_device = _module_device(lm_head, hidden_states.device)
    lm_head_dtype = _module_dtype(lm_head, hidden_states.dtype)
    same_device = lm_head_device == hidden_states.device

    if labels is not None and (not return_logits):
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = shift_labels.ne(ignore_index)
        if not valid_mask.any():
            zero = shift_hidden_states.sum() * 0.0
            return None, zero.to(hidden_states.device)
        shift_hidden_states = shift_hidden_states[valid_mask]
        if shift_hidden_states.device != lm_head_device or shift_hidden_states.dtype != lm_head_dtype:
            shift_hidden_states = shift_hidden_states.to(device=lm_head_device, dtype=lm_head_dtype)
        shift_labels = shift_labels[valid_mask].to(lm_head_device)
        shift_logits = lm_head(shift_hidden_states).float()
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        return None, loss.to(hidden_states.device)

    projected_hidden_states = hidden_states
    if not same_device:
        projected_hidden_states = hidden_states.to(device=lm_head_device, dtype=lm_head_dtype)

    logits = lm_head(projected_hidden_states).float()
    loss = None
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        if not same_device:
            loss = loss.to(hidden_states.device)
        if not return_logits:
            logits = None
    elif not same_device:
        logits = logits.to(hidden_states.device)

    return logits, loss


def _scalar_to_int(value) -> Optional[int]:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean_pool_visual_tokens(
    hidden_states: torch.Tensor,
    boi_ids: Optional[torch.Tensor],
    eoi_ids: Optional[torch.Tensor],
) -> torch.Tensor:
    """Pool only image-token hidden states so grounding cannot read instruction text."""
    batch_size, seq_len, hidden_size = hidden_states.shape
    if boi_ids is None or eoi_ids is None:
        return hidden_states.new_zeros((batch_size, hidden_size))

    pooled_rows = []
    for batch_index in range(batch_size):
        try:
            boi = _scalar_to_int(boi_ids[batch_index])
            eoi = _scalar_to_int(eoi_ids[batch_index])
        except (IndexError, TypeError):
            boi, eoi = None, None

        if boi is None or eoi is None:
            pooled_rows.append(hidden_states.new_zeros((hidden_size,)))
            continue

        start = max(0, boi)
        end = min(seq_len - 1, eoi)
        if start > end:
            pooled_rows.append(hidden_states.new_zeros((hidden_size,)))
            continue

        pooled_rows.append(hidden_states[batch_index, start:end + 1].mean(dim=0))

    return torch.stack(pooled_rows, dim=0)


class ReconConfig(Qwen2Config):
    model_type = "recon_qwen2"


if Qwen3Config is not None:
    class ReconQwen3Config(Qwen3Config):
        model_type = "recon_qwen3"
else:
    ReconQwen3Config = None


class ReconQwen2Model(ReconMetaModel, Qwen2Model):
    config_class = ReconConfig

    def __init__(self, config: Qwen2Config):
        super(ReconQwen2Model, self).__init__(config)


if Qwen3Model is not None and ReconQwen3Config is not None:
    class ReconQwen3Model(ReconMetaModel, Qwen3Model):
        config_class = ReconQwen3Config

        def __init__(self, config: Qwen3Config):
            super(ReconQwen3Model, self).__init__(config)
else:
    ReconQwen3Model = None


class ReconQwen2ForCausalLM(Qwen2ForCausalLM, ReconMetaForCausalLM):
    config_class = ReconConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = ReconQwen2Model(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.consistency_outcome_head = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
        )


        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        target_images: Optional[torch.FloatTensor] = None,
        text_recon_input_ids: Optional[torch.LongTensor] = None,
        text_recon_attention_mask: Optional[torch.Tensor] = None,
        text_recon_labels: Optional[torch.LongTensor] = None,
        consistency_true_input_ids: Optional[torch.LongTensor] = None,
        consistency_true_attention_mask: Optional[torch.Tensor] = None,
        consistency_fake_input_ids: Optional[torch.LongTensor] = None,
        consistency_fake_attention_mask: Optional[torch.Tensor] = None,
        consistency_action_input_ids: Optional[torch.LongTensor] = None,
        consistency_action_attention_mask: Optional[torch.Tensor] = None,
        consistency_pair_weight: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        origin_text = None
        if inputs_embeds is None:
            recon_return_flag = 0 
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                boi_ids,
                eoi_ids,
                cache_position,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                cache_position,
                recon_return_flag = 0,
            )
        else:
            boi_ids, eoi_ids = None, None

        return self.inner_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            boi_ids=boi_ids,
            eoi_ids=eoi_ids,
            images=images,
            target_images=target_images,
            text_recon_input_ids=text_recon_input_ids,
            text_recon_attention_mask=text_recon_attention_mask,
            text_recon_labels=text_recon_labels,
            consistency_true_input_ids=consistency_true_input_ids,
            consistency_true_attention_mask=consistency_true_attention_mask,
            consistency_fake_input_ids=consistency_fake_input_ids,
            consistency_fake_attention_mask=consistency_fake_attention_mask,
            consistency_action_input_ids=consistency_action_input_ids,
            consistency_action_attention_mask=consistency_action_attention_mask,
            consistency_pair_weight=consistency_pair_weight,
            cache_position=cache_position,
            origin_text=origin_text, 
        )

    def inner_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        boi_ids: Optional[torch.LongTensor] = None,
        eoi_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        target_images: Optional[torch.FloatTensor] = None,
        text_recon_input_ids: Optional[torch.LongTensor] = None,
        text_recon_attention_mask: Optional[torch.Tensor] = None,
        text_recon_labels: Optional[torch.LongTensor] = None,
        consistency_true_input_ids: Optional[torch.LongTensor] = None,
        consistency_true_attention_mask: Optional[torch.Tensor] = None,
        consistency_fake_input_ids: Optional[torch.LongTensor] = None,
        consistency_fake_attention_mask: Optional[torch.Tensor] = None,
        consistency_action_input_ids: Optional[torch.LongTensor] = None,
        consistency_action_attention_mask: Optional[torch.Tensor] = None,
        consistency_pair_weight: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        origin_text: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-Qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-Qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )


        hidden_states = outputs[0]
        need_logits = labels is None or not return_dict
        logits, loss = _project_logits_and_loss(
            self.lm_head,
            hidden_states,
            labels,
            vocab_size=self.config.vocab_size,
            return_logits=need_logits,
        )

        lm_loss = None
        if loss is not None:
            lm_loss = loss.detach().clone()

        vm_loss = None
        text_recon_loss = None
        consistency_aux_loss = None
        if self.training and getattr(self.config, 'recon_enable', False):
            if self.config.reconstruct_image_num == 2:
                vm_loss = self.compute_double_vm_loss(images, images, hidden_states, boi_ids, eoi_ids)
            elif self.config.reconstruct_image_num == 1:
                eps=1e-6
                vm_loss = self.compute_vm_loss(target_images, hidden_states, boi_ids, eoi_ids, eps, origin_text)
            if vm_loss is not None:# my change for reconstruct_image_num = 0
                loss = vm_loss if loss is None else (loss + vm_loss)

            if self.config.reconstruct_image == True:
                loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

        if (
            self.training
            and getattr(self.config, "enable_text_reconstruction", False)
            and text_recon_input_ids is not None
            and text_recon_labels is not None
        ):
            text_outputs = self.model(
                input_ids=text_recon_input_ids,
                attention_mask=text_recon_attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            text_hidden_states = text_outputs.last_hidden_state
            _, text_recon_loss = _project_logits_and_loss(
                self.lm_head,
                text_hidden_states,
                text_recon_labels,
                vocab_size=self.config.vocab_size,
                return_logits=False,
                ignore_index=-100,
            )

            text_recon_weight = float(getattr(self.config, "text_reconstruction_weight", 0.3))
            text_recon_weight = max(0.0, text_recon_weight)
            if text_recon_weight > 0:
                weighted_text_loss = text_recon_weight * text_recon_loss
                loss = weighted_text_loss if loss is None else (loss + weighted_text_loss)

        if (
            self.training
            and getattr(self.config, "enable_consistency_aux", False)
            and consistency_true_input_ids is not None
            and consistency_fake_input_ids is not None
            and consistency_action_input_ids is not None
        ):
            def _mean_pool(last_hidden_state, mask):
                if mask is None:
                    mask = torch.ones(last_hidden_state.shape[:2], device=last_hidden_state.device, dtype=last_hidden_state.dtype)
                mask = mask.to(last_hidden_state.dtype)
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                return (last_hidden_state * mask.unsqueeze(-1)).sum(dim=1) / denom

            grounding_emb = _mean_pool_visual_tokens(hidden_states, boi_ids, eoi_ids)
            true_outputs = self.model(
                input_ids=consistency_true_input_ids,
                attention_mask=consistency_true_attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            fake_outputs = self.model(
                input_ids=consistency_fake_input_ids,
                attention_mask=consistency_fake_attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            action_outputs = self.model(
                input_ids=consistency_action_input_ids,
                attention_mask=consistency_action_attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

            true_emb = _mean_pool(true_outputs.last_hidden_state, consistency_true_attention_mask)
            fake_emb = _mean_pool(fake_outputs.last_hidden_state, consistency_fake_attention_mask)
            action_emb = _mean_pool(action_outputs.last_hidden_state, consistency_action_attention_mask)

            grounding_emb = nn.functional.normalize(grounding_emb, dim=-1)
            true_emb = nn.functional.normalize(true_emb, dim=-1)
            fake_emb = nn.functional.normalize(fake_emb, dim=-1)
            action_emb = nn.functional.normalize(action_emb, dim=-1)

            grounding_true = (grounding_emb * true_emb).sum(dim=-1)
            grounding_fake = (grounding_emb * fake_emb).sum(dim=-1)
            executability_true = (action_emb * true_emb).sum(dim=-1)
            executability_fake = (action_emb * fake_emb).sum(dim=-1)
            outcome_true = self.consistency_outcome_head(torch.cat([true_emb, action_emb], dim=-1)).squeeze(-1)
            outcome_fake = self.consistency_outcome_head(torch.cat([fake_emb, action_emb], dim=-1)).squeeze(-1)

            alpha = float(getattr(self.config, "consistency_alpha", 0.4))
            beta = float(getattr(self.config, "consistency_beta", 0.3))
            gamma = float(getattr(self.config, "consistency_gamma", 0.3))
            margin = float(getattr(self.config, "consistency_margin", 0.2))
            weight = float(getattr(self.config, "consistency_aux_weight", 0.3))

            true_score = alpha * grounding_true + beta * executability_true + gamma * outcome_true
            fake_score = alpha * grounding_fake + beta * executability_fake + gamma * outcome_fake
            pair_loss = torch.relu(margin - true_score + fake_score)
            if consistency_pair_weight is not None and bool(getattr(self.config, "consistency_use_pair_weights", True)):
                pair_w = consistency_pair_weight.to(pair_loss.device).view(-1).to(pair_loss.dtype)
                pair_loss = pair_loss * pair_w
            consistency_aux_loss = pair_loss.mean()
            if weight > 0:
                weighted_consistency_loss = weight * consistency_aux_loss
                loss = weighted_consistency_loss if loss is None else (loss + weighted_consistency_loss)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastWithVM(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            lm_loss=lm_loss,
            vm_loss=vm_loss,
            text_recon_loss=text_recon_loss,
            consistency_aux_loss=consistency_aux_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            recon_return_flag = 1 
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                boi_ids,
                eoi_ids,
                cache_positions,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                recon_return_flag = 1, 
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        attention_mask=None,
        **kwargs,
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if image_sizes is not None:
            _inputs['image_sizes'] = image_sizes
        return _inputs


if Qwen3ForCausalLM is not None and ReconQwen3Model is not None and ReconQwen3Config is not None:
    class ReconQwen3ForCausalLM(Qwen3ForCausalLM, ReconMetaForCausalLM):
        config_class = ReconQwen3Config

        def __init__(self, config):
            super(Qwen3ForCausalLM, self).__init__(config)
            self.model = ReconQwen3Model(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.consistency_outcome_head = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, 1),
            )
            self.post_init()

        def get_model(self):
            return self.model

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            target_images: Optional[torch.FloatTensor] = None,
            text_recon_input_ids: Optional[torch.LongTensor] = None,
            text_recon_attention_mask: Optional[torch.Tensor] = None,
            text_recon_labels: Optional[torch.LongTensor] = None,
            consistency_true_input_ids: Optional[torch.LongTensor] = None,
            consistency_true_attention_mask: Optional[torch.Tensor] = None,
            consistency_fake_input_ids: Optional[torch.LongTensor] = None,
            consistency_fake_attention_mask: Optional[torch.Tensor] = None,
            consistency_action_input_ids: Optional[torch.LongTensor] = None,
            consistency_action_attention_mask: Optional[torch.Tensor] = None,
            consistency_pair_weight: Optional[torch.FloatTensor] = None,
            image_sizes: Optional[List[List[int]]] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
            origin_text = None
            if inputs_embeds is None:
                recon_return_flag = 0
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    boi_ids,
                    eoi_ids,
                    cache_position,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes,
                    cache_position,
                    recon_return_flag=0,
                )
            else:
                boi_ids, eoi_ids = None, None

            return self.inner_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                boi_ids=boi_ids,
                eoi_ids=eoi_ids,
                images=images,
                target_images=target_images,
                text_recon_input_ids=text_recon_input_ids,
                text_recon_attention_mask=text_recon_attention_mask,
                text_recon_labels=text_recon_labels,
                consistency_true_input_ids=consistency_true_input_ids,
                consistency_true_attention_mask=consistency_true_attention_mask,
                consistency_fake_input_ids=consistency_fake_input_ids,
                consistency_fake_attention_mask=consistency_fake_attention_mask,
                consistency_action_input_ids=consistency_action_input_ids,
                consistency_action_attention_mask=consistency_action_attention_mask,
                consistency_pair_weight=consistency_pair_weight,
                cache_position=cache_position,
                origin_text=origin_text,
            )

        def inner_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            boi_ids: Optional[torch.LongTensor] = None,
            eoi_ids: Optional[torch.LongTensor] = None,
            images: Optional[torch.FloatTensor] = None,
            target_images: Optional[torch.FloatTensor] = None,
            text_recon_input_ids: Optional[torch.LongTensor] = None,
            text_recon_attention_mask: Optional[torch.Tensor] = None,
            text_recon_labels: Optional[torch.LongTensor] = None,
            consistency_true_input_ids: Optional[torch.LongTensor] = None,
            consistency_true_attention_mask: Optional[torch.Tensor] = None,
            consistency_fake_input_ids: Optional[torch.LongTensor] = None,
            consistency_fake_attention_mask: Optional[torch.Tensor] = None,
            consistency_action_input_ids: Optional[torch.LongTensor] = None,
            consistency_action_attention_mask: Optional[torch.Tensor] = None,
            consistency_pair_weight: Optional[torch.FloatTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            origin_text: torch.LongTensor = None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = outputs[0]
            need_logits = labels is None or not return_dict
            logits, loss = _project_logits_and_loss(
                self.lm_head,
                hidden_states,
                labels,
                vocab_size=self.config.vocab_size,
                return_logits=need_logits,
            )

            lm_loss = None
            if loss is not None:
                lm_loss = loss.detach().clone()

            vm_loss = None
            text_recon_loss = None
            consistency_aux_loss = None
            if self.training and getattr(self.config, 'recon_enable', False):
                if self.config.reconstruct_image_num == 2:
                    vm_loss = self.compute_double_vm_loss(images, images, hidden_states, boi_ids, eoi_ids)
                elif self.config.reconstruct_image_num == 1:
                    eps = 1e-6
                    vm_loss = self.compute_vm_loss(target_images, hidden_states, boi_ids, eoi_ids, eps, origin_text)
                if vm_loss is not None:
                    loss = vm_loss if loss is None else (loss + vm_loss)

                if self.config.reconstruct_image is True:
                    loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

            if (
                self.training
                and getattr(self.config, "enable_text_reconstruction", False)
                and text_recon_input_ids is not None
                and text_recon_labels is not None
            ):
                text_outputs = self.model(
                    input_ids=text_recon_input_ids,
                    attention_mask=text_recon_attention_mask,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                text_hidden_states = text_outputs.last_hidden_state
                _, text_recon_loss = _project_logits_and_loss(
                    self.lm_head,
                    text_hidden_states,
                    text_recon_labels,
                    vocab_size=self.config.vocab_size,
                    return_logits=False,
                    ignore_index=-100,
                )

                text_recon_weight = float(getattr(self.config, "text_reconstruction_weight", 0.3))
                text_recon_weight = max(0.0, text_recon_weight)
                if text_recon_weight > 0:
                    weighted_text_loss = text_recon_weight * text_recon_loss
                    loss = weighted_text_loss if loss is None else (loss + weighted_text_loss)

            if (
                self.training
                and getattr(self.config, "enable_consistency_aux", False)
                and consistency_true_input_ids is not None
                and consistency_fake_input_ids is not None
                and consistency_action_input_ids is not None
            ):
                def _mean_pool(last_hidden_state, mask):
                    if mask is None:
                        mask = torch.ones(last_hidden_state.shape[:2], device=last_hidden_state.device, dtype=last_hidden_state.dtype)
                    mask = mask.to(last_hidden_state.dtype)
                    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                    return (last_hidden_state * mask.unsqueeze(-1)).sum(dim=1) / denom

                grounding_emb = _mean_pool_visual_tokens(hidden_states, boi_ids, eoi_ids)
                true_outputs = self.model(
                    input_ids=consistency_true_input_ids,
                    attention_mask=consistency_true_attention_mask,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                fake_outputs = self.model(
                    input_ids=consistency_fake_input_ids,
                    attention_mask=consistency_fake_attention_mask,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                action_outputs = self.model(
                    input_ids=consistency_action_input_ids,
                    attention_mask=consistency_action_attention_mask,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )

                true_emb = _mean_pool(true_outputs.last_hidden_state, consistency_true_attention_mask)
                fake_emb = _mean_pool(fake_outputs.last_hidden_state, consistency_fake_attention_mask)
                action_emb = _mean_pool(action_outputs.last_hidden_state, consistency_action_attention_mask)

                grounding_emb = nn.functional.normalize(grounding_emb, dim=-1)
                true_emb = nn.functional.normalize(true_emb, dim=-1)
                fake_emb = nn.functional.normalize(fake_emb, dim=-1)
                action_emb = nn.functional.normalize(action_emb, dim=-1)

                grounding_true = (grounding_emb * true_emb).sum(dim=-1)
                grounding_fake = (grounding_emb * fake_emb).sum(dim=-1)
                executability_true = (action_emb * true_emb).sum(dim=-1)
                executability_fake = (action_emb * fake_emb).sum(dim=-1)
                outcome_true = self.consistency_outcome_head(torch.cat([true_emb, action_emb], dim=-1)).squeeze(-1)
                outcome_fake = self.consistency_outcome_head(torch.cat([fake_emb, action_emb], dim=-1)).squeeze(-1)

                alpha = float(getattr(self.config, "consistency_alpha", 0.4))
                beta = float(getattr(self.config, "consistency_beta", 0.3))
                gamma = float(getattr(self.config, "consistency_gamma", 0.3))
                margin = float(getattr(self.config, "consistency_margin", 0.2))
                weight = float(getattr(self.config, "consistency_aux_weight", 0.3))

                true_score = alpha * grounding_true + beta * executability_true + gamma * outcome_true
                fake_score = alpha * grounding_fake + beta * executability_fake + gamma * outcome_fake
                pair_loss = torch.relu(margin - true_score + fake_score)
                if consistency_pair_weight is not None and bool(getattr(self.config, "consistency_use_pair_weights", True)):
                    pair_w = consistency_pair_weight.to(pair_loss.device).view(-1).to(pair_loss.dtype)
                    pair_loss = pair_loss * pair_w
                consistency_aux_loss = pair_loss.mean()
                if weight > 0:
                    weighted_consistency_loss = weight * consistency_aux_loss
                    loss = weighted_consistency_loss if loss is None else (loss + weighted_consistency_loss)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPastWithVM(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                lm_loss=lm_loss,
                vm_loss=vm_loss,
                text_recon_loss=text_recon_loss,
                consistency_aux_loss=consistency_aux_loss,
            )

        @torch.no_grad()
        def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            images: Optional[torch.Tensor] = None,
            image_sizes: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> Union[GenerateOutput, torch.LongTensor]:
            position_ids = kwargs.pop("position_ids", None)
            attention_mask = kwargs.pop("attention_mask", None)
            if "inputs_embeds" in kwargs:
                raise NotImplementedError("`inputs_embeds` is not supported")

            if images is not None:
                recon_return_flag = 1
                (
                    inputs,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _,
                    boi_ids,
                    eoi_ids,
                    cache_positions,
                ) = self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    image_sizes=image_sizes,
                    recon_return_flag=1,
                )
            else:
                inputs_embeds = self.get_model().embed_tokens(inputs)

            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs
            )

        def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            inputs_embeds=None,
            attention_mask=None,
            **kwargs,
        ):
            images = kwargs.pop("images", None)
            image_sizes = kwargs.pop("image_sizes", None)
            _inputs = super().prepare_inputs_for_generation(
                input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                **kwargs
            )
            if images is not None:
                _inputs['images'] = images
            if image_sizes is not None:
                _inputs['image_sizes'] = image_sizes
            return _inputs
else:
    ReconQwen3ForCausalLM = None


def get_recon_causallm_cls(version: str):
    normalized = (version or "").lower().replace("-", "_")
    if normalized.startswith("qwen_3") or normalized.startswith("qwen3"):
        if ReconQwen3ForCausalLM is None:
            raise ImportError(
                "当前 transformers 环境未提供 Qwen3 类，无法使用 --version qwen_3。"
            )
        return ReconQwen3ForCausalLM
    return ReconQwen2ForCausalLM



AutoConfig.register("recon_qwen2", ReconConfig)
AutoModelForCausalLM.register(ReconConfig, ReconQwen2ForCausalLM)

if ReconQwen3Config is not None and ReconQwen3ForCausalLM is not None:
    AutoConfig.register("recon_qwen3", ReconQwen3Config)
    AutoModelForCausalLM.register(ReconQwen3Config, ReconQwen3ForCausalLM)

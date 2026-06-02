"""该文件从 Reconvla 同步到 AuroraIG 本地运行目录。

说明：
- 主要用途是让 AuroraIG 可在本项目内直接运行 Reconvla 训练流程。
- 与上游版本保持兼容；若上游更新，建议重新同步并审查差异。
"""

import os
import io
import copy
import sys
from dataclasses import dataclass, field
from collections import OrderedDict
import datetime
import builtins
import json
import logging
import pathlib
import random
from typing import Dict, Optional, Sequence, List

import torch
import torch.distributed as dist

import transformers
import tokenizers
import megfile
from safetensors import safe_open

from recon.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from torch.utils.data import Dataset
from recon.recon_trainer import ReconTrainer

from recon import conversation as conversation_lib
from recon.model import *
from recon.mm_utils import tokenizer_image_token

from action_tokenizer import ActionTokenizer, encode_actions, encode_robot_obs
try:
    from kbit_training import build_bnb_skip_modules, prepare_model_for_kbit_training_oom_safe
except ImportError:
    from reconvla.kbit_training import build_bnb_skip_modules, prepare_model_for_kbit_training_oom_safe
from functools import partial
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


local_rank = None

LORA_MODULES_TO_SAVE = ["consistency_outcome_head"]


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    unfreeze_mm_vision_tower: Optional[bool] = field(default=False)
    mm_pixel_decoder: Optional[str] = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)   
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_mm_inv_mlp_adapter: Optional[str] = field(default=None)

    mm_projector_type: Optional[str] = field(default='linear')
    mm_inv_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    reconstruct_image_num: Optional[int] = field(default=1)
    reconstruct_image_savefolder: Optional[str] = field(default="./reconstructed_images")
    
@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    reconstruct_image: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    target_image_folder: Optional[str] = field(default=None)
    action_stat: str = None
    image_aspect_ratio: str = 'square'
    enable_text_reconstruction: bool = False
    text_reconstruction_weight: float = 0.3
    text_reconstruction_max_length: int = 128
    text_reconstruction_corrupt_ratio: float = 0.3
    enable_consistency_aux: bool = False
    consistency_aux_weight: float = 0.3
    consistency_margin: float = 0.2
    consistency_alpha: float = 0.4
    consistency_beta: float = 0.3
    consistency_gamma: float = 0.3
    consistency_max_length: int = 128
    consistency_use_pair_weights: bool = True
    consistency_min_pair_weight: float = 0.5
    consistency_type_weights_json: str = field(
        default='{"action_polarity_flip": 1.0, "neighbor_object_replacement": 0.9, "direction_replacement": 0.85, "hard_color_negative": 0.85, "subject_object_swap": 0.85, "spatial_replacement": 0.8, "color_replacement": 0.75, "easy_color_negative": 0.55, "content_simplification": 0.65, "other_rewrite": 0.7}'
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_inv_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    lm_head_cpu_offload: bool = field(default=False)
    lm_head_cpu_dtype: str = field(default="float32")
    kbit_skip_large_embedding_upcast: bool = field(default=False)
    kbit_keep_lm_head_in_fp16: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    save_steps: int = 5000
    save_total_limit: int = 1,


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    named_params = list(named_params)
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    for k, t in named_params:
        if any(f"{module_name}.modules_to_save." in k for module_name in LORA_MODULES_TO_SAVE):
            to_return[k] = t
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    # 兼容 4bit/8bit 量化线性层，避免仅识别 torch.nn.Linear 导致 LoRA 目标层为空或不稳定。
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'mm_inv_projector']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue

        cls_name = module.__class__.__name__
        is_linear_like = isinstance(module, torch.nn.Linear) or cls_name in {"Linear4bit", "Linear8bitLt"}
        if not is_linear_like:
            continue

        names = name.split('.')
        leaf_name = names[0] if len(names) == 1 else names[-1]
        if leaf_name.isdigit() or leaf_name == 'lm_head':
            continue
        lora_module_names.add(leaf_name)

    prioritized = [n for n in preferred if n in lora_module_names]
    return prioritized if prioritized else sorted(list(lora_module_names))


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        weight_mm_projector = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)

        keys_to_match = ['mm_inv_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        weight_mm_inv_projector = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)

        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, 'mm_projector')
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_mm_projector, os.path.join(mm_projector_folder, f'{current_folder}.bin'))

                mm_inv_projector_folder = os.path.join(parent_folder, 'mm_inv_projector')
                os.makedirs(mm_inv_projector_folder, exist_ok=True)
                torch.save(weight_mm_inv_projector, os.path.join(mm_inv_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_mm_projector, os.path.join(output_dir, f'mm_projector.bin'))
                torch.save(weight_mm_inv_projector, os.path.join(output_dir, f'mm_inv_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    all_state_dict = trainer.model.state_dict()
    state_dict = OrderedDict()
    for k, v in all_state_dict.items():
        if ".teacher_model." in k:
            print(f"=> skipped {k} for saving.")
            continue
        state_dict[k] = copy.deepcopy(v)
    del all_state_dict
    
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt.replace(conv.sep2, ''), tokenizer, return_tensors='pt') for prompt in
             conversations], dim=0)
    else:
        input_ids = tokenizer(
            [prompt.replace(conv.sep2, '') for prompt in conversations],
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = f'<|start_header_id|>{conv.roles[1]}<|end_header_id|>\n\n'
    for conversation, target in zip(conversations, targets):
        total_len = target.shape[0]

        rounds = conversation.split(conv.sep2)
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        rounds_len = []
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break

            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids)

            if i != 0 and not getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    comprehension, creation = False, False
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = []
        for prompt in conversations:
            if DEFAULT_IMAGE_TOKEN in prompt:
                input_ids.append(tokenizer_image_token(prompt, tokenizer, return_tensors='pt'))
            else:
                raise NotImplementedError
        input_ids = torch.stack(input_ids, dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for input_id, conversation, target in zip(input_ids, conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    comprehension, creation = False, False
    for source in sources:
        assert len(source) == 2
        if DEFAULT_IMAGE_TOKEN in source[0]['value']:
            # comprehension data
            comprehension = True
            source[0]['value'] = DEFAULT_IMAGE_TOKEN
            conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        assert comprehension or creation
        assert (not comprehension) or (not creation)
        conversations.append(conversation)
    # tokenize conversations
    tokenizer_fn = tokenizer_image_token
    input_ids = [tokenizer_fn(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def format_source_data(sources, conv, has_embody, action_tokenizer):
    """Convert the source QA data into the format expected.

    Args:
        sources (List[Dict]): One element list for current dialogue.
        conv (_type_): Target conversation template.
        has_embody (bool): Whether the item is a robotic data.

    Returns:
        conversations (List[Str]): One element list for formatted dialogue.
    """
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        if has_embody:
            conv.system = "A chat between a curious human and an artificial intelligence robot. The robot provides actions to follow out the user's instructions."
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            if sentence["from"] == "gpt":
                if has_embody:
                    real_action_token, sent_value = action_to_lang(sentence["value"], action_tokenizer)
                else:
                    sent_value = sentence["value"]
            else:
                if has_embody:
                    sent_value = sentence["value"]
                    sent_value_parts = sent_value.split("\n")
                    real_obs_token, robot_obs = robot_obs_lang(sent_value_parts[-1], action_tokenizer)
                    sent_value_parts[-1] = robot_obs
                    # sent_value_parts = sent_value_parts[:2]
                    sent_value = "\n".join(sent_value_parts)
                else:
                    sent_value = sentence["value"]
            conv.append_message(role, sent_value)
        conversations.append(conv.get_prompt())
    return conversations, real_action_token, real_obs_token

def mask_qwen_labels(input_ids):

    labels = torch.full_like(input_ids, IGNORE_INDEX)  
    index_25 = (input_ids == 25).nonzero(as_tuple=True)[1]
    if len(index_25) > 0:
        last_25_index = index_25[-1].item()  
    else:
        last_25_index = None  

    labels[0][last_25_index + 1:] = input_ids[0][last_25_index + 1:]
    
    return labels


def preprocess_qwen_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for input_id, conversation, target in zip(input_ids, conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)

        rounds_len = len(rounds)
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break

            parts[0] += sep

            if has_image:
                round_ids = tokenizer_image_token(rou, tokenizer)
                instruction_ids = tokenizer_image_token(parts[0], tokenizer)
                equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]

                instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
                round_len = len(round_ids)

            else:
                round_ids = tokenizer(rou).input_ids
                instruction_ids = tokenizer(parts[0]).input_ids
                equal_parts = [x == y for x, y in zip(round_ids, instruction_ids)]

                instruction_len = equal_parts.index(False) if False in equal_parts else len(equal_parts)
                round_len = len(round_ids)

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            if i == 0:
                target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            else:
                target[cur_len + 1: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len - 1:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len - 1}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_qwen_2_vla(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    action_tokenizer: ActionTokenizer,
    has_image: bool = False,
    has_embody: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()

    # Format conversations
    conversations, real_action_token, real_obs_token = format_source_data(sources, conv, has_embody, action_tokenizer)

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    index_35560 = (input_ids == 35560).nonzero(as_tuple=True)[1]
    index_25 = (input_ids == 25).nonzero(as_tuple=True)[1]
    last_index = input_ids.shape[1] - 1
    if len(index_35560) > 0:
        start_obs = max(index_35560[0] - 15, 0)  
        input_ids = torch.cat((input_ids[:, :start_obs], torch.tensor(real_obs_token).unsqueeze(0), input_ids[:, index_35560[0]:]), dim=1)

    if len(index_25) > 0:
        last_25 = index_25[-1]  
        if last_25 + 1 < last_index:  
            input_ids = torch.cat((input_ids[:, :last_25 + 1], torch.tensor(real_action_token).unsqueeze(0), input_ids[:, last_index:]), dim=1)

    input_id_copy = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    targets = mask_qwen_labels(input_id_copy)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        print("Preprocess: plain")
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        print("Preprocess: llama2")
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        print("Preprocess: plainv1")
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        print("Preprocess: mpt")
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith(("qwen_2", "qwen_3")):
        print("Preprocess: qwen_family")
        return preprocess_qwen_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("llama3"):
        print("Preprocess: llama3")
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def preprocess_action(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    action_tokenizer: ActionTokenizer,
    has_image: bool = False,
    has_embody: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith(("qwen_2", "qwen_3")):
        return preprocess_qwen_2_vla(sources, tokenizer, action_tokenizer, has_image=has_image, has_embody=has_embody)
    if conversation_lib.default_conversation.version.startswith("llama3"):
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 action_tokenizer: ActionTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        with megfile.smart_open(data_path, "r", encoding="utf-8") as file:
            list_data_dict = json.load(file)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.action_tokenizer = action_tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.consistency_type_weights = self._parse_consistency_type_weights(data_args.consistency_type_weights_json)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        sample_has_image = ('image' in sources[0]) and bool(self.data_args.is_multimodal)
        if sample_has_image:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            with megfile.smart_open(os.path.join(image_folder, image_file), "rb") as f:
                bytes_data = f.read()
            image = Image.open(io.BytesIO(bytes_data), 'r').convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                # center crop
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)

            target_image_file = self.list_data_dict[i]['image_target']
            target_image_folder = self.data_args.target_image_folder
            with megfile.smart_open(os.path.join(target_image_folder, target_image_file), "rb") as f:
                target_bytes_data = f.read()
            target_image = Image.open(io.BytesIO(target_bytes_data), 'r').convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                target_image = expand2square(target_image, tuple(int(x*255) for x in processor.image_mean))
                target_image = processor.preprocess(target_image, return_tensors='pt')['pixel_values'][0]
            else:
                # center crop
                target_image = processor.preprocess(target_image, return_tensors='pt')['pixel_values'][0]
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess_action(
            sources,
            self.tokenizer,
            self.action_tokenizer,
            has_image=sample_has_image,
            has_embody=('embody' in self.list_data_dict[i])
            )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        if self.data_args.enable_text_reconstruction:
            text_recon = self._build_text_reconstruction_item(self.list_data_dict[i])
            if text_recon is not None:
                data_dict.update(text_recon)

        if self.data_args.enable_consistency_aux:
            consistency_item = self._build_consistency_item(self.list_data_dict[i])
            if consistency_item is not None:
                data_dict.update(consistency_item)

        # image exist in the data
        if sample_has_image:
            data_dict['image'] = image
            data_dict['target_image'] = target_image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            data_dict['target_image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict

    def _build_text_reconstruction_item(self, sample: Dict) -> Optional[Dict[str, torch.Tensor]]:
        pre_true = sample.get("aux_true_instruction", "")
        pre_noisy = sample.get("aux_text_recon_noisy", "")

        if pre_true:
            instruction = str(pre_true).strip()
        else:
            conversations = sample.get("conversations", [])
            if not conversations:
                return None
            human_text = conversations[0].get("value", "")
            instruction = self._extract_instruction(human_text)
        if not instruction:
            return None

        noisy_instruction = str(pre_noisy).strip() if pre_noisy else self._corrupt_instruction(
            instruction,
            self.data_args.text_reconstruction_corrupt_ratio,
        )
        prompt = f"Recover instruction from noisy text: {noisy_instruction}\nRecovered:"

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.data_args.text_reconstruction_max_length,
        ).input_ids
        target_ids = self.tokenizer(
            instruction,
            add_special_tokens=False,
            truncation=True,
            max_length=self.data_args.text_reconstruction_max_length,
        ).input_ids
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = self.tokenizer.pad_token_id
        if eos_id is None:
            return None

        full_ids = (prompt_ids + target_ids + [eos_id])[: self.data_args.text_reconstruction_max_length]
        labels = [IGNORE_INDEX] * min(len(prompt_ids), len(full_ids))
        for idx in range(len(prompt_ids), len(full_ids)):
            labels.append(full_ids[idx])
        labels = labels[: len(full_ids)]

        return {
            "text_recon_input_ids": torch.tensor(full_ids, dtype=torch.long),
            "text_recon_labels": torch.tensor(labels, dtype=torch.long),
        }

    def _build_consistency_item(self, sample: Dict) -> Optional[Dict[str, torch.Tensor]]:
        conversations = sample.get("conversations", [])
        if len(conversations) < 2:
            return None

        human_text = conversations[0].get("value", "")
        action_text = conversations[1].get("value", "")
        true_instruction = str(sample.get("aux_true_instruction", "")).strip() or self._extract_instruction(human_text)
        if not true_instruction:
            return None

        fake_instruction = ""
        negative_type = "other_rewrite"

        fake_pool = sample.get("aux_fake_instruction_pool", [])
        type_pool = sample.get("aux_negative_type_pool", [])
        valid_pairs: List[tuple[str, str]] = []
        if isinstance(fake_pool, list):
            for idx, fake in enumerate(fake_pool):
                fake_text = str(fake).strip()
                if not fake_text:
                    continue
                pair_type = "other_rewrite"
                if isinstance(type_pool, list) and idx < len(type_pool):
                    pair_type = str(type_pool[idx]).strip() or "other_rewrite"
                valid_pairs.append((fake_text, pair_type))

        if valid_pairs:
            fake_instruction, negative_type = random.choice(valid_pairs)
        else:
            fake_instruction = str(sample.get("aux_fake_instruction", "")).strip()
            negative_type = str(sample.get("aux_negative_type", "other_rewrite")).strip() or "other_rewrite"

        if not fake_instruction:
            fake_instruction = self._make_fake_instruction(true_instruction)
            negative_type = "rule_fallback"
        if fake_instruction == true_instruction:
            return None

        max_len = self.data_args.consistency_max_length
        true_tok = self.tokenizer(
            true_instruction,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )
        fake_tok = self.tokenizer(
            fake_instruction,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )
        action_tok = self.tokenizer(
            action_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )

        pair_weight = self._resolve_consistency_pair_weight(negative_type)

        return {
            "consistency_true_input_ids": torch.tensor(true_tok.input_ids, dtype=torch.long),
            "consistency_true_attention_mask": torch.ones(len(true_tok.input_ids), dtype=torch.long),
            "consistency_fake_input_ids": torch.tensor(fake_tok.input_ids, dtype=torch.long),
            "consistency_fake_attention_mask": torch.ones(len(fake_tok.input_ids), dtype=torch.long),
            "consistency_action_input_ids": torch.tensor(action_tok.input_ids, dtype=torch.long),
            "consistency_action_attention_mask": torch.ones(len(action_tok.input_ids), dtype=torch.long),
            "consistency_pair_weight": torch.tensor(pair_weight, dtype=torch.float32),
        }

    def _resolve_consistency_pair_weight(self, negative_type: str) -> float:
        if not bool(self.data_args.consistency_use_pair_weights):
            return 1.0
        key = str(negative_type).strip() or "other_rewrite"
        default_w = self.consistency_type_weights.get("other_rewrite", 1.0)
        raw_w = float(self.consistency_type_weights.get(key, default_w))
        min_w = max(0.0, float(self.data_args.consistency_min_pair_weight))
        return max(min_w, raw_w)

    @staticmethod
    def _parse_consistency_type_weights(raw: str) -> Dict[str, float]:
        default = {
            "action_polarity_flip": 1.0,
            "neighbor_object_replacement": 0.9,
            "direction_replacement": 0.85,
            "hard_color_negative": 0.85,
            "subject_object_swap": 0.85,
            "spatial_replacement": 0.8,
            "color_replacement": 0.75,
            "easy_color_negative": 0.55,
            "content_simplification": 0.65,
            "other_rewrite": 0.7,
            "rule_fallback": 0.6,
        }
        if not raw:
            return default
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return default
            merged = dict(default)
            for k, v in parsed.items():
                merged[str(k)] = float(v)
            return merged
        except (ValueError, TypeError, json.JSONDecodeError):
            return default

    @staticmethod
    def _extract_instruction(human_text: str) -> str:
        lines = [x.strip() for x in human_text.split("\n") if x.strip()]
        for line in lines:
            if "<image>" in line:
                continue
            if len(line.split()) >= 3:
                return line
        return ""

    @staticmethod
    def _corrupt_instruction(text: str, ratio: float) -> str:
        words = text.split()
        if len(words) <= 2:
            return text

        ratio = max(0.0, min(0.9, float(ratio)))
        drop_n = max(1, int(len(words) * ratio))
        candidate_indices = list(range(len(words)))
        random.shuffle(candidate_indices)
        drop_set = set(candidate_indices[:drop_n])

        corrupted = [w for idx, w in enumerate(words) if idx not in drop_set]
        if not corrupted:
            return text
        return " ".join(corrupted)

    @staticmethod
    def _make_fake_instruction(text: str) -> str:
        swaps = [
            ("left", "right"),
            ("right", "left"),
            ("front", "back"),
            ("back", "front"),
            ("red", "blue"),
            ("blue", "red"),
            ("cup", "bottle"),
            ("bottle", "cup"),
            ("box", "chopsticks"),
            ("chopsticks", "box"),
            ("左", "右"),
            ("右", "左"),
            ("前", "后"),
            ("后", "前"),
            ("红", "蓝"),
            ("蓝", "红"),
            ("盒子", "筷子"),
            ("筷子", "盒子"),
        ]
        for src, dst in swaps:
            if src in text:
                return text.replace(src, dst, 1)
        return text


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if 'target_image' in instances[0]:
            target_images = [instance['target_image'] for instance in instances]
            if all(x is not None and x.shape == target_images[0].shape for x in target_images):
                batch['target_images'] = torch.stack(target_images)
            else:
                batch['target_images'] = target_images

        if 'text_recon_input_ids' in instances[0]:
            text_recon_input_ids = [instance['text_recon_input_ids'] for instance in instances]
            text_recon_labels = [instance['text_recon_labels'] for instance in instances]

            text_recon_input_ids = torch.nn.utils.rnn.pad_sequence(
                text_recon_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            text_recon_labels = torch.nn.utils.rnn.pad_sequence(
                text_recon_labels,
                batch_first=True,
                padding_value=IGNORE_INDEX,
            )
            batch['text_recon_input_ids'] = text_recon_input_ids
            batch['text_recon_labels'] = text_recon_labels
            batch['text_recon_attention_mask'] = text_recon_input_ids.ne(self.tokenizer.pad_token_id)

        if 'consistency_true_input_ids' in instances[0]:
            for prefix in [
                "consistency_true",
                "consistency_fake",
                "consistency_action",
            ]:
                ids_key = f"{prefix}_input_ids"
                mask_key = f"{prefix}_attention_mask"
                ids = [instance[ids_key] for instance in instances]
                masks = [instance[mask_key] for instance in instances]
                ids = torch.nn.utils.rnn.pad_sequence(
                    ids,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                )
                masks = torch.nn.utils.rnn.pad_sequence(
                    masks,
                    batch_first=True,
                    padding_value=0,
                )
                batch[ids_key] = ids
                batch[mask_key] = masks
            if 'consistency_pair_weight' in instances[0]:
                batch['consistency_pair_weight'] = torch.stack([
                    instance['consistency_pair_weight'] for instance in instances
                ])

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                action_tokenizer: ActionTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                action_tokenizer=action_tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def train(attn_implementation=None):
    global local_rank, action_to_lang, robot_obs_lang

    if attn_implementation is None:
        attn_implementation = os.getenv("ATTN_IMPLEMENTATION", "eager").strip() or "eager"

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    def _normalize_optional_ref(value):
        if value is None:
            return None
        if isinstance(value, str):
            v = value.strip()
            if v == "" or v.lower() in {"none", "null", "false", "0"}:
                return None
            return v
        return value

    model_args.vision_tower = _normalize_optional_ref(model_args.vision_tower)
    model_args.mm_pixel_decoder = _normalize_optional_ref(model_args.mm_pixel_decoder)

    if training_args.gradient_checkpointing and training_args.gradient_checkpointing_kwargs is None:
        # Avoid PyTorch checkpoint warnings from the legacy reentrant default.
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    action_to_lang = partial(encode_actions, statistics=None)
    robot_obs_lang = partial(encode_robot_obs, statistics=data_args.action_stat)
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_skip_modules = build_bnb_skip_modules(
            keep_lm_head_in_fp16=training_args.kbit_keep_lm_head_in_fp16
        )
        if training_args.kbit_keep_lm_head_in_fp16:
            rank0_print("Keeping lm_head out of bitsandbytes quantization for GPU fp16/bf16 training.")
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8, (my change for the 4-bit error because of the new version of transformers and bitsandbytes
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=bnb_skip_modules,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    model_cls = get_recon_causallm_cls(model_args.version)
    rank0_print(f"Using language model class: {model_cls.__name__} (version={model_args.version})")

    def resolve_pretrained_ref(path_or_repo: Optional[str]) -> Optional[str]:
        if not path_or_repo:
            return path_or_repo
        expanded = os.path.expanduser(path_or_repo)
        # Keep Hub repo ids unchanged; normalize local filesystem paths.
        if expanded.startswith("/") or expanded.startswith("."):
            return os.path.abspath(expanded)
        if os.path.isdir(expanded) or os.path.isfile(expanded):
            return os.path.abspath(expanded)
        return path_or_repo

    def _load_local_weight_tensor(model_dir: str, tensor_name: str) -> torch.Tensor:
        model_dir = os.path.abspath(os.path.expanduser(model_dir))
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                weight_index = json.load(f)
            shard_name = weight_index.get("weight_map", {}).get(tensor_name)
            if shard_name is None:
                raise KeyError(f"{tensor_name} not found in {index_path}")
            shard_path = os.path.join(model_dir, shard_name)
        else:
            shard_path = os.path.join(model_dir, "model.safetensors")
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(
                    f"Unable to find safetensors weights for {tensor_name} under {model_dir}"
                )
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            if tensor_name not in f.keys():
                raise KeyError(f"{tensor_name} not found in {shard_path}")
            return f.get_tensor(tensor_name)

    def maybe_offload_lm_head_to_cpu(model: torch.nn.Module):
        if not bool(training_args.lm_head_cpu_offload):
            return

        model_path = resolve_pretrained_ref(model_args.model_name_or_path)
        if not model_path or not os.path.isdir(model_path):
            raise ValueError(
                "lm_head_cpu_offload=True requires model_name_or_path to be a local checkpoint directory."
            )

        dtype_name = str(training_args.lm_head_cpu_dtype).strip().lower()
        if dtype_name not in {"float16", "float32"}:
            raise ValueError(f"Unsupported lm_head_cpu_dtype: {training_args.lm_head_cpu_dtype}")
        target_dtype = torch.float16 if dtype_name == "float16" else torch.float32

        lm_head_weight = _load_local_weight_tensor(model_path, "lm_head.weight").to(dtype=target_dtype)
        current_lm_head = model.get_output_embeddings()
        if current_lm_head is None:
            raise ValueError("Model does not expose output embeddings for lm_head offload.")
        expected_out_features = int(getattr(current_lm_head, "out_features", model.config.vocab_size))
        expected_in_features = int(getattr(current_lm_head, "in_features", model.config.hidden_size))
        expected_shape = (expected_out_features, expected_in_features)
        if tuple(lm_head_weight.shape) != expected_shape:
            raise ValueError(
                f"lm_head shape mismatch: expected={expected_shape} checkpoint={tuple(lm_head_weight.shape)}"
            )

        cpu_lm_head = torch.nn.Linear(
            lm_head_weight.shape[1],
            lm_head_weight.shape[0],
            bias=False,
            device="cpu",
            dtype=target_dtype,
        )
        cpu_lm_head.weight.data.copy_(lm_head_weight)
        cpu_lm_head.weight.requires_grad = False
        model.set_output_embeddings(cpu_lm_head)
        model.config.lm_head_cpu_offload = True
        model.config.lm_head_cpu_dtype = dtype_name
        rank0_print(f"lm_head offloaded to CPU with dtype={dtype_name}")

    config_cls = getattr(model_cls, "config_class", None) or transformers.PretrainedConfig
    model_config = config_cls.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    if model_args.vision_tower is not None:
        model_config.mm_vision_tower = resolve_pretrained_ref(model_args.vision_tower)
    else:
        # Explicitly clear checkpoint-carried multimodal refs for text-only runs.
        model_config.mm_vision_tower = None
    if model_args.mm_pixel_decoder is not None:
        model_config.mm_pixel_decoder = resolve_pretrained_ref(model_args.mm_pixel_decoder)
    else:
        model_config.mm_pixel_decoder = None

    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        config=model_config,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(
            compute_dtype
            if training_args.bits in [4, 8] and training_args.kbit_keep_lm_head_in_fp16
            else (torch.bfloat16 if training_args.bf16 else None)
        ),
        **bnb_model_from_pretrained_args
    )
    

    model.config.use_cache = False

    import json

    params_info = {k: v.shape for k, v in model.state_dict().items()}
    with open("./model_parameters.json", "w") as f:
        json.dump(params_info, f, indent=4)

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        if training_args.kbit_skip_large_embedding_upcast:
            rank0_print("Skipping fp32 upcast for large input/output embeddings during k-bit preparation.")
        model = prepare_model_for_kbit_training_oom_safe(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs,
            skip_large_embedding_upcast=training_args.kbit_skip_large_embedding_upcast,
        )

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    model.tokenizer = tokenizer
    
    action_tokenizer = ActionTokenizer(tokenizer)

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif "llama-3" in model_args.model_name_or_path.lower():
            tokenizer.pad_token = '<|end_of_text|>'

        else:  # use qwen
            tokenizer.legacy = False
            if tokenizer.pad_token is None:
                print(f"Adding pad token as '<|pad|>'")
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="<|pad|>"),
                    tokenizer=tokenizer,
                    model=model,
                )

        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    maybe_offload_lm_head_to_cpu(model)

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp,
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            if hasattr(model.get_model(), "mm_inv_projector"):
                for p in model.get_model().mm_inv_projector.parameters():
                    p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
            if hasattr(model.get_model(), "mm_inv_projector"):
                for p in model.get_model().mm_inv_projector.parameters():
                    p.requires_grad = False

        model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
        if training_args.unfreeze_mm_vision_tower:
            for p in model.get_model().vision_tower.parameters():
                p.requires_grad = True

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
            if hasattr(model.get_model(), "mm_inv_projector"):
                model.get_model().mm_inv_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_inv_projector_lr = training_args.mm_inv_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        model.config.pad_token_id = tokenizer.pad_token_id

    # Aux losses should be configurable even when vision modules are disabled.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.reconstruct_image_num = model_args.reconstruct_image_num
    model.config.reconstruct_image = data_args.reconstruct_image
    model.config.enable_text_reconstruction = data_args.enable_text_reconstruction
    model.config.text_reconstruction_weight = data_args.text_reconstruction_weight
    model.config.enable_consistency_aux = data_args.enable_consistency_aux
    model.config.consistency_aux_weight = data_args.consistency_aux_weight
    model.config.consistency_margin = data_args.consistency_margin
    model.config.consistency_alpha = data_args.consistency_alpha
    model.config.consistency_beta = data_args.consistency_beta
    model.config.consistency_gamma = data_args.consistency_gamma
    model.config.consistency_use_pair_weights = data_args.consistency_use_pair_weights
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        transformers.set_seed(training_args.seed)
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            modules_to_save=LORA_MODULES_TO_SAVE,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        rank0_print("LoRA adapters attached.")
        model.print_trainable_parameters()


    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    train_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() if p.requires_grad else 0 for p in model.parameters())
    print(f">> Total params: {total_params / 1.e6}M")
    print(f">> Train params: {train_params / 1.e6}M, Ratio {train_params / total_params * 100.:.2f}%")
    print(f">> Save every {training_args.save_steps} steps.")

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              action_tokenizer=action_tokenizer,
                                              data_args=data_args)
    


    trainer = ReconTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

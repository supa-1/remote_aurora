from __future__ import annotations

import inspect
import warnings
from typing import Optional

import torch


def build_bnb_skip_modules(keep_lm_head_in_fp16: bool = False) -> list[str]:
    skip_modules = ["mm_projector", "consistency_outcome_head"]
    if keep_lm_head_in_fp16:
        skip_modules.append("lm_head")
    return skip_modules


def _parameter_ids(module: Optional[torch.nn.Module]) -> set[int]:
    if module is None:
        return set()
    return {id(param) for param in module.parameters(recurse=True)}


def _large_embedding_parameter_ids(model: torch.nn.Module) -> set[int]:
    parameter_ids: set[int] = set()
    if hasattr(model, "get_input_embeddings"):
        parameter_ids.update(_parameter_ids(model.get_input_embeddings()))
    if hasattr(model, "get_output_embeddings"):
        parameter_ids.update(_parameter_ids(model.get_output_embeddings()))
    return parameter_ids


def prepare_model_for_kbit_training_oom_safe(
    model: torch.nn.Module,
    use_gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: Optional[dict] = None,
    skip_large_embedding_upcast: bool = False,
) -> torch.nn.Module:
    """Prepare a k-bit model while optionally keeping huge embeddings in fp16/bf16.

    PEFT's default helper casts all non-Params4bit fp16/bf16 parameters to fp32.
    For Qwen-sized vocabularies that can require a 2GB allocation for the input
    embedding alone, which is too large on 8GB cards. This mirrors PEFT's helper
    but can skip the input/output embedding matrices.
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    is_aqlm_quantized = getattr(model, "quantization_method", None) == "aqlm"
    is_eetq_quantized = getattr(model, "quantization_method", None) == "eetq"
    is_torchao_quantized = getattr(model, "quantization_method", None) == "torchao"
    is_hqq_quantized = getattr(model, "quantization_method", None) == "hqq" or getattr(model, "hqq_quantized", False)

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    for param in model.parameters():
        param.requires_grad = False

    skip_parameter_ids = _large_embedding_parameter_ids(model) if skip_large_embedding_upcast else set()
    if (
        not is_gptq_quantized
        and not is_aqlm_quantized
        and not is_eetq_quantized
        and not is_hqq_quantized
        and not is_torchao_quantized
    ):
        for param in model.parameters():
            if id(param) in skip_parameter_ids:
                continue
            if (
                (param.dtype == torch.float16) or (param.dtype == torch.bfloat16)
            ) and param.__class__.__name__ != "Params4bit":
                param.data = param.data.to(torch.float32)

    if (
        loaded_in_kbit
        or is_gptq_quantized
        or is_aqlm_quantized
        or is_eetq_quantized
        or is_hqq_quantized
        or is_torchao_quantized
    ) and use_gradient_checkpointing:
        if "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
            inspect.signature(model.gradient_checkpointing_enable).parameters
        )
        if not supports_gc_kwargs and len(gradient_checkpointing_kwargs) > 0:
            warnings.warn(
                "gradient_checkpointing_kwargs is not supported in this version of transformers. "
                "The passed kwargs will be ignored.",
                FutureWarning,
            )

        model.gradient_checkpointing_enable(
            **(
                {}
                if not supports_gc_kwargs
                else {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs}
            )
        )

    return model

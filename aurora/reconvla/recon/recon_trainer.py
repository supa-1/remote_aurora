import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled, # to check if SageMaker model parallelism is enabled
    get_parameter_names, # to get the names of the parameters of the model, used for setting up the optimizer
    has_length, # to check if the dataset has a length (i.e., implements __len__)
    ALL_LAYERNORM_LAYERS, # a constant that contains the names of all layer normalization layers, used for setting up the optimizer
    logger, # to log information during training and optimization setup
)
from typing import List, Optional

def safe_batch_decode(tokenizer, input_ids_batch, skip_special_tokens=True):
    decoded_texts = []
    for i, input_ids in enumerate(input_ids_batch):
        try:
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            tokens = [t for t in tokens if t is not None]
            text = tokenizer.convert_tokens_to_string(tokens)
            if skip_special_tokens:
                # manually remove special tokens 
                special_tokens = tokenizer.all_special_tokens
                for sp in special_tokens:
                    text = text.replace(sp, "")
            decoded_texts.append(text)
        except Exception as e:
            print(f"[Decode Error @ Sample {i}]: {e}")
            decoded_texts.append("[DECODE ERROR]")
    return decoded_texts

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=generator)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=generator)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator) # shuffle the indices to add some randomness(genenrator make sure can be achieved)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class ReconTrainer(Trainer):

    def _wrap_model(self, *args, **kwargs):
        """兼容 transformers 新版对 accelerate.unwrap_model 的参数调用差异。

        现象：
        - 新版 transformers 在内部会调用
          `accelerator.unwrap_model(model, keep_torch_compile=False)`。
        - 旧版 accelerate 的 `unwrap_model` 不接受该关键字参数。

        处理：
        - 首先按原逻辑调用 super()._wrap_model。
        - 若命中该签名不兼容错误，则临时包装 unwrap_model，丢弃
          `keep_torch_compile` 参数后重试。
        """
        try:
            return super()._wrap_model(*args, **kwargs)
        except TypeError as e:
            if "keep_torch_compile" not in str(e):
                raise

            unwrap_fn = getattr(self.accelerator, "unwrap_model", None)
            if unwrap_fn is None:
                raise

            def _unwrap_compat(*u_args, **u_kwargs):
                u_kwargs.pop("keep_torch_compile", None)
                return unwrap_fn(*u_args, **u_kwargs)

            self.accelerator.unwrap_model = _unwrap_compat
            try:
                return super()._wrap_model(*args, **kwargs)
            finally:
                self.accelerator.unwrap_model = unwrap_fn

    def _get_train_sampler(self, train_dataset=None) -> Optional[torch.utils.data.Sampler]:
        """兼容新旧 transformers 的 sampler 签名。

        说明：
        - 旧版 Trainer 调用: _get_train_sampler(self)
        - 新版 Trainer 调用: _get_train_sampler(self, train_dataset)
        这里统一兼容两种调用方式，并保持原有采样逻辑不变。
        """
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None or not has_length(dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = dataset.modality_lengths
            if not hasattr(self, "_length_grouped_generator") or self._length_grouped_generator is None:
                seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
                generator = torch.Generator()
                generator.manual_seed(int(seed))
                self._length_grouped_generator = generator
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                generator=self._length_grouped_generator,
                group_by_modality=True,
            )
        else:
            try:
                return super()._get_train_sampler(dataset)
            except TypeError:
                # 回退兼容旧版 transformers
                return super()._get_train_sampler()

    def training_step(self, model: nn.Module, inputs, num_items_in_batch=None):
        """兼容 optimizer 不含 train() 方法的情况。

        新版 transformers 在 `training_step` 内可能调用 `self.optimizer.train()`，
        但部分 torch/accelerate 组合下底层 AdamW 没有该方法。
        这里仅在缺失时注入 no-op，不改变原训练行为。
        """
        opt = getattr(self, "optimizer", None)
        if opt is not None and not hasattr(opt, "train"):
            setattr(opt, "train", lambda: None)

        # accelerate 优化器包装器内部常访问 `optimizer.optimizer`。
        inner_opt = getattr(opt, "optimizer", None) if opt is not None else None
        if inner_opt is not None and not hasattr(inner_opt, "train"):
            setattr(inner_opt, "train", lambda: None)

        # 兼容新旧 transformers:
        # - 新版: Trainer.training_step(model, inputs, num_items_in_batch)
        # - 旧版: Trainer.training_step(model, inputs)
        if num_items_in_batch is None:
            return super().training_step(model, inputs)

        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except TypeError as e:
            if "positional arguments" in str(e) or "unexpected" in str(e):
                return super().training_step(model, inputs)
            raise

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.mm_projector_lr is not None:
                lr_mapper["mm_projector"] = self.args.mm_projector_lr
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper["vision_tower"] = self.args.mm_vision_tower_lr
            if self.args.mm_inv_projector_lr is not None:
                lr_mapper["mm_inv_projector"] = self.args.mm_inv_projector_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            print(self.optimizer)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in']) # save to find the special image tokens
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

            # Only save Inv adapter
            keys_to_match = ['mm_inv_projector']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_inv_projector.bin'))
        else:
            super(ReconTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        print(f"=> saving to {output_dir}")
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(ReconTrainer, self)._save(output_dir, state_dict)

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        log_items = {}
        if outputs.get('vm_loss', None) is not None:
            assert outputs.get('lm_loss', None) is not None

            vm_loss = outputs['vm_loss']
            lm_loss = outputs['lm_loss']
            log_items.update({
                "vm_loss": round(vm_loss.item(), 4),
                "lm_loss": round(lm_loss.item(), 4),
            })

        if outputs.get('text_recon_loss', None) is not None:
            text_recon_loss = outputs['text_recon_loss']
            log_items["text_recon_loss"] = round(text_recon_loss.item(), 4)

        if outputs.get('consistency_aux_loss', None) is not None:
            consistency_aux_loss = outputs['consistency_aux_loss']
            log_items["consistency_aux_loss"] = round(consistency_aux_loss.item(), 4)

        if log_items and self.state.global_step % (self.args.logging_steps * self.args.gradient_accumulation_steps) == 0:
            self.log(log_items)

        return (loss, outputs) if return_outputs else loss

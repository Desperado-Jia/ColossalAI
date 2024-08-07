from typing import ContextManager, List, Optional, Tuple, Union
from typing_extensions import Self
from abc import ABC, abstractmethod
from contextlib import nullcontext
from enum import unique

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

import torch
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel

from colossalai.booster import Booster
from colossalai.booster.plugin import Plugin, LowLevelZeroPlugin, GeminiPlugin, HybridParallelPlugin
from colossalai.lazy.lazy_init import LazyInitContext
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR

from ..data.collator import DataCollatorForLanguageModeling
from ..data.dataset import load_tokenized_dataset
from ..data.sampler import SkippableSampler
from ..models import ChatLLM, ChatLLMConfig, ChatLLMType, fetch_llm_cls
from ..utils.generic import ExplicitEnum, IGNORE_INDEX, get_current_device

from accelerate import DeepSpeedPlugin
from transformers import TrainingArguments


@unique
class ParallelismStrategy(ExplicitEnum):
    """Acceptable values of parallelism strategies for LLMs/VLMs' training."""

    ZERO = "zero"


class TrainerArguments(BaseSettings):
    """A class used to represent all arguments for LLMs/VLMs training."""

    datasets: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Paths of tokenized training sample collections."
    )
    llm: Union[str, ChatLLMType] = Field(
        default=ChatLLMType.LLAMA3,
        description="Type of large language model, it must be a registered name in models."
    )
    pretrained_model_name_or_path: str = Field(
        default=...,
        description="A path to a directory containing configuration files, vocabulary files and "
                    "model weights used for chat LLM."
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Whether or not to allow for custom models defined on the Hub in their own modeling files. "
                    "This option should only be set to `True` for repositories you trust and in which "
                    "you have read the code, as it will execute code present on the Hub on your local machine."
    )
    output_dir: str = Field(
        default="./outputs",
        description="The output directory where the model predictions and checkpoints will be written."
    )
    resume_from_checkpoint: Optional[str] = Field(
        default=None,
        description="The path to a folder with a valid checkpoint for your model."
    )
    strategy: Union[str, ParallelismStrategy] = Field(
        default=ParallelismStrategy.ZERO,
        description="Parallel strategy for model training, for booster selection and instantiation."
    )
    zero_stage: int = Field(
        default=2,
        description="Possible options are 0,1,2,3; Default will be taken from environment variable. "
                    "This field is only valid when the value of `strategy` is \"zero\"."
    )
    cpu_offload: bool = Field(
        default=False,
        description=""
    )
    max_sequence_length: int = Field(
        default=4096,
        description="Maximum length of each tokenized (and packed if needed) sequence."
    )
    max_epochs: int = Field(
        default=1,
        description="Total number of training epochs to perform."
    )
    max_steps: Optional[int] = Field(
        default=None,
        description="If set to a positive number, the total number of training steps to perform. "
                    "Overrides `max_epochs`. For a finite dataset, training is reiterated through the dataset "
                    "(if all data is exhausted) until `max_steps` is reached."
    )
    per_device_train_batch_size: int = Field(
        default=8,
        description="The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training."
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of updates steps to accumulate the gradients for, before performing a backward/update pass."
    )
    bf16: bool = Field(
        default=True,
        description="Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
                    "architecture or using CPU (use_cpu) or Ascend NPU."
    )
    fp16: bool = Field(
        default=False,
        description="Whether to use fp16 (mixed) precision instead of 32-bit."
    )
    max_grad_norm: float = Field(
        default=1.0,
        description="Max gradient normalization scale for gradient clipping."
    )
    gradient_checkpointing: bool = Field(
        default=False,
        description="If True, use gradient checkpointing to save memory at the expense of slower backward pass."
    )
    ignore_index: int = Field(
        default=IGNORE_INDEX,
        description="Specifies a target value that is ignored and does not contribute to the "
                    "gradient of loss function in PyTorch."
    )
    dataloader_drop_last: bool = Field(
        default=True,
        description="Drop the last incomplete batch if it is not divisible by the batch size."
    )
    lr: float = Field(
        default=1e-6,
        description="The initial learning rate for [`AdamW`] optimizer."
    )
    lower_limit_lr: Optional[float] = Field(
        default=None,
        description="Lower limit of the learning rate."
    )
    lower_limit_lr_ratio: Optional[float] = Field(
        default=None,
        description="Ratio of the lower limit of the learning rate."
    )
    adam_beta1: float = Field(
        default=0.9,
        description="The beta1 hyperparameter for the [`AdamW`] optimizer."
    )
    adam_beta2: float = Field(
        default=0.999,
        description="The beta2 hyperparameter for the [`AdamW`] optimizer."
    )
    adam_epsilon: float = Field(
        default=1e-8,
        description="The epsilon hyperparameter for the [`AdamW`] optimizer."
    )
    weight_decay: float = Field(
        default=0.0,
        description="The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights "
                    "in [`AdamW`] optimizer."
    )
    warmup_steps: Optional[int] = Field(
        default=None,
        description=""
    )
    warmup_ratio: float = Field(
        default=0.0,
        description=""
    )
    seed: int = Field(
        default=42,
        description="Random seed that will be set at the beginning of training. To ensure reproducibility across runs."
    )
    initial_scale: float = Field(
        default=2**16,
        description="Initial scale used by Booster plugins."
    )

    @model_validator(mode="after")
    def verify_warmup(self) -> Self:
        if not self.warmup_steps and not self.warmup_ratio:
            raise RuntimeError()
        return self

    def setup_precision(self, return_torch_dtype: bool = False) -> Union[str, torch.dtype]:
        if not self.bf16 and not self.fp16:
            raise RuntimeError("Please make sure that only one of `bf16` and `fp16` is valid.")
        if self.bf16:
            # bf16 has a higher setting priority.
            return torch.bfloat16 if return_torch_dtype else "bf16"
        return torch.float16 if return_torch_dtype else "fp16"

    def setup_booster(self) -> Booster:
        if self.strategy == ParallelismStrategy.ZERO:
            plugin = LowLevelZeroPlugin
            kwargs = {
                "stage": self.zero_stage,
                "precision": self.setup_precision(return_torch_dtype=False),
                "initial_scale": self.initial_scale,
                "max_norm": self.max_grad_norm,
            }
            # TODO
        else:
            raise NotImplementedError()
        return Booster(plugin=plugin(**kwargs))

    def setup_optimization(self,
                           model: torch.nn.Module,
                           max_steps: int) -> Tuple[Optimizer, LRScheduler]:
        opt_model = self.unwrap_model(model=model)
        optimizer = HybridAdam(
            model_params=filter(lambda p: p.requires_grad, opt_model.parameters()),
            lr=self.lr,
            bias_correction=True,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon,
            weight_decay=self.weight_decay,
            adamw_mode=True
        )
        lr_scheduler = CosineAnnealingWarmupLR(
            optimizer=optimizer, total_steps=max_steps,
            warmup_steps=self.warmup_steps if self.warmup_steps else self.warmup_ratio * max_steps,
            eta_min=self.lr * self.lower_limit_lr_ratio if self.lower_limit_lr_ratio else self.lower_limit_lr  # TODO: verify
        )
        return optimizer, lr_scheduler

    @staticmethod
    def init_model_context_manager(booster: Booster) -> ContextManager:
        if isinstance(booster.plugin, (GeminiPlugin, HybridParallelPlugin)):
            return LazyInitContext(default_device=get_current_device())
        return nullcontext()

    @staticmethod
    def grad_accu_context_manager(updating: bool) -> ContextManager:
        raise NotImplementedError()

    @classmethod
    def unwrap_model(cls, model: torch.nn.Module) -> torch.nn.Module:
        """
        Recursively unwraps a model from potential wrappers (as used in distributed training).
        """
        if not hasattr(model, "module"):
            return model
        return cls.unwrap_model(model)

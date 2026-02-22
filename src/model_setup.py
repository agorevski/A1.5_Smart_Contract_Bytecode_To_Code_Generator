"""
Llama 3.2 3B Model Setup with LoRA Configuration

This module implements the model architecture and training setup as described in the paper,
including Low-Rank Adaptation (LoRA) fine-tuning with rank 16 targeting specific layers.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)
from datasets import Dataset as HFDataset
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for the Llama 3.2 3B model setup."""

    model_name: str = "meta-llama/Llama-3.2-3B"
    max_sequence_length: int = 2048  # Practical training length; paper max is 20000
    lora_rank: int = 16  # As specified in paper
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    use_quantization: bool = True
    load_in_4bit: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            # Target query, key, value, and projection layers as mentioned in paper
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

    def to_dict(self) -> dict:
        """Serialize config to a JSON-safe dictionary."""
        return {
            "model_name": self.model_name,
            "max_sequence_length": self.max_sequence_length,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "use_quantization": self.use_quantization,
            "load_in_4bit": self.load_in_4bit,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Deserialize config from a dictionary."""
        known_keys = {
            "model_name",
            "max_sequence_length",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "use_quantization",
            "load_in_4bit",
        }
        filtered = {k: v for k, v in d.items() if k in known_keys}
        return cls(**filtered)


class SmartContractDataset(Dataset):
    """
    Dataset class for TAC-to-Solidity function pairs.

    Implements the custom formatting template mentioned in the paper
    to clearly delineate TAC input from target Solidity output.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        template_format: str = "alpaca",
    ):
        """Initialize the dataset with training data.

        Args:
            data_path: Path to the JSONL file containing TAC-to-Solidity pairs.
            tokenizer: Pre-initialized tokenizer for encoding text.
            max_length: Maximum sequence length for tokenization.
            template_format: Prompt template format ('alpaca' or 'simple').
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_format = template_format
        self.data = self._load_data(data_path)

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load dataset from JSONL file."""
        data = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    data.append(item)
        return data

    def _format_prompt(
        self, tac_input: str, solidity_output: str, metadata: Dict
    ) -> str:
        """Format the training example using the template described in the paper."""
        if self.template_format == "alpaca":
            instruction = "Convert the following Three-Address Code (TAC) representation to readable Solidity code."

            metadata_str = ""
            if metadata:
                metadata_parts = []
                if metadata.get("function_name"):
                    metadata_parts.append(f"Function: {metadata['function_name']}")
                if metadata.get("visibility"):
                    metadata_parts.append(f"Visibility: {metadata['visibility']}")
                if metadata.get("is_payable"):
                    metadata_parts.append("Payable: true")
                if metadata.get("is_view"):
                    metadata_parts.append("View/Pure: true")

                if metadata_parts:
                    metadata_str = f"{', '.join(metadata_parts)}\n\n"

            prompt = f"""### Instruction:
{instruction}

### Input:
{metadata_str}{tac_input.strip()}

### Response:
{solidity_output.strip()}"""

        elif self.template_format == "simple":
            prompt = f"""[TAC]
{tac_input.strip()}
[/TAC]

[SOLIDITY]
{solidity_output.strip()}
[/SOLIDITY]"""

        else:
            raise ValueError(f"Unknown template format: {self.template_format}")

        return prompt

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example with dynamic-length tokenization."""
        item = self.data[idx]

        prompt = self._format_prompt(
            item["input"], item["output"], item.get("metadata", {})
        )

        # Tokenize without padding — the data collator handles padding per batch
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,  # Return lists, not tensors
        )

        # For causal LM, labels == input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized


class SmartContractModelTrainer:
    """
    Main trainer class for fine-tuning Llama 3.2 3B on smart contract decompilation.
    """

    def __init__(self, config: ModelConfig, output_dir: str = "models"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.tokenizer = None
        self.model = None
        self.peft_model = None

    def _get_hf_token(self) -> Optional[str]:
        """Get HuggingFace token from environment or settings."""
        token = os.getenv("HF_TOKEN")
        if token:
            return token
        # Try settings.yaml
        settings_path = Path(__file__).parent / "settings.yaml"
        if settings_path.exists():
            import yaml

            with open(settings_path, "r") as f:
                settings = yaml.safe_load(f) or {}
            token = settings.get("HF_TOKEN")
            if token and token != "your_huggingface_token_here":
                return token
        return None

    def setup_model(
        self, force_reload: bool = False, use_deepspeed: bool = False
    ) -> Tuple[AutoTokenizer, PeftModel]:
        """Set up the Llama 3.2 3B model with LoRA configuration."""
        if self.tokenizer is not None and self.peft_model is not None and not force_reload:
            return self.tokenizer, self.peft_model

        logger.info("Setting up model with LoRA...")

        # Enable TF32 for matmuls on Ampere+ GPUs (~2x faster than FP32 precision)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        hf_token = self._get_hf_token()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right",
            token=hf_token,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Determine dtype and device based on GPU availability
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_cap = torch.cuda.get_device_capability()
            use_bf16 = gpu_cap[0] >= 8  # Ampere+ supports bf16 natively
            compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        else:
            compute_dtype = torch.float32
        model_dtype = compute_dtype

        # Configure quantization
        quantization_config = None
        if self.config.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # Determine which GPU this process owns (for DDP / multi-GPU)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Load base model
        # Use SDPA (Scaled Dot-Product Attention) for fused attention kernels
        attn_impl = None
        if has_cuda:
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"  # PyTorch 2.x built-in efficient attention

        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": model_dtype,
            "use_cache": False,
            "token": hf_token,
        }
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            # Pin the quantized model to this process's GPU
            load_kwargs["device_map"] = {"": local_rank} if has_cuda else "auto"
        elif has_cuda:
            load_kwargs["device_map"] = "auto"
        # For CPU without quantization, don't set device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs,
        )

        if self.config.use_quantization:
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

        # Auto-detect target modules for LoRA if defaults don't exist in model
        target_modules = self.config.target_modules
        model_module_names = [name for name, _ in self.model.named_modules()]
        module_name_str = " ".join(model_module_names)
        # Check if any target module is present
        valid_targets = [t for t in target_modules if t in module_name_str]
        if not valid_targets:
            # Fall back to auto-detecting linear layers
            logger.info("Default target modules not found; using auto-detection for LoRA targets")
            target_modules = "all-linear"

        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules if target_modules != self.config.target_modules else self.config.target_modules,
            bias="none",
        )

        self.peft_model = get_peft_model(self.model, lora_config)

        # Resize embeddings if needed
        if len(self.tokenizer) > self.peft_model.config.vocab_size:
            self.peft_model.resize_token_embeddings(len(self.tokenizer))

        self.peft_model.print_trainable_parameters()

        logger.info("Model setup completed successfully")
        return self.tokenizer, self.peft_model

    def create_training_arguments(
        self,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 4,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
        max_grad_norm: float = 1.0,
        do_eval: bool = True,
        train_dataset_size: int = 0,
        deepspeed_config: Optional[str] = None,
    ) -> TrainingArguments:
        """Create training arguments based on the paper's optimization strategy.
        
        Automatically adapts configuration for small datasets to ensure
        training completes successfully.
        """
        # Auto-adjust for small datasets
        effective_batch = batch_size * gradient_accumulation_steps
        if train_dataset_size > 0 and train_dataset_size < effective_batch:
            # Reduce gradient_accumulation_steps so at least 1 optimizer step runs
            gradient_accumulation_steps = max(1, train_dataset_size // batch_size)
            if gradient_accumulation_steps == 0:
                gradient_accumulation_steps = 1
            logger.info(
                f"Auto-adjusted gradient_accumulation_steps to {gradient_accumulation_steps} "
                f"for dataset size {train_dataset_size}"
            )

        # Scale warmup proportionally for small datasets
        steps_per_epoch = max(1, train_dataset_size // (batch_size * gradient_accumulation_steps))
        total_steps = steps_per_epoch * num_epochs
        if warmup_steps > total_steps // 2:
            warmup_steps = max(0, total_steps // 5)
            logger.info(f"Auto-adjusted warmup_steps to {warmup_steps}")

        # For small datasets, use epoch-based saving/eval instead of step-based
        is_small = train_dataset_size > 0 and train_dataset_size < 200
        save_strategy = "epoch" if is_small else "steps"
        logging_steps_final = max(1, min(logging_steps, total_steps)) if is_small else logging_steps

        # Determine mixed-precision strategy
        has_cuda = torch.cuda.is_available()
        use_bf16 = False
        use_fp16 = False

        # When DeepSpeed is active, align with its config to avoid conflicts
        if deepspeed_config:
            try:
                with open(deepspeed_config, "r") as _ds_f:
                    ds_cfg = json.load(_ds_f)
                if ds_cfg.get("bf16", {}).get("enabled", False):
                    use_bf16 = True
                    use_fp16 = False
                elif ds_cfg.get("fp16", {}).get("enabled", False):
                    use_fp16 = True
                    use_bf16 = False
                # else: both False, full precision
            except Exception as e:
                logger.warning(f"Could not read DeepSpeed config for precision settings: {e}")
                # Fall back to auto-detection below
                if has_cuda:
                    gpu_cap = torch.cuda.get_device_capability()
                    if gpu_cap[0] >= 8:
                        use_bf16 = True
                    else:
                        use_fp16 = True
        elif has_cuda:
            gpu_cap = torch.cuda.get_device_capability()
            if gpu_cap[0] >= 8:  # Ampere+ supports bf16 natively
                use_bf16 = True
            else:
                use_fp16 = True

        args = {
            "output_dir": str(self.output_dir / "checkpoints"),
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "optim": "adamw_torch_fused" if has_cuda else "adamw_torch",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "warmup_steps": warmup_steps,
            "lr_scheduler_type": "linear",
            "logging_steps": logging_steps_final,
            "save_strategy": save_strategy,
            "dataloader_pin_memory": True,
            "dataloader_num_workers": 4,
            "dataloader_persistent_workers": True,
            "group_by_length": True,
            "remove_unused_columns": False,
            "push_to_hub": False,
            "report_to": "none",
            "seed": 42,
            "bf16": use_bf16,
            "fp16": use_fp16,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "dataloader_drop_last": False,
        }

        # For DDP with quantized models, disable unused parameter detection
        if self.config.use_quantization and has_cuda:
            args["ddp_find_unused_parameters"] = False

        # DeepSpeed integration
        if deepspeed_config:
            args["deepspeed"] = deepspeed_config
            logger.info(f"DeepSpeed enabled with config: {deepspeed_config}")

        if save_strategy == "steps":
            args["save_steps"] = save_steps

        if do_eval:
            if is_small:
                args["eval_strategy"] = "epoch"
            else:
                args["eval_steps"] = eval_steps
                args["eval_strategy"] = "steps"
            args["load_best_model_at_end"] = True
            args["metric_for_best_model"] = "eval_loss"
            args["greater_is_better"] = False
        else:
            args["eval_strategy"] = "no"

        return TrainingArguments(**args)

    def train(
        self,
        train_dataset_path: str,
        eval_dataset_path: Optional[str] = None,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        resume_from_checkpoint: Optional[str] = None,
        deepspeed_config: Optional[str] = None,
    ) -> str:
        """Train the model on the smart contract decompilation dataset."""
        tokenizer, peft_model = self.setup_model(
            use_deepspeed=deepspeed_config is not None
        )

        logger.info("Loading training dataset...")
        train_dataset = SmartContractDataset(
            train_dataset_path,
            tokenizer,
            max_length=self.config.max_sequence_length,
        )

        eval_dataset = None
        if eval_dataset_path and Path(eval_dataset_path).exists():
            logger.info("Loading evaluation dataset...")
            eval_dataset = SmartContractDataset(
                eval_dataset_path,
                tokenizer,
                max_length=self.config.max_sequence_length,
            )

        do_eval = eval_dataset is not None

        # Custom data collator that properly pads input_ids, attention_mask, and labels
        def custom_data_collator(features):
            # Find max length in this batch
            max_len = max(len(f["input_ids"]) for f in features)
            pad_token_id = tokenizer.pad_token_id

            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []

            for f in features:
                ids = f["input_ids"]
                mask = f.get("attention_mask", [1] * len(ids))
                labels = f["labels"]
                pad_len = max_len - len(ids)

                batch_input_ids.append(ids + [pad_token_id] * pad_len)
                batch_attention_mask.append(mask + [0] * pad_len)
                # Use -100 for padded label positions so they're ignored in loss
                batch_labels.append(labels + [-100] * pad_len)

            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.long),
            }

        data_collator = custom_data_collator

        training_args = self.create_training_arguments(
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            do_eval=do_eval,
            train_dataset_size=len(train_dataset),
            deepspeed_config=deepspeed_config,
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        logger.info(
            f"Starting training on {len(train_dataset)} examples for {num_epochs} epochs..."
        )
        if resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train()

        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))

        # Save model config so load_model knows which base model to use
        config_path = final_model_path / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Training completed. Model saved to {final_model_path}")
        return str(final_model_path)

    def save_model(self, path: str):
        """Save the trained model and tokenizer."""
        if self.peft_model is None or self.tokenizer is None:
            raise ValueError("Model not initialized. Call setup_model() first.")

        save_path = Path(path)
        save_path.mkdir(exist_ok=True)

        self.peft_model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

        config_path = save_path / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: str, for_inference: bool = True) -> Tuple[AutoTokenizer, PeftModel]:
        """Load a previously trained model.

        When *for_inference* is ``True`` (default) several GPU-specific
        optimisations are applied:

        * **KV-cache enabled** — avoids recomputing attention for previous
          tokens at every generation step (major speedup).
        * **BFloat16 compute** — faster and more numerically stable on
          Ampere+ GPUs (RTX 30xx / 40xx / A100 / H100).
        * **Flash Attention 2** — fused attention kernels that are both
          faster and more memory-efficient.  Falls back to eager attention
          if the ``flash_attn`` package is not installed.
        * **torch.compile()** — JIT-compiles the model graph, fusing
          operations and reducing kernel-launch overhead.

        On CPU-only machines the model is loaded in FP32 with none of the
        GPU-specific flags.
        """
        load_path = Path(path)

        config_path = load_path / "model_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            self.config = ModelConfig.from_dict(config_dict)

        hf_token = self._get_hf_token()

        # Try loading tokenizer from saved path first, fall back to base model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(load_path), token=hf_token)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, token=hf_token
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # For batched generation the tokenizer must pad on the LEFT so that
        # the generated tokens are contiguous on the right.
        if for_inference:
            self.tokenizer.padding_side = "left"

        has_cuda = torch.cuda.is_available()

        # ---- Determine best compute dtype ----
        # BFloat16 is faster and more stable on Ampere+ (sm_80+).
        # Fall back to FP16 otherwise.
        if has_cuda:
            gpu_cap = torch.cuda.get_device_capability()
            use_bf16 = gpu_cap[0] >= 8  # Ampere = sm_80
            compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
            model_dtype = compute_dtype
        else:
            compute_dtype = torch.float32
            model_dtype = torch.float32

        quantization_config = None
        if self.config.use_quantization and has_cuda:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # ---- Flash Attention 2 ----
        attn_impl = None
        if has_cuda and for_inference:
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                logger.info("Flash Attention 2 available — enabling")
            except ImportError:
                # Try SDPA (PyTorch 2.x built-in efficient attention)
                attn_impl = "sdpa"
                logger.info("flash_attn not installed — using PyTorch SDPA")

        # ---- Build load kwargs ----
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": model_dtype,
            "token": hf_token,
        }
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        elif has_cuda:
            load_kwargs["device_map"] = "auto"
        # CPU-only: no device_map, loads to CPU automatically

        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs,
        )

        self.peft_model = PeftModel.from_pretrained(base_model, str(load_path))

        # ---- Inference-time optimisations ----
        if for_inference:
            # Enable KV cache (was disabled for training)
            if hasattr(self.peft_model.config, "use_cache"):
                self.peft_model.config.use_cache = True
                logger.info("KV cache enabled for inference")

            # torch.compile for fused kernels (PyTorch 2.0+)
            if has_cuda:
                try:
                    self.peft_model = torch.compile(
                        self.peft_model,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    logger.info("torch.compile() applied (mode=reduce-overhead)")
                except Exception as e:
                    logger.warning("torch.compile() failed, skipping: %s", e)

        device = next(self.peft_model.parameters()).device
        dtype_name = str(compute_dtype).split(".")[-1]
        logger.info(
            "Model loaded from %s (device: %s, dtype: %s, attn: %s)",
            load_path, device, dtype_name, attn_impl or "eager",
        )
        return self.tokenizer, self.peft_model


class SmartContractDecompiler:
    """High-level interface for using the trained model for decompilation."""

    # Reserve tokens for the prompt template + generation headroom
    _PROMPT_OVERHEAD_TOKENS = 80
    _MAX_INPUT_TOKENS = 1500  # leaves ~500 tokens for output within 2048

    def __init__(self, model_path: str):
        self.trainer = SmartContractModelTrainer(ModelConfig())
        self.tokenizer, self.model = self.trainer.load_model(model_path)
        self.model.eval()

    # ------------------------------------------------------------------ #
    #  Token-aware TAC truncation
    # ------------------------------------------------------------------ #

    def _count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _truncate_tac(self, tac_text: str, max_tokens: int) -> str:
        """Truncate *tac_text* to fit within *max_tokens*.

        Strategy (applied in order until the text fits):
          1. Strip comment-only lines (``// …``)
          2. Remove dead-code blocks
          3. Hard-truncate remaining lines with a marker
        """
        # Fast path — already fits
        if self._count_tokens(tac_text) <= max_tokens:
            return tac_text

        lines = tac_text.splitlines()

        # Pass 1: remove pure-comment lines (keep block headers like "block_00a2:")
        stripped = [
            ln for ln in lines
            if not ln.strip().startswith("//")
        ]
        candidate = "\n".join(stripped)
        if self._count_tokens(candidate) <= max_tokens:
            return candidate

        # Pass 2: remove dead-code blocks (block header + its indented body)
        filtered: List[str] = []
        skip = False
        for ln in stripped:
            if "Dead code" in ln or "dead code" in ln:
                skip = True
                continue
            # A new block header ends the skip region
            if skip and ln.strip().endswith(":") and not ln.strip().startswith("//"):
                skip = False
            if not skip:
                filtered.append(ln)
        candidate = "\n".join(filtered)
        if self._count_tokens(candidate) <= max_tokens:
            return candidate

        # Pass 3: hard-truncate line-by-line
        kept: List[str] = []
        running = 0
        for ln in filtered:
            ln_tokens = self._count_tokens(ln)
            if running + ln_tokens > max_tokens - 10:  # leave room for marker
                break
            kept.append(ln)
            running += ln_tokens
        kept.append("  // ... truncated (TAC too large for context window)")
        return "\n".join(kept)

    # ------------------------------------------------------------------ #
    #  Single-function decompilation (original API, now token-safe)
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self,
        tac_input: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Build a decompilation prompt from TAC + optional metadata."""
        instruction = "Convert the following Three-Address Code (TAC) representation to readable Solidity code."

        metadata_str = ""
        if metadata:
            metadata_parts = []
            if metadata.get("function_name"):
                metadata_parts.append(f"Function: {metadata['function_name']}")
            if metadata.get("visibility"):
                metadata_parts.append(f"Visibility: {metadata['visibility']}")
            if metadata.get("is_payable"):
                metadata_parts.append("Payable: true")
            if metadata.get("is_view"):
                metadata_parts.append("View/Pure: true")
            if metadata_parts:
                metadata_str = f"{', '.join(metadata_parts)}\n\n"

        # Truncate TAC to fit context window
        budget = self._MAX_INPUT_TOKENS - self._PROMPT_OVERHEAD_TOKENS
        safe_tac = self._truncate_tac(tac_input, budget)

        return f"""### Instruction:
{instruction}

### Input:
{metadata_str}{safe_tac.strip()}

### Response:
"""

    def decompile_tac_to_solidity(
        self,
        tac_input: str,
        metadata: Optional[Dict] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        do_sample: bool = False,
        repetition_penalty: float = 1.15,
    ) -> str:
        """Decompile TAC representation to Solidity code.

        The TAC input is automatically truncated if it would exceed the
        model's context window.

        Defaults to **greedy decoding** (``do_sample=False``) for maximum
        GPU throughput.  Set ``do_sample=True`` for stochastic sampling.
        """
        prompt = self._build_prompt(tac_input, metadata)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return generated_text.strip()

    # ------------------------------------------------------------------ #
    #  Batched decompilation — process multiple functions at once
    # ------------------------------------------------------------------ #

    def decompile_batch(
        self,
        tac_inputs: List[str],
        metadatas: Optional[List[Optional[Dict]]] = None,
        max_new_tokens: int = 1024,
        repetition_penalty: float = 1.15,
    ) -> List[str]:
        """Decompile multiple TAC functions in a single batched forward pass.

        This keeps the GPU saturated by processing several prompts
        simultaneously.  The tokenizer pads on the left (set during
        ``load_model``) so that generated tokens are contiguous.

        Args:
            tac_inputs: List of TAC strings, one per function.
            metadatas: Optional parallel list of metadata dicts.
            max_new_tokens: Max tokens to generate per function.
            repetition_penalty: Repetition penalty factor.

        Returns:
            List of generated Solidity strings (same order as input).
        """
        if metadatas is None:
            metadatas = [None] * len(tac_inputs)

        prompts = [
            self._build_prompt(tac, meta)
            for tac, meta in zip(tac_inputs, metadatas)
        ]

        # Tokenize as a batch with left-padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._MAX_INPUT_TOKENS + self._PROMPT_OVERHEAD_TOKENS,
        ).to(self.model.device)

        prompt_lengths = [
            (inputs["attention_mask"][i] == 1).sum().item()
            for i in range(len(prompts))
        ]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        results: List[str] = []
        for i in range(len(prompts)):
            # The input was left-padded; skip all input tokens
            input_len = inputs["input_ids"].shape[1]
            gen_ids = outputs[i][input_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(text.strip())

        return results

    # ------------------------------------------------------------------ #
    #  Contract-level decompilation (per-function pipeline)
    # ------------------------------------------------------------------ #

    def decompile_contract(
        self,
        bytecode: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Dict:
        """Decompile an entire contract by processing each function independently.

        This is the recommended entry point for large contracts.  It avoids
        the token-length explosion that occurs when the full TAC is sent
        as a single prompt.

        Args:
            bytecode: Hex-encoded EVM bytecode (with or without ``0x`` prefix).
            max_new_tokens: Maximum tokens to generate per function.
            temperature: Sampling temperature.

        Returns:
            A dict with keys:
              - ``functions``: dict mapping function name → generated Solidity
              - ``solidity``: assembled full contract string
              - ``tac_per_function``: dict mapping function name → TAC used
              - ``analysis``: metadata about the analysis
        """
        from src.bytecode_analyzer import BytecodeAnalyzer

        import time
        t0 = time.time()

        analyzer = BytecodeAnalyzer(bytecode)
        func_tac_map = analyzer.generate_per_function_tac()

        tac_time = time.time() - t0

        num_instructions = len(analyzer.instructions)
        num_blocks = len(analyzer.basic_blocks)
        num_functions = len(analyzer.functions)

        logger.info(
            "Contract analysis: %d instructions, %d blocks, %d functions",
            num_instructions, num_blocks, num_functions,
        )

        # Decompile each function independently
        t1 = time.time()
        function_solidity: Dict[str, str] = {}
        function_errors: Dict[str, str] = {}

        for fname, tac_str in func_tac_map.items():
            func_meta = {}
            func_obj = analyzer.functions.get(fname)
            if func_obj:
                func_meta = {
                    "function_name": func_obj.name,
                    "visibility": func_obj.visibility,
                    "is_payable": func_obj.is_payable,
                    "is_view": func_obj.is_view,
                }

            try:
                sol = self.decompile_tac_to_solidity(
                    tac_str,
                    metadata=func_meta,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                function_solidity[fname] = sol
                logger.info("Decompiled %s (%d TAC tokens → %d output chars)",
                                 fname, self._count_tokens(tac_str), len(sol))
            except Exception as e:
                logger.error("Failed to decompile %s: %s", fname, e)
                function_errors[fname] = str(e)
                function_solidity[fname] = f"// Decompilation failed: {e}"

        gen_time = time.time() - t1

        # Assemble into a single contract
        assembled = self._assemble_contract(function_solidity, analyzer)

        return {
            "functions": function_solidity,
            "solidity": assembled,
            "tac_per_function": func_tac_map,
            "analysis": {
                "num_instructions": num_instructions,
                "num_basic_blocks": num_blocks,
                "num_functions": num_functions,
                "tac_generation_time_s": round(tac_time, 3),
                "solidity_generation_time_s": round(gen_time, 3),
                "function_errors": function_errors,
            },
        }

    @staticmethod
    def _assemble_contract(
        function_solidity: Dict[str, str],
        analyzer: "BytecodeAnalyzer",
    ) -> str:
        """Combine per-function Solidity outputs into a single contract string."""
        lines: List[str] = [
            "// SPDX-License-Identifier: UNKNOWN",
            "pragma solidity ^0.8.0;",
            "",
            "/// @notice Decompiled contract",
            f"/// @dev {len(function_solidity)} function(s) recovered from bytecode",
            "contract DecompiledContract {",
            "",
        ]

        for fname, sol in function_solidity.items():
            func_obj = analyzer.functions.get(fname)
            selector = func_obj.selector if func_obj else None
            if selector:
                lines.append(f"    // Function selector: {selector}")
            lines.append(f"    // {fname}")

            # Indent each line of the generated Solidity
            for sol_line in sol.splitlines():
                lines.append(f"    {sol_line}")
            lines.append("")

        lines.append("}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DPO Training Support (Smart-LLaMA-DPO approach, 2506.18245v1)
# ---------------------------------------------------------------------------


class DPOTrainingConfig:
    """Configuration for Direct Preference Optimization training."""

    def __init__(
        self,
        beta: float = 0.1,
        learning_rate: float = 1e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        max_length: int = 2048,
        max_prompt_length: int = 1024,
        gradient_accumulation_steps: int = 4,
    ):
        self.beta = beta
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.gradient_accumulation_steps = gradient_accumulation_steps


class DPODatasetBuilder:
    """
    Builds DPO preference pairs from decompilation outputs.

    Creates (prompt, chosen, rejected) triples where:
    - prompt: TAC input
    - chosen: high-quality decompiled Solidity (verified source)
    - rejected: lower-quality output (baseline model output)
    """

    @staticmethod
    def build_preference_pairs(
        dataset: List[Dict],
        baseline_outputs: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build DPO preference pairs from a dataset.

        Each item in dataset should have 'input' (TAC) and 'output' (ground truth Solidity).
        baseline_outputs provides the rejected completions (model's own outputs before DPO).
        """
        pairs = []
        for i, item in enumerate(dataset):
            prompt = item.get("input", "")
            chosen = item.get("output", "")

            if baseline_outputs and i < len(baseline_outputs):
                rejected = baseline_outputs[i]
            else:
                # Create a degraded version as rejected
                rejected = DPODatasetBuilder._degrade_output(chosen)

            if prompt and chosen and rejected and chosen != rejected:
                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                })

        return pairs

    @staticmethod
    def _degrade_output(solidity: str) -> str:
        """Create a degraded version of Solidity for rejected output."""
        degraded = solidity
        # Remove comments
        lines = [l for l in degraded.split("\n") if not l.strip().startswith("//")]
        degraded = "\n".join(lines)
        # Replace meaningful names with generic ones
        degraded = degraded.replace("owner", "var1")
        degraded = degraded.replace("balance", "var2")
        degraded = degraded.replace("transfer", "func1")
        return degraded


def main():
    """Example usage of the model training pipeline."""
    logging.basicConfig(level=logging.INFO)

    config = ModelConfig(
        max_sequence_length=2048,
        lora_rank=16,
        use_quantization=True,
    )

    trainer = SmartContractModelTrainer(config)
    tokenizer, model = trainer.setup_model()

    print("Model setup completed successfully!")
    print(f"Trainable parameters: {model.get_nb_trainable_parameters()}")


if __name__ == "__main__":
    main()
"""
Llama 3.2 3B Model Setup with LoRA Configuration

This module implements the model architecture and training setup as described in the paper,
including Low-Rank Adaptation (LoRA) fine-tuning with rank 16 targeting specific layers.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
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
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from datasets import Dataset as HFDataset
import numpy as np
from tqdm import tqdm

@dataclass
class ModelConfig:
    """Configuration for the Llama 3.2 3B model setup."""
    model_name: str = "meta-llama/Llama-3.2-3B"
    max_sequence_length: int = 20000  # As specified in paper
    lora_rank: int = 16  # As specified in paper
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    use_quantization: bool = True
    quantization_config: Optional[Dict] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            # Target query, key, value, and projection layers as mentioned in paper
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        if self.quantization_config is None:
            self.quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4"
            }

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
        max_length: int = 20000,
        template_format: str = "alpaca"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_format = template_format
        self.data = self._load_data(data_path)
        
        # Add special tokens for function boundaries and metadata
        self._add_special_tokens()
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load dataset from JSONL file."""
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        return data
    
    def _add_special_tokens(self):
        """Add special tokens for smart contract decompilation."""
        special_tokens = [
            "<TAC_START>", "<TAC_END>",
            "<SOLIDITY_START>", "<SOLIDITY_END>",
            "<FUNCTION_START>", "<FUNCTION_END>",
            "<METADATA_START>", "<METADATA_END>"
        ]
        
        # Add tokens if they don't exist
        new_tokens = []
        for token in special_tokens:
            if token not in self.tokenizer.get_vocab():
                new_tokens.append(token)
        
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
    
    def _format_prompt(self, tac_input: str, solidity_output: str, metadata: Dict) -> str:
        """
        Format the training example using the template described in the paper.
        
        Creates clear delineation between TAC input and Solidity output with
        special tokens for function boundaries and metadata.
        """
        if self.template_format == "alpaca":
            # Alpaca-style formatting
            instruction = "Convert the following Three-Address Code (TAC) representation to readable Solidity code."
            
            # Include metadata if available
            metadata_str = ""
            if metadata:
                metadata_parts = []
                if metadata.get('function_name'):
                    metadata_parts.append(f"Function: {metadata['function_name']}")
                if metadata.get('visibility'):
                    metadata_parts.append(f"Visibility: {metadata['visibility']}")
                if metadata.get('is_payable'):
                    metadata_parts.append("Payable: true")
                if metadata.get('is_view'):
                    metadata_parts.append("View/Pure: true")
                
                if metadata_parts:
                    metadata_str = f"<METADATA_START>\n{', '.join(metadata_parts)}\n<METADATA_END>\n\n"
            
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{metadata_str}<TAC_START>
{tac_input.strip()}
<TAC_END>

### Response:
<SOLIDITY_START>
{solidity_output.strip()}
<SOLIDITY_END>"""
        
        elif self.template_format == "simple":
            # Simple format with clear delimiters
            prompt = f"""<FUNCTION_START>
<TAC_START>
{tac_input.strip()}
<TAC_END>

<SOLIDITY_START>
{solidity_output.strip()}
<SOLIDITY_END>
<FUNCTION_END>"""
        
        else:
            raise ValueError(f"Unknown template format: {self.template_format}")
        
        return prompt
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        item = self.data[idx]
        
        # Format the prompt
        prompt = self._format_prompt(
            item['input'], 
            item['output'], 
            item.get('metadata', {})
        )
        
        # Tokenize
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # For causal language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["labels"].squeeze()
        }

class SmartContractModelTrainer:
    """
    Main trainer class for fine-tuning Llama 3.2 3B on smart contract decompilation.
    
    Implements the training approach described in the paper including LoRA adaptation,
    gradient checkpointing, and specialized optimization strategy.
    """
    
    def __init__(self, config: ModelConfig, output_dir: str = "models"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.peft_model = None
    
    def setup_model(self, force_reload: bool = False) -> Tuple[AutoTokenizer, PeftModel]:
        """
        Set up the Llama 3.2 3B model with LoRA configuration.
        
        Args:
            force_reload: Whether to force reload the model
            
        Returns:
            Tuple of (tokenizer, peft_model)
        """
        if self.tokenizer is None or self.peft_model is None or force_reload:
            self.logger.info("Setting up Llama 3.2 3B model with LoRA...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization for memory efficiency
            if self.config.use_quantization:
                quantization_config = BitsAndBytesConfig(**self.config.quantization_config)
            else:
                quantization_config = None
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                use_cache=False  # Disable for training
            )
            
            # Prepare model for training if using quantization
            if self.config.use_quantization:
                self.model = prepare_model_for_kbit_training(self.model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias="none"
            )
            
            # Apply LoRA to the model
            self.peft_model = get_peft_model(self.model, lora_config)
            
            # Resize embeddings if new tokens were added
            if len(self.tokenizer) > self.peft_model.config.vocab_size:
                self.peft_model.resize_token_embeddings(len(self.tokenizer))
            
            # Enable gradient checkpointing for memory efficiency
            self.peft_model.gradient_checkpointing_enable()
            
            # Print trainable parameters
            self.peft_model.print_trainable_parameters()
            
            self.logger.info("Model setup completed successfully")
        
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
        max_grad_norm: float = 1.0
    ) -> TrainingArguments:
        """
        Create training arguments based on the paper's optimization strategy.
        
        Implements AdamW optimizer with learning rate schedule including
        warmup period followed by linear decay.
        """
        return TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            warmup_steps=warmup_steps,
            lr_scheduler_type="linear",
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=["tensorboard"],
            seed=42,
            # Memory optimization
            fp16=True,
            gradient_checkpointing=True,
            dataloader_drop_last=True,
        )
    
    def train(
        self,
        train_dataset_path: str,
        eval_dataset_path: Optional[str] = None,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        resume_from_checkpoint: Optional[str] = None
    ) -> str:
        """
        Train the model on the smart contract decompilation dataset.
        
        Args:
            train_dataset_path: Path to training dataset (JSONL format)
            eval_dataset_path: Optional path to evaluation dataset
            batch_size: Training batch size
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            resume_from_checkpoint: Optional checkpoint to resume from
            
        Returns:
            Path to the final model
        """
        # Setup model if not already done
        tokenizer, peft_model = self.setup_model()
        
        # Create datasets
        self.logger.info("Loading training dataset...")
        train_dataset = SmartContractDataset(
            train_dataset_path, 
            tokenizer, 
            max_length=self.config.max_sequence_length
        )
        
        eval_dataset = None
        if eval_dataset_path:
            self.logger.info("Loading evaluation dataset...")
            eval_dataset = SmartContractDataset(
                eval_dataset_path,
                tokenizer,
                max_length=self.config.max_sequence_length
            )
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # We're doing causal language modeling
        )
        
        # Create training arguments
        training_args = self.create_training_arguments(
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs
        )
        
        # Create trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Start training
        self.logger.info("Starting training...")
        if resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train()
        
        # Save final model
        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))
        
        self.logger.info(f"Training completed. Model saved to {final_model_path}")
        return str(final_model_path)
    
    def save_model(self, path: str):
        """Save the trained model and tokenizer."""
        if self.peft_model is None or self.tokenizer is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        save_path = Path(path)
        save_path.mkdir(exist_ok=True)
        
        # Save the LoRA adapter
        self.peft_model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        
        # Save configuration
        config_path = save_path / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str) -> Tuple[AutoTokenizer, PeftModel]:
        """Load a previously trained model."""
        load_path = Path(path)
        
        # Load configuration
        config_path = load_path / "model_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                self.config = ModelConfig(**config_dict)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(load_path))
        
        # Load base model
        if self.config.use_quantization:
            quantization_config = BitsAndBytesConfig(**self.config.quantization_config)
        else:
            quantization_config = None
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load LoRA adapter
        self.peft_model = PeftModel.from_pretrained(base_model, str(load_path))
        
        self.logger.info(f"Model loaded from {load_path}")
        return self.tokenizer, self.peft_model

class SmartContractDecompiler:
    """
    High-level interface for using the trained model for decompilation.
    """
    
    def __init__(self, model_path: str):
        self.trainer = SmartContractModelTrainer(ModelConfig())
        self.tokenizer, self.model = self.trainer.load_model(model_path)
        self.model.eval()
    
    def decompile_tac_to_solidity(
        self, 
        tac_input: str, 
        metadata: Optional[Dict] = None,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        do_sample: bool = True
    ) -> str:
        """
        Decompile TAC representation to Solidity code.
        
        Args:
            tac_input: Three-address code representation
            metadata: Optional metadata about the function
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated Solidity code
        """
        # Format the input using the same template as training
        dataset = SmartContractDataset.__new__(SmartContractDataset)
        dataset.tokenizer = self.tokenizer
        dataset.template_format = "alpaca"
        
        # Create a dummy output for formatting (will be ignored)
        prompt = dataset._format_prompt(tac_input, "", metadata or {})
        
        # Remove the response part for inference
        prompt = prompt.split("### Response:")[0] + "### Response:\n<SOLIDITY_START>\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.encode("<SOLIDITY_END>")[0]
            )
        
        # Decode and extract Solidity code
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract Solidity code between markers
        if "<SOLIDITY_START>" in generated_text and "<SOLIDITY_END>" in generated_text:
            start_idx = generated_text.find("<SOLIDITY_START>") + len("<SOLIDITY_START>")
            end_idx = generated_text.find("<SOLIDITY_END>")
            solidity_code = generated_text[start_idx:end_idx].strip()
        else:
            # Fallback: take everything after the prompt
            solidity_code = generated_text[len(prompt):].strip()
        
        return solidity_code

def main():
    """Example usage of the model training pipeline."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create model configuration
    config = ModelConfig(
        max_sequence_length=20000,
        lora_rank=16,
        use_quantization=True
    )
    
    # Initialize trainer
    trainer = SmartContractModelTrainer(config)
    
    # Setup model
    tokenizer, model = trainer.setup_model()
    
    # Train model (assuming you have a dataset)
    # model_path = trainer.train("data/train_dataset.jsonl", "data/eval_dataset.jsonl")
    
    print("Model setup completed successfully!")
    print(f"Model has {model.num_parameters()} parameters")
    print(f"Trainable parameters: {model.get_nb_trainable_parameters()}")

if __name__ == "__main__":
    main()

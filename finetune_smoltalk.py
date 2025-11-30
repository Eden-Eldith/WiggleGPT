"""
WiggleGPT v3 - finetune_smoltalk.py
===================================

Fine-tuning Script for WiggleGPT using HuggingFace Trainer API
==============================================================

This script fine-tunes a pre-trained WiggleGPT model on the SmolTalk2 dataset
for instruction following. It adapts the bio-inspired model (oscillating neurons)
to conversational/instruction tasks.

Usage:
    python finetune_smoltalk.py
    python finetune_smoltalk.py --config config_finetune.py

Features:
- Loads pre-trained WiggleGPT checkpoint (with oscillating neurons)
- Uses SmolTalk2 SFT dataset from HuggingFace
- HuggingFace Trainer API for training loop
- Supports gradient checkpointing for memory efficiency
- Chat template formatting for conversations
- Configurable via command line or config file
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from datasets import load_dataset
import tiktoken
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# Import the WiggleGPT model
from model_bio import GPTConfig, GPT


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning WiggleGPT on SmolTalk"""
    
    # Pretrained model checkpoint
    pretrained_ckpt: str = "out-wigglegpt-pure-124m/ckpt.pt"
    
    # Output directory for fine-tuned model
    output_dir: str = "out-wigglegpt-smoltalk"
    
    # Dataset configuration
    dataset_name: str = "HuggingFaceTB/smoltalk2"
    dataset_subset: str = "SFT"  # SFT, Mid, or Preference
    dataset_split: str = "smoltalk-smollm3_smol-magpie-ultra_no_think"  # specific split
    max_samples: Optional[int] = None  # Limit samples for testing (None = use all)
    
    # Training hyperparameters
    num_train_epochs: float = 3.0
    max_steps: int = -1  # -1 means use num_train_epochs
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    
    # Learning rate configuration
    learning_rate: float = 2e-5  # Lower than pretraining for fine-tuning
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # Sequence length
    max_seq_length: int = 1024
    
    # Regularization
    dropout: float = 0.1  # Add dropout for fine-tuning
    
    # Optimization
    bf16: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = True
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    
    # Misc
    seed: int = 42
    dataloader_num_workers: int = 4


# =============================================================================
# WRAPPER FOR HUGGINGFACE TRAINER COMPATIBILITY
# =============================================================================

class WiggleGPTConfig(PretrainedConfig):
    """HuggingFace-compatible config wrapper for WiggleGPT"""
    model_type = "wigglegpt"
    
    def __init__(
        self,
        vocab_size: int = 50304,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        block_size: int = 1024,
        dropout: float = 0.0,
        bias: bool = False,
        use_bio_mlp: bool = True,
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        gradient_checkpointing: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias
        self.use_bio_mlp = use_bio_mlp
        self.use_rmsnorm = use_rmsnorm
        self.use_rope = use_rope
        self.gradient_checkpointing = gradient_checkpointing
        
        # Aliases for HuggingFace compatibility
        self.hidden_size = n_embd
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.max_position_embeddings = block_size


class WiggleGPTForCausalLM(PreTrainedModel):
    """
    HuggingFace-compatible wrapper for WiggleGPT.
    
    This wraps the native WiggleGPT model to work with the Trainer API.
    """
    config_class = WiggleGPTConfig
    base_model_prefix = "wigglegpt"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: WiggleGPTConfig):
        super().__init__(config)
        
        # Create the native GPTConfig
        gpt_config = GPTConfig(
            block_size=config.block_size,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            bias=config.bias,
            use_bio_mlp=config.use_bio_mlp,
            use_rmsnorm=config.use_rmsnorm,
            use_rope=config.use_rope,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        
        # Create the native WiggleGPT model
        self.model = GPT(gpt_config)
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        Forward pass compatible with HuggingFace Trainer.
        
        The native WiggleGPT model handles targets internally, so we adapt
        the interface here.
        """
        # The native model expects targets for loss computation
        logits, loss = self.model(input_ids, targets=labels)
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )
    
    def get_input_embeddings(self):
        return self.model.transformer.wte
    
    def set_input_embeddings(self, value):
        self.model.transformer.wte = value
        
    def get_output_embeddings(self):
        return self.model.lm_head
    
    def set_output_embeddings(self, value):
        self.model.lm_head = value
        
    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = value
        self.model.config.gradient_checkpointing = value
        
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None, **kwargs):
        """Generate text using the native WiggleGPT generate method"""
        return self.model.generate(input_ids, max_new_tokens, temperature, top_k)


# =============================================================================
# DATASET PREPARATION
# =============================================================================

class SmolTalkDataset(Dataset):
    """
    Dataset class for SmolTalk2 conversations.
    
    Formats multi-turn conversations into a single sequence for causal LM training.
    """
    
    # Chat template tokens (using GPT-2 style markers)
    SYSTEM_START = "<|system|>"
    SYSTEM_END = "<|/system|>"
    USER_START = "<|user|>"
    USER_END = "<|/user|>"
    ASSISTANT_START = "<|assistant|>"
    ASSISTANT_END = "<|/assistant|>"
    
    def __init__(
        self,
        dataset_name: str,
        dataset_subset: str,
        dataset_split: str,
        max_seq_length: int = 1024,
        max_samples: Optional[int] = None,
    ):
        self.max_seq_length = max_seq_length
        
        # Initialize tokenizer (GPT-2 tokenizer)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # Load dataset
        print(f"Loading dataset: {dataset_name}/{dataset_subset}")
        try:
            # Try loading with the specific split format
            self.dataset = load_dataset(
                dataset_name,
                dataset_subset,
                split=dataset_split,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Could not load specific split, trying train split: {e}")
            self.dataset = load_dataset(
                dataset_name,
                dataset_subset,
                split="train",
                trust_remote_code=True
            )
        
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} samples")
        
        # Check dataset structure
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            print(f"Sample keys: {sample.keys()}")
            
    def __len__(self):
        return len(self.dataset)
    
    def format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """
        Format a conversation into a single string.
        
        Expected format: list of dicts with 'role' and 'content' keys
        """
        formatted = ""
        
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            
            if role == "system":
                formatted += f"{self.SYSTEM_START}\n{content}\n{self.SYSTEM_END}\n"
            elif role == "user":
                formatted += f"{self.USER_START}\n{content}\n{self.USER_END}\n"
            elif role == "assistant":
                formatted += f"{self.ASSISTANT_START}\n{content}\n{self.ASSISTANT_END}\n"
        
        return formatted.strip()
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        
        # Handle different dataset formats
        if "messages" in sample:
            # Standard chat format
            text = self.format_conversation(sample["messages"])
        elif "conversations" in sample:
            # Alternative format
            text = self.format_conversation(sample["conversations"])
        elif "text" in sample:
            # Plain text format
            text = sample["text"]
        elif "prompt" in sample and "response" in sample:
            # Prompt-response format
            text = f"{self.USER_START}\n{sample['prompt']}\n{self.USER_END}\n{self.ASSISTANT_START}\n{sample['response']}\n{self.ASSISTANT_END}"
        else:
            # Try to find any text field
            for key in sample.keys():
                if isinstance(sample[key], str) and len(sample[key]) > 10:
                    text = sample[key]
                    break
            else:
                text = str(sample)
        
        # Tokenize
        tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        
        # Truncate or pad to max_seq_length
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        
        # Create input_ids and labels (shifted by 1 for causal LM)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        
        # Pad if necessary
        if len(tokens) < self.max_seq_length:
            padding_length = self.max_seq_length - len(tokens)
            input_ids = torch.cat([
                input_ids,
                torch.full((padding_length,), self.tokenizer.eot_token, dtype=torch.long)
            ])
            labels = torch.cat([
                labels,
                torch.full((padding_length,), -100, dtype=torch.long)  # -100 = ignore in loss
            ])
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": (input_ids != self.tokenizer.eot_token).long()
        }


class SimpleDataCollator:
    """Simple data collator that stacks tensors"""
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        }
        return batch


# =============================================================================
# CUSTOM CALLBACKS
# =============================================================================

class WiggleGPTCallback(TrainerCallback):
    """Custom callback for WiggleGPT-specific logging"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and state.is_local_process_zero:
            # Log additional WiggleGPT-specific metrics
            model = kwargs.get("model")
            if model is not None and hasattr(model, "model"):
                sparsity = model.model.get_avg_sparsity()
                if sparsity is not None:
                    logs["sparsity"] = sparsity


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def load_pretrained_wigglegpt(ckpt_path: str, finetune_config: FinetuneConfig) -> WiggleGPTForCausalLM:
    """
    Load a pre-trained WiggleGPT checkpoint and wrap it for fine-tuning.
    """
    print(f"Loading pretrained checkpoint from: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_args = checkpoint["model_args"]
    
    print(f"Model configuration: {model_args}")
    
    # Create HuggingFace-compatible config
    hf_config = WiggleGPTConfig(
        vocab_size=model_args.get("vocab_size", 50304),
        n_layer=model_args.get("n_layer", 12),
        n_head=model_args.get("n_head", 12),
        n_embd=model_args.get("n_embd", 768),
        block_size=model_args.get("block_size", 1024),
        dropout=finetune_config.dropout,  # Use fine-tuning dropout
        bias=model_args.get("bias", False),
        use_bio_mlp=model_args.get("use_bio_mlp", True),
        use_rmsnorm=model_args.get("use_rmsnorm", True),
        use_rope=model_args.get("use_rope", True),
        gradient_checkpointing=finetune_config.gradient_checkpointing,
    )
    
    # Create model
    model = WiggleGPTForCausalLM(hf_config)
    
    # Load weights
    state_dict = checkpoint["model"]
    
    # Fix key prefixes from torch.compile
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # Load into wrapped model
    model.model.load_state_dict(state_dict)
    
    print(f"Loaded model with {model.model.get_num_params()/1e6:.2f}M parameters")
    
    if hf_config.use_bio_mlp:
        print("✓ Bio-inspired neurons enabled (oscillating activations)")
    if hf_config.use_rmsnorm:
        print("✓ RMSNorm enabled")
    if hf_config.use_rope:
        print("✓ Rotary Position Embeddings enabled")
    
    return model


def train(config: FinetuneConfig):
    """Main training function"""
    
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    
    # Load model
    model = load_pretrained_wigglegpt(config.pretrained_ckpt, config)
    
    # Prepare datasets
    print("\n" + "=" * 60)
    print("PREPARING DATASETS")
    print("=" * 60)
    
    train_dataset = SmolTalkDataset(
        dataset_name=config.dataset_name,
        dataset_subset=config.dataset_subset,
        dataset_split=config.dataset_split,
        max_seq_length=config.max_seq_length,
        max_samples=config.max_samples,
    )
    
    # Use 10% for validation if not too small
    val_size = min(1000, len(train_dataset) // 10)
    if val_size > 0:
        # Simple split
        train_indices = list(range(val_size, len(train_dataset)))
        val_indices = list(range(val_size))
        
        # Create subset datasets
        full_dataset = train_dataset.dataset
        train_dataset.dataset = full_dataset.select(train_indices)
        
        val_dataset = SmolTalkDataset(
            dataset_name=config.dataset_name,
            dataset_subset=config.dataset_subset,
            dataset_split=config.dataset_split,
            max_seq_length=config.max_seq_length,
            max_samples=val_size,
        )
    else:
        val_dataset = None
    
    print(f"Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if val_dataset else None,
        eval_strategy=config.eval_strategy if val_dataset else "no",
        save_total_limit=config.save_total_limit,
        fp16=config.fp16 and torch.cuda.is_available(),
        bf16=config.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=config.gradient_checkpointing,
        dataloader_num_workers=config.dataloader_num_workers,
        remove_unused_columns=False,  # We handle columns ourselves
        report_to="none",  # Disable wandb/tensorboard by default
        seed=config.seed,
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss" if val_dataset else None,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=SimpleDataCollator(),
        callbacks=[WiggleGPTCallback()],
    )
    
    # Print training info
    print("\n" + "=" * 60)
    print("STARTING FINE-TUNING")
    print("=" * 60)
    print(f"Output directory: {config.output_dir}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.per_device_train_batch_size} x {config.gradient_accumulation_steps} = {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max sequence length: {config.max_seq_length}")
    print("=" * 60 + "\n")
    
    # Train
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(config.output_dir, "final"))
    
    # Also save in native format for compatibility with sample_bio.py
    save_native_checkpoint(model, config.output_dir, trainer.state)
    
    print(f"\n✓ Fine-tuning complete! Model saved to: {config.output_dir}")


def save_native_checkpoint(model: WiggleGPTForCausalLM, output_dir: str, trainer_state):
    """Save checkpoint in native WiggleGPT format for compatibility"""
    
    native_model = model.model
    
    # Get model args from config
    model_args = {
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_embd": model.config.n_embd,
        "block_size": model.config.block_size,
        "vocab_size": model.config.vocab_size,
        "dropout": model.config.dropout,
        "bias": model.config.bias,
        "use_bio_mlp": model.config.use_bio_mlp,
        "use_rmsnorm": model.config.use_rmsnorm,
        "use_rope": model.config.use_rope,
        "gradient_checkpointing": model.config.gradient_checkpointing,
    }
    
    checkpoint = {
        "model": native_model.state_dict(),
        "model_args": model_args,
        "iter_num": trainer_state.global_step,
        "best_val_loss": trainer_state.best_metric if trainer_state.best_metric else float("inf"),
        "config": {
            "dataset": "smoltalk2",
            "finetuned": True,
        },
    }
    
    ckpt_path = os.path.join(output_dir, "ckpt.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"Saved native checkpoint to: {ckpt_path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune WiggleGPT on SmolTalk")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--pretrained_ckpt", type=str, help="Path to pretrained checkpoint")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--max_samples", type=int, help="Limit dataset size for testing")
    parser.add_argument("--num_train_epochs", type=float, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    args = parser.parse_args()
    
    # Load config
    config = FinetuneConfig()
    
    # Load config file if provided
    if args.config:
        exec(open(args.config, encoding="utf-8").read())
    
    # Override with command line arguments
    if args.pretrained_ckpt:
        config.pretrained_ckpt = args.pretrained_ckpt
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_samples:
        config.max_samples = args.max_samples
    if args.num_train_epochs:
        config.num_train_epochs = args.num_train_epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    # Run training
    train(config)

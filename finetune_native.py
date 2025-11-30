"""
WiggleGPT v3 - finetune_native.py
=================================

Native Fine-tuning Script (Memory-Safe Streaming)
=================================================

This script fine-tunes a pre-trained WiggleGPT model using memory-efficient
streaming - same approach as prepare_openwebtext_streaming.py.

Design Philosophy:
- NEVER load full dataset in memory
- Stream dataset, tokenize chunks, write to .bin files
- Load .bin files via memmap (constant memory)
- Same pattern as pretraining data loading

Usage:
    python finetune_native.py
    python finetune_native.py config_finetune_native.py
"""

import os
import time
import math
import pickle
import random
from contextlib import nullcontext
from typing import Optional, Dict, List

import numpy as np
import torch
import tiktoken

from model_bio import GPTConfig, GPT

# Try to import datasets library for SmolTalk
try:
    from datasets import load_dataset as hf_load_dataset
    HAS_DATASETS = True
except ImportError:
    print("WARNING: 'datasets' library not installed. Install with: pip install datasets")
    HAS_DATASETS = False


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# I/O
out_dir = 'out-wigglegpt-finetune'
pretrained_ckpt = 'out-wigglegpt-pure-124m/ckpt.pt'
data_dir = 'data/smoltalk'  # Where to save processed .bin files
eval_interval = 250
log_interval = 10
eval_iters = 20  # Reduced to save memory
always_save_checkpoint = True

# wandb logging
wandb_log = False
wandb_project = 'wigglegpt-finetune'
wandb_run_name = 'wigglegpt-smoltalk'

# Data
dataset_name = 'HuggingFaceTB/smoltalk2'
dataset_subset = 'SFT'
dataset_split = 'smoltalk_smollm3_smol_magpie_ultra_no_think'
max_samples = None  # None = use all

# Batch configuration
batch_size = 1
block_size = 1024
gradient_accumulation_steps = 32

# Optimizer (fine-tuning uses lower LR than pretraining)
learning_rate = 2e-5  # Much lower than pretraining
weight_decay = 0.01  # Lower for fine-tuning
beta1 = 0.9
beta2 = 0.99  # Slightly different for fine-tuning
grad_clip = 1.0

# Fine-tuning schedule
max_iters = 10000  # Fewer iterations for fine-tuning
warmup_iters = 200
lr_decay_iters = 10000
min_lr = 2e-6
decay_lr = True
lr_schedule = 'cosine'

# Regularization
dropout = 0.1  # Add dropout for fine-tuning

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # torch.compile sometimes causes issues with fine-tuning
gradient_checkpointing = True  # Save VRAM at cost of speed

# Streaming config
chunk_size = 500  # Documents per chunk (SMALL = SAFE for RAM)
val_split = 0.05  # 5% for validation

# Memory saving: freeze early layers (only train last N layers + lm_head)
freeze_layers = 8  # Freeze first 8 of 12 layers (train last 4)

# =============================================================================
# Load configuration from file or command line
# =============================================================================

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open('configurator.py', encoding='utf-8').read())
config = {k: globals()[k] for k in config_keys}


# =============================================================================
# STREAMING DATA PREPARATION (Memory-Safe)
# =============================================================================

# Chat template markers
SYSTEM_START = "<|system|>\n"
SYSTEM_END = "\n<|/system|>\n"
USER_START = "<|user|>\n"
USER_END = "\n<|/user|>\n"
ASSISTANT_START = "<|assistant|>\n"
ASSISTANT_END = "\n<|/assistant|>\n"


def format_conversation(messages: List[Dict]) -> str:
    """Convert messages to formatted text"""
    formatted = ""
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        
        if role == "system":
            formatted += SYSTEM_START + content + SYSTEM_END
        elif role == "user":
            formatted += USER_START + content + USER_END
        elif role == "assistant":
            formatted += ASSISTANT_START + content + ASSISTANT_END
    
    return formatted.strip()


def get_text_from_sample(sample: Dict) -> str:
    """Extract text from various dataset formats"""
    if "messages" in sample:
        return format_conversation(sample["messages"])
    elif "conversations" in sample:
        return format_conversation(sample["conversations"])
    elif "text" in sample:
        return sample["text"]
    elif "prompt" in sample and "response" in sample:
        return (USER_START + sample["prompt"] + USER_END +
                ASSISTANT_START + sample["response"] + ASSISTANT_END)
    else:
        # Find any text field
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 20:
                return value
        return str(sample)


def prepare_smoltalk_streaming():
    """
    Stream SmolTalk dataset, tokenize in chunks, write to .bin files.
    Memory-safe: constant ~200-500MB footprint.
    """
    print("\n" + "=" * 70)
    print("PREPARING SMOLTALK DATA (Memory-Safe Streaming)")
    print("=" * 70)
    
    os.makedirs(data_dir, exist_ok=True)
    train_bin = os.path.join(data_dir, 'train.bin')
    val_bin = os.path.join(data_dir, 'val.bin')
    
    # Check if already prepared
    if os.path.exists(train_bin) and os.path.exists(val_bin):
        train_size = os.path.getsize(train_bin) / (1024**2)
        val_size = os.path.getsize(val_bin) / (1024**2)
        print(f"Found existing data files:")
        print(f"  train.bin: {train_size:.1f} MB")
        print(f"  val.bin: {val_size:.1f} MB")
        
        # Check if files are non-empty
        if train_size > 1 and val_size > 0.01:
            print("Skipping preparation, using existing files.")
            return True
        else:
            print("Files too small, re-preparing...")
    
    if not HAS_DATASETS:
        raise RuntimeError("datasets library required. Install with: pip install datasets")
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    print(f"Tokenizer: gpt2, vocab size: {enc.n_vocab:,}")
    
    # Load dataset in STREAMING mode
    print(f"\nLoading {dataset_name}/{dataset_subset} in streaming mode...")
    try:
        dataset = hf_load_dataset(
            dataset_name,
            dataset_subset,
            split=dataset_split,
            streaming=True,  # KEY: Streaming mode!
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Could not load split '{dataset_split}', trying 'train': {e}")
        dataset = hf_load_dataset(
            dataset_name,
            dataset_subset,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    
    print("Dataset loaded in streaming mode (low memory)")
    
    # Process with streaming
    train_temp = os.path.join(data_dir, 'train.tmp')
    val_temp = os.path.join(data_dir, 'val.tmp')
    
    random.seed(2357)
    start_time = time.time()
    
    train_tokens = 0
    val_tokens = 0
    doc_count = 0
    
    train_chunk = []
    val_chunk = []
    
    def process_and_write_chunk(texts, file_handle):
        """Tokenize chunk and write to file, return token count"""
        tokens = []
        for text in texts:
            # Truncate to block_size if too long
            encoded = enc.encode_ordinary(text)
            if len(encoded) > block_size:
                encoded = encoded[:block_size]
            tokens.extend(encoded)
            tokens.append(enc.eot_token)
        
        if tokens:
            arr = np.array(tokens, dtype=np.uint16)
            arr.tofile(file_handle)
            return len(tokens)
        return 0
    
    print(f"\nProcessing dataset (chunk_size={chunk_size})...")
    
    with open(train_temp, 'wb') as train_f, open(val_temp, 'wb') as val_f:
        try:
            for example in dataset:
                doc_count += 1
                
                # Apply max_samples limit
                if max_samples and doc_count > max_samples:
                    break
                
                text = get_text_from_sample(example)
                
                # Probabilistic train/val split
                if random.random() < val_split:
                    val_chunk.append(text)
                    if len(val_chunk) >= chunk_size:
                        val_tokens += process_and_write_chunk(val_chunk, val_f)
                        val_chunk = []
                else:
                    train_chunk.append(text)
                    if len(train_chunk) >= chunk_size:
                        train_tokens += process_and_write_chunk(train_chunk, train_f)
                        train_chunk = []
                
                # Progress update
                if doc_count % 5000 == 0:
                    elapsed = time.time() - start_time
                    docs_per_sec = doc_count / elapsed if elapsed > 0 else 0
                    print(f"  Processed {doc_count:,} docs | {train_tokens + val_tokens:,} tokens | {docs_per_sec:.0f} docs/s")
        
        except KeyboardInterrupt:
            print("\nInterrupted! Saving progress...")
        
        # Write remaining chunks
        if train_chunk:
            train_tokens += process_and_write_chunk(train_chunk, train_f)
        if val_chunk:
            val_tokens += process_and_write_chunk(val_chunk, val_f)
    
    # Rename temp files
    for split, temp, final in [("train", train_temp, train_bin), ("val", val_temp, val_bin)]:
        if os.path.exists(final):
            os.remove(final)
        os.rename(temp, final)
    
    elapsed = time.time() - start_time
    total_tokens = train_tokens + val_tokens
    
    print(f"\n" + "=" * 70)
    print(f"DATA PREPARATION COMPLETE")
    print(f"=" * 70)
    print(f"Documents: {doc_count:,}")
    print(f"Train tokens: {train_tokens:,}")
    print(f"Val tokens: {val_tokens:,}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Files saved to: {data_dir}/")
    print(f"=" * 70 + "\n")
    
    return True


# =============================================================================
# DATA LOADING (Same as train_bio.py - memmap)
# =============================================================================

class MemmapDataLoader:
    """Memory-mapped data loader - constant memory footprint"""
    
    def __init__(self, data_dir, split, batch_size, block_size, device, device_type):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.device_type = device_type
        
        data_path = os.path.join(data_dir, f'{split}.bin')
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        print(f"Loaded {split}.bin: {len(self.data):,} tokens")
    
    def get_batch(self):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i:i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i + 1:i + 1 + self.block_size]).astype(np.int64)) for i in ix])
        
        if self.device_type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        
        return x, y


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

def get_lr(it):
    """Learning rate scheduler"""
    if not decay_lr:
        return learning_rate
    
    # Warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    
    # After decay
    if it > lr_decay_iters:
        return min_lr
    
    # Cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    global config
    
    # Setup
    print("\n" + "=" * 70)
    print("WIGGLEGPT FINE-TUNING (Memory-Safe)")
    print("=" * 70)
    
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Prepare data (streaming, memory-safe)
    prepare_smoltalk_streaming()
    
    # Load pretrained model
    print(f"\nLoading pretrained model from: {pretrained_ckpt}")
    if not os.path.exists(pretrained_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_ckpt}")
    
    # Load to CPU first to avoid GPU memory fragmentation
    checkpoint = torch.load(pretrained_ckpt, map_location='cpu', weights_only=False)
    model_args = checkpoint['model_args']
    
    # Update dropout for fine-tuning
    model_args['dropout'] = dropout
    
    print(f"Model config: {model_args}")
    
    # Create model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Load weights
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    model.to(device)
    
    if model_args.get('use_bio_mlp'):
        print("Bio-inspired neurons enabled (oscillating activations)")
    
    # Freeze early layers to save memory (only train last few layers)
    if freeze_layers > 0:
        # Freeze embeddings
        for param in model.transformer.wte.parameters():
            param.requires_grad = False
        if hasattr(model.transformer, 'wpe'):
            for param in model.transformer.wpe.parameters():
                param.requires_grad = False
        
        # Freeze first N transformer blocks
        for i, block in enumerate(model.transformer.h):
            if i < freeze_layers:
                for param in block.parameters():
                    param.requires_grad = False
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Frozen {freeze_layers}/{len(model.transformer.h)} layers")
        print(f"Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")
    
    # Enable gradient checkpointing to save VRAM
    if gradient_checkpointing:
        model.config.gradient_checkpointing = True
        print("Gradient checkpointing enabled (saves VRAM)")
    
    # Create data loaders (memory-mapped)
    print("\nLoading data (memory-mapped)...")
    train_loader = MemmapDataLoader(data_dir, 'train', batch_size, block_size, device, device_type)
    val_loader = MemmapDataLoader(data_dir, 'val', batch_size, block_size, device, device_type)
    
    # Optimizer
    print("\nSetting up optimizer...")
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
    # GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # Compile if enabled
    if compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Wandb
    if wandb_log:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    
    # Training info
    tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
    print(f"\nTokens per iteration: {tokens_per_iter:,}")
    print(f"Training for {max_iters} iterations")
    print("=" * 70 + "\n")
    
    # Training loop
    iter_num = 0
    best_val_loss = float('inf')
    t0 = time.time()
    
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, loader in [('train', train_loader), ('val', val_loader)]:
            losses = []
            for _ in range(eval_iters):
                x, y = loader.get_batch()
                with ctx:
                    _, loss = model(x, y)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses) if losses else float('inf')
        model.train()
        # Clear CUDA cache after eval to free memory for training
        if device_type == 'cuda':
            torch.cuda.empty_cache()
        return out
    
    # Main training loop
    # Clear cache before starting
    if device_type == 'cuda':
        torch.cuda.empty_cache()
    
    while iter_num < max_iters:
        # Set learning rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluation (skip step 0 to avoid OOM before optimizer states are built)
        if iter_num % eval_interval == 0 and iter_num > 0:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if wandb_log:
                import wandb
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                })
            
            # Save checkpoint
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    ckpt = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))
        
        # Gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            X, Y = train_loader.get_batch()
            
            with ctx:
                _, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
        
        # Gradient clipping
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Timing
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % log_interval == 0:
            lossf = loss.item() * gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")
        
        iter_num += 1
    
    print("\n" + "=" * 70)
    print(f"Fine-tuning complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {out_dir}/ckpt.pt")
    print("=" * 70)


if __name__ == '__main__':
    main()

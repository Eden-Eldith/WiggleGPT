"""
WiggleGPT v3 - config_finetune_smoltalk.py
==========================================

Fine-tuning Configuration for WiggleGPT on SmolTalk2
====================================================

This configuration fine-tunes a pre-trained WiggleGPT (oscillating neurons)
model on the SmolTalk2 instruction-following dataset.

Usage:
    python finetune_smoltalk.py --config config_finetune_smoltalk.py

Or modify the default config and run:
    python finetune_smoltalk.py
"""

# =============================================================================
# PRETRAINED MODEL
# =============================================================================

# Path to your pre-trained WiggleGPT checkpoint
pretrained_ckpt = "out-wigglegpt-pure-124m/ckpt.pt"

# Output directory for fine-tuned model
output_dir = "out-wigglegpt-smoltalk-sft"

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# SmolTalk2 dataset from HuggingFace
dataset_name = "HuggingFaceTB/smoltalk2"
dataset_subset = "SFT"

# Available splits in SmolTalk2 SFT:
# NO_THINK (without reasoning traces):
#   - smoltalk-smollm3_smol-magpie-ultra_no_think (406K samples)
#   - smoltalk-multilingual-8languages_lang_5_no_think (254K samples)
#   - OpenThoughts3-1.2M_no_think_no_think (435K samples)
#   - OpenHermes-2.5_no_think (385K samples)
#   - smoltalk-smollm3_systemchats-30k_no_think (34K samples)
#   - tulu-3-sft-personas-instruction-following_no_think (30K samples)
#
# THINK (with reasoning traces):
#   - OpenThoughts3-1.2M_think (1.1M samples - chain of thought)
#   - smoltalk-multilingual8-Qwen3-32B_think (245K samples)
#   - multi-turn-reasoning-if_think (28K samples)

dataset_split = "smoltalk-smollm3_smol-magpie-ultra_no_think"

# Limit samples for testing (set to None for full dataset)
max_samples = None  # Use 1000 for quick testing

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Training epochs
num_train_epochs = 3.0
max_steps = -1  # -1 = use epochs, set positive number to override

# Batch configuration (effective batch = per_device * gradient_accumulation)
# For 8GB GPU: batch_size=2, grad_accum=16 -> effective batch of 32
# For 12GB GPU: batch_size=4, grad_accum=8 -> effective batch of 32
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
gradient_accumulation_steps = 16

# Sequence length (should match pretrained model)
max_seq_length = 1024

# =============================================================================
# LEARNING RATE
# =============================================================================

# Fine-tuning learning rate (typically 10x-100x lower than pretraining)
# Pretraining used 6e-4, so we use 2e-5 for fine-tuning
learning_rate = 2e-5

# Weight decay (lower for fine-tuning)
weight_decay = 0.01

# Warmup (ratio of total steps)
warmup_ratio = 0.03

# LR scheduler
lr_scheduler_type = "cosine"

# =============================================================================
# REGULARIZATION
# =============================================================================

# Add dropout for fine-tuning (pretraining used 0.0)
# This helps prevent overfitting on the smaller fine-tuning dataset
dropout = 0.1

# =============================================================================
# OPTIMIZATION
# =============================================================================

# Mixed precision (fp16 for most GPUs, bf16 for Ampere+)
fp16 = True
bf16 = False

# Gradient checkpointing (saves memory, slightly slower)
# Recommended for fine-tuning to fit larger batches
gradient_checkpointing = True

# =============================================================================
# LOGGING AND CHECKPOINTING
# =============================================================================

# Log every N steps
logging_steps = 10

# Save checkpoint every N steps
save_steps = 500

# Evaluate every N steps
eval_steps = 500

# Keep only N most recent checkpoints
save_total_limit = 3

# Evaluation strategy: "steps" or "epoch"
eval_strategy = "steps"

# =============================================================================
# SYSTEM
# =============================================================================

# Random seed for reproducibility
seed = 42

# Number of data loading workers
dataloader_num_workers = 4

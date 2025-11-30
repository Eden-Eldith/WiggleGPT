"""
WiggleGPT v3 - config_finetune_native.py
========================================

Configuration for Native Fine-tuning (finetune_native.py)
=========================================================

This config is for the native training loop style (like train_bio.py).
Use with: python finetune_native.py config=config_finetune_native.py

For the HuggingFace Trainer version, use finetune_smoltalk.py instead.
"""

# =============================================================================
# I/O
# =============================================================================

# Output directory for fine-tuned model
out_dir = 'out-wigglegpt-finetune-native'

# Pretrained checkpoint to fine-tune from
pretrained_ckpt = 'out-wigglegpt-pure-124m/ckpt.pt'

# Evaluation and logging intervals
eval_interval = 250
log_interval = 10
eval_iters = 20  # Reduced to save memory
always_save_checkpoint = True

# =============================================================================
# WANDB LOGGING (optional)
# =============================================================================

wandb_log = False
wandb_project = 'wigglegpt-finetune'
wandb_run_name = 'wigglegpt-smoltalk-native'

# =============================================================================
# DATASET
# =============================================================================

# SmolTalk2 from HuggingFace
dataset_name = 'HuggingFaceTB/smoltalk2'
dataset_subset = 'SFT'

# Available splits (see SmolTalk dataset documentation):
# NO_THINK (without reasoning traces) - Good for basic instruction following:    
#   - smoltalk_smollm3_smol_magpie_ultra_no_think (general instructions)  
#   - OpenHermes_2.5_no_think (diverse)
#   - smoltalk_smollm3_systemchats_30k_no_think (with system prompts)      
#
# THINK (with reasoning traces) - Good for chain-of-thought:
#   - OpenThoughts3_1.2M_think (extensive reasoning)
#   - multi_turn_reasoning_if_think (multi-turn reasoning)

dataset_split = 'smoltalk_smollm3_smol_magpie_ultra_no_think'# Limit dataset size (None = use all, set to small number for testing)
max_samples = None  # Try 1000 for quick test

# =============================================================================  
# BATCH CONFIGURATION
# =============================================================================  

# Minimal batch for fine-tuning (16GB VRAM constraint)
# batch_size=1 with grad_accum=64 = effective batch of 64
batch_size = 1

# REDUCED sequence length to save memory (512 vs 1024)
block_size = 512

# Gradient accumulation (effective batch = batch_size * gradient_accumulation_steps)
gradient_accumulation_steps = 64# =============================================================================
# OPTIMIZER
# =============================================================================

# Fine-tuning learning rate (much lower than pretraining!)
# Pretraining used 6e-4, fine-tuning typically uses 1e-5 to 5e-5
learning_rate = 2e-5

# Weight decay (lower for fine-tuning to preserve pretrained knowledge)
weight_decay = 0.01

# Adam betas (standard values work well)
beta1 = 0.9
beta2 = 0.99

# Gradient clipping
grad_clip = 1.0

# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

# Whether to decay learning rate
decay_lr = True

# Schedule type: 'cosine'
lr_schedule = 'cosine'

# Number of iterations for fine-tuning (fewer than pretraining)
max_iters = 10000

# Warmup iterations (shorter warmup for fine-tuning)
warmup_iters = 200

# LR decay iterations (usually = max_iters)
lr_decay_iters = 10000

# Minimum learning rate
min_lr = 2e-6

# =============================================================================
# REGULARIZATION
# =============================================================================

# Add dropout for fine-tuning (prevents overfitting on smaller dataset)
# Pretraining used 0.0, but we need 0.0 for memory constraints on 16GB
dropout = 0.0

# =============================================================================
# SYSTEM
# =============================================================================

device = 'cuda'

# Mixed precision (float16 to match pretraining)
dtype = 'float16'

# torch.compile (can cause issues with fine-tuning, often disabled)
compile = False

# Gradient checkpointing (saves VRAM at cost of speed)
gradient_checkpointing = True

# Freeze early layers to save VRAM (only train last N layers)
# Model has 12 layers, freeze 0 = train ALL layers (full fine-tuning)
freeze_layers = 0

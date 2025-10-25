"""
WiggleGPT 124M - PURE OSCILLATING NEURONS
=========================================

CLEAN TEST: Just oscillating activations vs GELU
No sparsity gating, no complexity, just the core hypothesis.

This is the scientific comparison:
- GPT-2: Linear → GELU → Linear
- WiggleGPT: Linear → sin(ω·x + φ)·tanh(x) → Linear

Same architecture, same parameters (~124M), one variable changed.
"""

# I/O
out_dir = 'out-wigglegpt-pure-124m'
eval_interval = 1000
log_interval = 10
eval_iters = 100

# Wandb
wandb_log = False
wandb_project = 'wigglegpt-pure'
wandb_run_name = 'wigglegpt-pure-124m'

# Data
dataset = 'openwebtext'

# Batch configuration
batch_size = 2
block_size = 1024
gradient_accumulation_steps = 16  # Effective batch: 128K tokens

# Model architecture - EXACT GPT-2 124M
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# Bio-inspired neurons - PURE VERSION
use_bio_mlp = True  # This ONLY changes activation function, nothing else

# Optimizations
use_rmsnorm = True
use_rope = True
gradient_checkpointing = False

# Optimizer
optimizer_type = 'adamw'

# Learning rate schedule
lr_schedule = 'cosine'
stable_iters = 0

# torch.compile
compile = False
compile_mode = 'default'

# Training hyperparameters
max_iters = 600000
lr_decay_iters = 600000
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Warmup
warmup_iters = 2000
min_lr = 6e-5

# System
device = 'cuda'
dtype = 'float16'

# EXPECTED RESULTS
# Parameters: ~124M (exactly GPT-2)
# Training time: ~3 days on 3070
# Target: Beat GPT-2's 3.12 val loss with oscillating neurons alone
# No sparsity metric - we removed that complexity
# GPT-2 with Bio Neurons - Single 3070 (8GB)
# You know the drill, you've done this before

out_dir = 'out-gpt2-bio'
eval_interval = 2000
log_interval = 10
eval_iters = 50

wandb_log = False  # Set True if you want
wandb_project = 'owt'
wandb_run_name = 'gpt2-bio-3070'

dataset = 'openwebtext'

# Tuned for 8GB VRAM
batch_size = 2
block_size = 1024
gradient_accumulation_steps = 16  # Effective batch: 128K tokens

# Standard GPT-2 124M architecture
n_layer = 4
n_head = 6
n_embd = 384
dropout = 0.0
bias = False

# Your bio neurons
use_bio_mlp = True
bio_compartments = 2
bio_threshold = 0.25

# Training
max_iters = 200000
lr_decay_iters = 200000
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# System
device = 'cuda'
dtype = 'float16'  # 3070 doesn't have bfloat16
compile = False

# Expected: ~2-3 days on 3070


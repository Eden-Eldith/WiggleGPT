"""
Pure WiggleGPT Training Script - Oscillating Neurons Only
==========================================================

Clean training script testing ONLY oscillating activations vs GELU.
All sparsity gating complexity has been removed.

Research Question: Can sin(ω·x + φ)·tanh(x) beat GELU in transformers?

Optimizations preserved:
1. Multiple optimizer support (AdamW, AdamW8bit, Lion, Adafactor)
2. Better data loading with prefetching and multiple workers
3. Improved torch.compile settings
4. Better learning rate schedules (WSD - Warmup-Stable-Decay)
5. Gradient accumulation optimizations
6. Memory profiling support

Pure test: One variable (activation function), everything else identical to GPT-2.
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Import from optimized bio-aware model
from model_bio import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2'  # 'run' + str(time.time())

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?

# bio-inspired neuron parameters
use_bio_mlp = False  # Set to True to use oscillating neurons (pure, no sparsity)

# optimization parameters
use_rmsnorm = False  # Use RMSNorm instead of LayerNorm
use_rope = False  # Use Rotary Position Embeddings
gradient_checkpointing = False  # Trade compute for memory

# optimizer
optimizer_type = 'adamw'  # 'adamw', 'adamw8bit', 'lion', 'adafactor'
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
lr_schedule = 'cosine'  # 'cosine', 'wsd' (warmup-stable-decay), 'constant'
warmup_iters = 2000  # how many steps to warm up for
stable_iters = 0  # for WSD schedule: stable phase after warmup
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.

# system
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16'
compile = True  # use PyTorch 2.0 to compile the model to be faster
compile_mode = 'default'  # 'default', 'reduce-overhead', 'max-autotune'

# profiling
profile_memory = False  # Enable memory profiling (for debugging)

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py', encoding='utf-8').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# Print configuration
print("\n" + "=" * 70)
if use_bio_mlp:
    print("OSCILLATING NEURONS ENABLED (Pure WiggleGPT)")
    print("  Features: sin(ω·x + φ)·tanh(x) activation only")
    print("  No sparsity gating - testing core hypothesis")
else:
    print("Standard GPT architecture")

print("\nTRAINING OPTIMIZATIONS:")
optimizations = []
if use_rmsnorm:
    optimizations.append("+ RMSNorm (faster than LayerNorm)")
else:
    optimizations.append("- RMSNorm disabled (using LayerNorm)")
    
if use_rope:
    optimizations.append("+ Rotary Position Embeddings (RoPE)")
else:
    optimizations.append("- RoPE disabled (using learned embeddings)")
    
if gradient_checkpointing:
    optimizations.append("+ Gradient Checkpointing (memory efficient)")
    
optimizations.append(f"+ Optimizer: {optimizer_type.upper()}")
optimizations.append(f"+ LR Schedule: {lr_schedule.upper()}")

if compile:
    optimizations.append(f"+ torch.compile (mode: {compile_mode})")
else:
    optimizations.append("- torch.compile DISABLED (missing 20-30% speedup!)")

for opt in optimizations:
    print(f"  {opt}")
print("=" * 70 + "\n")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Improved data loader with caching
data_dir = os.path.join('data', dataset)


class DataLoader:
    """Improved data loader with better memory management"""

    def __init__(self, data_dir, split, batch_size, block_size, device, device_type):
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.device_type = device_type

        # Load data
        data_path = os.path.join(data_dir, f'{split}.bin')
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')

    def get_batch(self):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i:i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i + 1:i + 1 + self.block_size]).astype(np.int64)) for i in ix])

        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y


# Initialize data loaders
train_loader = DataLoader(data_dir, 'train', batch_size, block_size, device, device_type)
val_loader = DataLoader(data_dir, 'val', batch_size, block_size, device, device_type)


def get_batch(split):
    """Wrapper for backward compatibility"""
    if split == 'train':
        return train_loader.get_batch()
    else:
        return val_loader.get_batch()


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout,
    use_bio_mlp=use_bio_mlp,  # Pure oscillating neurons, no threshold needed
    use_rmsnorm=use_rmsnorm, use_rope=use_rope,
    gradient_checkpointing=gradient_checkpointing
)

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # Also keep bio parameters and optimization flags if they were set
    for k in ['use_bio_mlp', 'use_rmsnorm', 'use_rope', 'gradient_checkpointing']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # Note: optimized features like RMSNorm and RoPE can't be used with pretrained weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer - use the new configure_optimizers with optimizer type support
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type, optimizer_type
)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

# compile the model with better settings
if compile:
    print(f"compiling the model with mode '{compile_mode}'... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model, mode=compile_mode)

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Improved learning rate schedulers
def get_lr(it):
    """
    Get learning rate for iteration with multiple schedule options.

    Schedules:
    - cosine: warmup -> cosine decay to min_lr
    - wsd: warmup -> stable -> decay (better for large models)
    - constant: warmup -> constant learning rate
    """
    if lr_schedule == 'constant':
        # Warmup then constant
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        return learning_rate

    elif lr_schedule == 'wsd':
        # Warmup-Stable-Decay (better for large models)
        # Phase 1: Linear warmup
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        # Phase 2: Stable learning rate
        if it < warmup_iters + stable_iters:
            return learning_rate
        # Phase 3: Cosine decay to min_lr
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters - stable_iters) / (lr_decay_iters - warmup_iters - stable_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    else:  # cosine (default)
        # Standard cosine with warmup
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Memory profiling setup
if profile_memory and torch.cuda.is_available():
    torch.cuda.memory._record_memory_history(max_entries=100000)

# training loop
X, Y = get_batch('train')  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()

        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            }
            wandb.log(log_dict)

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation

        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')

        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# Cleanup
if profile_memory and torch.cuda.is_available():
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

if ddp:
    destroy_process_group()
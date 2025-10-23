# WiggleGPT Training Optimizations Guide

## Overview

This document describes modern training optimizations implemented to speed up WiggleGPT training while **preserving the core model architecture** and bio-inspired features.

**Expected Speedup: 20-30% faster training** with similar or better convergence.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Optimization Categories](#optimization-categories)
3. [Detailed Optimizations](#detailed-optimizations)
4. [Configuration Guide](#configuration-guide)
5. [Benchmarks](#benchmarks)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Using Optimized Training

```bash
# Install optional dependencies (for advanced optimizers)
pip install lion-pytorch bitsandbytes

# Run optimized training (same command, new config)
python train_bio_optimized.py config_gpt2_bio_optimized.py
```

### What's Different?

The optimized version uses:
- **RMSNorm** instead of LayerNorm (~7% faster)
- **Better optimizer options** (Lion, 8-bit AdamW, Adafactor)
- **torch.compile enabled** with better settings (20-30% speedup)
- **Improved data loading** with better prefetching
- **Gradient checkpointing support** for memory-constrained setups

**Core model architecture is preserved:**
- Same number of layers, heads, embedding dimensions
- Same bio-inspired neurons with oscillating activations
- Same sparse event processing

---

## Optimization Categories

### 1. Architectural Optimizations (Preserve Core Config)

These optimizations **do not change** the model's core architecture (n_layer, n_head, n_embd) but make individual components faster.

#### A. RMSNorm (Root Mean Square Layer Normalization)

**What it does:** Simpler and faster alternative to LayerNorm
**Speedup:** ~7% faster forward/backward pass
**Memory:** Slightly lower (no bias parameters)

**Technical details:**
- Standard LayerNorm: `x = (x - mean) / std * gamma + beta`
- RMSNorm: `x = x / rms(x) * gamma` (no mean subtraction, no bias)
- Used in: LLaMA, GPT-NeoX, PaLM

**Enable in config:**
```python
use_rmsnorm = True
```

**Trade-offs:**
- ✅ Faster computation
- ✅ Fewer parameters
- ⚠️ Can't load GPT-2 pretrained weights (different normalization)
- ⚠️ Slightly different training dynamics (usually better or similar)

#### B. Rotary Position Embeddings (RoPE)

**What it does:** Encodes position through rotation instead of learned embeddings
**Speedup:** ~3-5% faster (no separate position embedding lookup/add)
**Memory:** Saves parameters (no position embedding table)

**Technical details:**
- Standard: Learn position embeddings, add to token embeddings
- RoPE: Apply rotation to Q/K in attention based on position
- Better length extrapolation (can handle longer sequences than trained on)
- Used in: LLaMA, GPT-NeoX, PaLM

**Enable in config:**
```python
use_rope = True
```

**Trade-offs:**
- ✅ Faster inference
- ✅ Better length extrapolation
- ✅ Fewer parameters
- ⚠️ Can't load GPT-2 pretrained weights
- ⚠️ Position encoding is in attention only (not in embeddings)

---

### 2. Optimizer Optimizations

#### A. Lion Optimizer

**What it does:** Momentum-based optimizer that's faster than AdamW
**Speedup:** 10-20% faster convergence (fewer iterations needed)
**Memory:** 33% less optimizer state memory (only momentum, no variance)

**Technical details:**
- Uses sign of gradient update (like SignSGD)
- Larger weight decay typically needed
- Often better final performance than AdamW

**Enable in config:**
```python
optimizer_type = 'lion'
# Adjust learning rate (Lion typically needs lower LR)
learning_rate = 3e-4  # vs 6e-4 for AdamW
weight_decay = 1e-1   # vs 1e-1 for AdamW (same or higher)
```

**Install:**
```bash
pip install lion-pytorch
```

**Trade-offs:**
- ✅ Faster convergence
- ✅ Less memory
- ✅ Often better final loss
- ⚠️ Requires tuning LR (typically 3-10x lower than AdamW)
- ⚠️ Less mature than AdamW

#### B. 8-bit AdamW

**What it does:** AdamW with 8-bit quantized optimizer states
**Speedup:** Minimal (~2-5% due to less memory traffic)
**Memory:** 75% less optimizer state memory

**Technical details:**
- Quantizes optimizer states (momentum, variance) to 8-bit
- Uses dynamic quantization to maintain precision
- Nearly identical convergence to full-precision AdamW

**Enable in config:**
```python
optimizer_type = 'adamw8bit'
# Same hyperparameters as regular AdamW
```

**Install:**
```bash
pip install bitsandbytes
```

**Trade-offs:**
- ✅ 4x less memory for optimizer states
- ✅ Same convergence as AdamW
- ⚠️ Slightly slower per-step (quantization overhead)
- ⚠️ Only works on CUDA GPUs

#### C. Adafactor

**What it does:** Memory-efficient optimizer with factorized second moments
**Speedup:** Similar to AdamW
**Memory:** 80% less optimizer state memory

**Technical details:**
- Factorizes variance matrix (reduces memory)
- No momentum (optional)
- Used in T5 and other large models

**Enable in config:**
```python
optimizer_type = 'adafactor'
# Slightly different hyperparameters
```

**Install:**
```bash
pip install transformers
```

**Trade-offs:**
- ✅ Very low memory
- ✅ Good for very large models
- ⚠️ Can be less stable than AdamW
- ⚠️ May need more tuning

---

### 3. Training Optimizations

#### A. torch.compile

**What it does:** PyTorch 2.0+ graph compilation for faster execution
**Speedup:** 20-30% faster training
**Memory:** Similar or slightly lower

**Technical details:**
- Fuses operations (less kernel launches)
- Better memory access patterns
- Removes Python overhead

**Enable in config:**
```python
compile = True
compile_mode = 'default'  # Options: 'default', 'reduce-overhead', 'max-autotune'
```

**Modes:**
- `default`: Balanced speed/compile time
- `reduce-overhead`: Fastest runtime, quick compile
- `max-autotune`: Tries many optimizations (slow compile, potentially fastest)

**Requirements:**
- PyTorch 2.0+
- First iteration is slow (compilation time)
- Some operations may not compile (fallback to eager)

**Trade-offs:**
- ✅ Significant speedup (20-30%)
- ✅ No model changes needed
- ⚠️ First run is slower (compilation)
- ⚠️ May use more CPU during compilation

#### B. Gradient Checkpointing

**What it does:** Trades compute for memory by recomputing activations
**Speedup:** 20-30% **slower** (but allows larger batch size)
**Memory:** 40-60% less activation memory

**Technical details:**
- Doesn't save intermediate activations during forward pass
- Recomputes them during backward pass
- Allows training larger models or larger batches

**Enable in config:**
```python
gradient_checkpointing = True
# Can now increase batch_size or block_size
batch_size = 4  # vs 2 without checkpointing
```

**When to use:**
- Out of memory with current batch size
- Want to use larger batch size for better convergence
- Training very deep models

**Trade-offs:**
- ✅ Much lower memory usage
- ✅ Enables larger batch sizes
- ⚠️ 20-30% slower per iteration
- ⚠️ Overall can be faster (due to larger batch)

#### C. Learning Rate Schedules

**What it does:** Different LR decay strategies for better convergence

**Options:**

1. **Cosine (default):** Smooth decay from peak to minimum
   ```python
   lr_schedule = 'cosine'
   ```
   - Used in GPT-2, BERT
   - Smooth, continuous decay
   - Good general-purpose choice

2. **WSD (Warmup-Stable-Decay):** Warmup → Stable → Decay
   ```python
   lr_schedule = 'wsd'
   stable_iters = 10000  # Stable phase length
   ```
   - Used in LLaMA, PaLM
   - Longer stable phase before decay
   - Better for large models
   - May improve final loss

3. **Constant:** Warmup → Constant
   ```python
   lr_schedule = 'constant'
   ```
   - Simplest schedule
   - Good for fine-tuning
   - May not reach best final loss

**Recommendation:** Start with `cosine` (default), try `wsd` if training large models.

---

## Configuration Guide

### Recommended Configurations by Goal

#### 1. Maximum Speed (Slightly different model)

```python
# Architecture optimizations
use_rmsnorm = True      # 7% faster
use_rope = True         # 3-5% faster

# Optimizer
optimizer_type = 'lion'
learning_rate = 3e-4    # Lower for Lion

# Training
compile = True
compile_mode = 'reduce-overhead'  # Fastest compile + good runtime

# Expected: 30-40% faster than baseline
```

#### 2. Balanced Speed + Compatibility (Recommended)

```python
# Architecture optimizations
use_rmsnorm = True      # 7% faster, preserves architecture
use_rope = False        # Keep learned embeddings for compatibility

# Optimizer
optimizer_type = 'adamw'  # Or 'lion' if you can retune LR

# Training
compile = True
compile_mode = 'default'

# Expected: 20-30% faster than baseline
```

#### 3. Memory Efficiency

```python
# Architecture
use_rmsnorm = True
use_rope = True
gradient_checkpointing = True  # Key for memory

# Optimizer
optimizer_type = 'adamw8bit'  # Or 'adafactor'

# Batch size
batch_size = 4  # Larger thanks to memory savings
gradient_accumulation_steps = 8  # Adjust to keep same effective batch

# Expected: 50% less memory, slightly slower per-iter, similar total time
```

#### 4. Best Final Loss

```python
# Architecture
use_rmsnorm = True
use_rope = False

# Optimizer
optimizer_type = 'lion'
learning_rate = 3e-4
weight_decay = 1e-1

# Learning rate
lr_schedule = 'wsd'
stable_iters = 20000  # Longer stable phase

# Batch (larger is better)
batch_size = 4  # If memory allows
gradient_accumulation_steps = 8

# Expected: Best convergence, similar speed to baseline
```

---

## Benchmarks

### Baseline (Original WiggleGPT)

**Config:** `config_gpt2_bio_3070.py`
```
Model: 45.31M parameters, 4 layers, 6 heads, 384 dim
Optimizer: AdamW (fused)
Compile: False
Normalization: LayerNorm
```

**Results:**
- Training time: ~1.6 days (200K iterations @ ~700ms/iter)
- Final train loss: 3.5533
- Final val loss: 3.5870
- Sparsity: 15.77%

### Optimized (RMSNorm + compile)

**Config:** `config_gpt2_bio_optimized.py`
```
Model: ~45M parameters (slightly fewer due to RMSNorm)
Optimizer: AdamW (fused)
Compile: True (default mode)
Normalization: RMSNorm
```

**Expected Results:**
- Training time: ~1.2 days (200K iterations @ ~530ms/iter) = **25% faster**
- Final train loss: ~3.55 (similar)
- Final val loss: ~3.59 (similar)
- Sparsity: ~16% (similar)

### Optimized (RMSNorm + RoPE + Lion + compile)

**Config:** Custom configuration
```
Model: ~44M parameters (RoPE saves embedding params)
Optimizer: Lion
Compile: True (reduce-overhead mode)
Normalization: RMSNorm
Position: RoPE
```

**Expected Results:**
- Training time: ~1.1 days (180K iterations @ ~520ms/iter) = **35% faster**
- Final train loss: ~3.50 (potentially better)
- Final val loss: ~3.57 (potentially better)
- Sparsity: ~16% (similar)

---

## Implementation Details

### What's Preserved

✅ **Model Architecture:**
- Number of layers (n_layer)
- Number of attention heads (n_head)
- Embedding dimension (n_embd)
- Block size / context length (block_size)
- Vocabulary size (vocab_size)

✅ **Bio-Inspired Features:**
- Oscillating activations (sin(ωx + φ)·tanh(x))
- Sparse event processing
- Learnable frequency and phase parameters
- Sparsity tracking and metrics

✅ **Training Fundamentals:**
- Loss function (cross-entropy)
- Gradient accumulation
- Mixed precision training (AMP)
- Flash Attention
- Gradient clipping
- Weight decay strategy

### What's Changed (Optional)

⚙️ **Architectural Components (if enabled):**
- LayerNorm → RMSNorm (different normalization method)
- Learned embeddings → RoPE (different position encoding)

⚙️ **Training Components:**
- Optimizer type (AdamW → Lion/8bit/Adafactor)
- LR schedule (cosine → WSD)
- Compilation (none → torch.compile)

---

## Troubleshooting

### Common Issues

#### 1. "torch.compile not available"

**Cause:** PyTorch < 2.0
**Solution:**
```bash
pip install torch>=2.0.0
# Or disable compile:
compile = False
```

#### 2. "Lion optimizer not found"

**Cause:** lion-pytorch not installed
**Solution:**
```bash
pip install lion-pytorch
# Or use AdamW:
optimizer_type = 'adamw'
```

#### 3. "Out of memory with RMSNorm"

**Cause:** Other memory issues (not RMSNorm, it uses less memory)
**Solution:**
```python
# Enable gradient checkpointing
gradient_checkpointing = True
# Or reduce batch size
batch_size = 1
```

#### 4. "Training diverged with Lion"

**Cause:** Learning rate too high for Lion
**Solution:**
```python
# Lower learning rate (Lion needs 3-10x lower LR)
learning_rate = 3e-4  # vs 6e-4 for AdamW
# Or increase warmup
warmup_iters = 4000  # vs 2000
```

#### 5. "Compile is very slow"

**Cause:** First iteration always slow (compilation time)
**Solution:**
- Wait 2-5 minutes for first iteration
- Subsequent iterations are fast
- Or use faster compile mode:
```python
compile_mode = 'reduce-overhead'
```

#### 6. "RMSNorm gives different loss"

**Cause:** Different normalization → different training dynamics
**Solution:**
- Expected! Loss curve may differ slightly
- Final loss should be similar or better
- If worse, try:
  - Slightly adjust learning rate
  - Longer warmup
  - Check that other hyperparameters are correct

---

## Migration Guide

### From Original to Optimized

**Step 1:** Install optional dependencies
```bash
pip install lion-pytorch bitsandbytes
```

**Step 2:** Choose optimization level

**Conservative (safest):**
```python
# Only enable torch.compile
compile = True
# Expected: 20% speedup, identical convergence
```

**Moderate (recommended):**
```python
# RMSNorm + compile
use_rmsnorm = True
compile = True
# Expected: 25% speedup, similar convergence
```

**Aggressive (maximum speed):**
```python
# All optimizations
use_rmsnorm = True
use_rope = True
optimizer_type = 'lion'
learning_rate = 3e-4
compile = True
compile_mode = 'reduce-overhead'
# Expected: 35% speedup, may need LR tuning
```

**Step 3:** Run training
```bash
python train_bio_optimized.py config_gpt2_bio_optimized.py
```

**Step 4:** Monitor first 1000 iterations
- Loss should decrease smoothly
- If unstable, reduce learning rate
- If OOM, enable gradient checkpointing

---

## Advanced Topics

### Custom Optimizers

You can add your own optimizer in `model_bio_optimized.py`:

```python
elif optimizer_type == 'my_optimizer':
    from my_optimizer import MyOptimizer
    optimizer = MyOptimizer(optim_groups, lr=learning_rate)
```

### Hybrid Approaches

Mix standard and optimized components:

```python
# Use RMSNorm in some layers, LayerNorm in others
# Useful for transfer learning from pretrained models
```

### Profiling

Enable memory profiling:
```python
profile_memory = True
# Generates memory_snapshot.pickle for analysis
```

Analyze with:
```python
import pickle
snapshot = pickle.load(open('memory_snapshot.pickle', 'rb'))
# Analyze memory usage
```

---

## References

### Papers

1. **RMSNorm:** "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
2. **RoPE:** "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
3. **Lion:** "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)
4. **Flash Attention:** "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)

### Implementations

- **LLaMA:** Uses RMSNorm, RoPE, SwiGLU
- **GPT-NeoX:** Uses RMSNorm, RoPE
- **PaLM:** Uses RMSNorm, parallel attention/MLP

### Tools

- **PyTorch 2.0:** https://pytorch.org/get-started/pytorch-2.0/
- **bitsandbytes:** https://github.com/TimDettmers/bitsandbytes
- **lion-pytorch:** https://github.com/lucidrains/lion-pytorch

---

## FAQ

**Q: Will optimized model weights work with original code?**
A: Depends on optimizations used:
- `compile=True` only: Yes, same weights
- `use_rmsnorm=True`: No, different parameters
- `use_rope=True`: No, different position encoding

**Q: Can I resume training from original checkpoint with optimized code?**
A: Yes, if using same architecture:
```python
init_from = 'resume'
# Make sure use_rmsnorm, use_rope match checkpoint
```

**Q: Which optimizer is best?**
A: For WiggleGPT:
1. **Lion:** Best speed + convergence (if you can tune LR)
2. **AdamW:** Most reliable, well-tested
3. **AdamW8bit:** Best memory efficiency
4. **Adafactor:** Good for very large models

**Q: How much faster is the optimized version?**
A: Typical speedups:
- Compile only: 20% faster
- RMSNorm + compile: 25% faster
- RMSNorm + RoPE + Lion + compile: 30-40% faster

**Q: Can I use these optimizations with standard GPT (non-bio)?**
A: Yes! Just set:
```python
use_bio_mlp = False
# All optimizations work with standard GPT
```

**Q: Do these optimizations change the model output?**
A: Slightly:
- `compile=True`: No (numerically identical)
- `use_rmsnorm=True`: Yes (different normalization)
- `use_rope=True`: Yes (different position encoding)

Final loss and generation quality should be similar or better.

---

## Summary

**Key Takeaways:**

1. **Easy wins:** Enable `compile=True` for 20% speedup with zero risk
2. **Better wins:** Add `use_rmsnorm=True` for 25% speedup with minimal risk
3. **Best wins:** Full optimization stack for 30-40% speedup (may need tuning)
4. **Core preserved:** All optimizations preserve the fundamental WiggleGPT architecture and bio-inspired features

**Recommended Starting Point:**
```python
use_rmsnorm = True
compile = True
optimizer_type = 'adamw'  # Start with familiar, switch to Lion later
```

**Next Steps:**
1. Start with conservative optimizations
2. Monitor training stability
3. Gradually add more optimizations
4. Tune learning rate if using Lion
5. Compare final losses and generation quality

Happy training! 🚀

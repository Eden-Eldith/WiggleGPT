# WiggleGPT 🧠⚡

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Ongoing%20🔬-blueviolet.svg)](#-future-work)
[![Val Loss](https://img.shields.io/badge/Val%20Loss-3.1621-purple.svg)](#wigglegpt-124m---pretraining-complete-)
[![Parameters](https://img.shields.io/badge/Parameters-124M-orange.svg)](#wigglegpt-124m---pretraining-complete-)

**Bio-Inspired Language Model with Oscillating Activation Functions**

WiggleGPT is an experimental language model that challenges a 56-year-old assumption in neural networks: **that neurons must use monotonic activation functions.**

By replacing standard ReLU/GELU activations with learnable oscillating activation functions (`sin(ω·x + φ)·tanh(x)`), WiggleGPT demonstrates that biologically-inspired neurons can achieve competitive performance with standard transformers—matching GPT-2 124M within 1.3% while using identical parameter counts.

> **📄 Read the full paper:** [WiggleGPT Paper](https://garden-backend-three.vercel.app/finalized-work/wiggle-gpt/wiggle-gpt-paper/)
> **Model Weights** | [HuggingFace](https://huggingface.co/edeneldith/WiggleGPT)
---

This project extends nanoGPT (MIT License, © Andrej Karpathy).  
All original modifications and bio-inspired components © 2025 Phillip C. O'Brien, MIT License.

---

## 🎯 Core Research Question

**Can a single neuron with oscillating activations solve XOR?**

Yes. Since Minsky and Papert's *Perceptrons* (1969), the field assumed multiple hidden layers were necessary for non-linearly separable problems like XOR. WiggleGPT demonstrates that by abandoning monotonic activations, a single neuron can solve XOR—and this principle scales to transformer-level language modeling.

---

## 📊 Results Summary

### WiggleGPT 124M - Pretraining (Complete ✅)

| Model | Parameters | Val Loss | Training Iters | Notes |
|-------|------------|----------|----------------|-------|
| **WiggleGPT 124M** | 124M | **3.1621** | 600K | Oscillating activation |
| GPT-2 124M (baseline) | 124M | ~3.12 | 600K | Standard GELU |

**Key Result:** WiggleGPT achieved validation loss within **1.3%** of the GPT-2 baseline, demonstrating that oscillating activations are a viable drop-in replacement for standard deep learning primitives at scale.

### Training Progression (124M)

| Step | Train Loss | Val Loss |
|------|------------|----------|
| 75,000 | 3.5357 | 3.4643 |
| 124,000 | 3.5069 | 3.4756 |
| 176,000 | 3.4597 | 3.4634 |
| 354,000 | 3.3162 | 3.3482 |
| 600,000 | 3.1749 | **3.1621** |

### Frequency Analysis: Did the Model Learn to Wiggle?

Analysis of all 36,864 oscillating neurons confirmed the model actively utilizes oscillation:

| Parameter | Initialization | Learned | Change |
|-----------|---------------|---------|--------|
| ω mean | 1.0 | 1.096 | +9.6% |
| ω std | 0.1 | **0.602** | **6x increase** |
| ω range | ~[0.8, 1.2] | [-0.19, 5.17] | Massive expansion |

- **95% of neurons retained active oscillation** (ω > 0.1)
- Only 5% linearized (ω ≈ 0)
- The 6x increase in frequency variance confirms the model learned diverse frequencies for different neurons

---

## 💬 Instruction Fine-Tuning (Complete ✅)

Following pretraining, WiggleGPT 124M was fine-tuned on the SmolTalk2 dataset for instruction-following capabilities.

### Dataset: SmolTalk2

| Statistic | Value |
|-----------|-------|
| Documents | 406,843 |
| Training Tokens | 386,033,668 |
| Validation Tokens | 20,145,529 |
| Average Turns | 6 |

### Fine-Tuning Results

| Step | Train Loss | Val Loss |
|------|------------|----------|
| 250 | 2.1191 | 2.1233 |
| 5,000 | 1.6619 | 1.5266 |
| 8,500 | 1.5143 | **1.3184** (best) |
| 10,000 | — | 1.5888 |

**Best validation loss: 1.3184** at step 8,500 (38% reduction from initial).

### Oscillation Parameter Stability

A critical finding: oscillation parameters (ω, φ) remained **virtually unchanged** during fine-tuning:

| Parameter | Metric | Pretrained | Fine-tuned | Change |
|-----------|--------|------------|------------|--------|
| ω | Mean | 1.0962 | 1.0964 | +0.0002 |
| φ | Mean | -0.0008 | -0.0008 | +0.0000 |

- **Mean absolute ω change:** 0.0013 (essentially unchanged)
- **Neurons with ω change > 0.1:** 0.0%

This suggests oscillation parameters encode fundamental representational patterns that remain task-agnostic, while other weights (attention, projections) adapt to specific tasks.

### Chat Usage

```bash
# Interactive chat with fine-tuned model
python chat.py

# With custom settings
python chat.py --temperature=0.5 --top_k=30
```

**Chat Template:**
```
<|user|>
{user message}
<|/user|>
<|assistant|>
{assistant response}
<|/assistant|>
```

---

## 🌟 Key Features

### 1. Oscillating Activations

```python
# Standard neuron: Linear + Static Activation
x → Linear → GELU(x) → output

# WiggleGPT neuron: Linear + Oscillating Activation  
x → Linear → sin(ω·x + φ)·tanh(x) → output
```

**Why it matters:**
- Single neurons can learn XOR and other non-linearly separable functions
- Learnable frequency (ω) and phase (φ) parameters per neuron
- The tanh envelope ensures training stability (bounded gradients)
- Provides richer representational capacity than static activations

### 2. Modern Transformer Features

- **RMSNorm**: Faster than LayerNorm
- **Rotary Position Embeddings (RoPE)**: Better length extrapolation
- **Flash Attention**: Efficient attention via PyTorch 2.0+ CUDA kernels
- **Weight Tying**: Input embeddings tied to output projection

### 3. Drop-in Compatibility

- Bio-neurons replace standard MLPs without changing transformer architecture
- Works with standard GPT-2 training procedures
- Scales to full GPT-2 configuration (12 layers, 12 heads, 768 dim)

---

## 🏗️ Architecture

### WiggleGPT Architecture

```
GPT-2 Transformer (124M params)
├── Token Embeddings (weight-tied with output)
├── Rotary Position Embeddings (RoPE)
└── Transformer Blocks (12 layers)
    ├── RMSNorm
    ├── Multi-Head Self-Attention (12 heads)
    │   └── Flash Attention (PyTorch 2.0+)
    ├── RMSNorm
    └── MLP with Oscillating Activation
        ├── Linear (768 → 3072)
        ├── sin(ω·x + φ)·tanh(x)  ← Learnable ω, φ per neuron
        └── Linear (3072 → 768)
```

### Standard MLP vs WiggleGPT MLP

**Standard:**
```
x → Linear(4x) → GELU → Linear → output
```

**WiggleGPT:**
```
x → Linear(4x) → sin(ω·x+φ)·tanh(x) → Linear → output
```

---

## 📦 Installation

### Requirements

```bash
# Python 3.8+
pip install torch numpy tiktoken datasets tqdm matplotlib
```

### Optional Dependencies

```bash
# For W&B logging
pip install wandb
```

### Quick Start

```bash
# Clone or download the project
git clone https://github.com/Eden-Eldith/WiggleGPT
cd WiggleGPT

# Prepare data (streaming mode, memory-efficient)
cd data/openwebtext
python prepare_openwebtext_streaming.py
cd ../..

# Train with bio-inspired neurons (single GPU)
python train_bio.py config_gpt2_bio_3070.py

# Sample from trained model
python sample_bio.py

# Chat with fine-tuned model
python chat.py
```

---

## 🚀 Usage

### Pretraining

```bash
# Train with oscillating neurons (default config)
python train_bio.py config_gpt2_bio_3070.py
```

### Fine-Tuning

```bash
# Fine-tune on SmolTalk2 for instruction following
python finetune_smoltalk.py

# Or use native fine-tuning script
python finetune_native.py
```

### Sampling & Chat

```bash
# Sample from pretrained model
python sample_bio.py --start="Hello WiggleGPT"

# Interactive chat with fine-tuned model
python chat.py --temperature=0.5 --top_k=30
```

---

## 📁 Project Structure

```
WiggleGPT/
├── model_bio.py              # WiggleGPT model with oscillating activations
├── model.py                  # Standard GPT-2 model
├── train_bio.py              # Pretraining script
├── sample_bio.py             # Sampling from pretrained model
├── chat.py                   # Interactive chat interface
├── finetune_smoltalk.py      # Fine-tuning on SmolTalk2
├── finetune_native.py        # Native fine-tuning script
├── config_gpt2_bio_3070.py   # Pretraining config
├── config_finetune_native.py # Fine-tuning config (native)
├── config_finetune_smoltalk.py # Fine-tuning config (SmolTalk)
├── configurator.py           # Config parsing utilities
├── LICENSE                   # MIT License
├── README.md                 # This file
│
├── Analysis scripts/         # Checkpoint analysis & visualization
│   ├── analyze_brainwaves.py         # Analyze brainwave/oscillation patterns
│   ├── analyze_wiggle-pretrain.py    # Analyze ω, φ from pretrained model
│   ├── analyze_wiggle-finetune.py    # Compare pretrain vs fine-tuned params
│   ├── generate_finetune_loss_chart.py
│   ├── generate_pretrain_loss_chart.py
│   ├── generate_social_media_visuals.py  # Social media optimized visuals
│   ├── Readme-Analysis-loss-scripts.md
│   ├── outputs-brainwave/            # Brainwave analysis outputs
│   ├── outputs-finetune/             # Fine-tuning analysis outputs
│   └── outputs-pretrain/             # Pretraining analysis outputs
│
├── data/
│   └── openwebtext/
│       └── prepare_openwebtext_streaming.py  # Memory-efficient data prep
│
├── optimizations/            # Experimental optimizations
│   ├── model_bio_optimized.py
│   ├── train_bio_optimized.py
│   ├── config_gpt2_bio_optimized.py
│   ├── requirements-optimizers.txt
│   ├── OPTIMIZATION_SUMMARY.md
│   └── TRAINING_OPTIMIZATIONS.md
│
├── Research docs/
│   ├── DENDRITIC_REMOVAL_STORY.md    # Post-mortem on dendritic routing
│   └── Wiggle-GPT paper/
│       ├── WiggleGPT Paper.md        # Full research paper
│       ├── pretrain_loss_chart.png   # Figure 1a
│       ├── finetune_loss_chart.png   # Figure 2b
│       ├── wiggle_analysis.png       # Figure 1b
│       └── wiggle_finetune_analysis.png  # Figure 2a
│
├── WiggleGPT-V1-OLD/         # Archive: v1 with dendritic compartments
└── WiggleGPT-V2-OLD/         # Archive: v2 with sparsity (45M model)
```

---

## 📊 Analysis Scripts

The `Analysis scripts/` folder contains tools for analyzing trained checkpoints and generating publication figures:

| Script | Purpose | Output |
|--------|---------|--------|
| `analyze_brainwaves.py` | Analyze brainwave/oscillation patterns | `outputs-brainwave/` |
| `analyze_wiggle-pretrain.py` | Analyze ω, φ distributions from pretrained checkpoint | `wiggle_analysis.png` |
| `analyze_wiggle-finetune.py` | Compare pretrained vs fine-tuned parameters | `wiggle_finetune_analysis.png` |
| `generate_pretrain_loss_chart.py` | Generate pretraining loss curves | `pretrain_loss_chart.png` |
| `generate_finetune_loss_chart.py` | Generate fine-tuning loss curves | `finetune_loss_chart.png` |
| `generate_social_media_visuals.py` | Generate social media optimized visuals | Various image formats |

See [`Analysis scripts/Readme-Analysis-loss-scripts.md`](./Analysis%20scripts/Readme-Analysis-loss-scripts.md) for detailed usage instructions.

---

## 🔧 Configuration

Key parameters in `config_gpt2_bio_3070.py`:

```python
# Model architecture (GPT-2 124M)
n_layer = 12         # Number of transformer layers
n_head = 12          # Number of attention heads  
n_embd = 768         # Embedding dimension

# Training batch configuration
batch_size = 2                     # Micro-batch size
block_size = 1024                  # Context length
gradient_accumulation_steps = 16   # Effective batch: 32 sequences

# Training hyperparameters
max_iters = 600000                # Total training iterations
learning_rate = 6e-4              # Peak learning rate
warmup_iters = 2000               # LR warmup steps
min_lr = 6e-5                     # Minimum LR (1/10th of peak)

# Bio-inspired neurons
use_bio_mlp = True                # Enable oscillating activations

# Optimizations
use_rmsnorm = True                # RMSNorm instead of LayerNorm
use_rope = True                   # Rotary Position Embeddings
compile = False                   # Disabled (Windows lacks Triton support)
```

**For fine-tuning** (from `config_finetune_native.py`):
```python
learning_rate = 2e-5              # Much lower than pretraining
max_iters = 10000                 # Fewer iterations needed
block_size = 512                  # Reduced for memory
batch_size = 1                    # Minimal batch
gradient_accumulation_steps = 64  # High accumulation compensates
gradient_checkpointing = True     # Trade compute for VRAM
dropout = 0.0                     # Keep at 0 for memory constraints
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` to 1-2
- Reduce `block_size` to 512
- Enable gradient checkpointing for fine-tuning

### RTX 50-series GPU Compatibility
```
CUDA capability sm_120 is not compatible with current PyTorch
```
**Solution:** Install PyTorch with CUDA 12.8 or 13.0

### Windows Performance
- Windows lacks full Triton support, so `compile=True` won't work
- MFU will be lower (~4%) compared to Linux (~20-30%)
- This is expected and doesn't indicate a problem

---

## 🔬 Research Transparency: The Dendritic Routing Detour

Early versions of WiggleGPT included "dendritic compartments"—complex routing mechanisms inspired by biological dendritic computation. This caused:

- **Parameter explosion**: 89M → 1,214M when scaling to GPT-2 size
- **42% slower training**: 1200ms → 700ms per iteration after removal
- **Violated experimental isolation**: Couldn't test oscillating neurons alone

The solution was simple: remove the complexity, keep the core idea (oscillating activations).

> Full post-mortem: [The Dendritic Routing Mistake](./Research%20docs/DENDRITIC_REMOVAL_STORY.md)

**Lesson learned:** When testing whether oscillating neurons improve transformers, test *only* oscillating neurons.

---

## 🔮 Future Work

### Potential Research Directions

1. **Shallower, Smarter Networks**
   - Can oscillating neurons reduce layer count while maintaining performance?
   - Trade depth for neuron complexity

2. **Sparsity at Scale**
   - Re-integrate lightweight sparsity (per-feature scalars) at 124M scale
   - Target biological sparsity levels (10-20%)

3. **Scaling Studies**
   - Test on larger models (350M-1B parameters)
   - Multi-GPU distributed training

4. **Neuromorphic Hardware**
   - Port to event-based hardware (Intel Loihi)
   - Exploit spike-based processing

---

## 📚 References

### Core Concepts

- Minsky, M., & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press.
- Buzsáki, G., & Draguhn, A. (2004). "Neuronal oscillations in cortical networks." *Science*, 304(5679), 1926–1929.
- Vaswani, A., et al. (2017). ["Attention is all you need."](https://arxiv.org/abs/1706.03762)
- [Neuron - Wikipedia](https://en.wikipedia.org/wiki/Neuron) — Biological neurons and oscillatory behavior
- [Artificial Neuron - Wikipedia](https://en.wikipedia.org/wiki/Artificial_neuron) — Why standard neurons can't solve XOR

### Code Attribution

- **Base Architecture**: [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy
- **Bio-Inspired Components**: Original implementation by Phillip C. O'Brien

---

## 🤝 Contributing

Contributions welcome:

- 🐛 **Bug Reports**: Open an issue with reproduction steps
- 💡 **Feature Requests**: Suggest bio-inspired mechanisms to implement
- 🔬 **Research Collaborations**: Share experimental results or theoretical insights
- 📝 **Documentation**: Improve explanations or add tutorials

**Research integrity:**
- When reporting results, always include full context (model size, iterations, hardware)
- If you find mistakes, report them transparently (as done throughout this project)
- Isolate variables in experiments—test one thing at a time

---

## 📖 Citation

If you use WiggleGPT in your research, please cite:

```bibtex
@misc{obrien2025wigglegpt,
  author       = {Phillip C. O'Brien},
  title        = {WiggleGPT: Revisiting the Monotonicity Assumption in Neural Networks via Oscillating Activation Functions},
  year         = {2025},
  howpublished = {\url{https://github.com/Eden-Eldith/WiggleGPT}},
  note         = {Transformer architecture with oscillating activation functions}
}
```

---

## 📬 Contact

- **GitHub Issues**: Technical questions and bug reports
- **Email**: pcobrien@hotmail.co.uk
- **ORCID**: 0009-0007-3961-1182

---

## License

**GNU GPLv3** (as of 2nd December 2025)

This project is a derivative of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy, originally released under the MIT License.

All modifications, including the oscillating activation function (BioMLP), are Copyright (C) 2025 Eden_Eldith (P.C. O'Brien) c: and licensed under GNU GPLv3.

This means any derivative works based on WiggleGPT must also be open source under GPLv3.

See the [LICENSE](LICENSE) file for full terms.

---

## 🙏 Acknowledgments

- **Andrej Karpathy** for nanoGPT - clean, educational GPT implementation
- **OpenWebText** community for the open dataset
- **HuggingFace** for the SmolTalk2 dataset
- **Neuroscience community** for research on oscillatory neural dynamics
- **PyTorch team** for the deep learning framework
- **Claude (Anthropic), GPT-4.5/GPT-5 (OpenAI), Gemini 2.5 Pro (Google)** for development assistance

---

<details>
<summary><b>📅 Development Log (October–November 2025)</b></summary>

### 124M Model Training (v3)

**12/11/2025 02:58AM** — Training finished! `step 600000: train loss 3.1749, val loss 3.1621`

**07/11/2025 05:26AM** — Training resumed with new GPU (RTX 5060 Ti 16GB). Config changed to `batch_size=4, gradient_accumulation_steps=8`. Required PyTorch with CUDA 12.8 for sm_120 compatibility.
> step 354000: train loss 3.3162, val loss 3.3482

**28/10/2025 01:36AM** — GPU order cancelled by seller 😭 Money stuck in eBay refund system for 3-5 days. Continuing on RTX 3070.
> step 176000: train loss 3.4597, val loss 3.4634

**26/10/2025 19:20PM** — Bought a 3090 for £550.44 including shipping. GPU arriving 30th.
> step 124000: train loss 3.5069, val loss 3.4756

**25/10/2025 15:51PM** — Training at step 75000 out of 600K. Researching second-hand 3090 prices—£500 is significant.
> step 75000: train loss 3.5357, val loss 3.4643

**23/10/2025 21:05PM** — Removed sparsity for later investigation. Running oscillating activations only at 124M params on single RTX 3070 for the full 600k iterations.

**23/10/2025** — Discovered Windows doesn't support Triton (`compile=True` unavailable). Identified second parameter explosion from sparsity implementation when scaling to GPT-2 config (~300M params instead of 124M). Developed lightweight sparsity alternative using per-feature scalars instead of full linear layers.

### 45M Model Training (v2)

**23/10/2025 00:44AM** — v2 model at iter 185800. Loss hovering around 3.4-3.6.

**22/10/2025 14:18PM** — `step 136000: train loss 3.5760, val loss 3.5547, sparsity 13.30%` (45.31M params). Already matching the 89M dendritic model—dendritic routing was definitely causing issues.

**22/10/2025** — Overnight training showing promising results. `step 108000: train loss 3.7259, val loss 3.7661, sparsity 12.08%`

**21/10/2025** — Began removing Dendritic Compartments. Model dropped from 89M to 49M params with same config (4 layers, 6 heads, 384 dim).

</details>

---

## 📝 Version History

| Version | Date | Highlights |
|---------|------|------------|
| **v3** | November 2025 | 124M pretraining complete (3.1621 val loss), instruction fine-tuning on SmolTalk2 (1.3184 val loss) |
| **v2** | October 2025 | Removed dendritic routing, 45M model with sparsity (3.587 val loss) |
| **v1** | October 2025 | Included dendritic compartments, 89M params, unscalable |

---

**Last Updated**: November 30, 2025  
**Status**: Research Complete - Pretraining & Fine-Tuning ✅

---

<p align="center">
  <b>WiggleGPT</b><br>
  Challenging a 56-year-old assumption, one wiggle at a time 🧠⚡
</p>

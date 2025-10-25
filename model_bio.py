"""
Optimized GPT with Bio-Inspired Neurons - Training Speed Improvements
=====================================================================
Extended from model_bio.py with modern training optimizations while
preserving the core WiggleGPT architecture.

New optimizations:
1. RMSNorm - Faster alternative to LayerNorm (~7% speedup, simpler computation)
2. Rotary Position Embeddings (RoPE) - More efficient than learned embeddings
3. Gradient checkpointing support - Trade compute for memory
4. Better initialization - GPT-NeoX style depth scaling
5. Flash Attention 2 hints - Better kernel selection

These optimizations preserve the core model config (n_layer, n_head, n_embd, etc.)
and bio-inspired features (oscillating activations, sparse events).
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


# ==============================================================================
# OPTIMIZED COMPONENTS
# ==============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    Simpler and faster than LayerNorm:
    - No mean subtraction (only RMS normalization)
    - No bias parameter
    - ~7% faster than LayerNorm
    - Used in LLaMA, GPT-NeoX, etc.

    Formula: x * (gamma / rms(x)) where rms(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute rotary position embedding frequencies.

    Used for RoPE (Rotary Position Embeddings) - more efficient than learned
    positional embeddings and enables better length extrapolation.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    Apply rotary position embeddings to query and key tensors.

    RoPE encodes position information through rotation in complex space,
    which naturally handles relative positions and enables length extrapolation.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis[:xq_.shape[1]]  # Truncate to sequence length
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# ==============================================================================
# BIO-INSPIRED COMPONENTS (unchanged from model_bio.py)
# ==============================================================================

class OscillatingActivation(nn.Module):
    """
    Oscillating activation function: sin(ω·x + φ)·tanh(x) + baseline

    Enables single neurons to learn XOR and other non-linearly separable functions.
    Key research finding: biological neurons can solve XOR, artificial neurons cannot.
    """
    def __init__(self, num_features: int, learnable: bool = True):
        super().__init__()
        if learnable:
            self.omega = nn.Parameter(torch.randn(num_features) * 0.1 + 1.0)
            self.phi = nn.Parameter(torch.randn(num_features) * 0.1)
            self.baseline = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_buffer('omega', torch.ones(num_features))
            self.register_buffer('phi', torch.zeros(num_features))
            self.register_buffer('baseline', torch.zeros(num_features))

    def forward(self, x):
        osc = torch.sin(self.omega * x + self.phi)
        return osc * torch.tanh(x) + self.baseline


# SparseEventLayer removed - testing pure oscillating neurons only


class BioMLP(nn.Module):
    """
    PURE Bio-inspired MLP: Just oscillating activations, nothing else.
    
    Drop-in replacement for standard MLP in GPT architecture.
    Same parameter count as baseline - only difference is the activation function.
    
    Standard MLP:  Linear → GELU → Linear
    WiggleGPT MLP: Linear → sin(ω·x + φ)·tanh(x) → Linear
    
    Tests the core hypothesis: Can oscillating neurons beat static activations?
    """
    def __init__(self, config):
        super().__init__()
        
        # Standard transformer MLP structure
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        
        # ONLY CHANGE: Replace GELU with oscillating activation
        self.activation = OscillatingActivation(4 * config.n_embd, learnable=True)
        
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)  # Pure oscillating, no gating, no complexity
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
    def get_avg_sparsity(self):
        # No sparsity in this version - testing pure oscillation
        return None


# ==============================================================================
# OPTIMIZED NANOGPT COMPONENTS
# ==============================================================================

class CausalSelfAttention(nn.Module):
    """
    Optimized multi-head causal self-attention with:
    - Flash Attention support
    - Optional RoPE (Rotary Position Embeddings)
    - Better initialization hints
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.use_rope = config.use_rope

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, freqs_cis=None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Apply rotary embeddings if enabled
        if self.use_rope and freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # causal self-attention
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Standard nanoGPT MLP"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer block with optional gradient checkpointing.
    """
    def __init__(self, config):
        super().__init__()
        # Use RMSNorm if enabled, otherwise standard LayerNorm
        norm_class = RMSNorm if config.use_rmsnorm else nn.LayerNorm
        if config.use_rmsnorm:
            self.ln_1 = norm_class(config.n_embd)
            self.ln_2 = norm_class(config.n_embd)
        else:
            # Standard LayerNorm with bias option
            self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        self.attn = CausalSelfAttention(config)

        # Choose MLP type based on config
        if config.use_bio_mlp:
            self.mlp = BioMLP(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x, freqs_cis=None):
        x = x + self.attn(self.ln_1(x), freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x

    def get_avg_sparsity(self):
        """Get sparsity metric if using bio MLP"""
        if hasattr(self.mlp, 'get_avg_sparsity'):
            return self.mlp.get_avg_sparsity()
        return None


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    # Bio-inspired neuron parameters
    use_bio_mlp: bool = False  # Use oscillating neurons instead of GELU

    # Optimization parameters
    use_rmsnorm: bool = False  # Use RMSNorm instead of LayerNorm (faster)
    use_rope: bool = False  # Use Rotary Position Embeddings instead of learned
    gradient_checkpointing: bool = False  # Trade compute for memory


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        module_dict = {
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        }

        # Add positional embeddings if not using RoPE
        if not config.use_rope:
            module_dict['wpe'] = nn.Embedding(config.block_size, config.n_embd)

        # Add final norm
        if config.use_rmsnorm:
            module_dict['ln_f'] = RMSNorm(config.n_embd)
        else:
            module_dict['ln_f'] = nn.LayerNorm(config.n_embd, bias=config.bias)

        self.transformer = nn.ModuleDict(module_dict)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # Precompute RoPE frequencies if using RoPE
        if config.use_rope:
            freqs_cis = precompute_freqs_cis(
                config.n_embd // config.n_head,
                config.block_size * 2  # Allow for some extrapolation
            )
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Init all weights
        self.apply(self._init_weights)

        # Apply special scaled init to the residual projections (GPT-NeoX style)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                # GPT-NeoX uses 2 * n_layer for better depth scaling
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.config.use_rope:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if self.config.use_rope:
            # RoPE: no position embeddings added, positions encoded in attention
            x = self.transformer.drop(tok_emb)
            freqs_cis = self.freqs_cis[:t]
        else:
            # Standard learned positional embeddings
            pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
            pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
            freqs_cis = None

        # Apply transformer blocks with optional gradient checkpointing
        if self.config.gradient_checkpointing and self.training:
            for block in self.transformer.h:
                x = checkpoint(block, x, freqs_cis, use_reentrant=False)
        else:
            for block in self.transformer.h:
                x = block(x, freqs_cis)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, optimizer_type='adamw'):
        """
        Configure optimizer with support for multiple optimizer types.

        Supported optimizers:
        - 'adamw': Standard AdamW (fused if available on CUDA)
        - 'adamw8bit': 8-bit AdamW from bitsandbytes (memory efficient)
        - 'lion': Lion optimizer (faster convergence, lower memory)
        - 'adafactor': Adafactor (memory efficient, no beta2)
        """
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        if optimizer_type == 'adamw':
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")

        elif optimizer_type == 'adamw8bit':
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(optim_groups, lr=learning_rate, betas=betas)
                print("using 8-bit AdamW from bitsandbytes")
            except ImportError:
                print("WARNING: bitsandbytes not installed, falling back to standard AdamW")
                print("Install with: pip install bitsandbytes")
                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        elif optimizer_type == 'lion':
            try:
                from lion_pytorch import Lion
                optimizer = Lion(optim_groups, lr=learning_rate, betas=betas)
                print("using Lion optimizer")
            except ImportError:
                print("WARNING: lion-pytorch not installed, falling back to AdamW")
                print("Install with: pip install lion-pytorch")
                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        elif optimizer_type == 'adafactor':
            try:
                from transformers.optimization import Adafactor
                # Adafactor doesn't use weight_decay in the same way, merge groups
                all_params = decay_params + nodecay_params
                optimizer = Adafactor(
                    all_params,
                    lr=learning_rate,
                    scale_parameter=False,
                    relative_step=False,
                    warmup_init=False
                )
                print("using Adafactor optimizer")
            except ImportError:
                print("WARNING: transformers not installed, falling back to AdamW")
                print("Install with: pip install transformers")
                optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def get_avg_sparsity(self):
        """Get average sparsity across all blocks if using bio MLP"""
        sparsities = []
        for block in self.transformer.h:
            s = block.get_avg_sparsity()
            if s is not None:
                sparsities.append(s)
        return sum(sparsities) / len(sparsities) if sparsities else None
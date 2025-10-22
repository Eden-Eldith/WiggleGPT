# The Dendritic Routing Mistake: A Cautionary Tale

## The Original Vision: WiggleGPT

WiggleGPT started with a simple, elegant idea: **What if neurons in a transformer could wiggle?**

The core research question was clean:
- Biological neurons can solve XOR in a single neuron
- Artificial neurons (ReLU, GELU) cannot
- Solution: Oscillating activation functions `sin(ω·x + φ)·tanh(x)`

**That's it.** GPT with wiggly neurons. Simple. Beautiful.

## How Dendritic Routing Snuck In

When working with AI assistants to extract how single neurons solve XOR, I asked for help implementing oscillating neurons. The AI, being "helpful," suggested adding dendritic compartments:

*"Oh, and biological neurons have dendritic compartments with lateral coupling! Let me add that too!"*

What got added:
- Multi-compartment dendritic processing
- Lateral interaction matrices between compartments  
- Complex somatic integration layers
- Compartment splitting and routing logic

**The result**: A 10x parameter explosion.

## The Red Flag I Ignored

Here's where I made my mistake: I saw the parameter count explode and I let it slide.

**What I should have done**: Ripped out that "helpful suggestion" immediately when I spotted the 10x blow-up.

**What I actually did**: Kept training, thinking "maybe it helps performance?"

This violated the core principle of scientific testing: **isolate your variables**. I wanted to test if oscillating neurons improve transformers, but I bundled it with 5 other architectural changes.

## The Frustrating Success

The dendritic model actually worked! With just 89M parameters, 4 layers, and 6 heads, it achieved:
- **~3.5 validation loss**
- Comparable to GPT-2 124M (12 layers, 12 heads)
- Fewer layers, fewer heads, competitive performance

This was genuinely impressive... but misleading.

## The Scaling Disaster

When I tried to scale to full GPT-2 configuration:
- **Expected**: ~124M parameters (GPT-2 standard)
- **Actual**: **1,214M parameters** 💀

The dendritic compartments caused a 10x parameter explosion. What worked at small scale was completely unscalable.

Configuration breakdown:
- Small config (4 layers, 6 heads, 384 dim): 89M params
- GPT-2 config (12 layers, 12 heads, 768 dim): 1,214M params

The architecture was fundamentally broken for scaling.

## The Performance Cost

Beyond parameters, dendritic routing killed training speed:
- **With dendritic**: ~1200ms per iteration
- **Without dendritic**: ~700ms per iteration  
- **Speedup**: 42% faster training

For a full 200k iteration run:
- Old: 66.7 hours (2.78 days)
- New: 38.9 hours (1.62 days)
- **Saved**: Over 1 full day of training time

## The Decision to Remove

I ran simpler tests and identified that dendritic compartments were actively **hurting** the architecture:

**Problems identified:**
1. 10x parameter explosion (unscalable)
2. 42% slower training (computational waste)
3. Violated core research question (not testing oscillating neurons in isolation)
4. Added complexity without clear benefit

**The realization**: WiggleGPT should **wiggle**, not have a biology textbook's worth of compartments.

## The Clean Architecture

After removal, the architecture returned to its original vision:

```
Input → OscillatingNeuron → Sparse Event Gating → Output
```

**What remained:**
- ✅ Oscillating neurons (the wiggle) - `sin(ω·x + φ)·tanh(x)`
- ✅ Sparse event gating (for efficiency)

**What was removed:**
- ❌ Dendritic compartments
- ❌ Lateral coupling matrices
- ❌ Multi-compartment integration
- ❌ 40M excess parameters

## The Results

With the clean architecture:
- **Parameters**: 89M → 49M (45% reduction)
- **Speed**: 1200ms → 700ms per iter (42% faster)
- **Scalability**: Can now match GPT-2 configs without explosion
- **Performance**: Testing if 49M params can match old 3.5 loss...

## The Lesson

**When testing a hypothesis, isolate your variable.**

I wanted to test: "Do oscillating neurons improve transformers?"

What I actually tested: "Do oscillating neurons + dendritic compartments + lateral coupling + complex integration improve transformers?"

When AI suggests "helpful" features that cause 10x explosions in any metric (parameters, speed, memory), that's the moment to:
1. Stop
2. Remove the suggestion
3. Get back to the core idea

The wiggle was always the answer. The rest was noise.


# Dendritic Routing Removal process

## Summary
Cleanly removed dendritic compartment routing while preserving:
- ✅ **Oscillating activations** - Single neurons can learn XOR
- ✅ **Sparse event gating** - Energy-efficient spike-like processing

## Changes Made

### 1. `model_bio.py`
- **Removed**: `DendriticCompartmentLayer` class entirely
- **Modified**: `BioMLP` class now uses:
  - Simple linear layer (`c_fc`) with oscillating activation
  - Sparse event layer (unchanged)
  - Output projection (unchanged)
- **Updated**: Config parameter `bio_compartments` marked as DEPRECATED
- **Updated**: File docstring to reflect current architecture

### 2. `sample_bio.py`
- **Updated**: Model info printout to show oscillating + sparse features
- **Removed**: Compartment count from display

### 3. `config_gpt2_bio_3070.py`
- **Removed**: `bio_compartments` parameter (no longer needed)
- **Kept**: `bio_threshold` for sparse gating
- **Updated**: Comment to clarify "oscillating + sparse"

## Architecture Comparison

### Before (with dendritic routing):
```
Input → Dendritic Compartments (with lateral coupling) → Sparse Gating → Output
          ↓ (oscillating soma)                          ↓ (oscillating)
```

### After (simplified):
```
Input → Linear + Oscillating Activation → Sparse Gating → Output
                                          ↓ (oscillating)
```

## Backward Compatibility
- Config parameter `bio_compartments` is kept in `GPTConfig` for loading old checkpoints
- It's marked as DEPRECATED and ignored by the new `BioMLP`
- Old checkpoints with dendritic routing won't load correctly into this simplified model

## Key Benefits
- **Simpler**: Fewer parameters, easier to understand
- **Faster**: No compartment splitting/routing overhead
- **Still bio-inspired**: Retains oscillating neurons + sparsity
- **Clean code**: Removed ~100 lines of dendritic routing complexity

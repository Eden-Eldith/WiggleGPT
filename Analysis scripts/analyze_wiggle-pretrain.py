"""
WiggleGPT Frequency Analysis
============================
Analyzes the learned omega (frequency) and phi (phase) parameters
to verify the model is actually using oscillation.

Based on Gemini's script with enhancements:
- Shows deviation from initialization (omega started at mean=1.0)
- Adds statistics summary
- Handles both wrapped and raw state dicts
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# Path to your checkpoint
ckpt_path = 'ckpt.pt'  # Change if your file is named differently

print(f"Loading checkpoint from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location='cpu')

# Handle both raw state_dicts and nanoGPT's wrapped dict
if 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint

# Lists to store our learned parameters
omegas = []
phis = []
baselines = []

print("\nScanning model parameters...")
print("-" * 60)

# Find all omega, phi, and baseline parameters
for key, value in state_dict.items():
    if 'omega' in key.lower():
        print(f"Found omega: {key} | Shape: {value.shape}")
        omegas.extend(value.float().numpy().flatten())
    
    if 'phi' in key.lower():
        print(f"Found phi:   {key} | Shape: {value.shape}")
        phis.extend(value.float().numpy().flatten())
    
    if 'baseline' in key.lower() and 'activation' in key.lower():
        print(f"Found baseline: {key} | Shape: {value.shape}")
        baselines.extend(value.float().numpy().flatten())

print("-" * 60)

if len(omegas) == 0:
    print("\nERROR: No parameters named 'omega' found!")
    print("\nAll keys in checkpoint:")
    for key in sorted(state_dict.keys()):
        print(f"  {key}")
else:
    omegas = np.array(omegas)
    phis = np.array(phis)
    
    print(f"\n📊 STATISTICS SUMMARY")
    print("=" * 60)
    print(f"Total omega parameters: {len(omegas):,}")
    print(f"Total phi parameters:   {len(phis):,}")
    print()
    
    # Omega statistics
    print("OMEGA (Frequency) Analysis:")
    print(f"  Initialization: mean=1.0, std=0.1")
    print(f"  Learned:        mean={omegas.mean():.4f}, std={omegas.std():.4f}")
    print(f"  Range:          [{omegas.min():.4f}, {omegas.max():.4f}]")
    print(f"  Deviation from init mean: {abs(omegas.mean() - 1.0):.4f}")
    
    # Key question: did it stay near initialization?
    near_init = np.sum(np.abs(omegas - 1.0) < 0.2) / len(omegas) * 100
    print(f"  % within ±0.2 of init (1.0): {near_init:.1f}%")
    print()
    
    # Phi statistics  
    print("PHI (Phase) Analysis:")
    print(f"  Initialization: mean=0.0, std=0.1")
    print(f"  Learned:        mean={phis.mean():.4f}, std={phis.std():.4f}")
    print(f"  Range:          [{phis.min():.4f}, {phis.max():.4f}]")
    print()
    
    # Interpretation
    print("=" * 60)
    print("🔍 INTERPRETATION:")
    if omegas.std() > 0.3:
        print("  ✅ HIGH VARIANCE in omega - model learned diverse frequencies!")
        print("     This supports the 'wiggle' hypothesis.")
    elif omegas.std() > 0.15:
        print("  ⚠️  MODERATE VARIANCE in omega - some frequency learning occurred.")
    else:
        print("  ❌ LOW VARIANCE in omega - model may not be using oscillation fully.")
    
    if abs(omegas.mean() - 1.0) > 0.2:
        print(f"  📈 Mean shifted from 1.0 to {omegas.mean():.3f} - active learning!")
    else:
        print(f"  📊 Mean stayed near initialization ({omegas.mean():.3f})")
    
    if np.any(np.abs(omegas) < 0.1):
        pct_near_zero = np.sum(np.abs(omegas) < 0.1) / len(omegas) * 100
        print(f"  ⚠️  {pct_near_zero:.1f}% of neurons have omega ≈ 0 (linearized)")
    
    print("=" * 60)
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Omega histogram
    ax1 = axes[0, 0]
    ax1.hist(omegas, bins=100, color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Init mean (1.0)')
    ax1.axvline(x=omegas.mean(), color='green', linestyle='-', linewidth=2, label=f'Learned mean ({omegas.mean():.3f})')
    ax1.set_title('Distribution of Learned Frequencies (ω)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Frequency Value')
    ax1.set_ylabel('Count of Neurons')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Phi histogram
    ax2 = axes[0, 1]
    ax2.hist(phis, bins=100, color='orange', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axvline(x=0.0, color='red', linestyle='--', linewidth=2, label='Init mean (0.0)')
    ax2.axvline(x=phis.mean(), color='green', linestyle='-', linewidth=2, label=f'Learned mean ({phis.mean():.3f})')
    ax2.set_title('Distribution of Learned Phases (φ)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Phase Value (Radians)')
    ax2.set_ylabel('Count of Neurons')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Omega deviation from initialization
    ax3 = axes[1, 0]
    deviation = omegas - 1.0
    ax3.hist(deviation, bins=100, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change from init')
    ax3.set_title('Omega Deviation from Initialization (ω - 1.0)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Deviation from Initial Value')
    ax3.set_ylabel('Count of Neurons')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Omega vs Phi scatter (sample if too many points)
    ax4 = axes[1, 1]
    if len(omegas) > 10000:
        idx = np.random.choice(len(omegas), 10000, replace=False)
        ax4.scatter(omegas[idx], phis[idx], alpha=0.3, s=1)
    else:
        ax4.scatter(omegas, phis, alpha=0.3, s=1)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_title('Omega vs Phi (sampled)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Omega (Frequency)')
    ax4.set_ylabel('Phi (Phase)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wiggle_analysis.png', dpi=150)
    print(f"\n💾 Analysis saved to 'wiggle_analysis.png'")
    plt.show()
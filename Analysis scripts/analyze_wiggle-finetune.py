"""
WiggleGPT Frequency Analysis - Fine-tuned Model
================================================
Analyzes the learned omega (frequency) and phi (phase) parameters
after instruction fine-tuning on SmolTalk2.

Compares pretrained vs fine-tuned parameter distributions to see
how SFT affected the oscillating neuron parameters.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths to checkpoints
pretrain_ckpt_path = '../out-wigglegpt-pure-124m/ckpt.pt'
finetune_ckpt_path = 'ckpt.pt'  # Current directory

def load_oscillation_params(ckpt_path, name="model"):
    """Load omega and phi parameters from a checkpoint"""
    print(f"\nLoading {name} from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Handle both raw state_dicts and nanoGPT's wrapped dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    omegas = []
    phis = []
    
    for key, value in state_dict.items():
        if 'omega' in key.lower():
            omegas.extend(value.float().numpy().flatten())
        if 'phi' in key.lower():
            phis.extend(value.float().numpy().flatten())
    
    return np.array(omegas), np.array(phis), checkpoint

# Load both checkpoints
pretrain_omegas, pretrain_phis, pretrain_ckpt = load_oscillation_params(pretrain_ckpt_path, "Pretrained")
finetune_omegas, finetune_phis, finetune_ckpt = load_oscillation_params(finetune_ckpt_path, "Fine-tuned")

print("\n" + "=" * 70)
print("📊 WIGGLEGPT FINE-TUNING OSCILLATION ANALYSIS")
print("=" * 70)

# Training info
if 'iter_num' in finetune_ckpt:
    print(f"\nFine-tuning iterations: {finetune_ckpt['iter_num']}")
if 'best_val_loss' in finetune_ckpt:
    print(f"Best validation loss: {finetune_ckpt['best_val_loss']:.4f}")

print(f"\nTotal omega parameters: {len(finetune_omegas):,}")
print(f"Total phi parameters:   {len(finetune_phis):,}")

# Compare statistics
print("\n" + "-" * 70)
print("OMEGA (Frequency) Comparison:")
print("-" * 70)
print(f"{'Metric':<30} {'Pretrained':>15} {'Fine-tuned':>15} {'Change':>15}")
print("-" * 70)
print(f"{'Mean':<30} {pretrain_omegas.mean():>15.4f} {finetune_omegas.mean():>15.4f} {finetune_omegas.mean() - pretrain_omegas.mean():>+15.4f}")
print(f"{'Std Dev':<30} {pretrain_omegas.std():>15.4f} {finetune_omegas.std():>15.4f} {finetune_omegas.std() - pretrain_omegas.std():>+15.4f}")
print(f"{'Min':<30} {pretrain_omegas.min():>15.4f} {finetune_omegas.min():>15.4f} {finetune_omegas.min() - pretrain_omegas.min():>+15.4f}")
print(f"{'Max':<30} {pretrain_omegas.max():>15.4f} {finetune_omegas.max():>15.4f} {finetune_omegas.max() - pretrain_omegas.max():>+15.4f}")

# Compute parameter drift
omega_drift = np.abs(finetune_omegas - pretrain_omegas)
print(f"\n{'Mean absolute change':<30} {omega_drift.mean():>15.4f}")
print(f"{'Max absolute change':<30} {omega_drift.max():>15.4f}")
print(f"{'% params changed > 0.1':<30} {100 * np.mean(omega_drift > 0.1):>14.1f}%")
print(f"{'% params changed > 0.5':<30} {100 * np.mean(omega_drift > 0.5):>14.1f}%")

print("\n" + "-" * 70)
print("PHI (Phase) Comparison:")
print("-" * 70)
print(f"{'Metric':<30} {'Pretrained':>15} {'Fine-tuned':>15} {'Change':>15}")
print("-" * 70)
print(f"{'Mean':<30} {pretrain_phis.mean():>15.4f} {finetune_phis.mean():>15.4f} {finetune_phis.mean() - pretrain_phis.mean():>+15.4f}")
print(f"{'Std Dev':<30} {pretrain_phis.std():>15.4f} {finetune_phis.std():>15.4f} {finetune_phis.std() - pretrain_phis.std():>+15.4f}")
print(f"{'Min':<30} {pretrain_phis.min():>15.4f} {finetune_phis.min():>15.4f} {finetune_phis.min() - pretrain_phis.min():>+15.4f}")
print(f"{'Max':<30} {pretrain_phis.max():>15.4f} {finetune_phis.max():>15.4f} {finetune_phis.max() - pretrain_phis.max():>+15.4f}")

phi_drift = np.abs(finetune_phis - pretrain_phis)
print(f"\n{'Mean absolute change':<30} {phi_drift.mean():>15.4f}")
print(f"{'Max absolute change':<30} {phi_drift.max():>15.4f}")
print(f"{'% params changed > 0.1':<30} {100 * np.mean(phi_drift > 0.1):>14.1f}%")
print(f"{'% params changed > 0.5':<30} {100 * np.mean(phi_drift > 0.5):>14.1f}%")

# Interpretation
print("\n" + "=" * 70)
print("🔍 INTERPRETATION:")
print("=" * 70)

if omega_drift.mean() < 0.05:
    print("  ✅ Omega parameters barely changed - oscillation patterns preserved!")
    print("     Fine-tuning adapted OTHER weights while keeping frequencies stable.")
elif omega_drift.mean() < 0.15:
    print("  ⚠️  Moderate omega drift - some frequency adaptation occurred.")
    print("     The model adjusted its oscillation to better fit instruction data.")
else:
    print("  📊 Significant omega drift - substantial frequency relearning.")
    print("     Instruction-following may require different oscillation patterns.")

if phi_drift.mean() < 0.05:
    print("  ✅ Phase parameters stable - timing patterns preserved.")
else:
    print(f"  📊 Phase shifted by {phi_drift.mean():.3f} on average.")

# Check if any neurons "died" (omega → 0)
near_zero_before = np.mean(np.abs(pretrain_omegas) < 0.1) * 100
near_zero_after = np.mean(np.abs(finetune_omegas) < 0.1) * 100
print(f"\n  Neurons with ω ≈ 0 (linearized):")
print(f"    Before fine-tuning: {near_zero_before:.2f}%")
print(f"    After fine-tuning:  {near_zero_after:.2f}%")

print("=" * 70)

# --- PLOTTING ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('WiggleGPT: Pretrained vs Fine-tuned Oscillation Parameters', fontsize=14, fontweight='bold')

# Plot 1: Omega comparison histogram
ax1 = axes[0, 0]
ax1.hist(pretrain_omegas, bins=80, alpha=0.6, label='Pretrained', color='blue', density=True)
ax1.hist(finetune_omegas, bins=80, alpha=0.6, label='Fine-tuned', color='red', density=True)
ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=1, label='Init (1.0)')
ax1.set_title('Omega (Frequency) Distribution')
ax1.set_xlabel('Omega Value')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Phi comparison histogram
ax2 = axes[0, 1]
ax2.hist(pretrain_phis, bins=80, alpha=0.6, label='Pretrained', color='blue', density=True)
ax2.hist(finetune_phis, bins=80, alpha=0.6, label='Fine-tuned', color='red', density=True)
ax2.axvline(x=0.0, color='black', linestyle='--', linewidth=1, label='Init (0.0)')
ax2.set_title('Phi (Phase) Distribution')
ax2.set_xlabel('Phi Value')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Parameter drift histogram
ax3 = axes[0, 2]
ax3.hist(omega_drift, bins=80, alpha=0.7, label='Omega drift', color='purple')
ax3.hist(phi_drift, bins=80, alpha=0.5, label='Phi drift', color='orange')
ax3.set_title('Parameter Drift (|Fine-tuned - Pretrained|)')
ax3.set_xlabel('Absolute Change')
ax3.set_ylabel('Count')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Pretrained omega vs Fine-tuned omega scatter
ax4 = axes[1, 0]
if len(pretrain_omegas) > 5000:
    idx = np.random.choice(len(pretrain_omegas), 5000, replace=False)
    ax4.scatter(pretrain_omegas[idx], finetune_omegas[idx], alpha=0.2, s=2)
else:
    ax4.scatter(pretrain_omegas, finetune_omegas, alpha=0.2, s=2)
lims = [min(pretrain_omegas.min(), finetune_omegas.min()), max(pretrain_omegas.max(), finetune_omegas.max())]
ax4.plot(lims, lims, 'r--', linewidth=1, label='No change line')
ax4.set_title('Omega: Pretrained vs Fine-tuned')
ax4.set_xlabel('Pretrained Omega')
ax4.set_ylabel('Fine-tuned Omega')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Pretrained phi vs Fine-tuned phi scatter
ax5 = axes[1, 1]
if len(pretrain_phis) > 5000:
    idx = np.random.choice(len(pretrain_phis), 5000, replace=False)
    ax5.scatter(pretrain_phis[idx], finetune_phis[idx], alpha=0.2, s=2)
else:
    ax5.scatter(pretrain_phis, finetune_phis, alpha=0.2, s=2)
lims = [min(pretrain_phis.min(), finetune_phis.min()), max(pretrain_phis.max(), finetune_phis.max())]
ax5.plot(lims, lims, 'r--', linewidth=1, label='No change line')
ax5.set_title('Phi: Pretrained vs Fine-tuned')
ax5.set_xlabel('Pretrained Phi')
ax5.set_ylabel('Fine-tuned Phi')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Fine-tuned omega vs phi scatter
ax6 = axes[1, 2]
if len(finetune_omegas) > 5000:
    idx = np.random.choice(len(finetune_omegas), 5000, replace=False)
    ax6.scatter(finetune_omegas[idx], finetune_phis[idx], alpha=0.2, s=2, c='green')
else:
    ax6.scatter(finetune_omegas, finetune_phis, alpha=0.2, s=2, c='green')
ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax6.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
ax6.set_title('Fine-tuned: Omega vs Phi')
ax6.set_xlabel('Omega (Frequency)')
ax6.set_ylabel('Phi (Phase)')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wiggle_finetune_analysis.png', dpi=150)
print(f"\n💾 Analysis saved to 'wiggle_finetune_analysis.png'")
plt.show()

"""
Generate fine-tuning loss chart for WiggleGPT paper.
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from finetune_log.md - validation checkpoints
steps = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 
         3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 
         6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000, 8250, 8500, 
         8750, 9000, 9250, 9500, 9750]

train_loss = [2.1191, 1.9307, 1.9094, 1.9563, 1.7044, 1.6534, 1.7463, 1.7408, 
              1.7182, 1.5984, 1.6565, 1.6883, 1.6842, 1.6502, 1.7235, 1.5844, 
              1.6378, 1.6298, 1.6788, 1.6619, 1.5353, 1.5778, 1.7215, 1.6783, 
              1.4848, 1.5975, 1.5523, 1.5224, 1.5291, 1.7821, 1.5803, 1.4409, 
              1.6926, 1.5143, 1.5065, 1.5394, 1.5022, 1.6195, 1.6527]

val_loss = [2.1233, 1.9708, 1.9433, 1.7124, 1.7303, 1.8193, 1.7900, 1.7040, 
            1.6267, 1.7254, 1.6257, 1.7028, 1.7842, 1.7025, 1.6451, 1.6615, 
            1.6670, 1.5931, 1.5635, 1.5266, 1.6073, 1.5472, 1.6970, 1.7003, 
            1.7681, 1.5638, 1.4391, 1.6125, 1.5258, 1.5451, 1.5112, 1.4131, 
            1.6089, 1.3184, 1.5930, 1.5115, 1.5449, 1.4941, 1.5888]

# Create figure with nice styling
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 6))

# Plot losses
ax.plot(steps, train_loss, 'b-', linewidth=2, label='Train Loss', alpha=0.8)
ax.plot(steps, val_loss, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)

# Mark best validation loss point
best_idx = np.argmin(val_loss)
best_step = steps[best_idx]
best_val = val_loss[best_idx]
ax.scatter([best_step], [best_val], color='green', s=150, zorder=5, 
           edgecolors='darkgreen', linewidths=2)
ax.annotate(f'Best: {best_val:.4f}\n(step {best_step})', 
            xy=(best_step, best_val), xytext=(best_step + 500, best_val - 0.15),
            fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# Mark final checkpoint
ax.scatter([steps[-1]], [val_loss[-1]], color='orange', s=100, zorder=5,
           edgecolors='darkorange', linewidths=2, marker='s')
ax.annotate(f'Final: {val_loss[-1]:.4f}', 
            xy=(steps[-1], val_loss[-1]), xytext=(steps[-1] - 1200, val_loss[-1] + 0.12),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.5))

# Add horizontal line at best val loss for reference
ax.axhline(y=best_val, color='green', linestyle='--', alpha=0.3, linewidth=1)

# Labels and title
ax.set_xlabel('Training Iterations', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.set_title('WiggleGPT Fine-Tuning Loss Progression\nSmolTalk2 Dataset (406K instruction-response pairs)', 
             fontsize=14, fontweight='bold')

# Legend
ax.legend(loc='upper right', fontsize=11)

# Set axis limits with some padding
ax.set_xlim(0, 10500)
ax.set_ylim(1.2, 2.3)

# Add annotation for overfitting region
ax.axvspan(8500, 10000, alpha=0.1, color='red')
ax.text(9250, 2.15, 'Overfitting\nRegion', ha='center', fontsize=10, 
        color='darkred', fontstyle='italic')

# Grid
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('finetune_loss_chart.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("✅ Loss chart saved to 'finetune_loss_chart.png'")

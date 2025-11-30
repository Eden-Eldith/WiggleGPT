"""
Generate pretraining loss chart for WiggleGPT paper.
Uses data from trainlog4_reconstructed.md
"""
import matplotlib.pyplot as plt
import numpy as np

# Data from trainlog4_reconstructed.md - key checkpoints for clean visualization
# Using sparse early points + dense later points
steps = [0, 11000, 13000, 27000, 75000, 124000, 176000, 240000, 
         273000, 294000, 300000, 312000, 332000, 340000,
         555000, 560000, 585000, 600000]

train_loss = [10.9410, 3.8124, 3.7640, 3.6428, 3.5357, 3.5069, 3.4597, 3.4279,
              3.3855, 3.3855, 3.3632, 3.3773, 3.3770, 3.3278,
              3.1810, 3.1721, 3.1742, 3.1749]

val_loss = [10.9413, 3.8164, 3.7825, 3.6579, 3.4643, 3.4756, 3.4634, 3.3895,
            3.3529, 3.3529, 3.3441, 3.3375, 3.3285, 3.3241,
            3.1460, 3.1332, 3.1315, 3.1621]

# Create figure with nice styling
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Full training curve (log scale for early drop)
ax1.semilogy(steps, train_loss, 'b-', linewidth=2, label='Train Loss', alpha=0.8, marker='o', markersize=4)
ax1.semilogy(steps, val_loss, 'r-', linewidth=2, label='Validation Loss', alpha=0.8, marker='s', markersize=4)

ax1.set_xlabel('Training Iterations', fontsize=11, fontweight='bold')
ax1.set_ylabel('Loss (log scale)', fontsize=11, fontweight='bold')
ax1.set_title('Full Pretraining Curve\n(600K iterations on OpenWebText)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Add annotations for key phases
ax1.axvspan(0, 50000, alpha=0.1, color='green')
ax1.text(25000, 6, 'Rapid\nDescent', ha='center', fontsize=9, color='darkgreen', fontstyle='italic')

# Right plot: Final convergence region (linear scale, last 100K)
# Get more granular data for final phase
steps_final = [555000, 556000, 557000, 558000, 559000, 560000, 561000, 562000, 
               563000, 564000, 565000, 566000, 567000, 568000, 569000, 570000,
               571000, 572000, 573000, 574000, 575000, 576000, 577000, 578000,
               579000, 580000, 581000, 582000, 583000, 584000, 585000, 586000,
               587000, 588000, 589000, 590000, 591000, 592000, 593000, 594000,
               595000, 596000, 597000, 598000, 599000, 600000]

train_final = [3.1810, 3.1647, 3.1826, 3.2001, 3.1850, 3.1721, 3.2052, 3.1594,
               3.1652, 3.1625, 3.1812, 3.1813, 3.1764, 3.1674, 3.1635, 3.1815,
               3.1812, 3.1774, 3.1313, 3.1790, 3.1751, 3.1724, 3.1344, 3.1682,
               3.1370, 3.1829, 3.1689, 3.1682, 3.1348, 3.1638, 3.1742, 3.1742,
               3.1563, 3.1522, 3.1513, 3.1628, 3.1367, 3.2052, 3.1689, 3.1504,
               3.1751, 3.1724, 3.1684, 3.1660, 3.1706, 3.1749]

val_final = [3.1460, 3.1572, 3.1797, 3.1872, 3.1473, 3.1332, 3.1656, 3.1530,
             3.1864, 3.1717, 3.1746, 3.1596, 3.1646, 3.1787, 3.1465, 3.1838,
             3.1746, 3.1707, 3.1952, 3.1838, 3.1467, 3.1837, 3.1470, 3.1580,
             3.1632, 3.1500, 3.1648, 3.1580, 3.1783, 3.1675, 3.1315, 3.1315,
             3.1603, 3.1741, 3.1451, 3.1437, 3.1966, 3.1616, 3.1539, 3.1743,
             3.1467, 3.1837, 3.1804, 3.1811, 3.1365, 3.1621]

ax2.plot(steps_final, train_final, 'b-', linewidth=1.5, label='Train Loss', alpha=0.7)
ax2.plot(steps_final, val_final, 'r-', linewidth=1.5, label='Validation Loss', alpha=0.7)

# Mark final point
ax2.scatter([600000], [3.1621], color='green', s=120, zorder=5, 
            edgecolors='darkgreen', linewidths=2)
ax2.annotate(f'Final: 3.1621', 
             xy=(600000, 3.1621), xytext=(590000, 3.21),
             fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

# Add GPT-2 baseline reference
ax2.axhline(y=3.12, color='purple', linestyle='--', alpha=0.5, linewidth=2, label='GPT-2 124M baseline (~3.12)')

ax2.set_xlabel('Training Iterations', fontsize=11, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
ax2.set_title('Final Convergence (Last 45K iterations)\nApproaching GPT-2 Baseline', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.set_ylim(3.05, 3.25)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pretrain_loss_chart.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.close()

print("✅ Pretraining loss chart saved to 'pretrain_loss_chart.png'")

"""
WiggleGPT EEG-Style Brainwave Analysis
======================================
Analyzes learned omega (frequency) parameters to see if WiggleGPT
developed emergent brainwave-like oscillation patterns.

Maps neural frequencies to EEG bands:
- Delta (δ): 0.5-4 Hz   - Deep sleep, unconscious
- Theta (θ): 4-8 Hz     - Drowsy, light sleep, meditation  
- Alpha (α): 8-13 Hz    - Relaxed, calm, alert
- Beta (β): 13-30 Hz    - Active thinking, focus, anxiety
- Gamma (γ): 30-100+ Hz - Higher cognition, perception, consciousness

Note: Our omega values aren't literal Hz, but we can analyze the
DISTRIBUTION and RATIOS to see if similar patterns emerged!

Run: python analyze_brainwaves.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import signal
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PRETRAIN_CKPT = "../out-wigglegpt-pure-124m/ckpt.pt"
FINETUNE_CKPT = "../out-wigglegpt-finetune-native/ckpt.pt"
OUTPUT_DIR = Path("./outputs-brainwave")
OUTPUT_DIR.mkdir(exist_ok=True)

# EEG-inspired frequency bands (mapped to omega ranges)
# We'll normalize omega to roughly match EEG Hz ranges for visualization
EEG_BANDS = {
    'Delta (δ)': {'range': (0.0, 0.4), 'color': '#9B59B6', 'desc': 'Deep patterns'},
    'Theta (θ)': {'range': (0.4, 0.8), 'color': '#3498DB', 'desc': 'Memory, learning'},
    'Alpha (α)': {'range': (0.8, 1.2), 'color': '#2ECC71', 'desc': 'Baseline, calm'},
    'Beta (β)':  {'range': (1.2, 1.8), 'color': '#F39C12', 'desc': 'Active processing'},
    'Gamma (γ)': {'range': (1.8, 3.0), 'color': '#E74C3C', 'desc': 'High cognition'},
}

plt.style.use('dark_background')

# ============================================================================
# LOAD CHECKPOINTS
# ============================================================================

def load_wiggle_params(ckpt_path):
    """Extract omega, phi parameters organized by layer."""
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    layer_params = {}
    
    for key, value in state_dict.items():
        if 'omega' in key.lower() or 'phi' in key.lower():
            parts = key.split('.')
            layer_num = None
            for i, p in enumerate(parts):
                if p == 'h' and i + 1 < len(parts):
                    try:
                        layer_num = int(parts[i + 1])
                        break
                    except:
                        pass
            
            if layer_num is None:
                layer_num = 0
                
            if layer_num not in layer_params:
                layer_params[layer_num] = {'omega': None, 'phi': None}
            
            if 'omega' in key.lower():
                layer_params[layer_num]['omega'] = value.float().numpy().flatten()
            elif 'phi' in key.lower():
                layer_params[layer_num]['phi'] = value.float().numpy().flatten()
    
    all_omega = np.concatenate([p['omega'] for p in layer_params.values() if p['omega'] is not None])
    all_phi = np.concatenate([p['phi'] for p in layer_params.values() if p['phi'] is not None])
    
    return layer_params, all_omega, all_phi


def classify_to_bands(omega):
    """Classify omega values into EEG-like bands."""
    band_counts = {}
    band_neurons = {}
    
    for band_name, band_info in EEG_BANDS.items():
        low, high = band_info['range']
        mask = (np.abs(omega) >= low) & (np.abs(omega) < high)
        band_counts[band_name] = np.sum(mask)
        band_neurons[band_name] = omega[mask]
    
    # Handle values outside defined ranges
    all_defined = sum(band_counts.values())
    band_counts['Other'] = len(omega) - all_defined
    
    return band_counts, band_neurons


# ============================================================================
# VISUALIZATION 1: EEG-STYLE POWER SPECTRUM
# ============================================================================

def create_eeg_power_spectrum(omega_pretrain, omega_finetune, output_path):
    """Create EEG-style power spectrum comparison."""
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor('#0a0a0a')
    
    for ax, omega, title, color_accent in zip(
        axes, 
        [omega_pretrain, omega_finetune],
        ['Pretrained', 'Finetuned'],
        ['#00d4ff', '#ff6b6b']
    ):
        ax.set_facecolor('#0a0a0a')
        
        # Create smooth density estimate (like EEG power spectrum)
        omega_abs = np.abs(omega)
        omega_clipped = np.clip(omega_abs, 0, 3)
        
        # KDE for smooth curve
        kde = gaussian_kde(omega_clipped, bw_method=0.05)
        x_range = np.linspace(0, 3, 500)
        density = kde(x_range)
        
        # Normalize to look like power
        power = density / density.max()
        
        # Fill under curve with gradient effect
        for band_name, band_info in EEG_BANDS.items():
            low, high = band_info['range']
            mask = (x_range >= low) & (x_range <= high)
            ax.fill_between(x_range[mask], 0, power[mask], 
                           color=band_info['color'], alpha=0.4, label=band_name)
            ax.fill_between(x_range[mask], 0, power[mask], 
                           color=band_info['color'], alpha=0.2)
        
        # Main power line
        ax.plot(x_range, power, color=color_accent, linewidth=2, alpha=0.9)
        
        # Add band labels
        for band_name, band_info in EEG_BANDS.items():
            mid = (band_info['range'][0] + band_info['range'][1]) / 2
            ax.axvline(x=band_info['range'][0], color='white', alpha=0.2, linestyle='--', linewidth=0.5)
            
            # Get power at band center
            idx = np.argmin(np.abs(x_range - mid))
            y_pos = power[idx] + 0.05
            
            ax.text(mid, min(y_pos, 0.95), band_name.split()[0], 
                   ha='center', va='bottom', fontsize=9, color=band_info['color'],
                   fontweight='bold', alpha=0.9)
        
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('Frequency (ω)', fontsize=12, color='#e0e1dd')
        ax.set_ylabel('Power (normalized)', fontsize=12, color='#e0e1dd')
        ax.set_title(f'WiggleGPT {title} - Neural Frequency Spectrum', 
                    fontsize=14, color='white', fontweight='bold', pad=10)
        ax.tick_params(colors='#778da9')
        ax.grid(True, alpha=0.1, color='#415a77')
        
        # Add badge
        ax.text(0.02, 0.95, f'🧠 {title.upper()}', transform=ax.transAxes,
               fontsize=11, color=color_accent, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#1a1a2a', alpha=0.9, edgecolor=color_accent))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()


# ============================================================================
# VISUALIZATION 2: EEG BAND POWER BAR CHART
# ============================================================================

def create_band_power_comparison(omega_pretrain, omega_finetune, output_path):
    """Compare band power distribution like EEG analysis."""
    
    bands_pretrain, _ = classify_to_bands(omega_pretrain)
    bands_finetune, _ = classify_to_bands(omega_finetune)
    
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    band_names = list(EEG_BANDS.keys())
    x = np.arange(len(band_names))
    width = 0.35
    
    # Calculate percentages
    total_pre = sum(bands_pretrain[b] for b in band_names)
    total_fine = sum(bands_finetune[b] for b in band_names)
    
    pct_pretrain = [bands_pretrain[b] / total_pre * 100 for b in band_names]
    pct_finetune = [bands_finetune[b] / total_fine * 100 for b in band_names]
    
    colors = [EEG_BANDS[b]['color'] for b in band_names]
    
    # Create bars
    bars1 = ax.bar(x - width/2, pct_pretrain, width, label='Pretrained',
                   color=colors, alpha=0.7, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, pct_finetune, width, label='Finetuned',
                   color=colors, alpha=0.4, edgecolor='white', linewidth=1, hatch='//')
    
    # Add value labels
    for bar, pct in zip(bars1, pct_pretrain):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, color='white')
    
    for bar, pct in zip(bars2, pct_finetune):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, color='#aaa')
    
    ax.set_xlabel('Neural Frequency Band', fontsize=12, color='#e0e1dd')
    ax.set_ylabel('% of Neurons', fontsize=12, color='#e0e1dd')
    ax.set_title('WiggleGPT Brainwave Band Distribution\nPretrained vs Finetuned', 
                fontsize=16, color='white', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{b}\n{EEG_BANDS[b]["desc"]}' for b in band_names], fontsize=10)
    ax.tick_params(colors='#778da9')
    ax.legend(loc='upper right', facecolor='#1a1a2a', edgecolor='#415a77', labelcolor='white')
    ax.grid(True, alpha=0.1, axis='y', color='#415a77')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()


# ============================================================================
# VISUALIZATION 3: EEG CHANNEL PLOT (Layer = Channel)
# ============================================================================

def create_eeg_channel_plot(layer_params_pretrain, layer_params_finetune, output_path):
    """Create EEG-style multi-channel plot where each layer is a 'channel'."""
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 12), dpi=150)
    fig.patch.set_facecolor('#0a0a0a')
    
    for ax, layer_params, title, color_accent in zip(
        axes,
        [layer_params_pretrain, layer_params_finetune],
        ['Pretrained', 'Finetuned'],
        ['#00d4ff', '#ff6b6b']
    ):
        ax.set_facecolor('#0a0a0a')
        
        n_layers = len(layer_params)
        t = np.linspace(0, 4 * np.pi, 500)  # Time axis
        
        for i, (layer_num, params) in enumerate(sorted(layer_params.items())):
            omega = params['omega']
            phi = params['phi']
            
            if omega is None:
                continue
            
            # Sample a few neurons from this layer to create composite wave
            n_sample = min(20, len(omega))
            idx = np.random.choice(len(omega), n_sample, replace=False)
            
            # Create composite oscillation (like EEG superposition)
            composite = np.zeros_like(t)
            for j in idx:
                composite += np.sin(omega[j] * t + phi[j]) / n_sample
            
            # Offset for display
            offset = (n_layers - 1 - i) * 2.5
            
            # Color by dominant frequency band
            mean_omega = np.abs(omega).mean()
            for band_name, band_info in EEG_BANDS.items():
                if band_info['range'][0] <= mean_omega < band_info['range'][1]:
                    color = band_info['color']
                    break
            else:
                color = '#888888'
            
            ax.plot(t, composite + offset, color=color, linewidth=0.8, alpha=0.9)
            ax.fill_between(t, offset - 0.1, composite + offset, alpha=0.1, color=color)
            
            # Layer label
            ax.text(-0.3, offset, f'L{layer_num}', ha='right', va='center',
                   fontsize=10, color='white', fontweight='bold')
            
            # Band label
            ax.text(t[-1] + 0.2, offset, f'μω={mean_omega:.2f}', ha='left', va='center',
                   fontsize=8, color='#778da9')
        
        ax.set_xlim(-1, t[-1] + 2)
        ax.set_ylim(-2, n_layers * 2.5 + 1)
        ax.set_xlabel('Time (simulated)', fontsize=12, color='#e0e1dd')
        ax.set_ylabel('Layer (Channel)', fontsize=12, color='#e0e1dd')
        ax.set_title(f'WiggleGPT {title} - EEG-Style Layer Activity', 
                    fontsize=14, color='white', fontweight='bold', pad=10)
        ax.tick_params(colors='#778da9')
        ax.set_yticks([])
        
        # Add badge
        ax.text(0.02, 0.98, f'🧠 {title.upper()}', transform=ax.transAxes,
               fontsize=11, color=color_accent, fontweight='bold', va='top',
               bbox=dict(boxstyle='round', facecolor='#1a1a2a', alpha=0.9, edgecolor=color_accent))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()


# ============================================================================
# VISUALIZATION 4: TOPOGRAPHIC "BRAIN MAP"
# ============================================================================

def create_brain_topography(layer_params_pretrain, layer_params_finetune, output_path):
    """Create topographic brain-map style visualization."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    fig.patch.set_facecolor('#0a0a0a')
    
    for ax, layer_params, title, cmap_name in zip(
        axes,
        [layer_params_pretrain, layer_params_finetune],
        ['Pretrained', 'Finetuned'],
        ['viridis', 'plasma']
    ):
        ax.set_facecolor('#0a0a0a')
        
        # Create a circular "brain" layout
        n_layers = len(layer_params)
        
        # Arrange layers in a brain-like oval
        theta = np.linspace(0.2 * np.pi, 0.8 * np.pi, n_layers)
        
        # Create background head shape
        head_theta = np.linspace(0, 2 * np.pi, 100)
        head_x = 1.3 * np.cos(head_theta)
        head_y = 1.4 * np.sin(head_theta)
        ax.plot(head_x, head_y, color='#444', linewidth=2)
        ax.fill(head_x, head_y, color='#1a1a2a', alpha=0.5)
        
        # Nose indicator
        ax.plot([0, 0], [1.4, 1.6], color='#444', linewidth=2)
        ax.plot([-0.1, 0, 0.1], [1.55, 1.7, 1.55], color='#444', linewidth=2)
        
        # Ears
        for sign in [-1, 1]:
            ear_x = sign * 1.3 + sign * 0.15 * np.cos(np.linspace(0, 2*np.pi, 20))
            ear_y = 0.15 * np.sin(np.linspace(0, 2*np.pi, 20))
            ax.plot(ear_x, ear_y, color='#444', linewidth=1.5)
        
        # Get all omega values for color scaling
        all_omega_std = []
        for params in layer_params.values():
            if params['omega'] is not None:
                all_omega_std.append(params['omega'].std())
        
        vmin, vmax = min(all_omega_std), max(all_omega_std)
        
        # Plot electrode-like circles for each layer
        for i, (layer_num, params) in enumerate(sorted(layer_params.items())):
            omega = params['omega']
            if omega is None:
                continue
            
            # Position in oval brain shape
            angle = theta[i]
            r = 0.8 + 0.3 * (i / n_layers)  # Layers spiral outward slightly
            x = r * np.cos(angle)
            y = r * np.sin(angle) * 0.9  # Slightly compressed vertically
            
            # Color by omega std (activity level)
            omega_std = omega.std()
            color_val = (omega_std - vmin) / (vmax - vmin + 1e-6)
            color = cm.get_cmap(cmap_name)(color_val)
            
            # Size by mean omega
            size = 200 + 300 * np.abs(omega.mean())
            
            ax.scatter([x], [y], s=size, c=[color], alpha=0.8, edgecolors='white', linewidths=1.5)
            ax.text(x, y, f'L{layer_num}', ha='center', va='center', fontsize=8, 
                   color='white', fontweight='bold')
        
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1.8, 2.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'WiggleGPT {title}\nNeural Frequency Topography', 
                    fontsize=14, color='white', fontweight='bold', pad=10)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
        cbar.set_label('Frequency Variance (σω)', color='#e0e1dd', fontsize=10)
        cbar.ax.yaxis.set_tick_params(color='#778da9')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#778da9')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()


# ============================================================================
# VISUALIZATION 5: FREQUENCY COHERENCE HEATMAP
# ============================================================================

def create_coherence_heatmap(layer_params_pretrain, layer_params_finetune, output_path):
    """Create layer-to-layer frequency coherence heatmap (like EEG coherence)."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    fig.patch.set_facecolor('#0a0a0a')
    
    for ax, layer_params, title, cmap in zip(
        axes,
        [layer_params_pretrain, layer_params_finetune],
        ['Pretrained', 'Finetuned'],
        ['Blues', 'Reds']
    ):
        ax.set_facecolor('#0a0a0a')
        
        n_layers = len(layer_params)
        coherence_matrix = np.zeros((n_layers, n_layers))
        
        layer_omegas = []
        for layer_num in sorted(layer_params.keys()):
            omega = layer_params[layer_num]['omega']
            if omega is not None:
                layer_omegas.append(omega)
        
        # Compute coherence (correlation of frequency distributions)
        for i in range(n_layers):
            for j in range(n_layers):
                # Use histogram-based similarity
                bins = np.linspace(0, 3, 50)
                hist_i, _ = np.histogram(np.abs(layer_omegas[i]), bins=bins, density=True)
                hist_j, _ = np.histogram(np.abs(layer_omegas[j]), bins=bins, density=True)
                
                # Correlation as coherence
                coherence = np.corrcoef(hist_i, hist_j)[0, 1]
                coherence_matrix[i, j] = coherence
        
        im = ax.imshow(coherence_matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        
        # Add coherence values
        for i in range(n_layers):
            for j in range(n_layers):
                val = coherence_matrix[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
        
        ax.set_xticks(range(n_layers))
        ax.set_yticks(range(n_layers))
        ax.set_xticklabels([f'L{i}' for i in range(n_layers)], fontsize=9)
        ax.set_yticklabels([f'L{i}' for i in range(n_layers)], fontsize=9)
        ax.tick_params(colors='#778da9')
        ax.set_xlabel('Layer', fontsize=11, color='#e0e1dd')
        ax.set_ylabel('Layer', fontsize=11, color='#e0e1dd')
        ax.set_title(f'WiggleGPT {title}\nInter-Layer Frequency Coherence', 
                    fontsize=13, color='white', fontweight='bold', pad=10)
        
        plt.colorbar(im, ax=ax, shrink=0.8, label='Coherence')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 WiggleGPT EEG-Style Brainwave Analysis")
    print("=" * 60)
    
    # Load both checkpoints
    print("Loading checkpoints...")
    layer_params_pre, omega_pre, phi_pre = load_wiggle_params(PRETRAIN_CKPT)
    layer_params_fine, omega_fine, phi_fine = load_wiggle_params(FINETUNE_CKPT)
    print(f"✅ Pretrained: {len(omega_pre):,} neurons | Finetuned: {len(omega_fine):,} neurons")
    
    # Band classification
    bands_pre, _ = classify_to_bands(omega_pre)
    bands_fine, _ = classify_to_bands(omega_fine)
    
    print("\n📊 Brainwave Band Distribution:")
    print("-" * 50)
    print(f"{'Band':<15} {'Pretrained':>12} {'Finetuned':>12} {'Change':>10}")
    print("-" * 50)
    for band in EEG_BANDS.keys():
        pre_pct = bands_pre[band] / len(omega_pre) * 100
        fine_pct = bands_fine[band] / len(omega_fine) * 100
        change = fine_pct - pre_pct
        print(f"{band:<15} {pre_pct:>11.1f}% {fine_pct:>11.1f}% {change:>+9.1f}%")
    print("-" * 50)
    
    print(f"\n📁 Output: {OUTPUT_DIR.absolute()}\n")
    
    print("⚡ [1/5] EEG Power Spectrum...")
    create_eeg_power_spectrum(omega_pre, omega_fine, OUTPUT_DIR / "eeg_power_spectrum.png")
    
    print("📊 [2/5] Band Power Comparison...")
    create_band_power_comparison(omega_pre, omega_fine, OUTPUT_DIR / "band_power_comparison.png")
    
    print("📈 [3/5] EEG Channel Plot...")
    create_eeg_channel_plot(layer_params_pre, layer_params_fine, OUTPUT_DIR / "eeg_channels.png")
    
    print("🧠 [4/5] Brain Topography Map...")
    create_brain_topography(layer_params_pre, layer_params_fine, OUTPUT_DIR / "brain_topography.png")
    
    print("🔗 [5/5] Coherence Heatmap...")
    create_coherence_heatmap(layer_params_pre, layer_params_fine, OUTPUT_DIR / "coherence_heatmap.png")
    
    print("\n" + "=" * 60)
    print("✨ Done! All brainwave visualizations saved!")
    print("=" * 60)
    
    # Print interpretation
    print("\n🔍 INTERPRETATION:")
    
    # Find dominant band
    dominant_pre = max(bands_pre.items(), key=lambda x: x[1] if x[0] != 'Other' else 0)
    dominant_fine = max(bands_fine.items(), key=lambda x: x[1] if x[0] != 'Other' else 0)
    
    print(f"   Pretrained dominant band:  {dominant_pre[0]}")
    print(f"   Finetuned dominant band:   {dominant_fine[0]}")
    
    # Check for interesting patterns
    alpha_pre = bands_pre.get('Alpha (α)', 0) / len(omega_pre) * 100
    alpha_fine = bands_fine.get('Alpha (α)', 0) / len(omega_fine) * 100
    
    if alpha_pre > 40:
        print(f"   ⚡ High Alpha presence ({alpha_pre:.1f}%) - Model learned stable baseline oscillations!")
    
    gamma_pre = bands_pre.get('Gamma (γ)', 0) / len(omega_pre) * 100
    gamma_fine = bands_fine.get('Gamma (γ)', 0) / len(omega_fine) * 100
    
    if gamma_fine > gamma_pre + 5:
        print(f"   🚀 Gamma increased after finetuning - Higher cognitive processing patterns!")
    
    print("\n   Note: These are analogies to brain waves, not literal Hz frequencies.")
    print("   The patterns show how the model distributed its oscillation frequencies!")

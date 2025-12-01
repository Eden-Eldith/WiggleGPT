"""
WiggleGPT Social Media Visualizations
======================================
Creates stunning visuals from your trained WiggleGPT checkpoint for social media.

Outputs:
1. wiggle_galaxy.png - Cosmic scatter plot of all neurons in omega/phi space
2. wiggle_waves.mp4 - Animated oscillation of all neurons (1080p high quality)
3. wiggle_fingerprint.png - Layer-by-layer frequency fingerprint
4. wiggle_3d_landscape.mp4 - 3D rotating visualization of learned parameters

Run: python generate_social_media_visuals.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CHECKPOINT_PATH = "../out-wigglegpt-pure-124m/ckpt.pt"  # Pretrained checkpoint
OUTPUT_DIR = Path("./outputs-pretrain")
MODEL_LABEL = "Pretrained"  # Label to show in visualizations

# Video quality settings (high quality for beefy PC)
VIDEO_DPI = 150          # Higher DPI = sharper video
VIDEO_FPS = 60           # Smooth 60fps
VIDEO_BITRATE = 20000    # 20 Mbps for crisp quality
WAVE_DURATION = 6        # seconds
LANDSCAPE_DURATION = 10  # seconds

OUTPUT_DIR.mkdir(exist_ok=True)

# Style settings
plt.style.use('dark_background')

# ============================================================================
# LOAD CHECKPOINT
# ============================================================================

def load_wiggle_params(ckpt_path):
    """Extract omega, phi, and baseline parameters from checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Organize by layer
    layer_params = {}
    
    for key, value in state_dict.items():
        if 'omega' in key.lower() or 'phi' in key.lower() or ('baseline' in key.lower() and 'activation' in key.lower()):
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
                layer_params[layer_num] = {'omega': None, 'phi': None, 'baseline': None}
            
            if 'omega' in key.lower():
                layer_params[layer_num]['omega'] = value.float().numpy().flatten()
            elif 'phi' in key.lower():
                layer_params[layer_num]['phi'] = value.float().numpy().flatten()
            elif 'baseline' in key.lower():
                layer_params[layer_num]['baseline'] = value.float().numpy().flatten()
    
    all_omega = np.concatenate([p['omega'] for p in layer_params.values() if p['omega'] is not None])
    all_phi = np.concatenate([p['phi'] for p in layer_params.values() if p['phi'] is not None])
    
    return layer_params, all_omega, all_phi


# ============================================================================
# VISUALIZATION 1: NEURAL GALAXY
# ============================================================================

def create_neural_galaxy(omega, phi, output_path, label):
    """Create a cosmic scatter plot of all neurons in omega/phi parameter space."""
    
    fig, ax = plt.subplots(figsize=(16, 16), dpi=200)
    fig.patch.set_facecolor('#0d1b2a')
    ax.set_facecolor('#0d1b2a')
    
    n_points = len(omega)
    if n_points > 50000:
        idx = np.random.choice(n_points, 50000, replace=False)
        omega_sample = omega[idx]
        phi_sample = phi[idx]
    else:
        omega_sample = omega
        phi_sample = phi
    
    magnitude = np.sqrt(omega_sample**2 + phi_sample**2)
    colors_galaxy = ['#1a1a2e', '#16213e', '#0f3460', '#533483', '#e94560', '#00d4ff', '#ffffff']
    cmap = LinearSegmentedColormap.from_list('wiggle_galaxy', colors_galaxy)
    
    for alpha, size in [(0.1, 80), (0.2, 40), (0.4, 15), (0.8, 5)]:
        ax.scatter(omega_sample, phi_sample, c=magnitude, cmap=cmap, 
                   alpha=alpha, s=size, edgecolors='none')
    
    ax.scatter([omega.mean()], [phi.mean()], c='white', s=500, alpha=0.3, edgecolors='none')
    ax.scatter([omega.mean()], [phi.mean()], c='#00d4ff', s=200, alpha=0.6, edgecolors='none')
    
    ax.set_xlabel('ω (Frequency)', fontsize=14, color='#e0e1dd', fontweight='bold')
    ax.set_ylabel('φ (Phase)', fontsize=14, color='#e0e1dd', fontweight='bold')
    ax.tick_params(colors='#778da9')
    
    # Title with label
    title = f"WiggleGPT Neural Galaxy ({label})\n{len(omega):,} Oscillating Neurons"
    ax.set_title(title, fontsize=20, color='white', fontweight='bold', pad=20)
    
    # Stats annotation with label badge
    stats_text = f"🏷️ {label.upper()}\nω range: [{omega.min():.2f}, {omega.max():.2f}]\nφ range: [{phi.min():.2f}, {phi.max():.2f}]\nω std: {omega.std():.3f}"
    ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=11, color='#e0e1dd', verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#1b263b', alpha=0.9, edgecolor='#00d4ff', linewidth=2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor='#0d1b2a', edgecolor='none',
                bbox_inches='tight', pad_inches=0.5)
    plt.close()


# ============================================================================
# VISUALIZATION 2: OSCILLATION WAVE ANIMATION (HIGH QUALITY)
# ============================================================================

def create_wave_animation(omega, phi, output_path, label, duration=WAVE_DURATION, fps=VIDEO_FPS):
    """Animate oscillating activation functions - HIGH QUALITY 1080p."""
    
    n_neurons = min(500, len(omega))
    idx = np.random.choice(len(omega), n_neurons, replace=False)
    omega_sample = omega[idx]
    phi_sample = phi[idx]
    
    sort_idx = np.argsort(omega_sample)
    omega_sample = omega_sample[sort_idx]
    phi_sample = phi_sample[sort_idx]
    
    # 1080p resolution
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=VIDEO_DPI)
    fig.patch.set_facecolor('#0d1b2a')
    ax.set_facecolor('#0d1b2a')
    
    x = np.linspace(-3, 3, 400)  # More points for smoother curves
    colors = cm.plasma(np.linspace(0, 1, n_neurons))
    frames = duration * fps
    
    def oscillating_activation(x, omega, phi, t):
        return np.sin(omega * x + phi + t) * np.tanh(x)
    
    lines = []
    for i in range(n_neurons):
        line, = ax.plot([], [], color=colors[i], alpha=0.4, linewidth=1.0)
        lines.append(line)
    
    hero_line, = ax.plot([], [], color='#00d4ff', alpha=1, linewidth=4)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Input x', fontsize=16, color='#e0e1dd', fontweight='bold')
    ax.set_ylabel('Activation', fontsize=16, color='#e0e1dd', fontweight='bold')
    ax.tick_params(colors='#778da9', labelsize=12)
    ax.grid(True, alpha=0.2, color='#415a77')
    
    # Label badge
    label_text = ax.text(0.02, 0.95, f'🏷️ {label.upper()}', transform=ax.transAxes,
                         fontsize=14, color='#00d4ff', fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='#1b263b', alpha=0.9, edgecolor='#00d4ff'))
    
    title = ax.set_title('', fontsize=22, color='white', fontweight='bold', pad=15)
    
    def init():
        for line in lines:
            line.set_data([], [])
        hero_line.set_data([], [])
        return lines + [hero_line, title]
    
    def animate(frame):
        t = frame / fps * 2 * np.pi
        
        for i, line in enumerate(lines):
            y = oscillating_activation(x, omega_sample[i], phi_sample[i], t)
            line.set_data(x, y)
        
        y_hero = oscillating_activation(x, omega_sample.mean(), phi_sample.mean(), t)
        hero_line.set_data(x, y_hero)
        
        title.set_text(f'WiggleGPT {label}: {n_neurons} Oscillating Neurons\nsin(ω·x + φ + t) · tanh(x)')
        
        return lines + [hero_line, title]
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                    frames=frames, interval=1000/fps, blit=True)
    
    writer = animation.FFMpegWriter(
        fps=fps, 
        bitrate=VIDEO_BITRATE,
        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'slow', '-crf', '18']
    )
    try:
        anim.save(output_path, writer=writer)
    except Exception as e:
        gif_path = output_path.replace('.mp4', '.gif')
        anim.save(gif_path, writer='pillow', fps=30)
    
    plt.close()


# ============================================================================
# VISUALIZATION 3: LAYER FINGERPRINT
# ============================================================================

def create_layer_fingerprint(layer_params, output_path, label):
    """Create layer-by-layer frequency distribution visualization."""
    
    n_layers = len(layer_params)
    fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    fig.patch.set_facecolor('#0d1b2a')
    ax.set_facecolor('#0d1b2a')
    
    cmap = cm.viridis
    
    for i, (layer_num, params) in enumerate(sorted(layer_params.items())):
        omega = params['omega']
        if omega is None:
            continue
        
        hist, bins = np.histogram(omega, bins=100, range=(omega.min() - 0.5, omega.max() + 0.5))
        hist = hist / hist.max()
        bin_centers = (bins[:-1] + bins[1:]) / 2
        color = cmap(i / n_layers)
        
        ax.fill_between(bin_centers, i - hist * 0.4, i + hist * 0.4, 
                        color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
        
        ax.text(omega.min() - 0.3, i, f'Layer {layer_num}', 
                color='white', fontsize=10, ha='right', va='center')
        ax.text(omega.max() + 0.1, i, f'μ={omega.mean():.2f} σ={omega.std():.2f}', 
                color='#778da9', fontsize=8, ha='left', va='center', fontfamily='monospace')
    
    ax.set_xlabel('Omega (ω) - Learned Frequency', fontsize=14, color='#e0e1dd', fontweight='bold')
    ax.set_ylabel('Transformer Layer', fontsize=14, color='#e0e1dd', fontweight='bold')
    ax.set_title(f'WiggleGPT Layer Fingerprint ({label})\nFrequency Distribution per Layer', 
                 fontsize=18, color='white', fontweight='bold', pad=20)
    
    ax.tick_params(colors='#778da9')
    ax.set_yticks([])
    ax.axvline(x=1.0, color='#e94560', linestyle='--', linewidth=2, alpha=0.7, label='Init mean (1.0)')
    ax.legend(loc='upper right', facecolor='#1b263b', edgecolor='#415a77', labelcolor='white')
    
    # Label badge
    ax.text(0.02, 0.98, f'🏷️ {label.upper()}', transform=ax.transAxes,
            fontsize=12, color='#00d4ff', fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#1b263b', alpha=0.9, edgecolor='#00d4ff'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor='#0d1b2a', edgecolor='none', bbox_inches='tight')
    plt.close()


# ============================================================================
# VISUALIZATION 4: 3D ROTATING LANDSCAPE (HIGH QUALITY)
# ============================================================================

def create_3d_landscape(omega, phi, output_path, label, duration=LANDSCAPE_DURATION, fps=VIDEO_FPS):
    """Create rotating 3D visualization - HIGH QUALITY 1080p."""
    
    n_points = min(15000, len(omega))
    idx = np.random.choice(len(omega), n_points, replace=False)
    omega_sample = omega[idx]
    phi_sample = phi[idx]
    
    height = np.abs(omega_sample - 1.0) + 0.1 * np.random.randn(n_points)
    
    # 1080p resolution
    fig = plt.figure(figsize=(19.2, 10.8), dpi=VIDEO_DPI)
    fig.patch.set_facecolor('#0d1b2a')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d1b2a')
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#1b263b')
    ax.yaxis.pane.set_edgecolor('#1b263b')
    ax.zaxis.pane.set_edgecolor('#1b263b')
    
    scatter = ax.scatter(omega_sample, phi_sample, height, 
                         c=height, cmap='plasma', s=8, alpha=0.7)
    
    ax.set_xlabel('ω (Frequency)', color='#e0e1dd', fontsize=12, fontweight='bold')
    ax.set_ylabel('φ (Phase)', color='#e0e1dd', fontsize=12, fontweight='bold')
    ax.set_zlabel('|ω - 1| (Deviation)', color='#e0e1dd', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#778da9')
    
    frames = duration * fps
    
    def animate(frame):
        ax.view_init(elev=20 + 10 * np.sin(frame / 40), azim=frame * 0.75)
        ax.set_title(f'WiggleGPT 3D Parameter Landscape ({label})\n{n_points:,} neurons', 
                     color='white', fontsize=16, fontweight='bold', pad=10)
        return [scatter]
    
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=1000/fps)
    
    writer = animation.FFMpegWriter(
        fps=fps, 
        bitrate=VIDEO_BITRATE,
        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'slow', '-crf', '18']
    )
    try:
        anim.save(output_path, writer=writer)
    except Exception as e:
        gif_path = output_path.replace('.mp4', '.gif')
        anim.save(gif_path, writer='pillow', fps=20)
    
    plt.close()


# ============================================================================
# VISUALIZATION 5: HERO IMAGE
# ============================================================================

def create_hero_image(layer_params, omega, phi, output_path, label):
    """Create composite hero image."""
    
    fig = plt.figure(figsize=(20, 12), dpi=200)
    fig.patch.set_facecolor('#0d1b2a')
    
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel 1: Galaxy
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_facecolor('#0d1b2a')
    
    n_sample = min(30000, len(omega))
    idx = np.random.choice(len(omega), n_sample, replace=False)
    magnitude = np.sqrt(omega[idx]**2 + phi[idx]**2)
    
    colors_galaxy = ['#1a1a2e', '#16213e', '#533483', '#e94560', '#00d4ff', '#ffffff']
    cmap = LinearSegmentedColormap.from_list('galaxy', colors_galaxy)
    
    ax1.scatter(omega[idx], phi[idx], c=magnitude, cmap=cmap, alpha=0.5, s=3, edgecolors='none')
    ax1.scatter([omega.mean()], [phi.mean()], c='#00d4ff', s=100, alpha=0.8, edgecolors='white')
    ax1.set_xlabel('ω', fontsize=12, color='#e0e1dd')
    ax1.set_ylabel('φ', fontsize=12, color='#e0e1dd')
    ax1.set_title('Neural Galaxy', fontsize=14, color='white', fontweight='bold')
    ax1.tick_params(colors='#778da9')
    
    # Panel 2: Activation Functions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#0d1b2a')
    
    x = np.linspace(-3, 3, 200)
    sample_idx = np.random.choice(len(omega), 50, replace=False)
    
    for i in sample_idx:
        y = np.sin(omega[i] * x + phi[i]) * np.tanh(x)
        ax2.plot(x, y, alpha=0.3, linewidth=0.5, color=cm.plasma(omega[i] / omega.max()))
    
    y_mean = np.sin(omega.mean() * x + phi.mean()) * np.tanh(x)
    ax2.plot(x, y_mean, color='#00d4ff', linewidth=2, label='Mean neuron')
    
    ax2.set_xlabel('x', fontsize=10, color='#e0e1dd')
    ax2.set_ylabel('Activation', fontsize=10, color='#e0e1dd')
    ax2.set_title('Oscillating Activations', fontsize=12, color='white', fontweight='bold')
    ax2.tick_params(colors='#778da9')
    ax2.grid(True, alpha=0.2, color='#415a77')
    
    # Panel 3: Omega histogram
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#0d1b2a')
    
    ax3.hist(omega, bins=80, color='#7b2cbf', alpha=0.8, edgecolor='#0d1b2a', linewidth=0.5)
    ax3.axvline(x=1.0, color='#e94560', linestyle='--', linewidth=2, label='Init')
    ax3.axvline(x=omega.mean(), color='#00d4ff', linestyle='-', linewidth=2, label='Learned')
    ax3.set_xlabel('ω', fontsize=10, color='#e0e1dd')
    ax3.set_ylabel('Count', fontsize=10, color='#e0e1dd')
    ax3.set_title('Frequency Distribution', fontsize=12, color='white', fontweight='bold')
    ax3.tick_params(colors='#778da9')
    ax3.legend(facecolor='#1b263b', edgecolor='#415a77', labelcolor='white', fontsize=8)
    
    # Panel 4: Layer heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#0d1b2a')
    
    n_layers = len(layer_params)
    stats_matrix = np.zeros((n_layers, 4))
    
    for i, (layer_num, params) in enumerate(sorted(layer_params.items())):
        if params['omega'] is not None:
            stats_matrix[i, 0] = params['omega'].mean()
            stats_matrix[i, 1] = params['omega'].std()
            stats_matrix[i, 2] = params['phi'].mean() if params['phi'] is not None else 0
            stats_matrix[i, 3] = params['phi'].std() if params['phi'] is not None else 0
    
    im = ax4.imshow(stats_matrix.T, aspect='auto', cmap='plasma')
    ax4.set_xlabel('Layer', fontsize=10, color='#e0e1dd')
    ax4.set_yticks([0, 1, 2, 3])
    ax4.set_yticklabels(['ω mean', 'ω std', 'φ mean', 'φ std'], fontsize=8)
    ax4.tick_params(colors='#778da9')
    ax4.set_title('Layer Statistics', fontsize=12, color='white', fontweight='bold')
    plt.colorbar(im, ax=ax4, shrink=0.8)
    
    # Panel 5: Phi distribution
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor('#0d1b2a')
    
    ax5.hist(phi, bins=80, color='#ff006e', alpha=0.8, edgecolor='#0d1b2a', linewidth=0.5)
    ax5.axvline(x=0.0, color='#e94560', linestyle='--', linewidth=2, label='Init')
    ax5.axvline(x=phi.mean(), color='#00d4ff', linestyle='-', linewidth=2, label='Learned')
    ax5.set_xlabel('φ', fontsize=10, color='#e0e1dd')
    ax5.set_ylabel('Count', fontsize=10, color='#e0e1dd')
    ax5.set_title('Phase Distribution', fontsize=12, color='white', fontweight='bold')
    ax5.tick_params(colors='#778da9')
    ax5.legend(facecolor='#1b263b', edgecolor='#415a77', labelcolor='white', fontsize=8)
    
    # Main title with label
    fig.suptitle(f'WiggleGPT: Bio-Inspired Oscillating Neural Networks ({label})\n' + 
                 f'{len(omega):,} neurons | {n_layers} layers | sin(ω·x + φ)·tanh(x)',
                 fontsize=20, color='white', fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=200, facecolor='#0d1b2a', edgecolor='none', bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 WiggleGPT Social Media Visualization Generator")
    print(f"   Model: {MODEL_LABEL} | Video: {VIDEO_FPS}fps @ {VIDEO_BITRATE/1000:.0f}Mbps")
    print("=" * 60)
    
    layer_params, omega, phi = load_wiggle_params(CHECKPOINT_PATH)
    print(f"✅ Loaded {len(omega):,} neurons across {len(layer_params)} layers")
    print(f"📁 Output: {OUTPUT_DIR.absolute()}\n")
    
    print("🌌 [1/5] Neural Galaxy...")
    create_neural_galaxy(omega, phi, OUTPUT_DIR / "wiggle_galaxy.png", MODEL_LABEL)
    
    print("🔍 [2/5] Layer Fingerprint...")
    create_layer_fingerprint(layer_params, OUTPUT_DIR / "wiggle_fingerprint.png", MODEL_LABEL)
    
    print("🎨 [3/5] Hero Image...")
    create_hero_image(layer_params, omega, phi, OUTPUT_DIR / "wiggle_hero.png", MODEL_LABEL)
    
    print("🌊 [4/5] Wave Animation (1080p)...")
    create_wave_animation(omega, phi, str(OUTPUT_DIR / "wiggle_waves.mp4"), MODEL_LABEL)
    
    print("🔄 [5/5] 3D Landscape (1080p)...")
    create_3d_landscape(omega, phi, str(OUTPUT_DIR / "wiggle_3d.mp4"), MODEL_LABEL)
    
    print("\n" + "=" * 60)
    print("✨ Done! All files saved to:", OUTPUT_DIR.absolute())
    print("=" * 60)

# WiggleGPT Analysis Scripts - User Guide

> These scripts accompany the [WiggleGPT paper](https://garden-backend-three.vercel.app/finalized-work/wiggle-gpt/wiggle-gpt-paper/) and generate the figures used in the publication.

This folder contains analysis and visualization scripts for WiggleGPT checkpoints. Below is a guide on how to use each script and where to place them before running.

---

## 📁 Scripts Overview

| Script | Purpose |
|--------|---------|
| `analyze_wiggle-pretrain.py` | Analyze oscillation parameters (ω, φ) from a pretrained checkpoint |
| `analyze_wiggle-finetune.py` | Compare oscillation parameters between pretrained and fine-tuned checkpoints |
| `analyze_brainwaves.py` | Analyze brainwave/oscillation patterns in the model |
| `generate_pretrain_loss_chart.py` | Generate loss curve visualization for pretraining |
| `generate_finetune_loss_chart.py` | Generate loss curve visualization for fine-tuning |
| `generate_social_media_visuals.py` | Generate visuals optimized for social media sharing |

---

## 🔧 Requirements

All scripts require:
- Python 3.x
- PyTorch (`torch`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)

Install dependencies:
```bash
pip install torch numpy matplotlib
```

---

## 📍 Script Placement & Usage

### 1. `analyze_wiggle-pretrain.py`

**Purpose:** Analyzes the learned omega (frequency) and phi (phase) parameters from a single pretrained checkpoint to verify the model is using oscillation.

**Placement:** Copy this script into the **same directory** as your checkpoint file (`ckpt.pt`).

**Example locations:**
- `out-wigglegpt-pure-124m/` (for pretrained model)
- Any checkpoint folder containing `ckpt.pt`

**How to run:**
```bash
cd out-wigglegpt-pure-124m
python analyze_wiggle-pretrain.py
```

**Expected output:**
- Console statistics showing omega/phi distributions
- Interpretation of whether oscillation is being used
- `wiggle_analysis.png` - visualization plots saved to the same directory

---

### 2. `analyze_wiggle-finetune.py`

**Purpose:** Compares pretrained vs fine-tuned oscillation parameters to see how SFT (Supervised Fine-Tuning) affected the omega/phi values.

**Placement:** Copy this script into your **fine-tuned checkpoint directory** (e.g., `out-wigglegpt-finetune-native/`).

**Required checkpoint paths (relative to script location):**
- `ckpt.pt` - Fine-tuned checkpoint (in the same directory as the script)
- `../out-wigglegpt-pure-124m/ckpt.pt` - Pretrained checkpoint (one level up, in the pretrain folder)

**Directory structure expected:**
```
WiggleGPT/
├── out-wigglegpt-pure-124m/
│   └── ckpt.pt                    ← Pretrained checkpoint
└── out-wigglegpt-finetune-native/
    ├── ckpt.pt                    ← Fine-tuned checkpoint
    ├── analyze_wiggle-finetune.py ← Script goes here
    └── wiggle_finetune_analysis.png ← Generated output
```

**How to run:**
```bash
cd out-wigglegpt-finetune-native
python analyze_wiggle-finetune.py
```

**Expected output:**
- Side-by-side comparison of pretrained vs fine-tuned statistics
- Parameter drift analysis
- `wiggle_finetune_analysis.png` - comparison visualization saved to the same directory

---

### 3. `generate_pretrain_loss_chart.py`

**Purpose:** Generates a publication-ready loss curve chart for the pretraining phase using data from training logs.

**Placement:** Can be run from **any directory**. The script contains hardcoded training data from `trainlog4_reconstructed.md`.

**How to run:**
```bash
python generate_pretrain_loss_chart.py
```

**Expected output:**
- `pretrain_loss_chart.png` - Two-panel chart showing:
  - Full training curve (log scale)
  - Final convergence region with GPT-2 baseline reference

**Updating the data:**
The hardcoded data comes from parsing training logs (e.g., `trainlog4_reconstructed.md`). To extract step printouts from your own logs:

1. **Before training:** Increase your Command Prompt history buffer! By default, CMD only keeps ~300 lines and your log will scroll out of history during long training runs (especially overnight). 
   - Right-click the CMD title bar → Properties → Options tab → Command History → Set "Buffer Size" to **32000**
   - Or: Right-click title bar → Defaults → same settings (applies to all future CMD windows)
   
2. Copy your full training log (can be very large)
2. Use [Google AI Studio](https://aistudio.google.com/) with its 1M token context window
3. Prompt it to extract only the step printouts (lines containing iteration number, train loss, and validation loss)
4. Format the extracted data into the `steps`, `train_loss`, and `val_loss` arrays in the script

This approach handles massive log files that exceed typical context limits.

---

### 4. `generate_finetune_loss_chart.py`

**Purpose:** Generates a publication-ready loss curve chart for the fine-tuning phase using data from `finetune_log.md`.

**Placement:** Can be run from **any directory**. The script contains hardcoded fine-tuning data.

**How to run:**
```bash
python generate_finetune_loss_chart.py
```

**Expected output:**
- `finetune_loss_chart.png` - Chart showing:
  - Train and validation loss over fine-tuning iterations
  - Best validation loss point highlighted
  - Overfitting region annotation

**Updating the data:**
The hardcoded data was generated by having an AI agent (GitHub Copilot in VS Code) read `Docs n' Logs/finetune_log.md` and automatically populate the arrays in the script. The log file is simply the terminal output copied (Ctrl+C) and pasted (Ctrl+V) into a markdown file during fine-tuning. Fine-tuning runs are short enough that the full terminal output fits easily.

To update with your own data, copy the following prompt and give it to an AI agent along with your log file:

```
Read the fine-tuning log and extract all the step checkpoints. 
Populate the `steps`, `train_loss`, and `val_loss` arrays in 
generate_finetune_loss_chart.py with this data. Each validation 
checkpoint line contains the step number, train loss, and val loss.
```

---

## 📊 Quick Reference Table

| Script | Place In | Requires Checkpoint(s) | Output File |
|--------|----------|------------------------|-------------|
| `analyze_wiggle-pretrain.py` | Checkpoint folder | `ckpt.pt` (same dir) | `wiggle_analysis.png` |
| `analyze_wiggle-finetune.py` | Fine-tune folder | `ckpt.pt` + `../out-wigglegpt-pure-124m/ckpt.pt` | `wiggle_finetune_analysis.png` |
| `analyze_brainwaves.py` | Checkpoint folder | `ckpt.pt` (same dir) | Brainwave analysis outputs |
| `generate_pretrain_loss_chart.py` | Anywhere | None (uses hardcoded data) | `pretrain_loss_chart.png` |
| `generate_finetune_loss_chart.py` | Anywhere | None (uses hardcoded data) | `finetune_loss_chart.png` |
| `generate_social_media_visuals.py` | Anywhere | None (uses hardcoded data) | Social media optimized images |

---

## 📂 Output Folders

The following folders store generated outputs:

| Folder | Contents |
|--------|----------|
| `outputs-brainwave/` | Output files from brainwave analysis |
| `outputs-finetune/` | Output files from fine-tuning analysis |
| `outputs-pretrain/` | Output files from pretraining analysis |

---

## ⚠️ Troubleshooting

### "No parameters named 'omega' found!"
- The checkpoint may not be a WiggleGPT model
- Check that you're using the correct checkpoint file

### Path errors for `analyze_wiggle-finetune.py`
- Ensure the pretrained checkpoint exists at `../out-wigglegpt-pure-124m/ckpt.pt`
- Modify the `pretrain_ckpt_path` variable in the script if your folder structure differs

### Matplotlib display issues
- On headless servers, the scripts will still save the PNG files
- If `plt.show()` causes issues, you can comment out that line

---

## 📝 Customization Tips

1. **Change checkpoint paths:** Edit the `ckpt_path` or `pretrain_ckpt_path` / `finetune_ckpt_path` variables at the top of the analysis scripts.

2. **Update loss data:** For the chart generation scripts, modify the `steps`, `train_loss`, and `val_loss` lists with your own training log data.

3. **Adjust plot styling:** All scripts use matplotlib - modify colors, figure sizes, and labels as needed.

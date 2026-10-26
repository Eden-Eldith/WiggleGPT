"""Portable native WiggleGPT loader and exact GPT-2 tokenizer.

The model class is the byte-preserved source in native/model_bio.py.
Model construction and all forwards are CUDA only. Tokenization is CPU only.
"""
from __future__ import annotations
import base64
import hashlib
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'native'))

DESKTOP_GPU_APPS = {
    "shellhost.exe", "crossdeviceresume.exe", "lghub_system_tray.exe", "dcv2.exe",
    "snippingtool.exe", "steelseriesggclient.exe", "explorer.exe", "textinputhost.exe",
    "startmenuexperiencehost.exe", "searchhost.exe", "shellexperiencehost.exe",
    "msedgewebview2.exe", "windowsterminal.exe", "chrome.exe", "msedge.exe", "firefox.exe",
    "chatgpt.exe", "codex-computer-use-swift.exe", "razerappengine.exe", "notepad++.exe",
    "obsidian.exe", "dwm.exe",
}

def get_encoder():
    import tiktoken
    from tiktoken_ext.openai_public import r50k_pat_str

    raw = (ROOT / "tokenizer" / "gpt2.tiktoken").read_bytes()
    # Exact canonical GPT-2/r50k merge table, independent of network/cache paths.
    if hashlib.sha256(raw).hexdigest() != "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930":
        raise ValueError("Bundled GPT-2 merge table is damaged.")
    ranks = {base64.b64decode(token): int(rank) for token, rank in
             (line.split() for line in raw.splitlines() if line)}
    return tiktoken.Encoding(name="gpt2", pat_str=r50k_pat_str, mergeable_ranks=ranks,
                             special_tokens={"<|endoftext|>": 50256}, explicit_n_vocab=50257)

def check_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    busy = []
    for line in result.stdout.splitlines():
        fields = line.split(",", 1)
        # Current Windows WDDM reports ordinary graphics clients as C+G too.
        # Desktop rendering is covered by the measured VRAM cap; other jobs wait.
        desktop = len(fields) == 2 and Path(fields[1].strip()).name.lower() in DESKTOP_GPU_APPS
        if fields[0].strip().isdigit() and int(fields[0]) != os.getpid() and not desktop:
            busy.append(line.strip())
    if busy:
        raise RuntimeError("Another GPU job is running. Finish it before loading chat: " + "; ".join(busy))

def load_model(checkpoint_path):
    check_gpu()
    import torch
    from model_bio import GPT, GPTConfig

    if not torch.cuda.is_available():
        raise RuntimeError("WiggleGPT needs CUDA here; CPU model execution is disabled.")
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    budget = min(total - int(2.5 * 1024**3), free - int(0.5 * 1024**3))
    if budget < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB is available after leaving desktop room.")
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    torch.set_float32_matmul_precision("high")
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    config = GPTConfig(**checkpoint["model_args"])
    # Construct directly on CUDA. In particular, never run a CPU model forward.
    with torch.device("cuda:0"):
        model = GPT(config)
    state = {key.removeprefix("_orig_mod."): value for key, value in checkpoint["model"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    metadata = {"checkpoint": str(checkpoint_path), "saved_step": checkpoint.get("iter_num", checkpoint.get("step")),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "model_args": checkpoint["model_args"], "torch": torch.__version__}
    del checkpoint, state
    encoder = get_encoder()
    print(f"CUDA: {torch.cuda.get_device_name(0)}; {metadata['parameters']:,} parameters; "
          f"{budget / 1024**3:.2f} GiB memory cap", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    return model, config, encoder, metadata

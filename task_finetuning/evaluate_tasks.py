"""Evaluate a trained task checkpoint on its labelled public held-out split.

Accepts either a complete training checkpoint or an inference-only release.
No optimizer or original pretrained checkpoint is needed for evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "native"))
import torch
from model_bio import GPT, GPTConfig
from task_runtime import check_gpu
from train_tasks import TaskModel, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-process-gib", type=float, default=9.0)
    parser.add_argument("--output", type=Path, help="Optional JSON with scores and every prediction.")
    args = parser.parse_args()
    if args.max_process_gib <= 0 or args.batch_size is not None and args.batch_size <= 0:
        parser.error("Memory cap and batch size must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU model execution is disabled.")
    check_gpu()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info()
    budget = min(int(args.max_process_gib * 1024**3), total - int(2.5 * 1024**3),
                 free - int(0.5 * 1024**3))
    if budget < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB is available after reserving desktop memory.")
    torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    torch.set_float32_matmul_precision("high")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    task = saved["task"]
    if task not in ("sst2", "race_middle"):
        raise ValueError(f"Unknown task: {task}")
    split = args.split or ("validation" if task == "sst2" else "test")
    if task == "sst2" and split == "test":
        raise ValueError("SST-2 public test labels are hidden. Use validation for labelled evaluation.")
    source_hash = saved.get("source_hashes", saved.get("provenance", {}).get("source_hashes", {})).get("model")
    if source_hash:
        actual_hash = hashlib.sha256((ROOT / "native" / "model_bio.py").read_bytes()).hexdigest()
        if source_hash != actual_hash:
            raise ValueError("Native model source differs from the checkpoint's recorded source.")
    config = GPTConfig(**saved["model_args"])
    config.gradient_checkpointing = False
    dropout = saved.get("config", {}).get("classifier_dropout", 0.1)
    with torch.device("cuda:0"):
        base = GPT(config)
        model = TaskModel(base, task, dropout)
    model.load_state_dict(saved["model"], strict=True)
    del saved
    cache_path = ROOT / "data" / f"{task}_{split}.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    data = {key: cache[key].to("cuda:0") for key in ("input_ids", "lengths", "labels")}
    if torch.any(data["labels"] < 0):
        raise ValueError("This split has withheld labels and cannot be scored.")
    result = evaluate(model, data, args.batch_size or (16 if task == "sst2" else 4))
    result.update(task=task, split=split, checkpoint=str(args.checkpoint),
                  example_ids=cache.get("example_ids"), source_sha256=cache.get("source_sha256"),
                  device=torch.cuda.get_device_name(0))
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("predictions", "example_ids")}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""CPU-only exact GPT-2 token caches for GPT-1-style supervised adaptation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TOKENIZER_SOURCE = ROOT / "task_runtime.py"
from task_runtime import get_encoder
START, DELIM, CLS, PAD = 50257, 50258, 50259, 50256
LIMIT = 512


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def encode_unique(encoder, strings: list[str]) -> dict[str, list[int]]:
    unique = list(dict.fromkeys(strings))
    encoded = encoder.encode_ordinary_batch(unique, num_threads=2)
    return dict(zip(unique, encoded))


def encode_split(encoder, task: str, split: str) -> dict:
    source = DATA / f"{task}_{split}.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    counts = {"examples_with_truncation": 0, "candidate_sequences_with_truncation": 0,
              "total_tokens_removed": 0, "max_tokens_removed_per_example": 0}
    if task == "sst2":
        encoded = encoder.encode_ordinary_batch([row["text"] for row in rows], num_threads=2)
        max_raw = max(map(len, encoded)) + 2
        lengths = np.array([[min(len(tokens), LIMIT - 2) + 2] for tokens in encoded], dtype=np.int64)
        pad_length = max(16, math.ceil(int(lengths.max()) / 16) * 16)
        input_ids = np.full((len(rows), 1, pad_length), PAD, dtype=np.int32)
        for index, tokens in enumerate(encoded):
            removed = max(0, len(tokens) - (LIMIT - 2))
            if removed:
                counts["examples_with_truncation"] += 1
                counts["candidate_sequences_with_truncation"] += 1
                counts["total_tokens_removed"] += removed
                counts["max_tokens_removed_per_example"] = max(counts["max_tokens_removed_per_example"], removed)
            sequence = [START] + tokens[:LIMIT - 2] + [CLS]
            input_ids[index, 0, :len(sequence)] = sequence
        context_policy = "[START] + review first 510 tokens + [CLS]; right-pad to the split maximum rounded up to a multiple of 16 (minimum 16)."
    else:
        article_ids = encode_unique(encoder, [row["article"] for row in rows])
        question_ids = encode_unique(encoder, [row["question"] for row in rows])
        answer_ids = encode_unique(encoder, [answer for row in rows for answer in row["options"]])
        pad_length = LIMIT
        input_ids = np.full((len(rows), 4, pad_length), PAD, dtype=np.int32)
        lengths = np.empty((len(rows), 4), dtype=np.int64)
        max_raw = 0
        for index, row in enumerate(rows):
            article = article_ids[row["article"]]
            question = question_ids[row["question"]]
            answers = [answer_ids[answer] for answer in row["options"]]
            overhead = 4 + len(question) + max(map(len, answers))
            if overhead > LIMIT:
                raise ValueError(f"RACE {split}/{index} {row['example_id']}: question and longest answer plus 4 markers need {overhead} tokens, beyond {LIMIT}.")
            budget = LIMIT - overhead
            shared_article = article[:budget]
            removed = max(0, len(article) - budget)
            if removed:
                counts["examples_with_truncation"] += 1
                counts["candidate_sequences_with_truncation"] += 4
                counts["total_tokens_removed"] += removed
                counts["max_tokens_removed_per_example"] = max(counts["max_tokens_removed_per_example"], removed)
            for choice, answer in enumerate(answers):
                sequence = [START] + shared_article + [DELIM] + question + [DELIM] + answer + [CLS]
                lengths[index, choice] = len(sequence)
                input_ids[index, choice, :len(sequence)] = sequence
                max_raw = max(max_raw, 4 + len(article) + len(question) + len(answer))
        context_policy = "[START] + shared article prefix + [DELIM] + complete question + [DELIM] + complete answer + [CLS]; all four choices retain the same article prefix, budgeted using the longest answer; right-pad to 512."
    summary = {"task": task, "split": split, "examples": len(rows), "choices": int(input_ids.shape[1]),
               "source_jsonl": str(source), "source_sha256": digest(source),
               "pad_length": pad_length, "max_sequence_length": int(lengths.max()),
               "min_sequence_length": int(lengths.min()), "max_untruncated_length": max_raw,
               "truncation": counts, "context_policy": context_policy,
               "input_ids_dtype": "int32", "lengths_dtype": "int64", "labels_dtype": "int64"}
    output = DATA / f"{task}_{split}.pt"
    temporary = output.with_suffix(".pt.partial")
    torch.save({"input_ids": torch.from_numpy(input_ids), "lengths": torch.from_numpy(lengths), "labels": labels,
                "example_ids": [row["example_id"] for row in rows], "metadata": summary,
                "source_sha256": summary["source_sha256"]}, temporary)
    temporary.replace(output)
    summary.update({"cache_path": str(output), "cache_sha256": digest(output), "cache_bytes": output.stat().st_size})
    print(f"{task}/{split}: shape {tuple(input_ids.shape)}, max length {lengths.max()}, truncated {counts['examples_with_truncation']:,} examples", flush=True)
    return summary


def main() -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    encoder = get_encoder()
    summary = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(),
               "encoder": "Exact bundled GPT-2/r50k via task_runtime.get_encoder; ordinary-string encoding only",
               "tokenizer_source": str(TOKENIZER_SOURCE), "tokenizer_source_sha256": digest(TOKENIZER_SOURCE),
               "merge_table_sha256": digest(ROOT / "tokenizer" / "gpt2.tiktoken"),
               "task_markers": {"START": START, "DELIM": DELIM, "CLS": CLS, "PAD": PAD},
               "limit": LIMIT, "cpu_threads": 2, "splits": {}}
    for task in ("sst2", "race_middle"):
        for split in ("train", "validation", "test"):
            summary["splits"][f"{task}_{split}"] = encode_split(encoder, task, split)
    path = DATA / "encoding_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Encoding summary: {path}", flush=True)


if __name__ == "__main__":
    main()

"""Download public benchmark parquets and preserve exact normalized task splits.

This script performs no model work. Public SST-2 test labels are withheld (-1);
the validation split is therefore the labelled held-out evaluation split.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent / "data"
SOURCES = [("sst2", "stanfordnlp/sst2", "default"), ("race_middle", "ehovy/race", "middle")]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fetch(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "WiggleGPT-benchmark-preparation/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            temporary.replace(path)
            return
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def normalize(task: str, row: dict, index: int, split: str) -> dict:
    if task == "sst2":
        return {"example_id": f"sst2/{split}/{row.get('idx', index)}", "text": row["sentence"], "label": int(row["label"])}
    answer = row["answer"]
    label = "ABCD".index(answer) if isinstance(answer, str) else int(answer)
    options = list(row["options"])
    if len(options) != 4 or not 0 <= label < 4:
        raise ValueError(f"Invalid RACE record {index}")
    return {"example_id": str(row.get("example_id", f"race_middle/{split}/{index}")),
            "article": row["article"], "question": row["question"], "options": options, "label": label}


def prepare_source(spec: tuple[str, str, str]) -> dict:
    task, dataset, config = spec
    metadata_url = "https://datasets-server.huggingface.co/parquet?" + urllib.parse.urlencode({"dataset": dataset})
    metadata_path = ROOT / "source_parquets" / task / "parquet_metadata.json"
    fetch(metadata_url, metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("partial"):
        raise RuntimeError(f"Partial parquet conversion for {dataset}")
    files = [entry for entry in metadata["parquet_files"] if entry["config"] == config]
    result = {"dataset": dataset, "config": config, "metadata_url": metadata_url, "splits": {}}
    for split in ("train", "validation", "test"):
        split_files = sorted([entry for entry in files if entry["split"] == split], key=lambda entry: entry["filename"])
        if not split_files:
            raise RuntimeError(f"Missing {dataset} {config} {split}")
        destination = ROOT / f"{task}_{split}.jsonl"
        temporary = destination.with_suffix(".jsonl.partial")
        count, labels, source_files = 0, Counter(), []
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for ordinal, entry in enumerate(split_files):
                parquet_path = ROOT / "source_parquets" / task / split / f"{ordinal:03d}_{entry['filename']}"
                fetch(entry["url"], parquet_path)
                if parquet_path.stat().st_size != entry["size"]:
                    raise RuntimeError(f"Size mismatch for {parquet_path}")
                source_files.append({"url": entry["url"], "path": str(parquet_path), "bytes": parquet_path.stat().st_size, "sha256": digest(parquet_path)})
                for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=1024):
                    for row in batch.to_pylist():
                        item = normalize(task, row, count, split)
                        output.write(json.dumps(item, ensure_ascii=False) + "\n")
                        labels[str(item["label"])] += 1
                        count += 1
        temporary.replace(destination)
        result["splits"][split] = {"path": str(destination), "rows": count, "sha256": digest(destination), "label_counts": dict(sorted(labels.items())), "labelled": all(int(label) >= 0 for label in labels), "source_parquets": source_files}
        print(f"{task} {split}: {count:,} rows; labels={dict(labels)}", flush=True)
    return result


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        prepared = list(pool.map(prepare_source, SOURCES))
    manifest = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "tasks": {spec[0]: result for spec, result in zip(SOURCES, prepared)},
                "split_policy": {"sst2": "Train on official train; validation is the only public labelled holdout. Official test labels are hidden (-1).",
                                 "race_middle": "Train on official middle/train; select using middle/validation; evaluate once on middle/test."},
                "normalization": "Exact input strings retained; sentiment labels are 0=negative,1=positive; RACE A/B/C/D mapped to integer 0/1/2/3; no examples generated or reassigned."}
    manifest_path = ROOT / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

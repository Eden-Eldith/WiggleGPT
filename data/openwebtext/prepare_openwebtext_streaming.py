"""
prepare_openwebtext_streaming.py
=================================

WiggleGPT - PROPER Streaming OpenWebText Preparation
Author: Claude (following Eden's proven streaming architecture)

Design Philosophy (learned from Token_usage_fast_enhanced.py):
---------------------------------------------------------------
• NEVER accumulate full dataset in memory
• Process in chunks, write immediately, discard
• Constant memory footprint (~200-500MB)
• Progress tracking with ETA
• Graceful error handling

Memory Usage:
  Peak: ~500MB (one chunk + encoding overhead)
  NOT: 16GB+ (loading everything like an idiot)

Output:
  ├── train.bin (~17 GB)
  └── val.bin   (~8 MB)
"""

import os
import numpy as np
from tqdm import tqdm
import random
from datasets import load_dataset
import tiktoken
import time

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
VAL_SPLIT = 0.0005           # 0.05% validation split
ENCODING = "gpt2"
DTYPE = np.uint16            # GPT-2 vocab < 65536
CACHE_DIR = "./openwebtext_cache"
CHUNK_SIZE = 1000            # Documents per chunk (SMALL = SAFE); reduce to 500 if still issues
APPROX_TOTAL_DOCS = 8013769  # Known size of OpenWebText train split

print(f"🧠 Using chunk size {CHUNK_SIZE}")

# ---------------------------------------------------------------------
# LOAD DATASET (STREAMING MODE)
# ---------------------------------------------------------------------
def load_openwebtext():
    print("📥 Loading OpenWebText in streaming mode...")
    dataset = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        trust_remote_code=True,
        streaming=True,
        cache_dir=CACHE_DIR,
    )
    
    print("✅ Dataset loaded in streaming mode (no full cache built)")
    print("   Processing will happen on-the-fly without high RAM usage")
    
    return dataset

# ---------------------------------------------------------------------
# STREAMING TOKENIZATION & WRITE
# ---------------------------------------------------------------------
def process_chunk(texts, enc):
    """
    Tokenize a chunk of texts.
    Returns flat token array, NOT accumulated in caller.
    """
    tokens = []
    for text in texts:
        tokens.extend(enc.encode_ordinary(text))
        tokens.append(enc.eot_token)
    return np.array(tokens, dtype=DTYPE)

def stream_tokenize_and_write(dataset, enc):
    """
    THE RIGHT WAY: Stream dataset iteratively, assign to train/val probabilistically,
    tokenize in chunks, write immediately.
    
    Memory usage: Constant (~200-500MB for current chunk only)
    Pattern: Stream example → Assign to split → Buffer chunk → Process → Write → DISCARD → Repeat
    """
    print("\n🔤 Streaming tokenization for train and val")
    print(f"   Processing ~{APPROX_TOTAL_DOCS:,} documents in chunks of {CHUNK_SIZE}")
    
    train_temp = "train.tmp"
    val_temp = "val.tmp"
    
    # Open files for streaming writes
    with open(train_temp, 'wb') as train_f, open(val_temp, 'wb') as val_f:
        total_tokens = 0
        train_tokens = 0
        val_tokens = 0
        start_time = time.time()
        
        train_chunk = []
        val_texts = []  # Collect all val in memory (small split: ~4K docs, low RAM)
        
        random.seed(2357)  # Same seed as original for reproducible split
        
        with tqdm(total=APPROX_TOTAL_DOCS, desc="Processing docs") as pbar:
            for example in dataset:
                text = example['text']
                
                if random.random() < VAL_SPLIT:
                    val_texts.append(text)
                else:
                    train_chunk.append(text)
                    if len(train_chunk) == CHUNK_SIZE:
                        try:
                            # Process and write train chunk (sequential to keep low mem)
                            tokens = process_chunk(train_chunk, enc)
                            tokens.tofile(train_f)
                            train_tokens += len(tokens)
                            total_tokens += len(tokens)
                            train_chunk = []  # Discard
                        except Exception as e:
                            print(f"\n⚠️  Error processing train chunk: {e}")
                            continue
                
                pbar.update(1)
                
                # Occasional stats update
                if pbar.n % (CHUNK_SIZE * 10) == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({
                        'tokens': f"{total_tokens:,}",
                        'tok/s': f"{tokens_per_sec:,.0f}"
                    })
        
        # Process remaining train chunk
        if train_chunk:
            tokens = process_chunk(train_chunk, enc)
            tokens.tofile(train_f)
            train_tokens += len(tokens)
            total_tokens += len(tokens)
        
        # Process val (in chunks if large, but won't be)
        val_chunk = []
        for text in val_texts:
            val_chunk.append(text)
            if len(val_chunk) == CHUNK_SIZE:
                tokens = process_chunk(val_chunk, enc)
                tokens.tofile(val_f)
                val_tokens += len(tokens)
                total_tokens += len(tokens)
                val_chunk = []
        if val_chunk:
            tokens = process_chunk(val_chunk, enc)
            tokens.tofile(val_f)
            val_tokens += len(tokens)
            total_tokens += len(tokens)
        
        # Stats
        elapsed = time.time() - start_time
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
        
        print(f"✅ Train: {train_tokens:,} tokens")
        print(f"✅ Val: {val_tokens:,} tokens")
        print(f"   Total time: {elapsed:.1f}s")
        print(f"   Speed: {tokens_per_sec:,.0f} tokens/sec")
        print(f"   Memory: Constant (~500MB max)")
    
    # Rename temp files to final
    for split, temp in [("train", train_temp), ("val", val_temp)]:
        final = f"{split}.bin"
        if os.path.exists(final):
            os.remove(final)
        os.rename(temp, final)
        
        # Verify file size
        file_size_gb = os.path.getsize(final) / (1024**3)
        print(f"   {split}.bin size: {file_size_gb:.2f} GB")
    
    return train_tokens, val_tokens

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("="*70)
    print("WiggleGPT - Streaming OpenWebText Preparation")
    print("="*70)
    print("\nMemory-safe design:")
    print("  ✅ Streaming mode enabled")
    print("  ✅ Process in chunks")
    print("  ✅ Write immediately")
    print("  ✅ Discard after writing")
    print("  ✅ Constant memory footprint")
    print("  ❌ NO loading everything into RAM like an idiot")
    print("="*70 + "\n")
    
    # Load dataset (streaming)
    dataset = load_openwebtext()
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding(ENCODING)
    print(f"📝 Tokenizer: {ENCODING}")
    print(f"   Vocab size: {enc.n_vocab:,}")
    print(f"   EOT token: {enc.eot_token}\n")
    
    # Process with streaming
    total_start = time.time()
    
    train_tokens, val_tokens = stream_tokenize_and_write(
        dataset,
        enc
    )
    
    # Final summary
    total_elapsed = time.time() - total_start
    total_tokens = train_tokens + val_tokens
    overall_speed = total_tokens / total_elapsed if total_elapsed > 0 else 0
    
    print("="*70)
    print("🎉 COMPLETE!")
    print("="*70)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Overall speed: {overall_speed:,.0f} tokens/sec")
    print(f"\nFiles created:")
    print(f"  ✅ train.bin ({train_tokens:,} tokens)")
    print(f"  ✅ val.bin ({val_tokens:,} tokens)")
    print("\nReady for WiggleGPT training! 🧠⚡")
    print("="*70)
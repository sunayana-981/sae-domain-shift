#!/usr/bin/env python
"""Tokenise each corpus once into fixed-length windows on disk.

Every arm of the sweep then reads the same windows through the same
train/eval index split, so conditions differ only in how the SAE was trained
-- not in which tokens it happened to stream.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from qsae.data import CORPORA, OOD_CORPORA, REFERENCE_CORPUS, build_token_cache

# The reference corpus is only ever *read* (activity estimation + anchor eval),
# never fine-tuned on, so it needs far fewer tokens than the OOD targets.
DEFAULT_TOKENS = {REFERENCE_CORPUS: 3_000_000}
DEFAULT_OOD_TOKENS = 4_600_000  # trimmed to 3.0M by CachedTokenWindows' split budget


def one(args_tuple):
    name, n_tokens, model, out_dir, seq_len, seed = args_tuple
    tok = AutoTokenizer.from_pretrained(model)
    print(f"[cache] {name}: target {n_tokens:,} tokens", flush=True)
    _, meta = build_token_cache(CORPORA[name], tok, out_dir, n_tokens,
                                seq_len=seq_len, seed=seed)
    return name, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpora", nargs="+",
                   default=[REFERENCE_CORPUS] + OOD_CORPORA)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    p.add_argument("--out_dir", default="out/token_cache")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokens", type=int, default=None,
                   help="Override the per-corpus token budget for all corpora.")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    jobs = []
    for name in args.corpora:
        n = args.tokens or DEFAULT_TOKENS.get(name, DEFAULT_OOD_TOKENS)
        if (Path(args.out_dir) / f"{name}.npy").exists():
            print(f"[cache] {name}: already built, skipping")
            continue
        jobs.append((name, n, args.model, args.out_dir, args.seq_len, args.seed))

    if not jobs:
        print("[cache] nothing to do")
        return

    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for name, meta in ex.map(one, jobs):
            print(f"[cache] done {name}: {meta['n_tokens']:,} tokens, "
                  f"{meta['docs_consumed']:,} docs, "
                  f"{meta['stream_restarts']} restart(s)")


if __name__ == "__main__":
    main()

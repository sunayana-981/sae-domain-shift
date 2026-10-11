#!/usr/bin/env python
"""Concatenate per-corpus shard results into the single CSVs stage 7 expects.

The multi-GPU driver shards stages 5 and 6 by corpus and gives each shard its
own output directory, because two processes appending to one results.csv would
clobber each other. This puts them back together, de-duplicating on the columns
that identify a cell so a re-run of one shard replaces its rows rather than
doubling them.
"""

import argparse
from pathlib import Path

import pandas as pd

MERGES = [
    ("results.csv", ["train_corpus", "eval_corpus", "backbone", "sae_condition"]),
    ("task_results.csv", ["train_corpus", "backbone", "sae_condition", "task"]),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard_root", default="out/results")
    p.add_argument("--out_dir", default="out/results")
    p.add_argument("--glob", default="shard_*")
    args = p.parse_args()

    root = Path(args.shard_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(d for d in root.glob(args.glob) if d.is_dir())
    if not shards:
        print(f"[merge] no shard dirs matching {root}/{args.glob}")
        return

    print(f"[merge] {len(shards)} shard(s): {', '.join(d.name for d in shards)}")
    for fname, keys in MERGES:
        frames = []
        for d in shards:
            f = d / fname
            if f.exists():
                df = pd.read_csv(f)
                frames.append(df)
                print(f"  {d.name}/{fname}: {len(df)} rows")
        if not frames:
            print(f"  [skip] no {fname} in any shard")
            continue
        merged = pd.concat(frames, ignore_index=True)
        before = len(merged)
        merged = merged.drop_duplicates(subset=keys, keep="last")
        target = out_dir / fname
        merged.to_csv(target, index=False)
        print(f"  -> {target}: {len(merged)} rows"
              f"{f' ({before - len(merged)} duplicate cells dropped)' if before != len(merged) else ''}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Clean table from the DKSplit result JSONs.

Filters by item count and requires the `chat_mode` field, so results from
earlier partial runs (200-item validation passes, files written before
chat_mode existed) cannot silently mix into the table.
"""

import argparse
import glob
import json
from pathlib import Path

CARD = {("zeroshot", False): (0.6382, 0.6716),
        ("lora", False): (0.8496, 0.8882)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="dksplit/out")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default="dksplit/out/SUMMARY.md")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(str(Path(args.dir) / "*.json"))):
        if any(k in Path(f).name for k in ("comparison", "probe", "best_per_arm",
                                           "SUMMARY")):
            continue
        d = json.load(open(f))
        if d.get("n") != args.n or "chat_mode" not in d:
            continue
        rows.append(d)

    if not rows:
        raise SystemExit(f"no {args.n}-item results with chat_mode in {args.dir}")

    rows.sort(key=lambda d: (d["arm"], bool(d.get("sae_repo")), d["chat_mode"]))
    lines = [f"# DKSplit: zero-shot vs LoRA, with and without the Qwen-Scope 9B SAE",
             "", f"`benchmark_5000.csv`, n={args.n}, greedy decoding, prompt=advanced.", ""]
    hdr = ("| arm | chat rendering | SAE | strict EM | lenient EM | mutation | "
           "empty | SAE FVE |")
    lines += [hdr, "|" + "---|" * 8]
    for d in rows:
        sae = "yes" if d.get("sae_repo") else "—"
        fve = f"{d['sae_fve']:.4f}" if d.get("sae_fve") is not None else "—"
        lines.append(f"| `{d['arm']}` | `{d['chat_mode']}` | {sae} | "
                     f"{d['strict_em_norm']*100:.2f}% | {d['lenient_em_norm']*100:.2f}% | "
                     f"{d['mutation']*100:.2f}% | {d['empty']*100:.2f}% | {fve} |")

    # best rendering per arm, no SAE -> the baseline comparison
    best = {}
    for d in rows:
        if d.get("sae_repo"):
            continue
        k = d["arm"]
        if k not in best or d["lenient_em_norm"] > best[k]["lenient_em_norm"]:
            best[k] = d
    lines += ["", "## Baseline (no SAE), each arm under its best rendering", ""]
    lines += ["| arm | rendering | strict | lenient | card strict | card lenient |",
              "|" + "---|" * 6]
    for arm in ("zeroshot", "lora"):
        if arm not in best:
            continue
        d = best[arm]
        cs, cl = CARD[(arm, False)]
        lines.append(f"| `{arm}` | `{d['chat_mode']}` | {d['strict_em_norm']*100:.2f}% | "
                     f"{d['lenient_em_norm']*100:.2f}% | {cs*100:.2f}% | {cl*100:.2f}% |")
    if "zeroshot" in best and "lora" in best:
        dz, dl = best["zeroshot"], best["lora"]
        lines += ["", f"**Fine-tuning gain: lenient EM "
                      f"{dz['lenient_em_norm']*100:.2f}% -> {dl['lenient_em_norm']*100:.2f}% "
                      f"({(dl['lenient_em_norm']-dz['lenient_em_norm'])*100:+.2f}pp)**", ""]

    # SAE effect, paired per arm on the same rendering
    lines += ["## Effect of splicing the SAE into the residual stream", ""]
    lines += ["| arm | rendering | lenient no SAE | lenient + SAE | delta | "
              "accuracy retained | SAE FVE | SAE cosine |",
              "|" + "---|" * 8]
    for arm in ("zeroshot", "lora"):
        with_sae = [d for d in rows if d["arm"] == arm and d.get("sae_repo")]
        for w in with_sae:
            base = next((d for d in rows if d["arm"] == arm and not d.get("sae_repo")
                         and d["chat_mode"] == w["chat_mode"]), None)
            if not base:
                continue
            ret = w['lenient_em_norm'] / max(base['lenient_em_norm'], 1e-9)
            lines.append(f"| `{arm}` | `{w['chat_mode']}` | "
                         f"{base['lenient_em_norm']*100:.2f}% | "
                         f"{w['lenient_em_norm']*100:.2f}% | "
                         f"{(w['lenient_em_norm']-base['lenient_em_norm'])*100:+.2f}pp | "
                         f"{ret*100:.1f}% | {w['sae_fve']:.4f} | "
                         f"{w.get('sae_cosine', float('nan')):.4f} |")

    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text)
    print(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

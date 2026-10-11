#!/usr/bin/env python
"""Paired comparison of the DKSplit arms, plus the card's reported numbers.

The arms see identical items, so the right significance test is McNemar's on
the discordant pairs -- not two independent binomial intervals, which would
throw away the pairing and overstate the uncertainty.
"""

import argparse
import json
import math
from pathlib import Path

CARD = {   # model card, 5,000-sample benchmark
    "zeroshot": {"strict_em": 0.6382, "lenient_em": 0.6716, "mutation": 0.0562},
    "lora":     {"strict_em": 0.8496, "lenient_em": 0.8882, "mutation": 0.0025},
    "bilstm_crf": {"strict_em": 0.869, "lenient_em": 0.904},
}


def load(path):
    per = [json.loads(l) for l in open(path)]
    return {r["prefix"]: r for r in per}, per


def mcnemar(a_correct, b_correct):
    """Exact-ish McNemar: b = a right/b wrong, c = a wrong/b right."""
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return b, c, float("nan"), 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n          # Yates-corrected
    # two-sided normal approximation on the discordant pairs
    z = (abs(b - c) - 1) / math.sqrt(n)
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return b, c, chi2, max(min(p, 1.0), 0.0)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zeroshot", required=True, help="zeroshot .jsonl")
    ap.add_argument("--lora", required=True, help="lora .jsonl")
    ap.add_argument("--metric", default="lenient_norm",
                    choices=["strict", "lenient", "strict_norm", "lenient_norm"])
    ap.add_argument("--out", default="dksplit/out/comparison.json")
    args = ap.parse_args()

    z_map, z_all = load(args.zeroshot)
    l_map, l_all = load(args.lora)
    keys = [k for k in z_map if k in l_map]
    print(f"paired on {len(keys)} items "
          f"(zeroshot {len(z_map)}, lora {len(l_map)})\n")

    rows = []
    for m in ["strict", "lenient", "strict_norm", "lenient_norm", "mutation"]:
        za = [z_map[k][m] for k in keys]
        la = [l_map[k][m] for k in keys]
        zr, lr = sum(za) / len(keys), sum(la) / len(keys)
        zlo, zhi = wilson(sum(za), len(keys))
        llo, lhi = wilson(sum(la), len(keys))
        b, c, chi2, p = mcnemar(za, la)
        rows.append(dict(metric=m, zeroshot=zr, lora=lr, delta=lr - zr,
                         zeroshot_ci=[zlo, zhi], lora_ci=[llo, lhi],
                         only_zeroshot_right=b, only_lora_right=c,
                         mcnemar_chi2=chi2, p_value=p))
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
        print(f"{m:14s} zero-shot {zr*100:6.2f}% [{zlo*100:5.2f},{zhi*100:5.2f}]   "
              f"lora {lr*100:6.2f}% [{llo*100:5.2f},{lhi*100:5.2f}]   "
              f"delta {(lr-zr)*100:+6.2f}pp   "
              f"discordant {b}/{c}  p={p:.2e} {star}")

    print("\nvs the model card (5,000-sample benchmark):")
    for arm, key in (("zeroshot", "zeroshot"), ("lora", "lora")):
        got = next(r for r in rows if r["metric"] == "lenient_norm")[arm]
        exp = CARD[key]["lenient_em"]
        print(f"  {arm:9s} lenient EM  measured {got*100:6.2f}%  "
              f"card {exp*100:6.2f}%  diff {(got-exp)*100:+6.2f}pp")
    print(f"  BiLSTM-CRF reference (card): strict {CARD['bilstm_crf']['strict_em']*100:.1f}% "
          f"lenient {CARD['bilstm_crf']['lenient_em']*100:.1f}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(n_paired=len(keys), rows=rows, card=CARD),
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

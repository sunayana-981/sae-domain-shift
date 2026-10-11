#!/usr/bin/env python
"""Do SAE latents represent "word boundary inside a concatenated string"?

The question: which SAE (and which backbone) encodes the segmentation concept
as an explicit, sparse feature. Measured the way the SAE literature measures
concept presence -- k-sparse probing (SAEBench's metric) -- against a gold
label we actually have, because DKSplit ships the correct segmentation.

Two corpora, deliberately the SAME task so they are comparable:

  dksplit  real domain prefixes, gold split from the benchmark's `truth`
  pseudo   FineWeb phrases with the spaces stripped out, gold split = the
           original words

A plain regular corpus will not do: there the boundary is given away by the
leading space on each token, so any probe scores ~1.0 and the comparison is
meaningless. Stripping the spaces keeps the task identical and changes only the
vocabulary -- domain-name coinages vs ordinary web prose.

Label: a token is positive if its character span *ends* where a gold word ends.
Probes are fit on a train split and scored on a held-out split, with a dense
logistic regression on the raw residual stream as the reference point -- an SAE
is only interesting here if a handful of its latents approach what all 4,096
dimensions can do.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from qsae.activations import ResidCapture, resolve_layer
from qsae.data import CachedTokenWindows
from qsae.sae import load_local_sae, load_qwenscope_sae
from eval_dksplit import ADAPTER, BASE, load_benchmark, render
from prompts import PROMPTS

SAE_REPO = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50"
WORD = re.compile(r"[a-z]{2,}")


def dksplit_items(n, train_half):
    df = load_benchmark("5000")
    sub = df.iloc[train_half:].head(n)
    out = []
    for _, r in sub.iterrows():
        words = str(r["truth"]).split()
        if len(words) < 2:
            continue
        if "".join(words) != str(r["prefix"]):
            continue          # gold must be a pure segmentation of the input
        out.append((str(r["prefix"]), words))
    return out


def pseudo_items(n, cache_dir, corpus, tokenizer, seed=0):
    """FineWeb phrases with the spaces removed -- same task, ordinary words."""
    w = CachedTokenWindows(cache_dir, corpus, "eval")
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        ids = w.next_batch(4)
        for row in ids:
            text = tokenizer.decode(row, skip_special_tokens=True).lower()
            toks = WORD.findall(text)
            i = 0
            while i + 3 <= len(toks) and len(out) < n:
                k = int(rng.integers(2, 4))
                words = toks[i:i + k]
                i += k
                joined = "".join(words)
                if 6 <= len(joined) <= 24 and joined not in seen:
                    seen.add(joined)
                    out.append((joined, words))
    return out[:n]


@torch.no_grad()
def capture(model, tok, items, layer, chat_mode, device, batch=16):
    """Activations for the prefix tokens only, with a gold-boundary label."""
    tok.padding_side = "left"
    A, Y = [], []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        texts = [render(tok, PROMPTS["advanced"], p, chat_mode) for p, _ in chunk]
        enc = tok(texts, return_tensors="pt", padding=True,
                  return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        with ResidCapture(model, layer) as cap:
            a = cap.run(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"], use_cache=False)
        mask = enc["attention_mask"].bool()
        for r, (text, (prefix, words)) in enumerate(zip(texts, chunk)):
            start = text.rfind(prefix)
            ends = set()
            c = 0
            for wd in words[:-1]:            # last word's end is the string end
                c += len(wd)
                ends.add(start + c)
            rows, labs = [], []
            for j in torch.nonzero(mask[r], as_tuple=True)[0].tolist():
                s, e = offs[r, j].tolist()
                if e <= s or s < start or e > start + len(prefix):
                    continue
                rows.append(j)
                labs.append(1 if e in ends else 0)
            if rows:
                A.append(a[r, rows].float().cpu())
                Y.extend(labs)
    return torch.cat(A), np.array(Y)


@torch.no_grad()
def feats(sae, acts, device, chunk=2048):
    out = []
    for i in range(0, acts.shape[0], chunk):
        _, f, _ = sae(acts[i:i + chunk].to(device))
        out.append(f.cpu())
    return torch.cat(out)


def sparse_probe(F, y, ks, seed=0):
    """k-sparse probe: rank latents by univariate separation on TRAIN only."""
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    if y[tr].sum() < 5 or y[te].sum() < 5:
        return {}
    Ftr = F[tr].numpy()
    pos, neg = Ftr[y[tr] == 1], Ftr[y[tr] == 0]
    score = np.abs(pos.mean(0) - neg.mean(0)) / (Ftr.std(0) + 1e-6)
    order = np.argsort(-score)
    res = {}
    for k in ks:
        sel = order[:k]
        X_tr, X_te = Ftr[:, sel], F[te].numpy()[:, sel]
        mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit((X_tr - mu) / sd, y[tr])
        p = clf.predict_proba((X_te - mu) / sd)[:, 1]
        res[f"k{k}"] = dict(auc=float(roc_auc_score(y[te], p)),
                            latents=[int(i) for i in sel[:min(k, 10)]])
    return res


def dense_probe(acts, y, seed=0):
    n = len(y)
    idx = np.random.default_rng(seed).permutation(n)
    cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    X = acts.numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    clf = LogisticRegression(max_iter=3000, C=0.1)
    clf.fit((X[tr] - mu) / sd, y[tr])
    p = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    return float(roc_auc_score(y[te], p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--models", nargs="+", default=["lora", "instruct"])
    ap.add_argument("--arms", nargs="+", default=["gsae", "masked", "ftsae"])
    ap.add_argument("--sae_dir", default="dksplit/out/sae")
    ap.add_argument("--n_items", type=int, default=700)
    ap.add_argument("--train_half", type=int, default=2500)
    ap.add_argument("--cache_dir", default="out/token_cache")
    ap.add_argument("--corpus", default="fineweb")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--chat_mode", default="no_think")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="dksplit/out/boundary_probe.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(ADAPTER, token=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    corpora = {"dksplit": dksplit_items(args.n_items, args.train_half),
               "pseudo": pseudo_items(args.n_items, args.cache_dir, args.corpus, tok)}
    for k, v in corpora.items():
        ex = v[0]
        print(f"[corpus] {k}: {len(v)} items, e.g. {ex[0]!r} -> {' '.join(ex[1])!r}")

    report = {"layer": args.layer, "ks": args.ks, "rows": [],
              "examples": {k: [[p, " ".join(w)] for p, w in v[:5]]
                           for k, v in corpora.items()}}

    for mname in args.models:
        m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                 device_map=args.device, token=False)
        if mname == "lora":
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, ADAPTER, token=False).merge_and_unload()
        m.eval()
        abs_layer = resolve_layer(m, args.layer)

        pools = {}
        for cname, items in corpora.items():
            A, y = capture(m, tok, items, args.layer, args.chat_mode, args.device)
            pools[cname] = (A, y)
            print(f"[{mname}/{cname}] {A.shape[0]} tokens, "
                  f"{y.sum()} boundaries ({100*y.mean():.1f}%)", flush=True)
        del m
        torch.cuda.empty_cache()

        for cname, (A, y) in pools.items():
            d = dense_probe(A, y)
            report["rows"].append(dict(model=mname, corpus=cname, sae="dense(4096d)",
                                       n_tokens=int(A.shape[0]),
                                       pos_rate=float(y.mean()),
                                       **{f"k{k}": None for k in args.ks},
                                       auc_dense=d))
            print(f"  {mname:9s} {cname:8s} dense residual      AUC {d:.4f}", flush=True)

        for arm in args.arms:
            if arm == "gsae":
                sae, _ = load_qwenscope_sae(SAE_REPO, abs_layer, device=args.device)
            else:
                p = Path(args.sae_dir) / arm / f"sae_9b_layer{abs_layer}_{arm}.pt"
                if not p.exists():
                    print(f"[skip] {arm}"); continue
                sae = load_local_sae(str(p), device=args.device)
            for cname, (A, y) in pools.items():
                F = feats(sae, A, args.device)
                res = sparse_probe(F, y, args.ks)
                row = dict(model=mname, corpus=cname, sae=arm,
                           n_tokens=int(A.shape[0]), pos_rate=float(y.mean()),
                           auc_dense=None)
                for k in args.ks:
                    row[f"k{k}"] = res.get(f"k{k}", {}).get("auc")
                row["top_latents"] = res.get(f"k{args.ks[-1]}", {}).get("latents")
                report["rows"].append(row)
                print(f"  {mname:9s} {cname:8s} {arm:11s} " +
                      "  ".join(f"k={k} AUC {row[f'k{k}']:.4f}" if row[f"k{k}"] else f"k={k} —"
                                for k in args.ks), flush=True)
                del F
            del sae
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

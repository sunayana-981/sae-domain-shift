#!/usr/bin/env python
"""Top-activating SAE latents on the DKSplit corpus, per SAE arm.

A rendered DKSplit prompt is 63 tokens of which 60 are the *identical* system
scaffold, so ranking latents over all tokens just surfaces the system prompt.
Activations are therefore tagged by region -- `domain` (the tokens of the
domain string being segmented) vs `scaffold` -- and latents are ranked within
each, so the domain-name concepts are visible separately.

Two passes over one cached set of activations:
  1. per-latent firing rate / mean / max, accumulated sparsely (Top-K means
     only k=50 of 65,536 latents are non-zero per token)
  2. for the top latents only, the tokens that activate them most, with context

For `masked`, each latent is tagged protected/free so it is visible whether the
top concepts come from the frozen reference set or the adapted capacity.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qsae.activations import ResidCapture, resolve_layer
from qsae.sae import load_local_sae, load_qwenscope_sae
from eval_dksplit import ADAPTER, BASE, load_benchmark, render
from prompts import PROMPTS

SAE_REPO = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50"


@torch.no_grad()
def capture(model, tok, prefixes, layer, chat_mode, device, batch=16):
    """Activations plus, per row, the token id and whether it is a domain token."""
    tok.padding_side = "left"
    A, T, D = [], [], []
    for i in range(0, len(prefixes), batch):
        chunk = prefixes[i:i + batch]
        texts = [render(tok, PROMPTS["advanced"], p, chat_mode) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True,
                  return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        with ResidCapture(model, layer) as cap:
            a = cap.run(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"], use_cache=False)
        mask = enc["attention_mask"].bool()
        # char span of the domain string inside each rendered prompt
        for r, (text, prefix) in enumerate(zip(texts, chunk)):
            start = text.rfind(prefix)
            end = start + len(prefix)
            keep = mask[r]
            for j in torch.nonzero(keep, as_tuple=True)[0].tolist():
                s, e = offs[r, j].tolist()
                D.append(bool(s >= start and e <= end and e > s))
        A.append(a[mask].float().cpu())
        T.append(enc["input_ids"][mask].cpu())
    return torch.cat(A), torch.cat(T), torch.tensor(D)


@torch.no_grad()
def latent_stats(sae, acts, device, chunk=4096):
    """Sparse accumulation of firing count / activation sum / max per latent."""
    d = sae.d_sae
    cnt = torch.zeros(d, device=device)
    tot = torch.zeros(d, device=device)
    mx = torch.zeros(d, device=device)
    for i in range(0, acts.shape[0], chunk):
        x = acts[i:i + chunk].to(device)
        _, f, _ = sae(x)
        nz = f > 1e-8
        cnt += nz.float().sum(0)
        tot += f.sum(0)
        mx = torch.maximum(mx, f.max(0).values)
    return cnt.cpu(), tot.cpu(), mx.cpu()


@torch.no_grad()
def top_tokens_for(sae, acts, toks, idxs, device, tokenizer, chunk=4096, topn=8):
    """For selected latents, the tokens that activate them most."""
    sel = torch.tensor(idxs, device=device)
    best_v = torch.full((len(idxs), topn), -1.0)
    best_i = torch.zeros((len(idxs), topn), dtype=torch.long)
    for i in range(0, acts.shape[0], chunk):
        x = acts[i:i + chunk].to(device)
        _, f, _ = sae(x)
        v = f[:, sel].cpu().t()                       # [n_sel, chunk]
        cat_v = torch.cat([best_v, v], dim=1)
        cat_i = torch.cat([best_i, torch.arange(i, i + x.shape[0]).expand(len(idxs), -1)], dim=1)
        top = cat_v.topk(topn, dim=1)
        best_v, best_i = top.values, torch.gather(cat_i, 1, top.indices)
    out = []
    for r, lat in enumerate(idxs):
        toks_r = [tokenizer.decode([toks[j].item()]) for j in best_i[r]]
        out.append(dict(latent=int(lat),
                        top_tokens=toks_r,
                        top_values=[round(float(x), 4) for x in best_v[r]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--sae_dir", default="dksplit/out/sae")
    ap.add_argument("--arms", nargs="+", default=["gsae", "masked", "ftsae"])
    ap.add_argument("--n_prompts", type=int, default=600)
    ap.add_argument("--train_half", type=int, default=2500)
    ap.add_argument("--top_k", type=int, default=25)
    ap.add_argument("--max_scaffold_tokens", type=int, default=4000,
                   help="The scaffold is the same 60 tokens repeated once per "
                        "prompt, so a sample is representative and keeps the "
                        "cached activations small.")
    ap.add_argument("--chat_mode", default="no_think")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="dksplit/out/top_concepts.json")
    args = ap.parse_args()

    df = load_benchmark("5000")
    prefixes = list(df["prefix"].iloc[args.train_half:].head(args.n_prompts))

    tok = AutoTokenizer.from_pretrained(ADAPTER, token=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                 device_map=args.device, token=False)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER, token=False).merge_and_unload()
    model.eval()
    abs_layer = resolve_layer(model, args.layer)

    acts, toks, is_dom = capture(model, tok, prefixes, args.layer,
                                 args.chat_mode, args.device)
    print(f"[acts] {acts.shape[0]} tokens "
          f"({int(is_dom.sum())} domain, {int((~is_dom).sum())} scaffold)", flush=True)
    # Keep every domain token, subsample the redundant scaffold.
    scaf = torch.nonzero(~is_dom, as_tuple=True)[0]
    if scaf.numel() > args.max_scaffold_tokens:
        g = torch.Generator().manual_seed(0)
        keep_scaf = scaf[torch.randperm(scaf.numel(), generator=g)[:args.max_scaffold_tokens]]
        keep = torch.zeros(acts.shape[0], dtype=torch.bool)
        keep[torch.nonzero(is_dom, as_tuple=True)[0]] = True
        keep[keep_scaf] = True
        acts, toks, is_dom = acts[keep], toks[keep], is_dom[keep]
        print(f"[acts] subsampled to {acts.shape[0]} tokens "
              f"({int(is_dom.sum())} domain, {int((~is_dom).sum())} scaffold)", flush=True)
    del model
    torch.cuda.empty_cache()

    report = {"layer": abs_layer, "n_prompts": len(prefixes),
              "n_tokens": int(acts.shape[0]), "n_domain": int(is_dom.sum()),
              "arms": {}}

    for arm in args.arms:
        if arm == "gsae":
            sae, _ = load_qwenscope_sae(SAE_REPO, abs_layer, device=args.device)
            protected = None
        else:
            p = Path(args.sae_dir) / arm / f"sae_9b_layer{abs_layer}_{arm}.pt"
            if not p.exists():
                print(f"[skip] {arm}: {p} missing"); continue
            sae = load_local_sae(str(p), device=args.device)
            blob = torch.load(p, map_location="cpu")
            pm = blob.get("extra", {}).get("protected_mask")
            protected = set(torch.nonzero(pm).flatten().tolist()) if pm is not None else None
        print(f"\n[{arm}] protected latents: "
              f"{len(protected) if protected else 0}", flush=True)

        entry = {"regions": {}}
        for region, sel in (("domain", is_dom), ("scaffold", ~is_dom)):
            sub = acts[sel]
            cnt, tot, mx = latent_stats(sae, sub, args.device)
            rate = cnt / max(sub.shape[0], 1)
            mean_when_on = tot / cnt.clamp_min(1)
            rank = torch.argsort(tot, descending=True)[:args.top_k].tolist()
            tops = top_tokens_for(sae, sub, toks[sel], rank, args.device, tok)
            for t in tops:
                lat = t["latent"]
                t["firing_rate"] = float(rate[lat])
                t["mean_act_when_active"] = float(mean_when_on[lat])
                t["total_act"] = float(tot[lat])
                t["max_act"] = float(mx[lat])
                t["protected"] = (None if protected is None else (lat in protected))
            entry["regions"][region] = {
                "n_tokens": int(sub.shape[0]),
                "n_alive": int((cnt > 0).sum()),
                "dead_frac": float(1 - (cnt > 0).float().mean()),
                "top": tops,
                "rank_curve": [float(x) for x in
                               torch.sort(tot, descending=True).values[:2000]],
            }
            print(f"  {region:9s} alive {int((cnt>0).sum()):6d}/{sae.d_sae}  "
                  f"top latent {rank[0]} "
                  f"({'prot' if protected and rank[0] in protected else 'free'})"
                  f" tokens={tops[0]['top_tokens'][:4]}", flush=True)
        report["arms"][arm] = entry
        del sae
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

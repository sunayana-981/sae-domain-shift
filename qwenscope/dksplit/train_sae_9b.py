#!/usr/bin/env python
"""Masked fine-tune of the 9B Qwen-Scope SAE onto the DKSplit LoRA model.

The problem this addresses, measured earlier: the released SAE explains 0.711
of variance on Qwen3.5-9B-instruct but **-0.033** on the same model with the
DKSplit LoRA merged in, because that adapter inflates the layer-24 residual RMS
from 1.66 to 115.4 -- a 63x scale shift the SAE's fixed b_dec and unit-norm
decoder cannot represent.

What is adapted to, and why it is not the DKSplit prompts: a rendered DKSplit
prompt is 63 tokens of which 60 are the *identical* system scaffold, so only
~3 tokens per item vary. An SAE trained on that would largely learn to
reconstruct one fixed prompt. The thing that actually broke the SAE is the
model, not the prompt distribution, and that shift is visible on any text. So
training activations come from the LoRA model over a general corpus, with a
`--dksplit_frac` share of DKSplit prompts mixed in so the chat scaffold is
represented too. Those prompts are drawn only from the SAE-train half of the
benchmark, so the accuracy eval stays clean.

Reference distribution for the protected mask is the *instruct* model (no
adapter) over the same general corpus -- the distribution the released SAE
already handles well. Protecting the units most active there and freeing the
rest is the question: can the SAE absorb the fine-tuned model without losing
the original one?

Arms:
    masked      protect the top --protect_frac reference-active units
    ftsae       same budget, nothing frozen (control)
    scratchsae  random init, same budget (matched-compute floor)
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qsae.activations import ResidCapture, resolve_layer
from qsae.data import CachedTokenWindows
from qsae.masked_trainer import MaskedTopKSAETrainer, compute_protected_mask
from qsae.sae import TopKSAE, load_qwenscope_sae
from eval_dksplit import ADAPTER, BASE, load_benchmark, render
from prompts import PROMPTS

SAE_REPO = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50"


@dataclass
class TrainCfg:
    total_training_tokens: int
    aux_k: int = 512
    aux_coefficient: float = 1 / 32
    dead_feature_window: int = 64
    log_every: int = 20


def build_model(with_lora, device):
    tok = AutoTokenizer.from_pretrained(ADAPTER, token=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                             device_map=device, token=False)
    if with_lora:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, ADAPTER, token=False).merge_and_unload()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, tok


class MixedActivationStore:
    """Residual activations from a general corpus, mixed with DKSplit prompts.

    `dksplit_frac` of each batch's tokens come from rendered DKSplit prompts
    (train half only) and the rest from the cached general corpus, so the SAE
    sees both the model's ordinary distribution and the chat scaffold it will
    actually be spliced under.
    """

    def __init__(self, model, tok, windows, prefixes, layer, tokens_per_batch,
                 seqs_per_forward, dksplit_frac, chat_mode, device, seed=42):
        self.model, self.tok, self.windows = model, tok, windows
        self.prefixes = list(prefixes)
        self.layer, self.device = layer, device
        self.tokens_per_batch = tokens_per_batch
        self.seqs_per_forward = seqs_per_forward
        self.dksplit_frac = dksplit_frac
        self.chat_mode = chat_mode
        self.rng = np.random.default_rng(seed)
        self._buf, self._n = [], 0
        self.tokens_served = 0

    @torch.no_grad()
    def _corpus_batch(self):
        ids = self.windows.next_batch(self.seqs_per_forward).to(self.device)
        with ResidCapture(self.model, self.layer) as cap:
            acts = cap.run(input_ids=ids, use_cache=False)
        return acts[:, 1:].reshape(-1, acts.shape[-1]).float()

    @torch.no_grad()
    def _dksplit_batch(self, n_prompts):
        idx = self.rng.choice(len(self.prefixes), n_prompts, replace=False)
        texts = [render(self.tok, PROMPTS["advanced"], self.prefixes[i],
                        self.chat_mode) for i in idx]
        self.tok.padding_side = "left"
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        with ResidCapture(self.model, self.layer) as cap:
            acts = cap.run(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"], use_cache=False)
        return acts[enc["attention_mask"].bool()].float()

    @torch.no_grad()
    def get_batch_activations(self):
        while self._n < self.tokens_per_batch:
            if self.dksplit_frac > 0 and self.rng.random() < self.dksplit_frac:
                flat = self._dksplit_batch(min(32, len(self.prefixes)))
            else:
                flat = self._corpus_batch()
            self._buf.append(flat)
            self._n += flat.shape[0]
        pool = torch.cat(self._buf)
        out, rest = pool[: self.tokens_per_batch], pool[self.tokens_per_batch:]
        self._buf = [rest] if rest.shape[0] else []
        self._n = rest.shape[0]
        self.tokens_served += out.shape[0]
        return out


class CorpusOnlyStore:
    """General-corpus activations only -- used for the reference activity pass."""

    def __init__(self, model, windows, layer, tokens_per_batch, seqs_per_forward, device):
        self.inner = MixedActivationStore(model, None, windows, [], layer,
                                          tokens_per_batch, seqs_per_forward,
                                          0.0, "no_think", device)

    def get_batch_activations(self):
        return self.inner.get_batch_activations()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True,
                   choices=["masked", "ftsae", "scratchsae"])
    p.add_argument("--sae_repo", default=SAE_REPO)
    p.add_argument("--layer", type=int, default=24)
    p.add_argument("--encode_variant", default="raw")
    p.add_argument("--cache_dir", default="out/token_cache")
    p.add_argument("--corpus", default="fineweb")
    p.add_argument("--dksplit_frac", type=float, default=0.2)
    p.add_argument("--chat_mode", default="no_think")
    p.add_argument("--train_half", type=int, default=2500,
                   help="First N benchmark items are the SAE-train half; the "
                        "rest are held out for the accuracy eval.")
    p.add_argument("--protect_frac", type=float, default=0.2)
    p.add_argument("--protect_mode", default="top",
                   choices=["top", "bottom", "random"])
    p.add_argument("--activity_n_batches", type=int, default=30)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--total_training_tokens", type=int, default=2_000_000)
    p.add_argument("--tokens_per_batch", type=int, default=4096)
    p.add_argument("--seqs_per_forward", type=int, default=4)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default="dksplit/out/sae")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out_dir) / args.condition
    out.mkdir(parents=True, exist_ok=True)

    df = load_benchmark("5000")
    train_prefixes = list(df["prefix"].head(args.train_half))
    print("=" * 66)
    print(f"9B SAE {args.condition}  layer {args.layer}  "
          f"corpus={args.corpus} dksplit_frac={args.dksplit_frac}")
    print(f"SAE-train prompts: {len(train_prefixes)} (held out for eval: "
          f"{len(df) - args.train_half})")
    print("=" * 66, flush=True)

    # ── Step A: protected mask, on the INSTRUCT model (no adapter) ───────────
    mask, mask_stats = None, {}
    if args.condition == "masked":
        print("[step A] reference activity on the instruct model (no adapter)",
              flush=True)
        instruct, _ = build_model(False, args.device)
        abs_layer = resolve_layer(instruct, args.layer)
        sae_ref, _ = load_qwenscope_sae(args.sae_repo, abs_layer,
                                        device=args.device,
                                        encode_variant=args.encode_variant)
        windows = CachedTokenWindows(args.cache_dir, args.corpus, "train",
                                     seed=args.seed)
        ref_store = CorpusOnlyStore(instruct, windows, args.layer,
                                    args.tokens_per_batch,
                                    args.seqs_per_forward, args.device)
        mask, _activity, mask_stats = compute_protected_mask(
            sae_ref, ref_store, protect_frac=args.protect_frac,
            n_batches=args.activity_n_batches, device=args.device,
            protect_mode=args.protect_mode, seed=args.seed)
        del sae_ref, ref_store, windows, instruct
        torch.cuda.empty_cache()

    # ── the SAE under training ──────────────────────────────────────────────
    lora_model, tok = build_model(True, args.device)
    abs_layer = resolve_layer(lora_model, args.layer)
    if args.condition == "scratchsae":
        ref, _ = load_qwenscope_sae(args.sae_repo, abs_layer, device=args.device,
                                    encode_variant=args.encode_variant)
        cfg_sae = ref.cfg
        del ref
        torch.cuda.empty_cache()
        sae = TopKSAE(cfg_sae, device=args.device)
        print(f"[sae] random init d_in={sae.d_model} d_sae={sae.d_sae} k={sae.k}")
    else:
        sae, _ = load_qwenscope_sae(args.sae_repo, abs_layer, device=args.device,
                                    encode_variant=args.encode_variant)
        print(f"[sae] loaded {sae.cfg.source}")

    # ── Step B: fine-tune on LoRA-model activations ─────────────────────────
    windows = CachedTokenWindows(args.cache_dir, args.corpus, "train",
                                 seed=args.seed + 7)
    store = MixedActivationStore(lora_model, tok, windows, train_prefixes,
                                 args.layer, args.tokens_per_batch,
                                 args.seqs_per_forward, args.dksplit_frac,
                                 args.chat_mode, args.device, seed=args.seed)
    cfg = TrainCfg(total_training_tokens=args.total_training_tokens)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1)))
    trainer = MaskedTopKSAETrainer(sae, store, opt, sched, cfg,
                                   protected_mask=mask, device=args.device)
    t0 = time.time()
    history = trainer.fit()

    ckpt = out / f"sae_9b_layer{abs_layer}_{args.condition}.pt"
    trainer.save(ckpt, extra={"condition": args.condition, "layer": abs_layer,
                              "sae_repo": args.sae_repo, "mask_stats": mask_stats,
                              "args": vars(args)})
    json.dump(dict(condition=args.condition, checkpoint=str(ckpt),
                   layer=abs_layer, sae_repo=args.sae_repo,
                   protect_frac=args.protect_frac if mask is not None else 0.0,
                   protect_mode=args.protect_mode if mask is not None else None,
                   mask_stats=mask_stats, dksplit_frac=args.dksplit_frac,
                   train_half=args.train_half,
                   tokens=trainer.n_training_tokens, steps=trainer.n_training_steps,
                   wall_seconds=time.time() - t0,
                   final=history[-1] if history else None),
              open(out / "manifest.json", "w"), indent=2)
    print(f"[done] {args.condition} in {(time.time()-t0)/60:.1f} min -> {ckpt}")


if __name__ == "__main__":
    main()

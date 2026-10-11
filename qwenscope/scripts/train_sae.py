#!/usr/bin/env python
"""Train one SAE arm of the Qwen-Scope masked-fine-tune experiment.

Arms (same names as patchsae's `SAE_CONDITIONS`):

  masked      fine-tune the pre-trained Qwen-Scope SAE on the target corpus
              with the top `protect_frac` reference-active units frozen
  ftsae       fine-tune the same pre-trained SAE, nothing frozen
  scratchsae  same architecture, random init, same token budget

`ftsae` and `scratchsae` are the controls that isolate what masking buys: they
share every hyperparameter, token budget, corpus and backbone with `masked`,
and differ only in the protected mask (empty) and the initialisation.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from qsae.activations import LMActivationStore, resolve_layer
from qsae.data import CachedTokenWindows, REFERENCE_CORPUS, get_corpus
from qsae.masked_trainer import MaskedTopKSAETrainer, compute_protected_mask
from qsae.sae import TopKSAE, load_qwenscope_sae

CONDITIONS = ("masked", "ftsae", "scratchsae")


@dataclass
class TrainCfg:
    total_training_tokens: int
    aux_k: int = 256
    aux_coefficient: float = 1 / 32
    dead_feature_window: int = 64
    log_every: int = 20


def load_backbone(model_id, adapter, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
        print(f"[train] merged LoRA adapter {adapter}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--corpus", required=True, help="target (OOD) corpus")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    p.add_argument("--sae_repo", default="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50")
    p.add_argument("--encode_variant", default="raw")
    p.add_argument("--layer", type=int, default=-4,
                   help="Decoder block whose resid_post the SAE reads. Negative "
                        "counts from the end, as patchsae's --block_layer does.")
    p.add_argument("--backbone_type", default="lora", choices=["base", "lora"])
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter dir; defaults to out/lora/<corpus>/adapter "
                        "when --backbone_type lora.")
    p.add_argument("--cache_dir", default="out/token_cache")
    p.add_argument("--out_dir", default="out/sae")

    # masking
    p.add_argument("--protect_frac", type=float, default=0.2)
    p.add_argument("--protect_mode", default="top", choices=["top", "bottom", "random"])
    p.add_argument("--activity_corpus", default=REFERENCE_CORPUS,
                   help="Corpus the protected units are selected on (the "
                        "reference distribution whose concepts get preserved).")
    p.add_argument("--activity_n_batches", type=int, default=50)
    p.add_argument("--freeze_b_dec", action="store_true")
    p.add_argument("--no_normalize_decoder", action="store_true")

    # optimisation
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--total_training_tokens", type=int, default=2_000_000)
    p.add_argument("--tokens_per_batch", type=int, default=4096)
    p.add_argument("--seqs_per_forward", type=int, default=4)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--aux_k", type=int, default=256)
    p.add_argument("--dead_feature_window", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--label", default=None,
                   help="Name this arm carries in the results matrix and on "
                        "disk. Defaults to --condition; the protect-mode and "
                        "protect-fraction ablations use it to sit alongside "
                        "the main `masked` arm (e.g. masked_bottom, "
                        "masked_pf0.4) without changing the algorithm.")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    label = args.label or args.condition
    tag = args.tag or f"{args.corpus}_{args.backbone_type}_{label}"
    out = Path(args.out_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    adapter = args.adapter
    if args.backbone_type == "lora" and adapter is None:
        adapter = f"out/lora/{args.corpus}/adapter"
    if args.backbone_type == "base":
        adapter = None

    print("=" * 66)
    print(f"ARM {tag}")
    print(f"  condition={args.condition} corpus={args.corpus} "
          f"backbone={args.backbone_type} layer={args.layer}")
    print("=" * 66)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_backbone(args.model, adapter, args.device)
    abs_layer = resolve_layer(model, args.layer)
    print(f"[train] hooking resid_post of block {abs_layer}")

    # ── the SAE under training ────────────────────────────────────────────
    if args.condition == "scratchsae":
        sae_ref, raw_cfg = load_qwenscope_sae(args.sae_repo, abs_layer,
                                              device=args.device,
                                              encode_variant=args.encode_variant)
        cfg_sae = sae_ref.cfg
        del sae_ref
        torch.cuda.empty_cache()
        sae = TopKSAE(cfg_sae, device=args.device)
        print(f"[train] random init: d_in={sae.d_model} d_sae={sae.d_sae} k={sae.k}")
    else:
        sae, raw_cfg = load_qwenscope_sae(args.sae_repo, abs_layer,
                                          device=args.device,
                                          encode_variant=args.encode_variant)
        print(f"[train] loaded {sae.cfg.source}: d_in={sae.d_model} "
              f"d_sae={sae.d_sae} k={sae.k}")

    # ── Step A: protected mask on the reference corpus ────────────────────
    mask, activity, mask_stats = None, None, {}
    if args.condition == "masked":
        ref_windows = CachedTokenWindows(args.cache_dir, args.activity_corpus,
                                         "train", seed=args.seed)
        ref_store = LMActivationStore(model, ref_windows, args.layer,
                                      tokens_per_batch=args.tokens_per_batch,
                                      seqs_per_forward=args.seqs_per_forward,
                                      device=args.device)
        print(f"[train] activity on {args.activity_corpus} "
              f"({args.activity_n_batches} x {args.tokens_per_batch} tokens)")
        mask, activity, mask_stats = compute_protected_mask(
            sae, ref_store, protect_frac=args.protect_frac,
            n_batches=args.activity_n_batches, device=args.device,
            protect_mode=args.protect_mode, seed=args.seed)
        del ref_store, ref_windows
        torch.cuda.empty_cache()

    # ── Step B: fine-tune on the target corpus ───────────────────────────
    corpus = get_corpus(args.corpus)
    train_windows = CachedTokenWindows(args.cache_dir, args.corpus, "train",
                                       seed=args.seed)
    store = LMActivationStore(model, train_windows, args.layer,
                              tokens_per_batch=args.tokens_per_batch,
                              seqs_per_forward=args.seqs_per_forward,
                              device=args.device)
    print(f"[train] target: {corpus.name} ({corpus.domain}) | {train_windows}")

    cfg = TrainCfg(total_training_tokens=args.total_training_tokens,
                   aux_k=args.aux_k, dead_feature_window=args.dead_feature_window)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1)))

    trainer = MaskedTopKSAETrainer(
        sae, store, opt, sched, cfg, protected_mask=mask, device=args.device,
        freeze_b_dec=args.freeze_b_dec,
        normalize_decoder=not args.no_normalize_decoder)

    t0 = time.time()
    history = trainer.fit()

    ckpt = out / f"sae_layer{abs_layer}_{label}.pt"
    trainer.save(ckpt, extra={
        "condition": args.condition, "label": label, "corpus": args.corpus,
        "backbone_type": args.backbone_type, "adapter": adapter,
        "layer": abs_layer, "block_layer_arg": args.layer,
        "activity_corpus": args.activity_corpus if args.condition == "masked" else None,
        "mask_stats": mask_stats, "args": vars(args),
    })
    manifest = dict(tag=tag, checkpoint=str(ckpt), condition=args.condition,
                    label=label,
                    corpus=args.corpus, backbone_type=args.backbone_type,
                    adapter=adapter, layer=abs_layer,
                    sae_repo=args.sae_repo, encode_variant=args.encode_variant,
                    d_sae=sae.d_sae, d_model=sae.d_model, k=sae.k,
                    protect_frac=args.protect_frac if args.condition == "masked" else 0.0,
                    protect_mode=args.protect_mode if args.condition == "masked" else None,
                    mask_stats=mask_stats, seed=args.seed,
                    total_training_tokens=trainer.n_training_tokens,
                    steps=trainer.n_training_steps,
                    train_epochs_over_cache=train_windows.epochs,
                    wall_seconds=time.time() - t0,
                    final=history[-1] if history else None)
    json.dump(manifest, open(out / "manifest.json", "w"), indent=2)
    print(f"[train] {tag} done in {(time.time() - t0) / 60:.1f} min -> {ckpt}")


if __name__ == "__main__":
    main()

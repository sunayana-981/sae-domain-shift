#!/usr/bin/env python
"""Unified {corpus, backbone, sae_condition, eval_corpus} eval matrix.

The LLM counterpart of `patchsae/tasks/eval_matrix.py`. Same sweep shape and
the same four reconstruction metrics (l0, dead_frac, recon_cosine, fve); what
replaces the vision experiment's `zeroshot_acc` is the task-level effect of
splicing the SAE into the residual stream:

  ce_clean        cross-entropy with no intervention
  ce_sae          cross-entropy reading the SAE reconstruction instead
  ce_zero/mean    the two ablation floors (see ResidIntervention)
  delta_ce        ce_sae - ce_clean
  ce_recovered    1 - delta_ce / (ce_zero - ce_clean)   <- headline number
  kl              mean KL(clean || sae) over next-token distributions

The extra axis over the vision version is `eval_corpus`: every SAE is scored
both on its own target corpus and on the reference corpus. That second block
is the retention test (patchsae's `imagenet_anchor_eval`) and is where masked
fine-tuning is supposed to differ from an unmasked one.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from qsae.activations import (LMActivationStore, ResidIntervention,
                              resolve_layer)
from qsae.data import CachedTokenWindows, OOD_CORPORA, REFERENCE_CORPUS, get_corpus
from qsae.sae import load_local_sae, load_qwenscope_sae

# `masked_*` names are ablation arms trained by the same code path with a
# different unit-selection rule (`--protect_mode`) or budget (`--protect_frac`),
# written to disk under their label so they sit alongside the main arm.
CONDITIONS = ["none", "gsae", "ftsae", "scratchsae", "masked",
              "masked_bottom", "masked_random"]
BACKBONES = ["base", "lora"]

RESULT_COLUMNS = [
    "train_corpus", "eval_corpus", "eval_role", "backbone", "sae_condition",
    "layer", "sae_path", "l0", "dead_frac", "recon_cosine", "fve",
    "ce_clean", "ce_sae", "ce_zero", "ce_mean", "delta_ce",
    "ce_recovered", "ce_recovered_mean", "kl", "n_eval_tokens", "skipped",
]


def is_valid_cell(backbone, condition):
    """`ftsae`/`scratchsae`/`masked` SAEs are trained on the LoRA backbone's
    activations, so scoring them on the base backbone would compare an SAE
    against a residual stream it never saw. patchsae's eval matrix guards the
    same way (`controlled_common.is_valid_cell`)."""
    if condition in ("none", "gsae"):
        return True
    return backbone == "lora"


# ═════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def recon_metrics(model, windows, layer, sae, tokens_per_batch, seqs_per_forward,
                  n_batches, device, l0_threshold=1e-8):
    """One pass with NO hook registered: capture the true activations and run
    them through the SAE directly, so nothing the SAE does can feed back into
    the activations it is being measured on."""
    windows.reset()
    store = LMActivationStore(model, windows, layer,
                              tokens_per_batch=tokens_per_batch,
                              seqs_per_forward=seqs_per_forward, device=device)
    ever_active = torch.zeros(sae.d_sae, dtype=torch.bool, device=device)
    l0_sum = l0_tokens = 0
    cos_sum = cos_n = 0
    err_sum = var_sum = 0.0

    for _ in range(n_batches):
        x = store.get_batch_activations()
        x_hat, acts, _ = sae(x)
        active = acts > l0_threshold
        l0_sum += active.float().sum().item()
        l0_tokens += active.shape[0]
        ever_active |= active.any(dim=0)
        cos = F.cosine_similarity(x, x_hat, dim=-1)
        cos_sum += cos.sum().item(); cos_n += cos.numel()
        err_sum += (x - x_hat).pow(2).sum(-1).sum().item()
        var_sum += (x - sae.b_dec).pow(2).sum(-1).sum().item()

    return dict(
        l0=l0_sum / max(l0_tokens, 1),
        dead_frac=1.0 - ever_active.float().mean().item(),
        recon_cosine=cos_sum / max(cos_n, 1),
        fve=1.0 - err_sum / max(var_sum, 1e-12),
        n_eval_tokens=l0_tokens,
    )


@torch.no_grad()
def _ce_and_logits(model, ids, keep_positions):
    out = model(input_ids=ids, labels=ids, use_cache=False)
    ce = out.loss.item()
    kept = out.logits[:, keep_positions, :].float()
    del out
    return ce, kept


@torch.no_grad()
def lm_metrics(model, windows, layer, sae, n_batches, seqs_per_forward, device,
               kl_stride=4):
    """Cross-entropy and next-token KL under each intervention.

    Logits are the memory constraint here (248k vocab), so KL is measured on
    every `kl_stride`-th position rather than all of them, and the kept slice
    is the only thing cast to float32.
    """
    windows.reset()
    seq_len = windows.seq_len
    keep = torch.arange(0, seq_len, kl_stride)

    tot = {"clean": 0.0, "sae": 0.0, "zero": 0.0, "mean": 0.0}
    kl_sum, kl_n, n = 0.0, 0, 0

    for _ in range(n_batches):
        ids = windows.next_batch(seqs_per_forward).to(device)

        ce_clean, lg_clean = _ce_and_logits(model, ids, keep)
        tot["clean"] += ce_clean

        if sae is not None:
            with ResidIntervention(model, layer, sae=sae, mode="sae"):
                ce_sae, lg_sae = _ce_and_logits(model, ids, keep)
            tot["sae"] += ce_sae
            lp_clean = F.log_softmax(lg_clean, dim=-1)
            lp_sae = F.log_softmax(lg_sae, dim=-1)
            kl = (lp_clean.exp() * (lp_clean - lp_sae)).sum(-1)
            kl_sum += kl.sum().item(); kl_n += kl.numel()
            del lg_sae, lp_clean, lp_sae
        del lg_clean

        for mode in ("zero", "mean"):
            with ResidIntervention(model, layer, mode=mode):
                tot[mode] += model(input_ids=ids, labels=ids,
                                   use_cache=False).loss.item()
        n += 1

    ce_clean = tot["clean"] / n
    ce_zero = tot["zero"] / n
    ce_mean = tot["mean"] / n
    if sae is None:
        return dict(ce_clean=ce_clean, ce_sae=ce_clean, ce_zero=ce_zero,
                    ce_mean=ce_mean, delta_ce=0.0, ce_recovered=1.0,
                    ce_recovered_mean=1.0, kl=0.0)

    ce_sae = tot["sae"] / n
    delta = ce_sae - ce_clean
    return dict(
        ce_clean=ce_clean, ce_sae=ce_sae, ce_zero=ce_zero, ce_mean=ce_mean,
        delta_ce=delta,
        ce_recovered=1.0 - delta / max(ce_zero - ce_clean, 1e-9),
        ce_recovered_mean=1.0 - delta / max(ce_mean - ce_clean, 1e-9),
        kl=kl_sum / max(kl_n, 1),
    )


# ═════════════════════════════════════════════════════════════════════════
# Model / SAE resolution
# ═════════════════════════════════════════════════════════════════════════

def load_backbone(model_id, adapter, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return model


def resolve_sae(condition, train_corpus, backbone, args, abs_layer):
    if condition == "none":
        return None, ""
    if condition == "gsae":
        return "__pretrained__", args.sae_repo
    path = Path(args.sae_dir) / f"{train_corpus}_{backbone}_{condition}" / \
        f"sae_layer{abs_layer}_{condition}.pt"  # condition == the arm's label
    return (str(path), str(path)) if path.exists() else (None, str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpora", nargs="+", default=OOD_CORPORA)
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    p.add_argument("--backbones", nargs="+", default=BACKBONES, choices=BACKBONES)
    p.add_argument("--reference_corpus", default=REFERENCE_CORPUS)
    p.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    p.add_argument("--sae_repo", default="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50")
    p.add_argument("--encode_variant", default="raw")
    p.add_argument("--layer", type=int, default=-4)
    p.add_argument("--sae_dir", default="out/sae")
    p.add_argument("--lora_dir", default="out/lora")
    p.add_argument("--cache_dir", default="out/token_cache")
    p.add_argument("--out_dir", default="out/results")
    p.add_argument("--recon_batches", type=int, default=30)
    p.add_argument("--lm_batches", type=int, default=20)
    p.add_argument("--tokens_per_batch", type=int, default=4096)
    p.add_argument("--seqs_per_forward", type=int, default=4)
    p.add_argument("--lm_seqs_per_forward", type=int, default=1)
    p.add_argument("--kl_stride", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.csv"

    # Group the sweep by (train_corpus, backbone) so each backbone is built once.
    rows = []
    done = set()
    if results_path.exists():
        prev = pd.read_csv(results_path)
        rows = prev.to_dict("records")
        done = {(r["train_corpus"], r["eval_corpus"], r["backbone"],
                 r["sae_condition"]) for r in rows}
        print(f"[eval] resuming: {len(done)} cells already in {results_path}")

    for train_corpus in args.corpora:
        for backbone in args.backbones:
            cells = [(train_corpus, ec, backbone, c)
                     for c in args.conditions
                     for ec in (train_corpus, args.reference_corpus)
                     if is_valid_cell(backbone, c)]
            todo = [c for c in cells if c not in done]
            if not todo:
                continue

            adapter = (str(Path(args.lora_dir) / train_corpus / "adapter")
                       if backbone == "lora" else None)
            if adapter and not Path(adapter).exists():
                print(f"[eval] SKIP {train_corpus}/{backbone}: no adapter at {adapter}")
                continue

            print(f"\n{'=' * 66}\n[eval] backbone={backbone} "
                  f"train_corpus={train_corpus} ({len(todo)} cells)\n{'=' * 66}")
            if args.dry_run:
                for c in todo:
                    print("  would eval", c)
                continue

            model = load_backbone(args.model, adapter, args.device)
            abs_layer = resolve_layer(model, args.layer)

            # Cache one SAE object per condition across both eval corpora.
            sae_cache = {}
            for (tc, eval_corpus, bb, cond) in todo:
                t0 = time.time()
                spec, sae_path = resolve_sae(cond, tc, bb, args, abs_layer)
                if cond != "none" and spec is None:
                    print(f"  [skip] {cond} on {eval_corpus}: missing {sae_path}")
                    rows.append({**{c: None for c in RESULT_COLUMNS},
                                 "train_corpus": tc, "eval_corpus": eval_corpus,
                                 "eval_role": "target" if eval_corpus == tc else "reference",
                                 "backbone": bb, "sae_condition": cond,
                                 "layer": abs_layer, "sae_path": sae_path,
                                 "skipped": "missing_checkpoint"})
                    continue

                if cond not in sae_cache:
                    if cond == "none":
                        sae_cache[cond] = None
                    elif cond == "gsae":
                        sae_cache[cond] = load_qwenscope_sae(
                            args.sae_repo, abs_layer, device=args.device,
                            encode_variant=args.encode_variant)[0]
                    else:
                        sae_cache[cond] = load_local_sae(
                            spec, device=args.device,
                            encode_variant=args.encode_variant)
                sae = sae_cache[cond]

                windows = CachedTokenWindows(args.cache_dir, eval_corpus, "eval",
                                             seed=args.seed)
                rec = dict(l0=None, dead_frac=None, recon_cosine=None, fve=None,
                           n_eval_tokens=None)
                if sae is not None:
                    rec = recon_metrics(model, windows, args.layer, sae,
                                        args.tokens_per_batch,
                                        args.seqs_per_forward,
                                        args.recon_batches, args.device)
                lm = lm_metrics(model, windows, args.layer, sae, args.lm_batches,
                                args.lm_seqs_per_forward, args.device,
                                kl_stride=args.kl_stride)

                row = {"train_corpus": tc, "eval_corpus": eval_corpus,
                       "eval_role": "target" if eval_corpus == tc else "reference",
                       "backbone": bb, "sae_condition": cond, "layer": abs_layer,
                       "sae_path": sae_path, **rec, **lm, "skipped": None}
                rows.append(row)
                print(f"  {cond:11s} eval={eval_corpus:8s} "
                      f"fve={_f(row['fve'])} cos={_f(row['recon_cosine'])} "
                      f"dead={_f(row['dead_frac'])} "
                      f"dCE={_f(row['delta_ce'])} rec={_f(row['ce_recovered'])} "
                      f"kl={_f(row['kl'])} ({time.time() - t0:.0f}s)", flush=True)

                pd.DataFrame(rows)[RESULT_COLUMNS].to_csv(results_path, index=False)

            del model, sae_cache
            torch.cuda.empty_cache()

    if args.dry_run:
        return
    df = pd.DataFrame(rows)[RESULT_COLUMNS]
    df.to_csv(results_path, index=False)
    json.dump(vars(args), open(out_dir / "eval_args.json", "w"), indent=2)
    print(f"\n[eval] wrote {results_path} ({len(df)} rows)")


def _f(v):
    return "   nan" if v is None or (isinstance(v, float) and v != v) else f"{v:6.3f}"


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Causal latent steering for Qwen-Scope SAEs -- the language counterpart of
patchsae's causal-steering experiment (tables/controlled_steering.tex,
tables/controlled_h5.tex).

For each (corpus, SAE condition), the top-k SAE latents by activity on that
corpus's own activations are treated as "the dictionary's domain-specific
latents" (the language analogue of "top-activating latent per class" in the
vision version, since a text corpus has no discrete classes). Necessity is
measured by zeroing those latents (LatentIntervention, an error-preserving
edit -- see qsae/activations.py) and reading the drop in the corpus's paired
task accuracy, against a matched random-k control. If gsae's top latents are
not causally load-bearing for the task (ablating them costs little more than
ablating random ones) while a domain-adapted SAE's (masked/ftsae) top latents
are, that reproduces the vision experiment's central causal claim in an LLM.

Usage:
    python scripts/eval_steering.py --corpora pubmed_central github \
        --conditions gsae masked ftsae --k 5 25
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from qsae.activations import LatentIntervention, LMActivationStore
from qsae.data import CachedTokenWindows
from qsae.masked_trainer import estimate_unit_activity
from qsae.sae import load_local_sae, load_qwenscope_sae
from qsae.tasks import TASKS, run_task

CONDITIONS = ["gsae", "ftsae", "masked", "scratchsae"]


def load_backbone(model_id, adapter, device):
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                             device_map=device)
    if adapter:
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.eval()
    m.config.use_cache = True
    return m


def find_local_sae_path(sae_dir, corpus, condition):
    d = Path(sae_dir) / f"{corpus}_lora_{condition}"
    matches = sorted(d.glob("*.pt")) if d.exists() else []
    if not matches:
        matches = sorted(d.glob("**/*.pt")) if d.exists() else []
    return matches[-1] if matches else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpora", nargs="+", default=["pubmed_central", "github"])
    p.add_argument("--conditions", nargs="+", default=["gsae", "masked", "ftsae"],
                   choices=CONDITIONS)
    p.add_argument("--k", type=int, nargs="+", default=[5, 25],
                   help="Top-k / random-k latent-set sizes, matching the vision "
                        "experiment's k=5,25 (tables/controlled_h5.tex).")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    p.add_argument("--sae_repo", default="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50")
    p.add_argument("--layer", type=int, default=20,
                   help="Positive layer index (block 20 of 24, matching "
                        "Qwen-Scope's layer20.sae.pt). Must be positive: "
                        "load_qwenscope_sae checks `layer in cfg['layers']` "
                        "against the release's positive-indexed layer list "
                        "and does not resolve negative indices the way "
                        "resolve_layer() (used for the hook point) does.")
    p.add_argument("--sae_dir", default="out/sae")
    p.add_argument("--lora_dir", default="out/lora")
    p.add_argument("--cache_dir", default="out/token_cache")
    p.add_argument("--out_dir", default="out/results")
    p.add_argument("--n_activity_batches", type=int, default=50)
    p.add_argument("--activity_tokens_per_batch", type=int, default=4096)
    p.add_argument("--n_task", type=int, default=None,
                   help="Override the task's default item count (smaller = faster).")
    p.add_argument("--keep_completions", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "steering_results.csv"
    samples_path = out_dir / "steering_samples.json"

    domain_task = {t.domain: name for name, t in TASKS.items() if t.domain != "general"}

    rows, samples, done = [], {}, set()
    if csv_path.exists():
        rows = pd.read_csv(csv_path).to_dict("records")
        done = {(r["corpus"], r["condition"], r["k"]) for r in rows}
        if samples_path.exists():
            samples = json.load(open(samples_path))
        print(f"[steering] resuming: {len(done)} cells already scored")

    tok = AutoTokenizer.from_pretrained(args.model)

    for corpus in args.corpora:
        if corpus not in domain_task:
            print(f"[steering] SKIP {corpus}: no domain benchmark registered")
            continue
        task = TASKS[domain_task[corpus]]

        adapter = str(Path(args.lora_dir) / corpus / "adapter")
        if not Path(adapter).exists():
            print(f"[steering] SKIP {corpus}: no LoRA adapter at {adapter}")
            continue

        todo_conditions = [c for c in args.conditions
                           if not all((corpus, c, k) in done for k in args.k)]
        if not todo_conditions:
            print(f"[steering] {corpus}: all conditions already done, skipping load")
            continue

        print(f"\n[steering] === {corpus} (adapter={adapter}) ===")
        model = load_backbone(args.model, adapter, args.device)

        token_source = CachedTokenWindows(args.cache_dir, corpus, split="eval")
        store = LMActivationStore(model, token_source, args.layer, device=args.device,
                                  tokens_per_batch=args.activity_tokens_per_batch)

        baseline_cache = {}

        for condition in todo_conditions:
            if all((corpus, condition, k) in done for k in args.k):
                continue
            print(f"[steering]   condition={condition}")
            if condition == "gsae":
                sae, _ = load_qwenscope_sae(args.sae_repo, args.layer, device=args.device)
            else:
                sae_path = find_local_sae_path(args.sae_dir, corpus, condition)
                if sae_path is None:
                    print(f"[steering]   SKIP {corpus}/{condition}: no local SAE checkpoint")
                    continue
                sae = load_local_sae(str(sae_path), device=args.device)

            t0 = time.time()
            activity, fire_rate = estimate_unit_activity(
                sae, store, n_batches=args.n_activity_batches)
            print(f"[steering]   activity estimated in {time.time()-t0:.0f}s "
                  f"(mean fire_rate={fire_rate.mean().item():.4f})")

            if "baseline" not in baseline_cache:
                base_res = run_task(task, model, tok, n=args.n_task, seed=args.seed,
                                    device=args.device, keep_completions=args.keep_completions)
                baseline_cache["baseline"] = base_res
                print(f"[steering]   baseline (no intervention) acc={base_res['accuracy']:.4f}")
            base_res = baseline_cache["baseline"]

            rng = np.random.default_rng(args.seed)
            for k in args.k:
                if (corpus, condition, k) in done:
                    continue
                top_idx = torch.topk(activity, k).indices.cpu()
                remaining = np.setdiff1d(np.arange(sae.d_sae), top_idx.numpy())
                random_idx = torch.tensor(rng.choice(remaining, size=k, replace=False))

                with LatentIntervention(model, args.layer, sae, top_idx, mode="ablate"):
                    top_res = run_task(task, model, tok, n=args.n_task, seed=args.seed,
                                       device=args.device,
                                       keep_completions=args.keep_completions)
                with LatentIntervention(model, args.layer, sae, random_idx, mode="ablate"):
                    rand_res = run_task(task, model, tok, n=args.n_task, seed=args.seed,
                                        device=args.device,
                                        keep_completions=args.keep_completions)

                top_drop = base_res["accuracy"] - top_res["accuracy"]
                rand_drop = base_res["accuracy"] - rand_res["accuracy"]
                row = dict(corpus=corpus, condition=condition, k=k, task=task.name,
                          metric=task.metric, n=base_res["n"],
                          baseline_acc=base_res["accuracy"],
                          top_ablate_acc=top_res["accuracy"],
                          random_ablate_acc=rand_res["accuracy"],
                          top_drop=top_drop, random_drop=rand_drop,
                          necessity_gap=top_drop - rand_drop,
                          mean_top_fire_rate=fire_rate[top_idx].mean().item())
                print(f"[steering]     k={k}: baseline={base_res['accuracy']:.4f} "
                      f"top_ablate={top_res['accuracy']:.4f} rand_ablate={rand_res['accuracy']:.4f} "
                      f"necessity_gap={row['necessity_gap']:+.4f}")
                rows.append(row)
                samples[f"{corpus}/{condition}/k{k}/top"] = top_res.get("samples", [])
                samples[f"{corpus}/{condition}/k{k}/random"] = rand_res.get("samples", [])

                pd.DataFrame(rows).to_csv(csv_path, index=False)
                json.dump(samples, open(samples_path, "w"), indent=2, default=str)

            del sae
            torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

    print(f"\n[steering] wrote {csv_path}")
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

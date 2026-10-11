#!/usr/bin/env python
"""Task accuracy with the SAE spliced into the residual stream.

The counterpart of `eval_matrix.py`: that script measures how well the SAE
reconstructs activations, this one measures whether the model can still do its
job while reading those reconstructions. It is the direct analogue of
patchsae's `zeroshot_acc`, which is the metric the vision experiment leads with.

Each corpus is scored on its own domain's benchmark and on GSM8K, which no
domain owns -- so every arm reports both "did adaptation help the target task"
and "did it damage general reasoning".
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from qsae.activations import ResidIntervention, resolve_layer
from qsae.data import OOD_CORPORA
from qsae.sae import load_local_sae, load_qwenscope_sae
from qsae.tasks import TASKS, run_task

CONDITIONS = ["none", "gsae", "ftsae", "scratchsae", "masked",
              "masked_bottom", "masked_random"]
COLUMNS = ["train_corpus", "backbone", "sae_condition", "task", "task_role",
           "metric", "accuracy", "n", "layer", "sae_path", "seconds"]


def is_valid_cell(backbone, condition):
    return condition in ("none", "gsae") or backbone == "lora"


def load_backbone(model_id, adapter, device):
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
                                             device_map=device)
    if adapter:
        m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.eval()
    m.config.use_cache = True
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpora", nargs="+", default=OOD_CORPORA)
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    p.add_argument("--backbones", nargs="+", default=["base", "lora"])
    p.add_argument("--general_task", default="gsm8k",
                   help="Task no domain owns, run as the general-capability "
                        "retention check.")
    p.add_argument("--general_conditions", nargs="+",
                   default=["none", "gsae", "ftsae", "scratchsae", "masked"],
                   help="Conditions that get the general task. The protect_mode "
                        "ablation arms are excluded by default: they exist to "
                        "isolate *which* units are frozen, a question the "
                        "domain task answers, and GSM8K is the slowest task "
                        "here by a wide margin.")
    p.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    p.add_argument("--sae_repo", default="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50")
    p.add_argument("--encode_variant", default="raw")
    p.add_argument("--layer", type=int, default=-4)
    p.add_argument("--sae_dir", default="out/sae")
    p.add_argument("--lora_dir", default="out/lora")
    p.add_argument("--out_dir", default="out/results")
    p.add_argument("--n_domain", type=int, default=None,
                   help="Override the domain task's item count.")
    p.add_argument("--n_general", type=int, default=120)  # GSM8K items
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--keep_completions", type=int, default=3,
                   help="Sample generations saved per cell, for eyeballing.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "task_results.csv"
    samples_path = out_dir / "task_samples.json"

    domain_task = {t.domain: name for name, t in TASKS.items()
                   if t.domain != "general"}

    # On the base backbone there is no adapter, so a (condition, task) cell is
    # literally the same computation for every corpus -- same weights, same
    # SAE, same items. GSM8K is run for every corpus, so without this memo the
    # base/none and base/gsae GSM8K cells would be recomputed three times.
    memo = {}

    rows, samples, done = [], {}, set()
    if csv_path.exists():
        rows = pd.read_csv(csv_path).to_dict("records")
        done = {(r["train_corpus"], r["backbone"], r["sae_condition"], r["task"])
                for r in rows}
        if samples_path.exists():
            samples = json.load(open(samples_path))
        print(f"[tasks] resuming: {len(done)} cells already scored")

    tok = AutoTokenizer.from_pretrained(args.model)

    for corpus in args.corpora:
        if corpus not in domain_task:
            print(f"[tasks] SKIP {corpus}: no domain benchmark registered")
            continue
        task_names = [domain_task[corpus]]
        if args.general_task:
            task_names.append(args.general_task)

        for backbone in args.backbones:
            cells = [(corpus, backbone, c, t) for c in args.conditions
                     for t in task_names if is_valid_cell(backbone, c)
                     and (t != args.general_task or c in args.general_conditions)]
            todo = [c for c in cells if c not in done]
            if not todo:
                continue

            adapter = (str(Path(args.lora_dir) / corpus / "adapter")
                       if backbone == "lora" else None)
            if adapter and not Path(adapter).exists():
                print(f"[tasks] SKIP {corpus}/{backbone}: no adapter")
                continue

            print(f"\n{'=' * 66}\n[tasks] {corpus} / {backbone} "
                  f"({len(todo)} cells)\n{'=' * 66}", flush=True)
            if args.dry_run:
                for c in todo:
                    print("  would run", c)
                continue

            model = load_backbone(args.model, adapter, args.device)
            abs_layer = resolve_layer(model, args.layer)
            sae_cache = {}

            for (_, _, cond, tname) in todo:
                task = TASKS[tname]
                if args.batch_size:
                    task.batch_size = args.batch_size
                n = (args.n_general if task.domain == "general"
                     else (args.n_domain or task.n_default))

                sae_path = ""
                if cond not in sae_cache:
                    if cond == "none":
                        sae_cache[cond] = None
                    elif cond == "gsae":
                        sae_cache[cond] = load_qwenscope_sae(
                            args.sae_repo, abs_layer, device=args.device,
                            encode_variant=args.encode_variant)[0]
                    else:
                        path = (Path(args.sae_dir) / f"{corpus}_{backbone}_{cond}"
                                / f"sae_layer{abs_layer}_{cond}.pt")
                        if not path.exists():
                            print(f"  [skip] {cond}/{tname}: missing {path}")
                            sae_cache[cond] = "__missing__"
                        else:
                            sae_cache[cond] = load_local_sae(
                                str(path), device=args.device,
                                encode_variant=args.encode_variant)
                sae = sae_cache[cond]
                if sae == "__missing__":
                    continue
                if cond not in ("none", "gsae"):
                    sae_path = str(Path(args.sae_dir) / f"{corpus}_{backbone}_{cond}"
                                   / f"sae_layer{abs_layer}_{cond}.pt")
                elif cond == "gsae":
                    sae_path = args.sae_repo

                memo_key = (backbone, cond, tname, n) if backbone == "base" else None
                if memo_key is not None and memo_key in memo:
                    prev_res, prev_samples, dt = memo[memo_key]
                    res = dict(prev_res)
                    samples[f"{corpus}|{backbone}|{cond}|{tname}"] = prev_samples
                    rows.append({
                        "train_corpus": corpus, "backbone": backbone,
                        "sae_condition": cond, "task": tname,
                        "task_role": "target" if task.domain == corpus else "general",
                        "metric": res["metric"], "accuracy": res["accuracy"],
                        "n": res["n"], "layer": abs_layer,
                        "sae_path": args.sae_repo if cond == "gsae" else "",
                        "seconds": 0.0})
                    print(f"  {cond:14s} {tname:18s} "
                          f"{res['metric']} = {res['accuracy'] * 100:5.1f}%  "
                          f"(n={res['n']}, reused base-backbone result)", flush=True)
                    pd.DataFrame(rows)[COLUMNS].to_csv(csv_path, index=False)
                    json.dump(samples, open(samples_path, "w"), indent=1)
                    continue

                t0 = time.time()
                if sae is None:
                    res = run_task(task, model, tok, n=n, seed=args.seed,
                                   device=args.device,
                                   keep_completions=args.keep_completions)
                else:
                    with ResidIntervention(model, args.layer, sae=sae, mode="sae"):
                        res = run_task(task, model, tok, n=n, seed=args.seed,
                                       device=args.device,
                                       keep_completions=args.keep_completions)
                dt = time.time() - t0

                key = f"{corpus}|{backbone}|{cond}|{tname}"
                samples[key] = res.pop("samples", [])
                row = {"train_corpus": corpus, "backbone": backbone,
                       "sae_condition": cond, "task": tname,
                       "task_role": "target" if task.domain == corpus else "general",
                       "metric": res["metric"], "accuracy": res["accuracy"],
                       "n": res["n"], "layer": abs_layer, "sae_path": sae_path,
                       "seconds": round(dt, 1)}
                rows.append(row)
                print(f"  {cond:14s} {tname:18s} "
                      f"{res['metric']} = {res['accuracy'] * 100:5.1f}%  "
                      f"(n={res['n']}, {dt:.0f}s)", flush=True)

                if memo_key is not None:
                    memo[memo_key] = (dict(res), samples[key], dt)

                pd.DataFrame(rows)[COLUMNS].to_csv(csv_path, index=False)
                json.dump(samples, open(samples_path, "w"), indent=1)

            del model, sae_cache
            torch.cuda.empty_cache()

    if not args.dry_run and rows:
        pd.DataFrame(rows)[COLUMNS].to_csv(csv_path, index=False)
        print(f"\n[tasks] wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

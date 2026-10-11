#!/usr/bin/env python
"""Turn out/results/results.csv into the tables the experiment is about.

Three views, per OOD corpus:
  1. target   -- did the SAE adapt to the new domain?
  2. reference-- did it keep working on the distribution it came from?
  3. trade-off-- both, as deltas against the untouched Qwen-Scope SAE (gsae),
                 which is the only arm that has not been trained on anything.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ORDER = ["none", "gsae", "ftsae", "scratchsae", "masked",
         "masked_bottom", "masked_random"]
MIN_EFFECT = 0.005   # smallest FVE change worth normalising by
METRICS = ["fve", "recon_cosine", "dead_frac", "l0",
           "ce_clean", "ce_sae", "delta_ce", "ce_recovered", "kl"]


def md_table(df, cols, floatfmt="{:.4f}"):
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, rule]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("—" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append("—" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def order_conditions(df):
    df = df.copy()
    df["__o"] = df["sae_condition"].map({c: i for i, c in enumerate(ORDER)})
    return df.sort_values("__o").drop(columns="__o")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="out/results/results.csv")
    p.add_argument("--out", default="out/results/SUMMARY.md")
    p.add_argument("--backbone", default="lora")
    p.add_argument("--reference", default="fineweb")
    p.add_argument("--tasks", default="out/results/task_results.csv")
    args = p.parse_args()

    df = pd.read_csv(args.results)
    df = df[df["skipped"].isna()]
    tasks = None
    if Path(args.tasks).exists():
        tasks = pd.read_csv(args.tasks)
        print(f"[summarize] merged {len(tasks)} task-accuracy rows")
    if df.empty:
        sys.exit("no completed rows in results.csv")

    lines = ["# Masked SAE fine-tuning on Qwen-Scope — results", ""]
    lines.append(f"Backbone shown: `{args.backbone}`. Reference corpus: "
                 f"`{args.reference}`. `gsae` = the released Qwen-Scope SAE, "
                 f"untouched.")
    lines.append("")

    summary = {}
    for corpus in sorted(df["train_corpus"].unique()):
        sub = df[(df.train_corpus == corpus) & (df.backbone == args.backbone)]
        if sub.empty:
            continue
        lines.append(f"## {corpus}")
        for role, label in (("target", f"Target domain ({corpus})"),
                            ("reference", f"Reference domain ({args.reference})")):
            blk = order_conditions(sub[sub.eval_role == role])
            if blk.empty:
                continue
            lines += [f"", f"### {label}", "",
                      md_table(blk, ["sae_condition"] + METRICS), ""]

        tk = tasks[tasks.train_corpus == corpus] if tasks is not None else None
        if tk is not None and not tk.empty:
            tk = tk[tk.backbone == args.backbone]
            for role, label in (("target", "Domain task accuracy, SAE spliced in"),
                                ("general", "General-capability retention (GSM8K), SAE spliced in")):
                blk = order_conditions(tk[tk.task_role == role])
                if blk.empty:
                    continue
                name = blk["task"].iloc[0]
                lines += [f"### {label} — `{name}`", "",
                          md_table(blk, ["sae_condition", "task", "metric",
                                         "accuracy", "n"], "{:.4f}"), ""]

        tgt = sub[sub.eval_role == "target"].set_index("sae_condition")
        ref = sub[sub.eval_role == "reference"].set_index("sae_condition")
        if "gsae" in tgt.index and "gsae" in ref.index:
            rows = []
            for cond in ORDER:
                if cond in ("none", "gsae") or cond not in tgt.index \
                        or cond not in ref.index:
                    continue
                rows.append({
                    "sae_condition": cond,
                    "d_fve_target": tgt.loc[cond, "fve"] - tgt.loc["gsae", "fve"],
                    "d_fve_reference": ref.loc[cond, "fve"] - ref.loc["gsae", "fve"],
                    "d_ce_recovered_target": tgt.loc[cond, "ce_recovered"] - tgt.loc["gsae", "ce_recovered"],
                    "d_ce_recovered_reference": ref.loc[cond, "ce_recovered"] - ref.loc["gsae", "ce_recovered"],
                    "d_kl_target": tgt.loc[cond, "kl"] - tgt.loc["gsae", "kl"],
                    "d_kl_reference": ref.loc[cond, "kl"] - ref.loc["gsae", "kl"],
                })
            if rows:
                tr = pd.DataFrame(rows)
                lines += ["### Adaptation vs retention (change from `gsae`)", "",
                          md_table(tr, list(tr.columns), "{:+.4f}"), ""]

                # The headline ratio: `ftsae` is the unmasked fine-tune, so it
                # sets both the most adaptation available at this budget and
                # the forgetting that comes with it. An arm is worth having if
                # it keeps most of the first and little of the second.
                base = tr[tr.sae_condition == "ftsae"]
                if not base.empty:
                    gain_ft = float(base["d_fve_target"].iloc[0])
                    loss_ft = float(base["d_fve_reference"].iloc[0])
                    # Only normalise by a denominator that is really there. If
                    # the unmasked fine-tune did not lose reference FVE, there
                    # was no forgetting to prevent on this corpus and the ratio
                    # is undefined, not zero -- dividing by a near-zero or
                    # wrong-signed baseline manufactures huge meaningless
                    # numbers.
                    has_gain = gain_ft > MIN_EFFECT
                    has_loss = loss_ft < -MIN_EFFECT
                    eff = []
                    for _, r in tr.iterrows():
                        eff.append({
                            "sae_condition": r["sae_condition"],
                            "adaptation_kept": (r["d_fve_target"] / gain_ft
                                                if has_gain else float("nan")),
                            "forgetting_incurred": (r["d_fve_reference"] / loss_ft
                                                    if has_loss else float("nan")),
                        })
                    ef = pd.DataFrame(eff)
                    lines += ["As a fraction of what the unmasked fine-tune "
                              "(`ftsae`) does — 1.00 means 'same as ftsae', so a "
                              "good arm is high on the first column and low on "
                              "the second:", "",
                              md_table(ef, list(ef.columns), "{:.2f}"), ""]
                    if not has_loss:
                        lines += [f"`forgetting_incurred` is undefined here: "
                                  f"`ftsae` changed reference FVE by "
                                  f"{loss_ft:+.4f}, i.e. it did not measurably "
                                  f"forget, so there is nothing for masking to "
                                  f"prevent on this corpus.", ""]
                    if not has_gain:
                        lines += [f"`adaptation_kept` is undefined here: `ftsae` "
                                  f"changed target FVE by {gain_ft:+.4f}.", ""]
                    summary[corpus] = {"deltas": tr.to_dict("records"),
                                       "relative_to_ftsae": ef.to_dict("records"),
                                       "ftsae_fve_gain_target": gain_ft,
                                       "ftsae_fve_loss_reference": loss_ft}
                else:
                    summary[corpus] = {"deltas": tr.to_dict("records")}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    json.dump(summary, open(out.with_suffix(".json"), "w"), indent=2)
    print("\n".join(lines))
    print(f"\n[summarize] wrote {out}")


if __name__ == "__main__":
    main()

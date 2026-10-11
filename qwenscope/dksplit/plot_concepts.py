#!/usr/bin/env python
"""Top-activating SAE latents on the DKSplit domain tokens, per SAE arm.

Three views:
  A  rank curve   -- how concentrated each SAE's representation is
  B  top latents  -- the concepts themselves, labelled by the tokens that fire
                     them; for `masked`, protected latents are hatched
  C  overlap      -- how much of each arm's top set it still shares with the
                     released SAE

Colours are the reference categorical slots 1-3 (blue/orange/aqua), validated:
worst normal-vision dE 24.0 (floor 15), worst protan/deutan dE 10.0 (target 8).
Aqua sits at 2.74:1 on the light surface, below the 3:1 mark, so every bar
carries a direct value label and a table view ships beside the figure.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ARMS = ["gsae", "masked", "ftsae"]
NICE = {"gsae": "gsae (released)", "masked": "masked FT", "ftsae": "unmasked FT"}

THEME = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e",
                  muted="#8d8c86", grid="#e6e5e1",
                  series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark":  dict(surface="#1a1a19", text="#ffffff", text2="#c3c2b7",
                  muted="#8d8c86", grid="#333331",
                  series=["#3987e5", "#d95926", "#199e70"]),
}


def tokline(toks, n=4):
    out, seen = [], set()
    for t in toks:
        s = t.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) == n:
            break
    return " ".join(f"{t}" for t in out) or "·"


def build(data, mode, out_png, top_n=8):
    C = THEME[mode]
    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"], "text.color": C["text"],
        "axes.labelcolor": C["text2"], "xtick.color": C["text2"],
        "ytick.color": C["text2"], "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C["grid"],
    })
    fig = plt.figure(figsize=(14.5, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.35],
                          hspace=0.42, wspace=0.30,
                          left=0.125, right=0.985, top=0.885, bottom=0.075)

    n_dom = {a: data["arms"][a]["regions"]["domain"]["n_tokens"] for a in ARMS}

    # ── A: rank curve ────────────────────────────────────────────────────────
    axA = fig.add_subplot(gs[0, :2])
    for i, arm in enumerate(ARMS):
        r = data["arms"][arm]["regions"]["domain"]
        y = np.array(r["rank_curve"]) / max(r["n_tokens"], 1)
        y = y[y > 0]
        axA.plot(np.arange(1, len(y) + 1), y, lw=2, color=C["series"][i],
                 label=f"{NICE[arm]}  ({r['n_alive']:,} alive)",
                 solid_capstyle="round")
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel("latent rank on domain tokens")
    axA.set_ylabel("mean activation per token")
    axA.set_title("A  How concentrated each SAE is on this corpus",
                  loc="left", color=C["text"], fontsize=10.5, fontweight="bold", pad=8)
    axA.grid(True, which="major", color=C["grid"], lw=0.8, alpha=0.9)
    axA.set_axisbelow(True)
    leg = axA.legend(frameon=False, loc="upper right", fontsize=9)
    for t in leg.get_texts():
        t.set_color(C["text2"])
    # ── C: overlap with gsae's top set ───────────────────────────────────────
    axC = fig.add_subplot(gs[0, 2])
    gtop = {t["latent"] for t in data["arms"]["gsae"]["regions"]["domain"]["top"]}
    labels, vals, cols = [], [], []
    for i, arm in enumerate(ARMS[1:], start=1):
        tops = data["arms"][arm]["regions"]["domain"]["top"]
        shared = len({t["latent"] for t in tops} & gtop)
        labels.append(NICE[arm]); vals.append(shared); cols.append(C["series"][i])
    ypos = np.arange(len(labels))
    axC.barh(ypos, vals, height=0.42, color=cols)
    for y, v in zip(ypos, vals):
        axC.text(v + 0.35, y, f"{v}/{len(gtop)}", va="center",
                 color=C["text2"], fontsize=9)
    axC.set_yticks(ypos); axC.set_yticklabels(labels, color=C["text2"])
    axC.set_xlim(0, len(gtop) * 1.1)
    axC.invert_yaxis()
    axC.set_xlabel(f"top-{len(gtop)} latents still shared with gsae")
    axC.set_title("C  Concept turnover", loc="left", color=C["text"],
                  fontsize=10.5, fontweight="bold", pad=8)
    axC.grid(True, axis="x", color=C["grid"], lw=0.8)
    axC.set_axisbelow(True)

    # ── B: the concepts themselves ───────────────────────────────────────────
    vmax = max(t["total_act"] / n_dom[a]
               for a in ARMS for t in data["arms"][a]["regions"]["domain"]["top"][:top_n])
    for i, arm in enumerate(ARMS):
        ax = fig.add_subplot(gs[1, i])
        tops = data["arms"][arm]["regions"]["domain"]["top"][:top_n]
        vals = [t["total_act"] / n_dom[arm] for t in tops]
        ypos = np.arange(len(tops))
        for y, (t, v) in zip(ypos, zip(tops, vals)):
            prot = bool(t.get("protected"))
            ax.barh(y, v, height=0.62, color=C["series"][i],
                    hatch="///" if prot else None,
                    edgecolor=C["surface"] if prot else "none", linewidth=0.0)
            ax.text(v + vmax * 0.015, y, f"{v:.1f}", va="center",
                    color=C["text2"], fontsize=8)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"L{t['latent']}  {tokline(t['top_tokens'], 3)}" for t in tops],
                           fontsize=8, color=C["text2"])
        ax.invert_yaxis()
        ax.set_xlim(0, vmax * 1.22)
        ax.grid(True, axis="x", color=C["grid"], lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("mean activation per domain token")
        head = "B  Top concepts" if i == 0 else ""
        ax.set_title(f"{head}{'   ' if head else ''}{NICE[arm]}", loc="left",
                     color=C["text"], fontsize=10.5,
                     fontweight="bold" if i == 0 else "normal", pad=8)
        if arm == "masked":
            ax.legend(handles=[Patch(facecolor=C["series"][i], hatch="///",
                                     edgecolor=C["surface"], label="protected (frozen)"),
                               Patch(facecolor=C["series"][i], label="free (adapted)")],
                      frameon=False, fontsize=8, loc="center right",
                      bbox_to_anchor=(1.0, 0.42), labelcolor=C["text2"])

    fig.suptitle("Top-activating SAE latents on DKSplit domain names — "
                 "Qwen3.5-9B + DKSplit LoRA, layer 24",
                 x=0.035, ha="left", color=C["text"], fontsize=13, fontweight="bold")
    fig.text(0.035, 0.925,
             f"{data['n_domain']:,} domain-name tokens from {data['n_prompts']} held-out prompts; "
             f"system-scaffold tokens excluded. d_sae=65,536, k=50.",
             ha="left", color=C["muted"], fontsize=9)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"wrote {out_png}")


def table(data, out_md, top_n=8):
    L = ["# Top-activating latents on DKSplit domain tokens\n",
         f"{data['n_domain']:,} domain tokens, layer {data['layer']}, d_sae=65,536, k=50.\n"]
    for arm in ARMS:
        r = data["arms"][arm]["regions"]["domain"]
        L.append(f"\n## {NICE[arm]} — {r['n_alive']:,} alive latents, "
                 f"dead fraction {r['dead_frac']:.3f}\n")
        L.append("| latent | frozen? | firing rate | mean act (when on) | "
                 "mean act / token | top tokens |")
        L.append("|---|---|---|---|---|---|")
        for t in r["top"][:top_n]:
            prot = {True: "protected", False: "free", None: "—"}[t.get("protected")]
            L.append(f"| L{t['latent']} | {prot} | {t['firing_rate']:.3f} | "
                     f"{t['mean_act_when_active']:.2f} | "
                     f"{t['total_act']/r['n_tokens']:.2f} | "
                     f"`{tokline(t['top_tokens'], 6)}` |")
    Path(out_md).write_text("\n".join(L) + "\n")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dksplit/out/top_concepts.json")
    ap.add_argument("--out_dir", default="dksplit/out")
    args = ap.parse_args()
    d = json.load(open(args.data))
    o = Path(args.out_dir)
    build(d, "light", o / "top_concepts.png")
    build(d, "dark", o / "top_concepts_dark.png")
    table(d, o / "top_concepts.md")

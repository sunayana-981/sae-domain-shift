#!/usr/bin/env python
"""Is "word boundary in a concatenated string" a sparse SAE concept?

k-sparse probe AUC against k, faceted by backbone x corpus. The dashed line is
a dense logistic regression on all 4,096 residual dimensions -- the reference an
SAE has to approach with a handful of latents to be said to carry the concept.

Palette: reference categorical slots 1-3 (blue/orange/aqua); worst normal-vision
dE 24.0, worst protan/deutan dE 10.0. Every point is direct-labelled at k=20 and
a table view ships alongside.
"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ["gsae", "masked", "ftsae"]
NICE = {"gsae": "gsae (released)", "masked": "masked FT", "ftsae": "unmasked FT"}
CORP = {"dksplit": "DKSplit domain names", "pseudo": "FineWeb phrases, spaces stripped"}
MODEL = {"instruct": "Qwen3.5-9B instruct", "lora": "instruct + DKSplit LoRA"}
THEME = {
 "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#8d8c86",
               grid="#e6e5e1", series=["#2a78d6", "#eb6834", "#1baf7a"]),
 "dark":  dict(surface="#1a1a19", text="#ffffff", text2="#c3c2b7", muted="#8d8c86",
               grid="#333331", series=["#3987e5", "#d95926", "#199e70"]),
}

def build(d, mode, out):
    C = THEME[mode]; ks = d["ks"]
    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "savefig.facecolor": C["surface"], "text.color": C["text"],
        "axes.labelcolor": C["text2"], "xtick.color": C["text2"],
        "ytick.color": C["text2"], "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C["grid"]})
    models = ["instruct", "lora"]; corpora = ["dksplit", "pseudo"]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), sharey=True,
                             gridspec_kw=dict(hspace=0.40, wspace=0.10,
                                              left=0.075, right=0.955,
                                              top=0.800, bottom=0.085))
    for r, mdl in enumerate(models):
        for c, cor in enumerate(corpora):
            ax = axes[r][c]
            dense = next((x["auc_dense"] for x in d["rows"]
                          if x["model"] == mdl and x["corpus"] == cor and x["auc_dense"]), None)
            if dense:
                ax.axhline(dense, ls=(0, (5, 3)), lw=1.6, color=C["muted"], zorder=1)
                ax.text(ks[0], dense + 0.012, f"dense 4096d  {dense:.3f}",
                        va="bottom", ha="left", color=C["muted"], fontsize=8)
            ends = []
            for i, arm in enumerate(ARMS):
                row = next((x for x in d["rows"] if x["model"] == mdl
                            and x["corpus"] == cor and x["sae"] == arm), None)
                if not row: continue
                y = [row[f"k{k}"] for k in ks]
                ax.plot(ks, y, lw=2, marker="o", ms=6.5, color=C["series"][i],
                        label=NICE[arm], zorder=3, solid_capstyle="round",
                        markeredgecolor=C["surface"], markeredgewidth=1.4)
                ends.append([y[-1], y[-1], C["series"][i]])
            # Arms can finish within a hair of each other (gsae vs masked differ
            # by 0.002 in places), so push the end labels apart vertically while
            # leaving the markers where the data is.
            ends.sort(key=lambda e: e[0])
            MIN_GAP = 0.022
            for j in range(1, len(ends)):
                if ends[j][1] - ends[j - 1][1] < MIN_GAP:
                    ends[j][1] = ends[j - 1][1] + MIN_GAP
            for val, ypos, col in ends:
                ax.annotate(f"{val:.3f}", xy=(ks[-1], ypos),
                            xytext=(7, 0), textcoords="offset points",
                            color=col, fontsize=8.5, va="center",
                            annotation_clip=False)
            ax.set_xscale("log"); ax.set_xticks(ks)
            ax.set_xlim(ks[0] * 0.85, ks[-1] * 1.45)
            ax.set_xticklabels([str(k) for k in ks])
            ax.set_ylim(0.5, 0.95)
            ax.grid(True, color=C["grid"], lw=0.8); ax.set_axisbelow(True)
            ax.set_title(f"{MODEL[mdl]}   ·   {CORP[cor]}", loc="left",
                         color=C["text"], fontsize=10, fontweight="bold", pad=7)
            if r == 1: ax.set_xlabel("k  (latents given to the probe)")
            if c == 0: ax.set_ylabel("boundary-probe AUC")
    h, l = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(h, l, frameon=False, loc="upper left", ncol=3,
                     bbox_to_anchor=(0.072, 0.875), fontsize=9,
                     handlelength=1.8, columnspacing=2.2)
    for t in leg.get_texts(): t.set_color(C["text2"])
    fig.suptitle('Is "word boundary" a sparse SAE concept?  k-sparse probe, layer 24',
                 x=0.075, ha="left", color=C["text"], fontsize=13, fontweight="bold")
    fig.text(0.075, 0.905,
             "Gold boundaries from the benchmark's own segmentation. Both corpora pose the "
             "identical task — find hidden word breaks — and differ only in vocabulary.  "
             "AUC 0.5 = chance.",
             ha="left", color=C["muted"], fontsize=8.5)
    fig.savefig(out, dpi=170); plt.close(fig); print("wrote", out)

def table(d, out):
    L = ["# Word-boundary sparse probe (layer 24)\n",
         "AUC for a logistic probe on the k most separating SAE latents; "
         "`dense` uses all 4,096 residual dimensions.\n",
         "| backbone | corpus | representation | k=1 | k=5 | k=20 | dense |",
         "|---|---|---|---|---|---|---|"]
    for mdl in ["instruct", "lora"]:
        for cor in ["dksplit", "pseudo"]:
            dense = next((x["auc_dense"] for x in d["rows"]
                          if x["model"] == mdl and x["corpus"] == cor and x["auc_dense"]), None)
            for arm in ARMS:
                row = next((x for x in d["rows"] if x["model"] == mdl
                            and x["corpus"] == cor and x["sae"] == arm), None)
                if not row: continue
                L.append(f"| {MODEL[mdl]} | {CORP[cor]} | `{arm}` | "
                         f"{row['k1']:.4f} | {row['k5']:.4f} | {row['k20']:.4f} | "
                         f"{dense:.4f} |")
    Path(out).write_text("\n".join(L) + "\n"); print("wrote", out)

if __name__ == "__main__":
    d = json.load(open("dksplit/out/boundary_probe.json"))
    build(d, "light", "dksplit/out/boundary_probe.png")
    build(d, "dark", "dksplit/out/boundary_probe_dark.png")
    table(d, "dksplit/out/boundary_probe.md")

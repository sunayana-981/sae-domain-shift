# Masked SAE fine-tuning with Qwen-Scope SAEs on Qwen3.5

Language-model recreation of the masked SAE fine-tuning experiment in
[`patchsae/`](../patchsae): freeze the SAE units that carry a reference
distribution's concepts, let only the remaining capacity adapt to a new
domain, and measure whether that buys adaptation without forgetting.

The vision version fine-tunes an ImageNet-trained G-SAE on LoRA-CLIP
activations over 11 image datasets. This version fine-tunes the released
**Qwen-Scope** SAE on **Qwen3.5-2B-Base** residual-stream activations over
**three OOD text corpora**.

## What maps to what

| patchsae (vision) | here (language) |
|---|---|
| CLIP ViT-B/16, `block_layer -2` (layer 10 of 12) | Qwen3.5-2B-Base, `--layer -4` (block 20 of 24) |
| G-SAE trained on ImageNet (`data/sae_weight/base/out.pt`) | `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50`, `layer20.sae.pt` |
| ReLU/Gated SAE, expansion 64, L1 8e-5 | Top-K SAE, k=50, expansion 16, no L1 term (L0 is fixed by k) |
| ImageNet = reference distribution for activity | FineWeb-Edu = reference distribution for activity (a *proxy* — see below) |
| 11 target image datasets | 3 OOD corpora: biomedical, code, legal (Pile subsets) |
| per-dataset LoRA-CLIP backbone | per-corpus LoRA Qwen3.5 backbone |
| `zeroshot_acc` with the SAE spliced in | **generative task accuracy** with the SAE spliced in (`eval_tasks.py`), plus `delta_ce` / `ce_recovered` / `kl` |
| `l0`, `dead_frac`, `recon_cosine`, `fve` | identical definitions |
| `imagenet_anchor_eval` (retention) | every SAE also scored on the reference corpus |

## Arms

| condition | init | trained on | frozen |
|---|---|---|---|
| `none` | — | — | — (no SAE; gives `ce_clean` and the ablation floors) |
| `gsae` | Qwen-Scope release | nothing | everything |
| `ftsae` | Qwen-Scope release | target corpus | nothing |
| `scratchsae` | random | target corpus | nothing |
| `masked` | Qwen-Scope release | target corpus | top 20% most reference-active units |
| `masked_bottom` | Qwen-Scope release | target corpus | bottom 20% (ablation: same budget, wrong units) |
| `masked_random` | Qwen-Scope release | target corpus | random 20% (ablation: does unit identity matter?) |

`ftsae` and `scratchsae` are the controls that isolate what masking buys: same
corpus, backbone, layer, token budget, learning rate and seed as `masked`,
differing only in the protected mask and the initialisation.

## Corpora

| name | source | role | paired benchmark |
|---|---|---|---|
| `fineweb` | `HuggingFaceFW/fineweb-edu` (sample-10BT) | reference | — |
| `pubmed_central` | The Pile / PubMed Central (`ArmelR/the-pile-splitted`) | OOD, biomedical | PubMedQA |
| `github` | The Pile / GitHub | OOD, source code | HumanEval pass@1 |
| `freelaw` | The Pile / FreeLaw (US court opinions) | OOD, legal | LegalBench `definition_extraction` |
| `lean` | `bigcode/the-stack-smol-xl` (`data/lean`) | OOD, formal proofs | — |
| `amharic` | `HuggingFaceFW/fineweb-2` (`amh_Ethi`) | OOD, Amharic (Ethiopic) | — |
| — | — | general | GSM8K (8-shot CoT) |

**Why these.** The Pile's per-domain components are the standard corpora for
measuring LM domain shift ([Gao et al. 2021](https://arxiv.org/abs/2101.00027)),
and each one here is paired with a benchmark from its own domain so the same
shift is measured twice — once as reconstruction fidelity, once as task
accuracy. GSM8K belongs to no domain and acts as the task-level counterpart of
the reference-corpus retention test: a domain-adapted SAE should not damage
reasoning it was never meant to touch.

**`lean` and `amharic` are shifts within a capability, not benchmark suites.**
Capability-cliff benchmarks (OSWorld, VITA-Bench, MedXpertQA-MM, Hypersim …)
measure where a model fails, but they are a few hundred eval items in an
interactive or multimodal setting: there is nothing there to LoRA fine-tune on,
and this pipeline is text-only. These two are corpora of millions of tokens
whose token distribution is far from general web text — Lean is a
dependently-typed proof language rather than imperative source, so it is not
interchangeable with `github`, and Amharic is non-Latin script.

Measured by how much a LoRA adaptation can gain (2M tokens, identical
hyperparameters), they are the two most out-of-distribution corpora of the
five:

| corpus | base CE | after LoRA | gain |
|---|---|---|---|
| `lean` | 1.661 | 1.328 | **+0.333** |
| `amharic` | 1.596 | 1.511 | +0.085 |
| `freelaw` | 2.155 | 2.081 | +0.074 |
| `pubmed_central` | 1.813 | 1.754 | +0.059 |
| `github` | 1.333 | 1.306 | +0.027 |

Note that base CE alone is a poor cross-script OOD measure — Amharic's 1.596
is *lower* than FreeLaw's 2.155 because byte-level tokens are individually
easy to predict. Adaptation gain is the more meaningful signal.

**An earlier version of this experiment used `TimSchopf/medical_abstracts`,
`code_search_net` and `open-web-math`, and they were the wrong choices.** The
medical set has only 11,550 documents (~3.15M tokens), which capped the token
budget for the whole experiment; `code_search_net` is a 2019 code-*search*
dataset rather than a code corpus; and open-web-math turned out not to be OOD
at all — the untouched SAE explained 0.768 of variance on it against 0.772 on
the reference corpus, so there was no forgetting for masking to prevent. Those
runs are kept under `out/run1_archive/`.

**The reference corpus is a proxy, and that matters.** Qwen-Scope's paper
([arXiv 2605.11887](https://arxiv.org/abs/2605.11887)) states its SAEs were
trained on undisclosed *in-house pretraining data*, so FineWeb-Edu is not the
distribution these SAEs were actually fitted to — it is a stand-in for general
web text. Every "retention" number below should be read as retention on general
web text, not as retention on the SAE's own training distribution. SAE Boost
([arXiv 2507.12990](https://arxiv.org/html/2507.12990)) uses FineWeb-Edu for the
identical purpose, so the choice is conventional, but it is still a proxy.

Each corpus is tokenised once into packed 512-token windows
(`out/token_cache/`) and every arm reads the same windows through the same
train/eval index split — 2.7M train + 300k eval tokens. All four caches build
in a single pass over their source, so no arm trains on duplicated text.
Packing rather than truncating-and-padding means every cached token is real.

## Two data bugs that were found by checking, not by looking at the code

Both were caught because a number looked wrong, not because anything crashed.

**The GitHub corpus was 15% degenerate.** The base model scored cross-entropy
0.178 (perplexity 1.19) on the GitHub eval split — implausibly low for source
code. The cause was machine-generated data files in the Pile's GitHub subset;
one is literally the number `26` on every line for thousands of tokens, which a
language model predicts at near-zero loss. 87 of 585 eval windows were
degenerate. `build_token_cache` now drops such documents, which removed them.
Left in, it would have shrunk every arm's measured delta on this domain for
reasons unrelated to SAEs.

**And the first version of that filter was itself wrong.** It counted unique
*tokens*, which made it tokenizer-dependent. Qwen's tokenizer has poor Ethiopic
coverage, so Amharic prose becomes long runs of byte-level tokens — and the
filter rejected **41% of the Amharic corpus**, including ordinary Wikipedia
articles, while preferentially dropping longer documents (median 5,302 tokens
dropped versus 1,346 kept). `qsae.data.is_degenerate` now uses zlib compression
ratio on the raw text, which is a property of the text rather than of the
tokenizer and separates cleanly: degenerate `26` file 0.057, pubmed_central
0.188, lean 0.337, amharic 0.455, fineweb/freelaw 0.511, so a 0.10 cut catches
the data files and leaves every real corpus alone. Under it, 0 of 150 sampled
Amharic documents are dropped and Amharic needs 860 documents rather than 2,240
for 3M tokens — the length bias is gone.

**The eval split was read contiguously.** Packing is sequential within a
document, so a contiguous slice of the eval range can be one or two long files
— and scripts that consume only part of the split (the LoRA eval reads 80 of
585 windows) were scoring the model on a couple of documents rather than on the
corpus. On the GitHub cache the contiguous head gave mean CE 0.29 against 0.71
for a scattered sample of the *same* split. Both splits are now read in a
seeded permutation, so any prefix is representative and every arm still sees an
identical sequence.

## Two things that are verified rather than assumed

**The encoder convention.** Qwen-Scope's card gives `W_enc/b_enc/W_dec/b_dec`
but not whether the encoder reads `x` or the centred `x - b_dec`.
`scripts/probe_sae_convention.py` measures both on the reference corpus before
anything is trained:

| layer | variant | FVE | cosine | L0 |
|---|---|---|---|---|
| 20 | `raw` | **0.772** | **0.893** | 50.0 |
| 20 | `centered` | 0.572 | 0.798 | 50.0 |
| 22 | `raw` | 0.738 | 0.872 | 50.0 |
| 22 | `centered` | 0.642 | 0.822 | 50.0 |

So `--encode_variant raw`, and layer 20 — which is also the same ~85% relative
depth as patchsae's `block_layer -2` on a 12-block ViT. The probe also confirms
the released decoder columns are exactly unit-norm (mean 1.0000, std 0.0000),
which is why training renormalises them after every step.

**That protection is actually exact.** The trainer keeps immutable copies of
the protected slices and re-asserts bit-equality after the first step and at
the end of training; a nonzero delta raises rather than warns. Gradient masking
alone would not be enough, because decoder renormalisation happens outside the
optimizer and would otherwise move protected columns.

## One deliberate departure from the vision code

patchsae's fast path runs the protected slice under `no_grad` so backward cost
scales with the free-unit count. That is only valid for ReLU SAEs. Qwen-Scope
ships **Top-K** SAEs, and top-k selection is *global* across all 32,768 units
for a token — taking top-k within the free and protected slices separately
would select a different active set than the dense forward, i.e. a different
algorithm, not a faster one. patchsae's own trainer falls back to the dense
forward + gradient-zeroing path for `topk` for exactly this reason, and that is
the only path implemented here. Dead units are revived with Gao et al.'s
auxiliary reconstruction loss (`aux_k`) instead of ghost gradients, which are
ReLU-specific.

LoRA rank is raised from CLIP-LoRA's `r=2` to `r=8` because the backbone is 2B
parameters rather than 86M and `r=2` barely moves the residual stream. Adapters
go on the q/k/v projections at every layer, matching CLIP-LoRA's
`--params q k v`; on Qwen3.5's hybrid stack that means `q_proj/k_proj/v_proj`
in full-attention layers and the fused `in_proj_qkv`/`in_proj_z` in
linear-attention layers.

## Results

Regenerated by stage 7 into [`out/results/SUMMARY.md`](out/results/SUMMARY.md),
from [`out/results/results.csv`](out/results/results.csv) (fidelity) and
[`out/results/task_results.csv`](out/results/task_results.csv) (accuracy).

The superseded first run — different corpora, no task-accuracy metric — is in
`out/run1_archive/`. Its headline finding was that freezing the top 20%
reference-active units kept 71–86% of an unmasked fine-tune's adaptation for
14–18% of its reference-domain loss, and that the effect came from *which*
units were frozen rather than how many: freezing the least-active 20% instead
reproduced the unmasked fine-tune almost exactly. Whether that survives on the
Pile corpora, and whether it shows up in task accuracy rather than only in
reconstruction fidelity, is what this run tests.

## Running it

```bash
./run_experiment.sh          # all stages, each idempotent
./run_experiment.sh 5 6 7    # just re-eval and re-summarise
```

Needs one 24GB GPU. The environment is `/path/to/envs/qwenscope`
(transformers 5.17 — `qwen3_5` does not exist in 4.57.x).

| stage | what | ~time on one RTX 3090 |
|---|---|---|
| 0 | token cache (network-bound) | 10 min |
| 1 | encoder-convention probe | 2 min |
| 2 | 3x LoRA, 2M tokens each | 30 min |
| 3 | 9x SAE arms, 2M tokens each | 45 min |
| 4 | 6x ablation arms | 30 min |
| 5 | eval matrix (fidelity) | 30 min |
| 6 | task accuracy, SAE spliced in | ~3 h |
| 7 | summary tables | seconds |

## Running it on multiple GPUs

`run_experiment_multigpu.sh` treats each stage as a list of independent
single-GPU work units — one LoRA per corpus, one SAE arm per (corpus,
condition), one eval shard per corpus — and feeds them through a slot pool that
pins each unit to its own `CUDA_VISIBLE_DEVICES`. Stage boundaries are barriers,
since stage 3 needs stage 2's adapters. Stages 5 and 6 write into per-shard
directories and are merged in stage 7. `grad_accum` holds tokens-per-step fixed,
so wall time and memory change but the optimisation being run does not.
Resubmitting after a timeout resumes: every stage skips work whose checkpoint or
manifest is already on disk.

## Layout

```
qsae/
  sae.py              Qwen-Scope Top-K SAE: loading + trainable module
  data.py             corpus specs, token-cache builder, train/eval window split
  activations.py      resid_post capture, SAE/zero/mean interventions
  masked_trainer.py   Step A (activity -> mask) and Step B (masked fine-tune)
  tasks.py            generative benchmarks + greedy decoding + scoring
scripts/
  build_token_cache.py   stage 0
  probe_sae_convention.py stage 1
  lora_finetune.py       stage 2
  train_sae.py           stages 3-4 (one arm per invocation)
  eval_matrix.py         stage 5 (fidelity)
  eval_tasks.py          stage 6 (task accuracy)
  summarize.py           stage 7
out/
  token_cache/        packed token windows + manifests
  probe/              encoder-convention measurements
  lora/<corpus>/      adapter + CE before/after
  sae/<corpus>_lora_<label>/  checkpoint, manifest, loss history
  results/results.csv        reconstruction fidelity, one row per cell
  results/task_results.csv   task accuracy, one row per cell
  results/task_samples.json  sample generations per cell
  results/SUMMARY.md         tables
  run1_archive/              the superseded first run
```

## Metrics

Reconstruction metrics are computed with **no hook registered** — activations
are captured clean and pushed through the SAE separately, so nothing the SAE
does can feed back into what it is measured on.

- `fve` — `1 - E‖x-x̂‖² / E‖x-b_dec‖²`
- `recon_cosine`, `l0`, `dead_frac` — as in patchsae's `compute_sae_quality`
- `ce_clean` / `ce_sae` — LM cross-entropy without / with the reconstruction
  spliced into the residual stream
- `ce_zero` / `ce_mean` — ablation floors (zero, and the broadcast batch-mean
  activation). All interventions leave position 0 untouched, because the SAE is
  never trained on position 0 and a floor that clobbered it would not be
  comparable.
- `delta_ce` = `ce_sae - ce_clean`; `ce_recovered` = `1 - delta_ce/(ce_zero - ce_clean)`
- `kl` — mean `KL(clean ‖ sae)` over next-token distributions, on every 4th
  position (the 248k-token vocabulary makes all-position logits too large to
  hold twice)

`delta_ce` and `kl` are the sharp metrics; `ce_recovered` sits near 1 for every
working SAE because zero-ablating a mid-stack residual is catastrophic
(CE ≈ 15 vs ≈ 2), so read it alongside `ce_recovered_mean`.

### Task accuracy (`scripts/eval_tasks.py`)

Fidelity metrics alone are not enough, and this is the metric patchsae actually
leads with. `zeroshot_acc` there is task accuracy measured while the SAE's
reconstruction replaces the model's activations; the analogue here is benchmark
accuracy under the same intervention, decoded greedily so a condition's score is
deterministic and differences between arms come from the intervention rather
than from sampling.

This matters because the fidelity metrics are exactly the set SAE Boost reports
(explained variance, LLM cross-entropy, L0) and exactly what
[SAEBench](https://arxiv.org/html/2503.09532v4) calls Explained Variance and
Loss Recovered — but SAEBench's own argument is that a single fidelity number
can be traded off against others, and SAEBench contains *no* metric that scores
benchmark accuracy with the SAE spliced into the forward pass. Domain-SAE work
that does report downstream accuracy
([arXiv 2508.09363](https://arxiv.org/pdf/2508.09363)) reports it alongside FVE,
not instead of it, which is what happens here.

One subtlety the implementation depends on: during cached generation every
decode step's hidden state is `[B, 1, d]` for a single *later* position. The
position-0 exemption that reconstruction metrics need must therefore be
conditioned on `seq_len > 1`, or the intervention silently becomes a no-op for
the entire continuation — the exact regime these benchmarks measure.

HumanEval's pass@1 is defined by running the generated function against the
benchmark's unit tests, so this executes model-written code; each candidate runs
in its own short-lived subprocess with a hard timeout.

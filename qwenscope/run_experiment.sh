#!/usr/bin/env bash
# =============================================================================
# run_experiment.sh
#
# Masked fine-tune of the released Qwen-Scope SAE (Qwen/SAE-Res-Qwen3.5-2B-
# Base-W32K-L0_50, resid_post) on three OOD corpora, with the same arms and
# controls as patchsae/run_masked_finetune_all_datasets.sh.
#
# Stages (each idempotent -- re-running skips finished work):
#   0  token cache        one packed-token file per corpus, shared by all arms
#   1  probe              verify the SAE's encoder convention before training
#   2  LoRA               domain-adapt Qwen3.5-2B per corpus (the `lora` backbone)
#   3  SAE arms           masked / ftsae / scratchsae on each corpus
#   4  ablation           protect_mode = bottom | random at the same budget
#   5  eval matrix        reconstruction fidelity: {corpus} x {base,lora} x
#                         {condition} x {target,reference}
#   6  task accuracy      generative benchmarks with the SAE spliced in --
#                         the analogue of patchsae's `zeroshot_acc`
#   7  summary            tables in out/results/SUMMARY.md
#
# Usage:  ./run_experiment.sh            # everything
#         ./run_experiment.sh 5 6        # only the listed stages
# =============================================================================

set -uo pipefail

# This GPU is shared with other jobs; expandable segments keeps a fragmented
# allocator from failing on small allocations when memory is tight.
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # older torch spelling
PY=/path/to/envs/qwenscope/bin/python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL="Qwen/Qwen3.5-2B-Base"
SAE_REPO="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"

# --- Fixed hyperparameters (never vary across conditions) --------------------
LAYER=-4                      # resid_post of block 20 of 24; best-reconstructing
                              # layer in the probe, and the same ~85% relative
                              # depth as patchsae's --block_layer -2 on ViT-B/16
ENCODE_VARIANT=raw            # verified by stage 1, not assumed
PROTECT_FRAC=0.2              # patchsae's PROTECT_FRAC
PROTECT_MODE=top
ACTIVITY_CORPUS=fineweb       # reference distribution (patchsae: imagenet)
ACTIVITY_N_BATCHES=50
SAE_TOKENS=2000000            # per arm, identical for every arm
LORA_TOKENS=2000000
LR=4e-4                       # patchsae's LR
SEED=42
GSM8K_N=100                   # general-capability retention check size

CORPORA=(pubmed_central github freelaw)
CONDITIONS=(masked ftsae scratchsae)
LOGS="out/logs"
mkdir -p "$LOGS" out/results

STAGES=("$@")
want() { [ ${#STAGES[@]} -eq 0 ] && return 0; for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
hdr() { echo; echo "############ $1  ($(date '+%F %T'))"; echo; }

# ── 0: token cache ──────────────────────────────────────────────────────────
if want 0; then
  hdr "STAGE 0 — token cache"
  $PY -u scripts/build_token_cache.py 2>&1 | tee -a "$LOGS/stage0_cache.log"
fi

# ── 1: encoder-convention probe ─────────────────────────────────────────────
if want 1; then
  hdr "STAGE 1 — SAE convention probe"
  if [ -f out/probe/sae_convention.json ]; then
    echo "[skip] out/probe/sae_convention.json exists"
  else
    $PY -u scripts/probe_sae_convention.py --sae_repo "$SAE_REPO" \
        --layers 20 22 2>&1 | tee "$LOGS/probe.log"
  fi
fi

# ── 2: per-corpus LoRA ──────────────────────────────────────────────────────
if want 2; then
  hdr "STAGE 2 — LoRA domain adaptation"
  for c in "${CORPORA[@]}"; do
    if [ -d "out/lora/$c/adapter" ]; then echo "[skip] lora/$c"; continue; fi
    echo "--- lora: $c"
    $PY -u scripts/lora_finetune.py --corpus "$c" --model "$MODEL" \
        --total_tokens $LORA_TOKENS --seed $SEED \
        > "$LOGS/lora_$c.log" 2>&1
    tail -2 "$LOGS/lora_$c.log"
  done
fi

# ── 3: the three SAE arms ───────────────────────────────────────────────────
run_arm() {   # corpus condition label [extra args...]
  local c=$1 cond=$2 label=$3; shift 3
  local dir="out/sae/${c}_lora_${label}"
  if [ -f "$dir/manifest.json" ]; then echo "[skip] $dir"; return; fi
  echo "--- sae: $c / $label"
  $PY -u scripts/train_sae.py \
      --condition "$cond" --label "$label" --corpus "$c" \
      --backbone_type lora --model "$MODEL" --sae_repo "$SAE_REPO" \
      --encode_variant "$ENCODE_VARIANT" --layer $LAYER \
      --protect_frac $PROTECT_FRAC --protect_mode $PROTECT_MODE \
      --activity_corpus "$ACTIVITY_CORPUS" \
      --activity_n_batches $ACTIVITY_N_BATCHES \
      --lr $LR --total_training_tokens $SAE_TOKENS --seed $SEED "$@" \
      > "$LOGS/sae_${c}_${label}.log" 2>&1
  tail -2 "$LOGS/sae_${c}_${label}.log"
}

if want 3; then
  hdr "STAGE 3 — SAE arms (masked / ftsae / scratchsae)"
  for c in "${CORPORA[@]}"; do
    for cond in "${CONDITIONS[@]}"; do run_arm "$c" "$cond" "$cond"; done
  done
fi

# ── 4: unit-selection ablation ──────────────────────────────────────────────
if want 4; then
  hdr "STAGE 4 — protect_mode ablation (same 20% budget, different units)"
  for c in "${CORPORA[@]}"; do
    run_arm "$c" masked masked_bottom --protect_mode bottom
    run_arm "$c" masked masked_random --protect_mode random
  done
fi

# ── 5: eval matrix ──────────────────────────────────────────────────────────
if want 5; then
  hdr "STAGE 5 — eval matrix"
  $PY -u scripts/eval_matrix.py --corpora "${CORPORA[@]}" --model "$MODEL" \
      --sae_repo "$SAE_REPO" --encode_variant "$ENCODE_VARIANT" \
      --layer $LAYER --seed $SEED 2>&1 | tee -a "$LOGS/eval_matrix.log"
fi

# ── 6: generative task accuracy with the SAE spliced in ─────────────────────
if want 6; then
  hdr "STAGE 6 — task accuracy (SAE spliced into the residual stream)"
  $PY -u scripts/eval_tasks.py --corpora "${CORPORA[@]}" --model "$MODEL" \
      --sae_repo "$SAE_REPO" --encode_variant "$ENCODE_VARIANT" \
      --layer $LAYER --seed $SEED --n_general $GSM8K_N 2>&1 \
      | tee -a "$LOGS/eval_tasks.log"
fi

# ── 7: summary ──────────────────────────────────────────────────────────────
if want 7; then
  hdr "STAGE 7 — summary"
  $PY -u scripts/summarize.py > "$LOGS/summarize.log" 2>&1
  tail -40 "$LOGS/summarize.log"
  echo
  echo "Fidelity: out/results/results.csv"
  echo "Accuracy: out/results/task_results.csv"
  echo "Tables:   out/results/SUMMARY.md"
fi

hdr "DONE"

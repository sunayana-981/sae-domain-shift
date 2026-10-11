#!/usr/bin/env bash
# =============================================================================
# run_experiment_multigpu.sh
#
# Same experiment as run_experiment.sh, spread over N GPUs (default 2).
#
# Every stage is a list of independent single-GPU work units fed through a slot
# pool, so the GPUs stay busy without any multi-GPU code in the scripts
# themselves -- each unit gets its own CUDA_VISIBLE_DEVICES and behaves exactly
# as it does in the serial driver. Stage boundaries are barriers: stage 3 needs
# stage 2's adapters, stage 5/6 need stage 3/4's checkpoints.
#
# Work units:
#   2  one LoRA per corpus                              (3 units)
#   3  one SAE arm per (corpus, condition)              (9 units)
#   4  one ablation arm per (corpus, protect_mode)      (6 units)
#   5  fidelity eval, sharded by corpus                 (3 units)
#   6  task accuracy, sharded by corpus                 (3 units)
#   7  merge shards + summarise                         (serial)
#
# Stage 5 and 6 write into per-shard directories and are merged in stage 7,
# because two processes appending to one results.csv would clobber each other.
#
# Usage:  NGPU=2 ./run_experiment_multigpu.sh          # all stages
#         NGPU=2 ./run_experiment_multigpu.sh 6 7      # only these
# =============================================================================

set -uo pipefail

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${PY:-/path/to/envs/qwenscope/bin/python}"
NGPU="${NGPU:-2}"

MODEL="Qwen/Qwen3.5-2B-Base"
SAE_REPO="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"

LAYER=-4
ENCODE_VARIANT=raw
PROTECT_FRAC=0.2
ACTIVITY_CORPUS=fineweb
ACTIVITY_N_BATCHES=50
SAE_TOKENS=2000000
LORA_TOKENS=2000000
LR=4e-4
SEED=42
GSM8K_N=100

# Dedicated GPUs: no gradient checkpointing, bigger micro-batches. These change
# wall time and memory only -- tokens-per-step and every optimisation detail are
# held fixed against the serial driver.
LORA_EXTRA="--no_grad_checkpoint --seqs_per_step 4 --grad_accum 2"
SAE_SEQS=8
EVAL_SEQS=8
EVAL_LM_SEQS=2
TASK_BATCH=8

# Overridable so a subset (e.g. two newly added corpora) can be run without
# touching the others: CORPORA="lean amharic" ./run_experiment_multigpu.sh 2
read -r -a CORPORA <<< "${CORPORA:-pubmed_central github freelaw}"
CONDITIONS=(masked ftsae scratchsae)
LOGS="out/logs"
mkdir -p "$LOGS" out/results

STAGES=("$@")
want() { [ ${#STAGES[@]} -eq 0 ] && return 0; for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
hdr() { echo; echo "############ $1  ($(date '+%F %T'))"; echo; }

# ── GPU slot pool ───────────────────────────────────────────────────────────
declare -a SLOT_PID SLOT_NAME
FAILED=()
for ((g = 0; g < NGPU; g++)); do SLOT_PID[$g]=0; SLOT_NAME[$g]=""; done

# Sets FREE_SLOT. Must run in the current shell -- a $(...) subshell would
# discard the SLOT_PID bookkeeping it does.
reap() {
    while true; do
        for ((g = 0; g < NGPU; g++)); do
            if [ "${SLOT_PID[$g]}" = "0" ]; then FREE_SLOT=$g; return; fi
            if ! kill -0 "${SLOT_PID[$g]}" 2>/dev/null; then
                wait "${SLOT_PID[$g]}"; local rc=$?
                if [ $rc -ne 0 ]; then
                    echo "[FAIL] ${SLOT_NAME[$g]} (exit $rc)"
                    FAILED+=("${SLOT_NAME[$g]}")
                else
                    echo "[ok]   ${SLOT_NAME[$g]}  ($(date +%T))"
                fi
                SLOT_PID[$g]=0; FREE_SLOT=$g; return
            fi
        done
        sleep 5
    done
}

launch() {   # name logfile command...
    local name=$1 log=$2; shift 2
    reap
    local g=$FREE_SLOT
    echo "[gpu$g] start $name  ($(date +%T))"
    CUDA_VISIBLE_DEVICES=$g "$@" > "$log" 2>&1 &
    SLOT_PID[$g]=$!
    SLOT_NAME[$g]=$name
}

drain() {
    for ((g = 0; g < NGPU; g++)); do
        if [ "${SLOT_PID[$g]}" != "0" ]; then
            wait "${SLOT_PID[$g]}"; local rc=$?
            if [ $rc -ne 0 ]; then
                echo "[FAIL] ${SLOT_NAME[$g]} (exit $rc)"; FAILED+=("${SLOT_NAME[$g]}")
            else
                echo "[ok]   ${SLOT_NAME[$g]}  ($(date +%T))"
            fi
            SLOT_PID[$g]=0
        fi
    done
}

echo "GPUs: $NGPU | python: $PY"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

# ── 0/1: token cache + probe (cheap, serial) ────────────────────────────────
if want 0; then
  hdr "STAGE 0 — token cache"
  $PY -u scripts/build_token_cache.py --tokens 3000000 2>&1 | tee -a "$LOGS/stage0_cache.log"
fi
if want 1; then
  hdr "STAGE 1 — SAE convention probe"
  if [ -f out/probe/sae_convention.json ]; then echo "[skip] probe exists"; else
    CUDA_VISIBLE_DEVICES=0 $PY -u scripts/probe_sae_convention.py \
        --sae_repo "$SAE_REPO" --layers 20 22 2>&1 | tee "$LOGS/probe.log"
  fi
fi

# ── 2: LoRA, one unit per corpus ────────────────────────────────────────────
if want 2; then
  hdr "STAGE 2 — LoRA domain adaptation ($NGPU-way)"
  for c in "${CORPORA[@]}"; do
    [ -d "out/lora/$c/adapter" ] && { echo "[skip] lora/$c"; continue; }
    launch "lora:$c" "$LOGS/lora_$c.log" \
        $PY -u scripts/lora_finetune.py --corpus "$c" --model "$MODEL" \
            --total_tokens $LORA_TOKENS --seed $SEED $LORA_EXTRA
  done
  drain
fi

# ── 3 & 4: SAE arms, one unit per (corpus, label) ───────────────────────────
arm() {   # corpus condition label [extra...]
  local c=$1 cond=$2 label=$3; shift 3
  [ -f "out/sae/${c}_lora_${label}/manifest.json" ] && { echo "[skip] ${c}/${label}"; return; }
  launch "sae:${c}/${label}" "$LOGS/sae_${c}_${label}.log" \
      $PY -u scripts/train_sae.py \
          --condition "$cond" --label "$label" --corpus "$c" \
          --backbone_type lora --model "$MODEL" --sae_repo "$SAE_REPO" \
          --encode_variant "$ENCODE_VARIANT" --layer $LAYER \
          --protect_frac $PROTECT_FRAC \
          --activity_corpus "$ACTIVITY_CORPUS" \
          --activity_n_batches $ACTIVITY_N_BATCHES \
          --seqs_per_forward $SAE_SEQS \
          --lr $LR --total_training_tokens $SAE_TOKENS --seed $SEED "$@"
}

if want 3; then
  hdr "STAGE 3 — SAE arms ($NGPU-way)"
  for c in "${CORPORA[@]}"; do
    for cond in "${CONDITIONS[@]}"; do arm "$c" "$cond" "$cond" --protect_mode top; done
  done
  drain
fi

if want 4; then
  hdr "STAGE 4 — protect_mode ablation ($NGPU-way)"
  for c in "${CORPORA[@]}"; do
    arm "$c" masked masked_bottom --protect_mode bottom
    arm "$c" masked masked_random --protect_mode random
  done
  drain
fi

# ── 5: fidelity eval, sharded by corpus ─────────────────────────────────────
if want 5; then
  hdr "STAGE 5 — eval matrix ($NGPU-way, sharded by corpus)"
  for c in "${CORPORA[@]}"; do
    launch "eval:$c" "$LOGS/eval_matrix_$c.log" \
        $PY -u scripts/eval_matrix.py --corpora "$c" --model "$MODEL" \
            --sae_repo "$SAE_REPO" --encode_variant "$ENCODE_VARIANT" \
            --layer $LAYER --seed $SEED \
            --seqs_per_forward $EVAL_SEQS --lm_seqs_per_forward $EVAL_LM_SEQS \
            --out_dir "out/results/shard_$c"
  done
  drain
fi

# ── 6: task accuracy, sharded by corpus ─────────────────────────────────────
# github carries HumanEval and is by far the longest shard, so it goes first --
# with 3 shards on 2 GPUs the last one to start determines the stage's wall time.
if want 6; then
  hdr "STAGE 6 — task accuracy ($NGPU-way, sharded by corpus)"
  for c in github pubmed_central freelaw; do
    printf '%s\n' "${CORPORA[@]}" | grep -qx "$c" || continue
    launch "tasks:$c" "$LOGS/eval_tasks_$c.log" \
        $PY -u scripts/eval_tasks.py --corpora "$c" --model "$MODEL" \
            --sae_repo "$SAE_REPO" --encode_variant "$ENCODE_VARIANT" \
            --layer $LAYER --seed $SEED --n_general $GSM8K_N \
            --batch_size $TASK_BATCH \
            --out_dir "out/results/shard_$c"
  done
  drain
fi

# ── 7: merge + summarise ────────────────────────────────────────────────────
if want 7; then
  hdr "STAGE 7 — merge shards and summarise"
  $PY -u scripts/merge_shards.py --shard_root out/results --out_dir out/results \
      2>&1 | tee "$LOGS/merge.log"
  $PY -u scripts/summarize.py > "$LOGS/summarize.log" 2>&1
  tail -40 "$LOGS/summarize.log"
  echo
  echo "Fidelity: out/results/results.csv"
  echo "Accuracy: out/results/task_results.csv"
  echo "Tables:   out/results/SUMMARY.md"
fi

if [ ${#FAILED[@]} -gt 0 ]; then
  hdr "DONE WITH FAILURES: ${FAILED[*]}"
  exit 1
fi
hdr "DONE"

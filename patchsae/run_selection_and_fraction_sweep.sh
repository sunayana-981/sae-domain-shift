#!/usr/bin/env bash
# =============================================================================
# run_selection_and_fraction_sweep.sh
#
# Two controlled ablations over masked SAE fine-tuning, sharing one set of
# hyperparameters (seed / layer / token budget / lr / l1 / batch size) so every
# arm is directly comparable. All runs use --freeze_b_dec, which makes "frozen"
# exact: the SAE encodes (x - b_dec), so a trainable shared b_dec moves the
# protected units' pre-activations too.
#
#   EXP3  selection rule at a FIXED free budget (protect_frac=0.8, free=20%):
#           protect_mode=top     -> free set is the bottom 20% (original recipe)
#           protect_mode=bottom  -> free set is the top 20%
#           protect_mode=random  -> free set is a random 20%
#         Tests whether the identity of the free units matters or only how many.
#
#   EXP2  protection fraction at a fixed selection rule (protect_mode=top):
#           protect_frac in {0.0, 0.5, 0.8, 0.9, 0.95, 0.98}
#         0.0 is full fine-tuning; 0.98 leaves ~1k free units.
#
# Two datasets spanning domain distance: caltech101 (CLIP MMD^2 0.028, near)
# and pathmnist (0.423, far).
#
# Usage:
#   nohup ./run_selection_and_fraction_sweep.sh > out/logs/selection_sweep/driver.log 2>&1 &
# =============================================================================

set -uo pipefail

PROJECT_ROOT="/path/to/sae-domain-shift/patchsae"
LORA_ROOT="/path/to/sae-domain-shift/lora_weights/vitb16"
SAE_CHECKPOINT="${PROJECT_ROOT}/data/sae_weight/base/out.pt"
REGISTRY_PATH="${PROJECT_ROOT}/out/controlled/sae_registry.json"

declare -A LORA_PATHS
LORA_PATHS["caltech101"]="${LORA_ROOT}/caltech101/16shots/seed1/lora_weights.pt"
LORA_PATHS["pathmnist"]="${LORA_ROOT}/medmnist/16shots/seed1/lora_weights.pt"

declare -A TRAIN_KEY
TRAIN_KEY["caltech101"]="caltech101"
TRAIN_KEY["pathmnist"]="medmnist"

DATASETS=(caltech101 pathmnist)

# --- Fixed hyperparameters (identical to run_masked_finetune_protection_sweep.sh) ---
BLOCK_LAYER=-2
EXPANSION_FACTOR=64
L1_COEFFICIENT=0.00008
LR=0.0004
BATCH_SIZE=16
LR_WARM_UP_STEPS=500
CLIP_DIM=768
MODEL_NAME="openai/clip-vit-base-patch16"
SEED=42
TOTAL_TOKENS=100000
ACTIVITY_N_BATCHES=50
ACTIVITY_DATASET="imagenet"
DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=0

cd "$PROJECT_ROOT"
mkdir -p out/logs/selection_sweep "$(dirname "$REGISTRY_PATH")"

register_sae() {
    local dataset=$1 ckpt=$2 condition=$3
    python3 - "$REGISTRY_PATH" "$dataset" "$ckpt" "$TOTAL_TOKENS" "$BLOCK_LAYER" "$condition" <<'PYEOF'
import json, os, sys, time
registry_path, dataset, ckpt, tokens, layer, condition = sys.argv[1:7]
os.makedirs(os.path.dirname(registry_path), exist_ok=True)
records = json.load(open(registry_path)) if os.path.exists(registry_path) else []
records = [r for r in records if not (r["dataset"] == dataset and r["condition"] == condition)]
records.append({"dataset": dataset, "vit_type": "lora", "condition": condition,
                "checkpoint_path": ckpt, "tokens": int(tokens), "layer": int(layer),
                "registered_at": time.strftime("%Y-%m-%d %H:%M:%S")})
json.dump(records, open(registry_path, "w"), indent=2)
print(f"[REGISTRY] {dataset}/{condition} -> {ckpt}")
PYEOF
}

run_one() {
    local dataset=$1 frac=$2 mode=$3 condition=$4
    local ckpt_dir="${PROJECT_ROOT}/out/checkpoints/selection_sweep/${condition}/${dataset}"
    local log="${PROJECT_ROOT}/out/logs/selection_sweep/${condition}_${dataset}.log"
    mkdir -p "$ckpt_dir"

    shopt -s nullglob
    local existing=("${ckpt_dir}"/*/final_sparse_autoencoder_*/*_${BLOCK_LAYER}_resid_*.pt)
    shopt -u nullglob
    if [ ${#existing[@]} -gt 0 ]; then
        echo "[SKIP] ${dataset}/${condition} -> ${existing[-1]}"
        register_sae "$dataset" "${existing[-1]}" "$condition"
        return 0
    fi

    echo ""
    echo "=============================================================="
    echo "RUN ${dataset} | protect_frac=${frac} | protect_mode=${mode} | ${condition}"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================="

    local start; start=$(date +%s)
    local start_iso; start_iso=$(date -d "@${start}" +"%Y-%m-%d %H:%M:%S")

    python3 tasks/train_sae_masked_finetune.py \
        --sae_checkpoint_path "${SAE_CHECKPOINT}" \
        --lora_checkpoint_path "${LORA_PATHS[$dataset]}" \
        --model_name "${MODEL_NAME}" \
        --clip_dim ${CLIP_DIM} \
        --block_layer ${BLOCK_LAYER} \
        --dataset "${TRAIN_KEY[$dataset]}" \
        --protect_frac ${frac} \
        --protect_mode ${mode} \
        --freeze_b_dec \
        --activity_n_batches ${ACTIVITY_N_BATCHES} \
        --activity_dataset "${ACTIVITY_DATASET}" \
        --expansion_factor ${EXPANSION_FACTOR} \
        --l1_coefficient ${L1_COEFFICIENT} \
        --lr ${LR} \
        --batch_size ${BATCH_SIZE} \
        --lr_warm_up_steps ${LR_WARM_UP_STEPS} \
        --total_training_tokens ${TOTAL_TOKENS} \
        --use_ghost_grads \
        --checkpoint_path "${ckpt_dir}" \
        --n_checkpoints 1 \
        --seed ${SEED} \
        --device ${DEVICE} \
        --run_name "${condition}_${dataset}" \
        > "$log" 2>&1
    local rc=$?

    local mins=$(( ($(date +%s) - start) / 60 ))
    if [ $rc -eq 0 ]; then
        local new_ckpt
        new_ckpt=$(find "${ckpt_dir}" -newermt "${start_iso}" -path "*/final_sparse_autoencoder_*" \
                   -name "*_${BLOCK_LAYER}_resid_*.pt" 2>/dev/null | sort | tail -1)
        if [ -n "$new_ckpt" ]; then
            echo "[OK] ${dataset}/${condition} in ${mins}m -> ${new_ckpt}"
            register_sae "$dataset" "$new_ckpt" "$condition"
            SUCCEEDED+=("${dataset}/${condition}")
        else
            echo "[WARN] ${dataset}/${condition} exited 0 but produced no checkpoint (see $log)"
            FAILED+=("${dataset}/${condition}")
        fi
    else
        echo "[FAIL] ${dataset}/${condition} exit ${rc} after ${mins}m (see $log)"
        FAILED+=("${dataset}/${condition}")
    fi
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}

SUCCEEDED=()
FAILED=()
OVERALL_START=$(date +%s)

# ---- EXP3: selection rule at fixed free budget (20% free) ----
for dataset in "${DATASETS[@]}"; do
    for mode in top bottom random; do
        run_one "$dataset" 0.8 "$mode" "sel_p80_${mode}"
    done
done

# ---- EXP2: protection fraction, selection rule fixed to top ----
for dataset in "${DATASETS[@]}"; do
    for frac in 0.0 0.5 0.9 0.95 0.98; do
        tag="p$(awk -v f="$frac" 'BEGIN{printf "%.0f", f*100}')"
        run_one "$dataset" "$frac" top "frac_${tag}_top"
    done
done

echo ""
echo "=============================================================="
echo "DONE in $(( ($(date +%s) - OVERALL_START) / 60 ))m — ${#SUCCEEDED[@]} ok, ${#FAILED[@]} failed"
for r in "${SUCCEEDED[@]}"; do echo "  ok   $r"; done
for r in "${FAILED[@]}";    do echo "  FAIL $r"; done
echo "Registry: ${REGISTRY_PATH}"

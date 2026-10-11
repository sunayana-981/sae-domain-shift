#!/usr/bin/env bash
# =============================================================================
# run_eurosat_p8090_multiseed.sh
#
# Confirms whether masked_p90 (the current best-trade-off masked FT model for
# eurosat: +7.40 own-domain, only -2.06 ImageNet at seed 42) holds up across
# seeds, the same way seed 42's masked-20% number (-16.52) turned out to be
# the lucky outlier of a true -30.48 mean. Adds seeds 123/456 for BOTH
# masked_p80 and masked_p90 so the comparison stays matched.
#
# Registers under "masked_p80_s<seed>" / "masked_p90_s<seed>".
#
# Usage:
#   nohup ./run_eurosat_p8090_multiseed.sh > out/logs/eurosat_p8090_multiseed/driver.log 2>&1 &
# =============================================================================

set -uo pipefail

PROJECT_ROOT="/path/to/sae-domain-shift/patchsae"
LORA_PATH="/path/to/sae-domain-shift/lora_weights/vitb16/eurosat/16shots/seed1/lora_weights.pt"
SAE_CHECKPOINT="${PROJECT_ROOT}/data/sae_weight/base/out.pt"
REGISTRY_PATH="${PROJECT_ROOT}/out/controlled/sae_registry.json"
CHECKPOINT_ROOT="${PROJECT_ROOT}/out/checkpoints/eurosat_multiseed"
LOG_ROOT="${PROJECT_ROOT}/out/logs/eurosat_p8090_multiseed"

BLOCK_LAYER=-2
EXPANSION_FACTOR=64
L1_COEFFICIENT=0.00008
LR=0.0004
BATCH_SIZE=16
LR_WARM_UP_STEPS=500
CLIP_DIM=768
MODEL_NAME="openai/clip-vit-base-patch16"
TOTAL_TOKENS=100000
ACTIVITY_N_BATCHES=50
ACTIVITY_DATASET="imagenet"
DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=0

SEEDS=(123 456)
CONDITIONS_FRACS="0.8:masked_p80 0.9:masked_p90"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT" "$(dirname "$REGISTRY_PATH")"

register_sae() {
    local dataset=$1 ckpt=$2 tokens=$3 condition=$4
    python3 - "$REGISTRY_PATH" "$dataset" "$ckpt" "$tokens" "$BLOCK_LAYER" "$condition" <<'PYEOF'
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

TOTAL=$(( ${#SEEDS[@]} * 2 )); CURRENT=0
SUCCEEDED=(); FAILED=()
OVERALL_START=$(date +%s)

for pair in $CONDITIONS_FRACS; do
    FRAC="${pair%%:*}"
    BASE_COND="${pair##*:}"
    for SEED in "${SEEDS[@]}"; do
        CURRENT=$((CURRENT + 1))
        CONDITION="${BASE_COND}_s${SEED}"
        CKPT_DIR="${CHECKPOINT_ROOT}/${CONDITION}"
        RUN_LOG="${LOG_ROOT}/${CONDITION}.log"
        mkdir -p "$CKPT_DIR"

        echo ""
        echo "### RUN ${CURRENT}/${TOTAL}: eurosat / ${CONDITION} (protect_frac=${FRAC}, seed=${SEED}) — $(date '+%Y-%m-%d %H:%M:%S')"

        shopt -s nullglob
        existing=("${CKPT_DIR}"/*/final_sparse_autoencoder_*/*_${BLOCK_LAYER}_resid_*.pt)
        shopt -u nullglob
        if [ ${#existing[@]} -gt 0 ]; then
            echo "[SKIP] ${CONDITION}: already exists -> ${existing[-1]}"
            register_sae "eurosat" "${existing[-1]}" "$TOTAL_TOKENS" "$CONDITION"
            SUCCEEDED+=("$CONDITION")
            continue
        fi

        START=$(date +%s); START_ISO=$(date -d "@${START}" +"%Y-%m-%d %H:%M:%S")
        python3 tasks/train_sae_masked_finetune.py \
            --sae_checkpoint_path "$SAE_CHECKPOINT" \
            --lora_checkpoint_path "$LORA_PATH" \
            --model_name "$MODEL_NAME" --clip_dim $CLIP_DIM --block_layer $BLOCK_LAYER \
            --dataset eurosat \
            --protect_frac "$FRAC" \
            --activity_n_batches $ACTIVITY_N_BATCHES --activity_dataset "$ACTIVITY_DATASET" \
            --expansion_factor $EXPANSION_FACTOR --l1_coefficient $L1_COEFFICIENT \
            --lr $LR --batch_size $BATCH_SIZE --lr_warm_up_steps $LR_WARM_UP_STEPS \
            --total_training_tokens $TOTAL_TOKENS --use_ghost_grads \
            --checkpoint_path "$CKPT_DIR" --n_checkpoints 1 \
            --seed "$SEED" --device $DEVICE \
            --run_name "eurosat_${CONDITION}" \
            > "$RUN_LOG" 2>&1
        RC=$?
        MIN=$(( ($(date +%s) - START) / 60 ))

        if [ $RC -eq 0 ]; then
            NEW=$(find "$CKPT_DIR" -newermt "$START_ISO" -path "*/final_sparse_autoencoder_*" \
                  -name "*_${BLOCK_LAYER}_resid_*.pt" 2>/dev/null | sort | tail -1)
            if [ -n "$NEW" ]; then
                echo "[OK] ${CONDITION} in ${MIN}m -> $NEW"
                register_sae "eurosat" "$NEW" "$TOTAL_TOKENS" "$CONDITION"
                SUCCEEDED+=("$CONDITION")
            else
                echo "[WARN] ${CONDITION} exited 0 but no checkpoint found"
                FAILED+=("$CONDITION")
            fi
        else
            echo "[FAIL] ${CONDITION} exit ${RC} after ${MIN}m (see $RUN_LOG)"
            FAILED+=("$CONDITION")
        fi
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        [ $CURRENT -lt $TOTAL ] && { echo "[INFO] pausing 15s"; sleep 15; }
    done
done

echo ""
echo "### DONE in $(( ($(date +%s) - OVERALL_START) / 60 ))m -- ${#SUCCEEDED[@]}/${TOTAL} ok"
for r in "${SUCCEEDED[@]}"; do echo "  ok   $r"; done
for r in "${FAILED[@]}"; do echo "  FAIL $r"; done
[ ${#FAILED[@]} -gt 0 ] && exit 1
exit 0

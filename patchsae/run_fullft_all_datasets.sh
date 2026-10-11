#!/usr/bin/env bash
# =============================================================================
# run_fullft_all_datasets.sh
#
# Full (unmasked) fine-tune of the pre-trained ImageNet G-SAE on LoRA-CLIP
# activations, for all 11 workshop-paper datasets -- protect_frac=0.0, i.e.
# every unit (W_enc, b_enc, W_dec, b_dec) is trainable. Same fixed
# hyperparameters as run_masked_finetune_all_datasets.sh ("masked" / 20%
# protected) and its masked_p80/masked_p90 siblings, so this condition slots
# directly into the same ImageNet-degradation comparison across protection
# levels {0%, 20%, 80%, 90%}. No --freeze_b_dec, matching those three.
#
# Registers under condition "fullft".
#
# Usage:
#   chmod +x run_fullft_all_datasets.sh
#   nohup ./run_fullft_all_datasets.sh > out/logs/fullft_all/driver.log 2>&1 &
# =============================================================================

set -uo pipefail

PROJECT_ROOT="/path/to/sae-domain-shift/patchsae"
LORA_WEIGHTS_ROOT="/path/to/sae-domain-shift/lora_weights/vitb16"
CHECKPOINT_ROOT="${PROJECT_ROOT}/out/checkpoints/fullft_all"
LOG_ROOT="${PROJECT_ROOT}/out/logs/fullft_all"
REGISTRY_PATH="${PROJECT_ROOT}/out/controlled/sae_registry.json"
SAE_CHECKPOINT="${PROJECT_ROOT}/data/sae_weight/base/out.pt"

declare -A LORA_PATHS
LORA_PATHS["caltech101"]="${LORA_WEIGHTS_ROOT}/caltech101/16shots/seed1/lora_weights.pt"
LORA_PATHS["cityscapes"]="${LORA_WEIGHTS_ROOT}/cityscapes/16shots/seed1/lora_weights.pt"
LORA_PATHS["cub2002011"]="${LORA_WEIGHTS_ROOT}/cub2002011/16shots/seed1/lora_weights.pt"
LORA_PATHS["dtd"]="${LORA_WEIGHTS_ROOT}/dtd/16shots/seed42/lora_weights.pt"
LORA_PATHS["eurosat"]="${LORA_WEIGHTS_ROOT}/eurosat/16shots/seed1/lora_weights.pt"
LORA_PATHS["fgvc"]="${LORA_WEIGHTS_ROOT}/fgvc/16shots/seed1/lora_weights.pt"
LORA_PATHS["kitti"]="${LORA_WEIGHTS_ROOT}/kitti/16shots/seed1/lora_weights.pt"
LORA_PATHS["pathmnist"]="${LORA_WEIGHTS_ROOT}/medmnist/16shots/seed1/lora_weights.pt"
LORA_PATHS["officehome"]="${LORA_WEIGHTS_ROOT}/officehome/16shots/seed1/lora_weights.pt"
LORA_PATHS["pets"]="${LORA_WEIGHTS_ROOT}/oxford_pets/16shots/seed1/lora_weights.pt"
LORA_PATHS["ucf101"]="${LORA_WEIGHTS_ROOT}/ucf101/16shots/seed1/lora_weights.pt"

declare -A TRAIN_DATASET_KEY
TRAIN_DATASET_KEY["caltech101"]="caltech101"
TRAIN_DATASET_KEY["cityscapes"]="cityscapes"
TRAIN_DATASET_KEY["cub2002011"]="cub2002011"
TRAIN_DATASET_KEY["dtd"]="dtd"
TRAIN_DATASET_KEY["eurosat"]="eurosat"
TRAIN_DATASET_KEY["fgvc"]="fgvc"
TRAIN_DATASET_KEY["kitti"]="kitti"
TRAIN_DATASET_KEY["pathmnist"]="medmnist"
TRAIN_DATASET_KEY["officehome"]="officehome"
TRAIN_DATASET_KEY["pets"]="oxford_pets"
TRAIN_DATASET_KEY["ucf101"]="ucf101"

ALL_DATASETS=(caltech101 cityscapes cub2002011 dtd eurosat fgvc kitti pathmnist officehome pets ucf101)

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
PROTECT_FRAC=0.0
ACTIVITY_N_BATCHES=50
ACTIVITY_DATASET="imagenet"
WANDB_PROJECT="masked_sae_finetune_all"
WANDB_LOG_FREQ=20
DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=0

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log_header() { echo ""; echo "###############################################"; echo "# $1"; echo "# $(timestamp)"; echo "###############################################"; echo ""; }

register_sae() {
    local dataset=$1 ckpt=$2 tokens=$3
    python3 - "$REGISTRY_PATH" "$dataset" "$ckpt" "$tokens" "$BLOCK_LAYER" <<'PYEOF'
import json, os, sys, time
registry_path, dataset, ckpt, tokens, layer = sys.argv[1:6]
os.makedirs(os.path.dirname(registry_path), exist_ok=True)
records = json.load(open(registry_path)) if os.path.exists(registry_path) else []
records = [r for r in records if not (r["dataset"] == dataset and r["condition"] == "fullft")]
records.append({"dataset": dataset, "vit_type": "lora", "condition": "fullft",
                "checkpoint_path": ckpt, "tokens": int(tokens), "layer": int(layer),
                "registered_at": time.strftime("%Y-%m-%d %H:%M:%S")})
json.dump(records, open(registry_path, "w"), indent=2)
print(f"[REGISTRY] {dataset}/fullft -> {ckpt}")
PYEOF
}

cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT" "$(dirname "$REGISTRY_PATH")"

log_header "PRE-FLIGHT"
for dataset in "${ALL_DATASETS[@]}"; do
    [ -f "${LORA_PATHS[$dataset]}" ] || { echo "[FATAL] missing LoRA for $dataset"; exit 1; }
done
echo "[OK] all ${#ALL_DATASETS[@]} LoRA checkpoints found"

TOTAL=${#ALL_DATASETS[@]}; CURRENT=0; SUCCEEDED=(); FAILED=()
OVERALL_START=$(date +%s)

for dataset in "${ALL_DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))
    log_header "RUN ${CURRENT}/${TOTAL}: ${dataset^^} (fullft, protect_frac=0.0)"

    DATASET_CKPT_DIR="${CHECKPOINT_ROOT}/${dataset}"
    RUN_LOG="${LOG_ROOT}/${dataset}.log"
    mkdir -p "$DATASET_CKPT_DIR"

    shopt -s nullglob
    existing=("${DATASET_CKPT_DIR}"/*/final_sparse_autoencoder_*/*_${BLOCK_LAYER}_resid_*.pt)
    shopt -u nullglob
    if [ ${#existing[@]} -gt 0 ]; then
        echo "[SKIP] ${dataset}: already exists -> ${existing[-1]}"
        register_sae "$dataset" "${existing[-1]}" "$TOTAL_TOKENS"
        SUCCEEDED+=("$dataset")
        continue
    fi

    START=$(date +%s); START_ISO=$(date -d "@${START}" +"%Y-%m-%d %H:%M:%S")
    python3 tasks/train_sae_masked_finetune.py \
        --sae_checkpoint_path "$SAE_CHECKPOINT" \
        --lora_checkpoint_path "${LORA_PATHS[$dataset]}" \
        --model_name "$MODEL_NAME" --clip_dim $CLIP_DIM --block_layer $BLOCK_LAYER \
        --dataset "${TRAIN_DATASET_KEY[$dataset]}" \
        --protect_frac $PROTECT_FRAC \
        --activity_n_batches $ACTIVITY_N_BATCHES --activity_dataset "$ACTIVITY_DATASET" \
        --expansion_factor $EXPANSION_FACTOR --l1_coefficient $L1_COEFFICIENT \
        --lr $LR --batch_size $BATCH_SIZE --lr_warm_up_steps $LR_WARM_UP_STEPS \
        --total_training_tokens $TOTAL_TOKENS --use_ghost_grads \
        --checkpoint_path "$DATASET_CKPT_DIR" --n_checkpoints 1 \
        --seed $SEED --device $DEVICE \
        --run_name "fullft_${dataset}" \
        > "$RUN_LOG" 2>&1
    RC=$?
    MIN=$(( ($(date +%s) - START) / 60 ))

    if [ $RC -eq 0 ]; then
        NEW=$(find "$DATASET_CKPT_DIR" -newermt "$START_ISO" -path "*/final_sparse_autoencoder_*" \
              -name "*_${BLOCK_LAYER}_resid_*.pt" 2>/dev/null | sort | tail -1)
        if [ -n "$NEW" ]; then
            echo "[OK] ${dataset} in ${MIN}m -> $NEW"
            register_sae "$dataset" "$NEW" "$TOTAL_TOKENS"
            SUCCEEDED+=("$dataset")
        else
            echo "[WARN] ${dataset} exited 0 but no checkpoint found"
            FAILED+=("$dataset")
        fi
    else
        echo "[FAIL] ${dataset} exit ${RC} after ${MIN}m (see $RUN_LOG)"
        FAILED+=("$dataset")
    fi
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    [ $CURRENT -lt $TOTAL ] && { echo "[INFO] pausing 15s"; sleep 15; }
done

log_header "DONE in $(( ($(date +%s) - OVERALL_START) / 60 ))m -- ${#SUCCEEDED[@]}/${TOTAL} ok"
for r in "${SUCCEEDED[@]}"; do echo "  ok   $r"; done
for r in "${FAILED[@]}"; do echo "  FAIL $r"; done
[ ${#FAILED[@]} -gt 0 ] && exit 1
exit 0

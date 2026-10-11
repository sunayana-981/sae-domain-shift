#!/usr/bin/env bash
# Local half of the Table-1 masked-FT expansion: CLIP (real LoRA) on the 6 new
# datasets, plus DINO/ALIGN/SigLIP2 (base backbone, topk SAE) on the 2 new
# datasets whose raw data is only local (food101, stanford_cars -- multi-GB,
# not worth pushing to the cluster). Sequential on this machine's single GPU.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

UW=/path/to/sae-domain-shift/unified_weights/vitb16
LOG_ROOT=out/logs/local_expanded_table1
mkdir -p "$LOG_ROOT"

run_clip() {
    local dataset=$1 lora_dir=$2
    local ckpt_dir="out/checkpoints/masked_finetune_table1_clip/${dataset}"
    if find "$ckpt_dir" -path "*/final_sparse_autoencoder_*/*_-2_resid_*.pt" 2>/dev/null | grep -q .; then
        echo "[SKIP] clip/${dataset}: already done"; return 0
    fi
    echo ""; echo "=== CLIP / ${dataset} -- $(date) ==="
    python3 tasks/train_sae_masked_finetune.py \
        --sae_checkpoint_path data/sae_weight/base/out.pt \
        --lora_checkpoint_path "${UW}/${lora_dir}/16shots/seed1/lora_adapter_weights.pt" \
        --backbone clip \
        --block_layer -2 \
        --dataset "$dataset" \
        --protect_frac 0.2 \
        --activity_n_batches 50 --activity_dataset imagenet \
        --expansion_factor 64 \
        --l1_coefficient 0.00008 --lr 0.0004 --batch_size 16 --lr_warm_up_steps 200 \
        --total_training_tokens 32768 --use_ghost_grads \
        --checkpoint_path "$ckpt_dir" --n_checkpoints 1 \
        --seed 42 --device cuda \
        --run_name "masked_table1_clip_${dataset}" 2>&1 | tee "${LOG_ROOT}/clip_${dataset}.log"
}

run_backbone() {
    local backbone=$1 dataset=$2
    local base_sae
    base_sae=$(find "out/checkpoints/base_expanded/${backbone}_topk" -path "*final_sparse_autoencoder*" -name "*.pt" | sort | tail -1)
    local ckpt_dir="out/checkpoints/masked_finetune_table1_${backbone}/${dataset}"
    if find "$ckpt_dir" -path "*/final_sparse_autoencoder_*" -name "*.pt" 2>/dev/null | grep -q .; then
        echo "[SKIP] ${backbone}/${dataset}: already done"; return 0
    fi
    echo ""; echo "=== ${backbone} / ${dataset} -- $(date) ==="
    python3 tasks/train_sae_masked_finetune.py \
        --sae_checkpoint_path "$base_sae" \
        --backbone "$backbone" \
        --block_layer -2 \
        --dataset "$dataset" \
        --protect_frac 0.2 \
        --activity_n_batches 50 --activity_dataset imagenet \
        --expansion_factor 64 --topk_sae --topk_k 32 \
        --l1_coefficient 0.00008 --lr 0.0004 --batch_size 16 --lr_warm_up_steps 200 \
        --total_training_tokens 32768 --use_ghost_grads \
        --checkpoint_path "$ckpt_dir" --n_checkpoints 1 \
        --seed 42 --device cuda \
        --run_name "masked_table1_${backbone}_${dataset}" 2>&1 | tee "${LOG_ROOT}/${backbone}_${dataset}.log"
}

# --- CLIP: 6 new datasets ---
run_clip sun397 sun397
run_clip imagenet_a imagenet_a
run_clip "imagenet-sketch" imagenet_sketch
run_clip food101 food101
run_clip stanfordcars stanford_cars
run_clip oxford_flowers oxford_flowers

# --- DINO/ALIGN/SigLIP2: food101 + stanfordcars (local-only data) ---
for bb in dino align siglip2; do
    run_backbone "$bb" food101
    run_backbone "$bb" stanfordcars
done

echo ""
echo "ALL_LOCAL_RUNS_DONE -- $(date)"

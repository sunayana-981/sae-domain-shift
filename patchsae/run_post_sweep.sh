#!/usr/bin/env bash
# Runs after run_selection_and_fraction_sweep.sh finishes, on a free GPU:
#   1. re-train pathmnist/frac_p0_top, which OOM'd when an eval ran alongside it
#   2. evaluate every fraction arm on both datasets
# Evaluations are kept strictly sequential and never overlap training -- the
# concurrency is what caused the original failure.
set -uo pipefail
cd /path/to/sae-domain-shift/patchsae
source ~/miniconda3/etc/profile.d/conda.sh && conda activate patchsae

# Wait for the training driver to exit.
while pgrep -f "run_selection_and_[f]raction_sweep" >/dev/null; do sleep 30; done
echo "[post] training driver finished at $(date '+%F %T')"

if ! ls out/checkpoints/selection_sweep/frac_p0_top/pathmnist/*/final_sparse_autoencoder_*/*_-2_resid_*.pt >/dev/null 2>&1; then
    echo "[post] retraining pathmnist/frac_p0_top"
    python3 tasks/train_sae_masked_finetune.py \
        --sae_checkpoint_path "$PWD/data/sae_weight/base/out.pt" \
        --lora_checkpoint_path "/path/to/sae-domain-shift/lora_weights/vitb16/medmnist/16shots/seed1/lora_weights.pt" \
        --model_name "openai/clip-vit-base-patch16" --clip_dim 768 --block_layer -2 \
        --dataset medmnist --protect_frac 0.0 --protect_mode top --freeze_b_dec \
        --activity_n_batches 50 --activity_dataset imagenet \
        --expansion_factor 64 --l1_coefficient 0.00008 --lr 0.0004 \
        --batch_size 16 --lr_warm_up_steps 500 --total_training_tokens 100000 \
        --use_ghost_grads --n_checkpoints 1 --seed 42 --device cuda \
        --checkpoint_path "$PWD/out/checkpoints/selection_sweep/frac_p0_top/pathmnist" \
        --run_name "frac_p0_top_pathmnist" \
        > out/logs/selection_sweep/frac_p0_top_pathmnist_retry.log 2>&1
    rc=$?
    echo "[post] retrain exit=$rc"
    if [ $rc -eq 0 ]; then
        ckpt=$(ls -t out/checkpoints/selection_sweep/frac_p0_top/pathmnist/*/final_sparse_autoencoder_*/*_-2_resid_*.pt 2>/dev/null | head -1)
        python3 - "$PWD/out/controlled/sae_registry.json" pathmnist "$ckpt" 100000 -2 frac_p0_top <<'PYEOF'
import json, os, sys, time
rp, ds, ckpt, tok, layer, cond = sys.argv[1:7]
recs = json.load(open(rp)) if os.path.exists(rp) else []
recs = [r for r in recs if not (r["dataset"] == ds and r["condition"] == cond)]
recs.append({"dataset": ds, "vit_type": "lora", "condition": cond,
             "checkpoint_path": ckpt, "tokens": int(tok), "layer": int(layer),
             "registered_at": time.strftime("%Y-%m-%d %H:%M:%S")})
json.dump(recs, open(rp, "w"), indent=2)
print(f"[REGISTRY] {ds}/{cond} -> {ckpt}")
PYEOF
    fi
fi

FRACS="frac_p0_top frac_p50_top frac_p90_top frac_p95_top frac_p98_top"
for ds in caltech101 pathmnist; do
    echo "[post] evaluating fraction arms for $ds"
    python3 -u eval_masked_finetune_sweep.py --datasets "$ds" --conditions $FRACS \
        --imagenet_n_samples 2000 \
        --save_path "out/eval_fraction_sweep/${ds}.json" \
        > "out/eval_fraction_${ds}.log" 2>&1
    echo "[post] $ds eval exit=$?"
done
echo "POST_SWEEP_COMPLETE"

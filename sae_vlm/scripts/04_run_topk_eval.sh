#! /bin/bash

PYTHONPATH=./ python tasks/classification_with_top_k_masking.py \
    --root_dir ./ \
    --dataset_name caltech101 \
    --sae_path /path/to/sae-domain-shift/patchsae/data/sae_weight/base/out.pt \
    --cls_wise_sae_activation_path ./out/feature_data/sae_openai/base/caltech101/cls_sae_cnt.npy \
    --vit_type base \
    > logs/04_test_topk_eval.txt


#  PYTHONPATH=./ python tasks/classification_with_top_k_masking.py     --root_dir ./     --dataset_name imagenet     --sae_path /path/to/sae-domain-shift/patchsae/data/sae_weight/lora/out.pt     --cls_wise_sae_activation_path ./out/feature_data/sae_openai/lora/imagenet/cls_sae_cnt.npy     --vit_type lora     --model_path /path/to/sae-domain-shift/lora_weights/vitb16/caltech101/16shots/seed1/lora_weights.pt 


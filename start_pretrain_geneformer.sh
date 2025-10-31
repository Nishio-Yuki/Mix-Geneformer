#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

RUN_ROOT=/work/mouse-Geneformer++/runs/20251031
mkdir -p "$RUN_ROOT"

deepspeed --num_gpus=7 pretrain_geneformer_ortholog_dynamic.py \
  --run_dir "$RUN_ROOT" \
  --dataset_path /work/dataset/mix-genecorpus-50M-label \
  --token_dictionary_path /work/mouse-Geneformer++/mix_token_dictionary_v3.pkl \
  --correspondence_tsv /work/mouse-Geneformer++/correspondence.tsv \
  --pretext_task MLM \
  --xspec_enable \
  --xspec_rate 1.0 \
  --lambda_xspec 0.5 \
  --tau_cross 0.1 \
  --batch_size 6 \
  --num_train_epochs 20 \
  --warmup_steps 4000 \
  --example_lengths_file /work/mouse-Geneformer++/runs/20250927/training/computed_lengths.pkl \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --deepspeed /work/mouse-Geneformer++/ds_config.json \
  > "$RUN_ROOT/train.log" 2>&1

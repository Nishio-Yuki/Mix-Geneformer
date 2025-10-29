#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
deepspeed --num_gpus=8 pretrain_geneformer.py --deepspeed ds_config.json --resume_from_checkpoint /work/mouse-geneformer/models/20250504_085947/checkpoint-112500
#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import os
import sys
import random
import pickle
from time import time

import numpy as np
import pytz
import torch
import torch.nn as nn
from transformers import (
    Trainer,
    TrainingArguments,
    BertConfig,
    BertForMaskedLM,
    BertForNextSentencePrediction,
    BertForPreTraining,
    logging as hf_logging,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk

from geneformer import GeneformerPretrainer
from SimCSE_util import (
    BertForSimCSE,
    BertForSimCSEpp,
    BertForMaskedMLandSimCSE,
    BertForMaskedMLandSimCSEpp,
)
os.environ["WANDB_DISABLED"] = "true"

# ────────────────
# Trainer.save_model を上書きして safe_serialization=False
# ────────────────
def _save_model_no_safe(self, output_dir=None, _internal_call=False):
    output_dir = output_dir or self.args.output_dir
    self.model.save_pretrained(output_dir, safe_serialization=False)
    if getattr(self, "tokenizer", None):
        self.tokenizer.save_pretrained(output_dir)
Trainer.save_model = _save_model_no_safe

def main(
    mouse_geneformer_flag: bool,
    use_pretrained: bool,
    change_dropout_rate: bool,
    resume_from_checkpoint: str = None,
):
    # ───── 時間・ディレクトリ設定 ─────
    tz = pytz.timezone("Asia/Tokyo")
    now = datetime.datetime.now(tz)
    datestamp = now.strftime("%y%m%d_%H%M%S")
    rootdir = "/work/mouse-Geneformer++"
    run_name = (
        f"{datestamp}_sim-geneformer_PM-NUse_50M_DV1_"
        f"TMLM-SimCSE_L12_emb256_SL2048_E10_B8_LR0.0001_LScosine_"
        f"WU10000_DR0.1_ACTsilu_Oadamw_DS8"
    )
    training_output_dir = os.path.join(rootdir, "models", run_name)
    logging_dir = os.path.join(rootdir, "runs", run_name)
    model_output_dir = os.path.join(training_output_dir, "models")

    # 複数プロセスで mkdir が競合しても大丈夫なよう exist_ok=True
    os.makedirs(training_output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(model_output_dir, exist_ok=True)

    # ───── モデル設定 ─────
    task = "MLM-SimCSE"
    num_layers = 6
    num_embed_dim = 256
    intermed_size = num_embed_dim * 2
    activ_fn = "silu"
    initializer_range = 0.02
    layer_norm_eps = 1e-12
    attention_probs_dropout_prob = 0.1
    hidden_dropout_prob = 0.1
    max_input_size = 2**11
    num_attn_heads = 4

    config_dict = {
        "hidden_size": num_embed_dim,
        "num_hidden_layers": num_layers,
        "initializer_range": initializer_range,
        "layer_norm_eps": layer_norm_eps,
        "attention_probs_dropout_prob": attention_probs_dropout_prob,
        "hidden_dropout_prob": hidden_dropout_prob,
        "intermediate_size": intermed_size,
        "hidden_act": activ_fn,
        "max_position_embeddings": max_input_size,
        "model_type": "bert",
        "num_attention_heads": num_attn_heads,
    }
    if mouse_geneformer_flag and task != "MLM":
        print(f"geneformer doesn't train by {task} task.")
        sys.exit(1)

    # ───── TrainingArguments ─────
    training_args = TrainingArguments(
        output_dir=training_output_dir,
        logging_dir=logging_dir,
        learning_rate=1e-3,
        per_device_train_batch_size=8,
        num_train_epochs=10,
        lr_scheduler_type="cosine",
        warmup_steps=10_000,
        weight_decay=0.001,
        group_by_length=True,
        length_column_name="length",
        disable_tqdm=False,
        save_strategy="epoch",
        logging_steps=3,
        optim="adamw_torch",
        report_to="all",  # v5 以降の警告回避
        # **DeepSpeed の設定は CLI (--deepspeed) に任せる**
    )

    # ───── データロード ─────
    dataset_path = "/work/dataset/mix-genecorpus-50M"
    dataset_length_path = os.path.join(dataset_path, "mix-genecorpus_length.pkl")
    token_dictionary_path = "/work/mouse-geneformer/dictionary_pickle/mix_token_dictionary_v3.pkl"

        # データセット読み込み
    train_ds = load_from_disk(dataset_path)

    # 語彙辞書読み込み
    with open(token_dictionary_path, "rb") as f:
        token_dict = pickle.load(f)

    # 長さ情報の読み込み or フォールバック計算
    try:
        with open(dataset_length_path, "rb") as f:
            lengths = pickle.load(f)
        print(f"Loaded lengths from {dataset_length_path}")
        example_lengths_file = dataset_length_path
    except Exception as e:
        print(f"Warning: length file not found or corrupted ({e})．Computing lengths from input_ids.")
        lengths = [len(example["input_ids"]) for example in train_ds]
        # フォールバック用の lengths を一時ファイルに保存
        computed_path = os.path.join(training_output_dir, "computed_lengths.pkl")
        with open(computed_path, "wb") as f2:
            pickle.dump(lengths, f2)
        example_lengths_file = computed_path

    # ───── モデル初期化 ─────
    if not use_pretrained:
        cfg = BertConfig(**config_dict, 
                         pad_token_id=token_dict.get("<pad>"),
                         vocab_size=len(token_dict))
        if task == "MLM":
            model = BertForMaskedLM(cfg)
        elif task == "NSP":
            model = BertForNextSentencePrediction(cfg)
        elif task == "BERT":
            model = BertForPreTraining(cfg)
        elif task == "SimCSE":
            model = BertForSimCSE(cfg)
        elif task == "SimCSEpp":
            model = BertForSimCSEpp(cfg)
        else:  # MLM-SimCSE
            model = BertForMaskedMLandSimCSE(cfg)
    else:
        model_path = "/path/to/pretrained/models"
        if task in ("SimCSE", "SimCSEpp"):
            cls = BertForSimCSE if task == "SimCSE" else BertForSimCSEpp
            model = cls.from_pretrained(model_path)
        else:
            model = BertForMaskedMLandSimCSE.from_pretrained(model_path)
        if change_dropout_rate:
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = hidden_dropout_prob

    model.train()

    # ───── Trainer定義 ─────
    trainer = GeneformerPretrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        example_lengths_file=example_lengths_file,
        token_dictionary=token_dict,
        pretext_task=task,
    )

    # ───── チェックポイント再開 or 新規開始 ─────
    last_ckpt = get_last_checkpoint(training_output_dir)
    ckpt = resume_from_checkpoint or last_ckpt
    if ckpt:
        print(f"Resuming from checkpoint: {ckpt}")
        trainer.train(resume_from_checkpoint=ckpt)
    else:
        print("Starting training from scratch")
        trainer.train()

    # ───── 最終モデル保存 ─────
    trainer.save_model(model_output_dir)
    print(f"Saved model to {model_output_dir}")

if __name__ == "__main__":
    hf_logging.set_verbosity_info()

    parser = argparse.ArgumentParser(
        description="Pretrain Geneformer w/ DeepSpeed"
    )
    parser.add_argument(
        "--mouse_geneformer_flag", action="store_true",
        help="Use mouse‐Geneformer settings"
    )
    parser.add_argument(
        "--use_pretrained", action="store_true",
        help="Load pretrained model"
    )
    parser.add_argument(
        "--change_dropout_rate", action="store_true",
        help="Adjust dropout after loading pretrained"
    )
    parser.add_argument(
        "--resume_from_checkpoint", "-r",
        type=str, default=None,
        help="Path to checkpoint dir to resume from"
    )
    # DeepSpeed が自動付与するオプションを無視
    parser.add_argument("--local_rank", type=int, default=0,
                        help="(DeepSpeed が渡す) 無視してOK")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="(DeepSpeed が渡す) 無視してOK")

    args = parser.parse_args()

    # 乱数シード固定
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    main(
        mouse_geneformer_flag=args.mouse_geneformer_flag,
        use_pretrained=args.use_pretrained,
        change_dropout_rate=args.change_dropout_rate,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

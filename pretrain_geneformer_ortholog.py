#!/usr/bin/env python
# coding: utf-8
"""
pretrain_geneformer_xspec.py (stable, MLM+xspec ready)

- cross-species InfoNCE をベース損失に加算（MLM だけ/MLM+SimCSE のどちらでも可）。
- 「task=mlm」時に SimCSE 成分をベース損失から明示的に除去（親の挙動に依存しない）。
- データ前処理は分散初期化前に完了させ、全 rank で成功/失敗を揃える（all_reduce は backend に応じて CPU/CUDA を切替）。
- --debug_n 指定時は example_lengths_file を無視してサブセット後に lengths を再計算。
- DataLoader 不整合対策として dataloader_drop_last=False（必要に応じて切替）。
- ログの出力量を抑制（Transformers/Datasets を ERROR、logging_steps を大きく、tqdm を無効化）。
"""

import os
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")          # うるさければ WARN/EMPTY
os.environ.setdefault("NCCL_IB_DISABLE", "1")        # IB 未使用環境の既知のハング回避
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

import argparse
import datetime
import json
import random
import pickle
from typing import Optional, Dict, Any

import numpy as np
import pytz
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    BertConfig,
    logging as hf_logging,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

# 既存依存
from geneformer import GeneformerPretrainer
from SimCSE_util import BertForMaskedMLandSimCSE

# ─────────────────────────────────────────────────────────────
# xspec ユーティリティ
# ─────────────────────────────────────────────────────────────

def _build_gene_id_tables(token_dict: dict):
    pad_id = int(token_dict.get("<pad>", 0))
    gene2id_human: Dict[str, int] = {}
    gene2id_mouse: Dict[str, int] = {}

    # max id を安全に推定
    max_id = 0
    for v in token_dict.values():
        if isinstance(v, (int, np.integer)):
            max_id = max(max_id, int(v))
        elif isinstance(v, str) and v.isdigit():
            max_id = max(max_id, int(v))
    id2species = torch.zeros(max_id + 1, dtype=torch.long)

    for tok, tid in token_dict.items():
        if not isinstance(tok, str):
            continue
        if isinstance(tid, (int, np.integer)):
            tid_int = int(tid)
        elif isinstance(tid, str) and tid.isdigit():
            tid_int = int(tid)
        else:
            continue

        if tok.startswith("ENSG"):
            gene2id_human[tok] = tid_int
            id2species[tid_int] = 1
        elif tok.startswith("ENSMUSG"):
            gene2id_mouse[tok] = tid_int
            id2species[tid_int] = 2
    return gene2id_human, gene2id_mouse, id2species, pad_id


def load_ortholog_maps_by_token(
    path_tsv: str,
    gene2id_mouse: dict,
    gene2id_human: dict,
    vocab_size: int,
):
    import pandas as pd
    df = pd.read_csv(path_tsv, sep="\t")
    if not {"mouse_gene", "human_gene"}.issubset(df.columns):
        raise ValueError("correspondence.tsv は 'mouse_gene' と 'human_gene' 列が必要です。")
    df = df.dropna(subset=["mouse_gene", "human_gene"]).astype(str)

    # 1:1 対応のみ
    m_counts = df["mouse_gene"].value_counts()
    h_counts = df["human_gene"].value_counts()
    df = df[df["mouse_gene"].map(m_counts).eq(1) & df["human_gene"].map(h_counts).eq(1)]

    m2h = torch.full((vocab_size,), -1, dtype=torch.long)
    h2m = torch.full((vocab_size,), -1, dtype=torch.long)

    kept = miss_m = miss_h = 0
    for m_tok, h_tok in zip(df["mouse_gene"], df["human_gene"]):
        mid = gene2id_mouse.get(m_tok)
        hid = gene2id_human.get(h_tok)
        if mid is None:
            miss_m += 1
            continue
        if hid is None:
            miss_h += 1
            continue
        m2h[mid] = hid
        h2m[hid] = mid
        kept += 1

    stats = {"pairs_kept": kept, "miss_mouse": miss_m, "miss_human": miss_h}
    return m2h, h2m, stats


class CrossSpeciesProjector:
    def __init__(self, m2h: torch.Tensor, h2m: torch.Tensor, id2species: torch.Tensor, pad_token_id: int):
        self.m2h = m2h.cpu()
        self.h2m = h2m.cpu()
        self.id2sp = id2species.cpu()  # 0=other, 1=human, 2=mouse
        self.pad = int(pad_token_id)

    @torch.no_grad()
    def project_one(self, input_ids: torch.Tensor, species: str):
        ids = input_ids.detach().clone().cpu()
        spv = self.id2sp[ids]

        if species == "mouse":
            gene_mask = spv.eq(2)
            mapped = self.m2h[ids[gene_mask]]
        else:
            gene_mask = spv.eq(1)
            mapped = self.h2m[ids[gene_mask]]

        valid = mapped.ge(0)
        num_pad = int((~valid).sum().item())
        ids[gene_mask] = torch.where(valid, mapped, torch.full_like(mapped, self.pad))
        use = bool(valid.any().item())
        return ids, use, num_pad


def info_nce(a: torch.Tensor, p: torch.Tensor, temperature: float = 0.05):
    a = F.normalize(a, dim=-1)
    p = F.normalize(p, dim=-1)
    logits = a @ p.t() / temperature
    targets = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, targets)

# ─────────────────────────────────────────────────────────────
# 便利関数（分散，長さ，データ前処理）
# ─────────────────────────────────────────────────────────────

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return (not is_dist_avail_and_initialized()) or dist.get_rank() == 0

def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()

def maybe_init_distributed(backend: str = "nccl", timeout_sec: int = 900):
    # torchrun/DeepSpeed 環境なら env:// で初期化，単機ならスキップ
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "-1")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return -1, 1
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    from datetime import timedelta
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(seconds=timeout_sec),
    )
    return local_rank, world_size

def _normalize_label_fn(batch: Dict[str, Any], col: str):
    vals = []
    for v in batch[col]:
        if isinstance(v, (bytes, bytearray)):
            v = v.decode("utf-8")
        if isinstance(v, str):
            s = v.strip().lower()
            vals.append(1 if ("mouse" in s or s == "m") else 0)
        else:
            vals.append(int(v))
    return {col: vals}

def _maybe_normalize_labels(ds: Dataset | DatasetDict, label_col: str) -> Dataset | DatasetDict:
    def _needs_map(one: Dataset) -> bool:
        if label_col not in one.column_names:
            return False
        if len(one) == 0:
            return False
        sample = one[0][label_col]
        try:
            int(sample)
            return False
        except Exception:
            return True

    if isinstance(ds, DatasetDict):
        out = {}
        for k, split in ds.items():
            out[k] = split.map(_normalize_label_fn, batched=True, fn_kwargs={"col": label_col},
                               desc=f"normalize '{label_col}' in split '{k}'") if _needs_map(split) else split
        return DatasetDict(out)
    else:
        return ds.map(_normalize_label_fn, batched=True, fn_kwargs={"col": label_col},
                      desc=f"normalize '{label_col}'") if _needs_map(ds) else ds

def _apply_debug_subset(ds: Dataset | DatasetDict, debug_n: int = 0, debug_fraction: float = 0.0):
    if (debug_n is None or debug_n <= 0) and (debug_fraction is None or debug_fraction <= 0.0):
        return ds

    def _subset(one: Dataset) -> Dataset:
        n = len(one)
        if debug_n and debug_n > 0:
            m = min(debug_n, n)
            return one.select(range(m))
        if debug_fraction and 0.0 < debug_fraction < 1.0:
            m = max(1, int(n * debug_fraction))
            return one.select(range(m))
        return one

    if isinstance(ds, DatasetDict):
        return DatasetDict({k: _subset(v) for k, v in ds.items()})
    else:
        return _subset(ds)

def _compute_lengths(ds: Dataset) -> list[int]:
    if "length" in ds.column_names:
        return [int(x) for x in ds["length"]]
    return [len(ex["input_ids"]) for ex in ds]

# safe_serialization=False に統一
def _save_model_no_safe(self, output_dir=None, _internal_call=False):
    output_dir = output_dir or self.args.output_dir
    self.model.save_pretrained(output_dir, safe_serialization=False)
    if getattr(self, "tokenizer", None):
        self.tokenizer.save_pretrained(output_dir)

Trainer.save_model = _save_model_no_safe

# ─────────────────────────────────────────────────────────────
# Pretrainer 拡張
# ─────────────────────────────────────────────────────────────
class GradientNormCallback(TrainerCallback):
    def __init__(self, log_key: str = "grad_norm"):
        self._log_key = log_key

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        trainer = kwargs.get("trainer")
        if model is None or trainer is None:
            return control

        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().norm(2))

        if not grads:
            return control

        total_norm = torch.norm(torch.stack(grads), 2).item()
        if state.is_world_process_zero():
            trainer.log({self._log_key: total_norm})
        return control


class GeneformerPretrainerXSpec(GeneformerPretrainer):
    def __init__(self, *args, projector=None, label_field="label",
                 xspec_enable=False, xspec_rate=1.0, tau_cross=0.05, lambda_xspec=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self._xspec_enable = bool(xspec_enable)
        self._projector = projector
        self._label_field = label_field
        self._xspec_rate = float(xspec_rate)
        self._tau_cross = float(tau_cross)
        self._lambda_xspec = float(lambda_xspec)
        self._seen = 0
        self._used = 0
        self._pad_replaced = 0

    def _labels_to_species(self, labels):
        species_list = []
        if isinstance(labels, (list, tuple)):
            for v in labels:
                if isinstance(v, (bytes, bytearray)):
                    v = v.decode("utf-8")
                if isinstance(v, str):
                    s = v.strip().lower()
                    species_list.append("mouse" if ("mouse" in s or s == "m") else "human")
                else:
                    species_list.append("mouse" if int(v) == 1 else "human")
        elif torch.is_tensor(labels):
            for v in labels.view(-1).tolist():
                species_list.append("mouse" if int(v) == 1 else "human")
        else:
            raise ValueError("未知の label フォーマットです。")
        return species_list

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """
        base(GeneformerPretrainer) 側で計算される ベース損失（MLM or MLM+SimCSE）に、
        オルソログ置換ペアの InfoNCE（xspec）を加算する。
        - proj_ids は projector で human↔mouse に変換してから attention/labels を作成
        - 置換側は labels=-100 で MLM 損失を無効化し、表現のみ取得
        """
        # ---- 1) ベース損失（親に委譲） ----
        lbl = inputs.get(self._label_field, None)
        inputs_wo_label = dict(inputs); inputs_wo_label.pop(self._label_field, None)
        model_inputs = dict(inputs_wo_label)
        model_inputs.setdefault("output_hidden_states", True)
        model_inputs.setdefault("return_dict", True)

        base_ret = super().compute_loss(model, model_inputs, return_outputs=True)
        if isinstance(base_ret, tuple):
            loss_base, outputs = base_ret
        else:
            loss_base, outputs = base_ret, None

        def _extract_loss_component(out, name):
            if out is None:
                return None
            value = getattr(out, name, None)
            if value is None and isinstance(out, dict):
                value = out.get(name)
            return value

        loss_mlm_component = _extract_loss_component(outputs, "masked_lm_loss")
        # SimCSE 名称の揺れ対策：両方拾う
        loss_simcse_component = _extract_loss_component(outputs, "contrastive_learning_loss")
        loss_simcse_dim_component = (
            _extract_loss_component(outputs, "dimensino_contrastive_learning_loss")
            or _extract_loss_component(outputs, "dimension_contrastive_learning_loss")
        )

        def _to_float(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return float(x.detach().item())
            return float(x)

        # ===== 追加：task=mlm のときは SimCSE 成分をベース損失から除去 =====
        def _as_tensor(x, device):
            if x is None:
                return None
            return x if torch.is_tensor(x) else torch.tensor(float(x), device=device)

        pretext = getattr(self, "pretext_task", None)
        if pretext is None:
            pretext = getattr(self.args, "pretext_task", None)
        is_mlm_only = isinstance(pretext, str) and ("simcse" not in pretext.lower())

        if is_mlm_only:
            dev = loss_base.device if torch.is_tensor(loss_base) else torch.device("cpu")
            sim = _as_tensor(loss_simcse_component, dev) or torch.tensor(0.0, device=dev)
            sim_dim = _as_tensor(loss_simcse_dim_component, dev) or torch.tensor(0.0, device=dev)
            loss_base = loss_base - sim - sim_dim
            # ログでは 0 として扱う
            zero = torch.tensor(0.0, device=dev)
            loss_simcse_component = zero
            loss_simcse_dim_component = zero
        # ===== ここまで =====

        def _log_progress(loss_total_tensor, loss_xspec_tensor, applied_rate, valid_examples=None):
            if getattr(self.args, "process_index", 0) != 0:
                return
            try:
                payload = {
                    "loss_total": _to_float(loss_total_tensor),
                    "loss_base": _to_float(loss_base),
                    "loss_xspec": _to_float(loss_xspec_tensor),
                    "xspec_applied_rate": applied_rate,
                }
                if valid_examples is not None:
                    payload["xspec_valid_examples"] = int(valid_examples)
                if loss_mlm_component is not None:
                    payload["loss_mlm"] = _to_float(loss_mlm_component)
                if loss_simcse_component is not None:
                    payload["loss_simcse"] = _to_float(loss_simcse_component)
                if loss_simcse_dim_component is not None:
                    payload["loss_simcse_dim"] = _to_float(loss_simcse_dim_component)
                self.log(payload)
            except Exception:
                pass

        # xspec が無効/不成立ならそのまま返す
        if not (self._xspec_enable and self._projector is not None and self._lambda_xspec > 0.0 and lbl is not None):
            return (loss_base, outputs) if return_outputs else loss_base
        if random.random() > self._xspec_rate:
            _log_progress(loss_base, 0.0, 0.0)
            return (loss_base, outputs) if return_outputs else loss_base

        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return (loss_base, outputs) if return_outputs else loss_base

        device = input_ids.device
        B = input_ids.size(0)

        # ---- 2) 置換 ID を先に作る ----
        species = self._labels_to_species(lbl)
        proj_list, valid_list = [], []
        for i in range(B):
            p_ids, use, _num_pad = self._projector.project_one(input_ids[i], species[i])
            proj_list.append(p_ids.unsqueeze(0))
            valid_list.append(use)
        proj_ids = torch.cat(proj_list, dim=0).to(device)                        # [B, L]
        valid_mask = torch.tensor(valid_list, dtype=torch.bool, device=device)    # [B]

        # 置換が1件も成立しなければ base のみ
        if not bool(valid_mask.any().item()):
            _log_progress(loss_base, 0.0, 1.0, valid_examples=0)
            return (loss_base, outputs) if return_outputs else loss_base

        # ---- 3) 置換側の attention と “MLM無効化” labels を作る ----
        pad_id = getattr(model.config, "pad_token_id", 0)
        attn_proj = (proj_ids != pad_id).to(device)                               # [B, L]
        ignore_labels = torch.full_like(proj_ids, fill_value=-100)                # [B, L]

        # ---- 4) 表現を取得 ----
        def _cls(o):
            if o is None:
                raise RuntimeError("CLS representation not found in model output")
            po = getattr(o, "pooler_output", None)
            if po is None and isinstance(o, dict):
                po = o.get("pooler_output")
            if po is not None:
                return po
            lho = getattr(o, "last_hidden_state", None)
            if lho is None and isinstance(o, dict):
                lho = o.get("last_hidden_state")
            if lho is not None:
                return lho[:, 0]
            hidden_states = getattr(o, "hidden_states", None)
            if hidden_states is None and isinstance(o, dict):
                hidden_states = o.get("hidden_states")
            if hidden_states is not None and len(hidden_states) > 0:
                return hidden_states[-1][:, 0]
            if isinstance(o, tuple) and len(o) > 0:
                first = o[0]
                if isinstance(first, torch.Tensor) and first.ndim >= 2:
                    return first[:, 0]
            raise RuntimeError("CLS representation not found in model output")

        anchor_rep = _cls(outputs)                                                # [B, H]

        out_proj = model(input_ids=proj_ids,
                         attention_mask=attn_proj,
                         labels=ignore_labels,        # MLM 無効（lossに寄与させない）
                         output_hidden_states=True,
                         return_dict=True)
        proj_rep = _cls(out_proj)                                                 # [B, H]

        # ---- 5) 有効サンプルのみで InfoNCE を計算・加算 ----
        idx = valid_mask.nonzero(as_tuple=True)[0]                                 # [B_valid]
        tau = getattr(self.args, "tau_cross", self._tau_cross)
        lam = getattr(self.args, "lambda_xspec", self._lambda_xspec)

        if idx.numel() > 0:
            loss_xspec = info_nce(anchor_rep[idx], proj_rep[idx], temperature=tau)
            loss_total = loss_base + lam * loss_xspec
        else:
            loss_xspec = torch.tensor(0.0, device=device)
            loss_total = loss_base

        # ---- 6) ログ（rank0 のみ）----
        _log_progress(loss_total, loss_xspec, 1.0, valid_examples=idx.numel())

        return (loss_total, outputs) if return_outputs else loss_total


# ─────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────

def _resolve_deepspeed_cfg(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    maybe = os.path.abspath(path)
    if os.path.isfile(maybe):
        return maybe
    print(f"[warn] DeepSpeed config not found: {path} → {maybe}. DeepSpeed 無効化。")
    return None

def main():
    # ログ静音化
    hf_logging.set_verbosity_error()
    try:
        import datasets as _ds
        _ds.logging.set_verbosity_error()
    except Exception:
        pass

    tz = pytz.timezone("Asia/Tokyo")
    now = datetime.datetime.now(tz)
    datestamp = now.strftime("%y%m%d_%H%M%S")

    ap = argparse.ArgumentParser(description="Pretrain Mix-Geneformer with cross-species loss (label=mouse/human)")
    # データ
    ap.add_argument("--local_rank", type=int, default=-1)  # DeepSpeed/torchrun が付ける
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--token_dictionary_path", type=str, required=True)
    ap.add_argument("--resume_from_checkpoint", type=str, default=None)
    ap.add_argument("--example_lengths_file", type=str, default=None,
                    help="事前計算した length pkl（--debug_n 指定時は無視して再計算）")
    # 学習設定
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--warmup_steps", type=int, default=1_000)
    ap.add_argument("--weight_decay", type=float, default=0.001)
    # モデルロード
    ap.add_argument("--use_pretrained", action="store_true")
    ap.add_argument("--pretrained_path", type=str, default="")
    # xspec
    ap.add_argument("--xspec_enable", action="store_true")
    ap.add_argument("--correspondence_tsv", type=str, default="")
    ap.add_argument("--xspec_rate", type=float, default=1.0)
    ap.add_argument("--lambda_xspec", type=float, default=0.3)
    ap.add_argument("--tau_cross", type=float, default=0.05)
    ap.add_argument("--label_field", type=str, default="label")
    # 分散/DeepSpeed
    ap.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed 設定ファイル（絶対パス推奨）")
    ap.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    ap.add_argument("--no_deepspeed", action="store_true", help="単機・小規模デバッグ用に DeepSpeed を使わない")
    # デバッグ用サブセット
    ap.add_argument("--debug_n", type=int, default=0, help="各 split から最大 N サンプルのみ使用")
    ap.add_argument("--debug_fraction", type=float, default=0.0, help="各 split の先頭 frac を使用（0-1）")

    args = ap.parse_args()

    # 分散初期化（必要時のみ）
    local_rank, world_size = maybe_init_distributed(backend=args.backend)

    # datestamp を rank0 から全ランクに共有
    if is_dist_avail_and_initialized():
        stamp_list = [datestamp] if is_main_process() else [None]
        dist.broadcast_object_list(stamp_list, src=0)
        datestamp = stamp_list[0]

    # 乱数初期化を rank でずらす（broadcast 前でも安全）
    base_seed = 1234
    torch.manual_seed(base_seed + (dist.get_rank() if is_dist_avail_and_initialized() else 0))
    np.random.seed(base_seed)
    random.seed(base_seed)

    # 出力ディレクトリ
    rootdir = "/work/mouse-Geneformer++"
    run_name = f"{datestamp}_mix-geneformer_xspec_label"
    training_output_dir = os.path.join(rootdir, "models", run_name)
    logging_dir = os.path.join(rootdir, "runs", run_name)
    model_output_dir = os.path.join(training_output_dir, "models")
    if is_main_process():
        os.makedirs(training_output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(model_output_dir, exist_ok=True)
    barrier()

    # ── 分散初期化前にコケがちな前処理を「全 rank 同期で完了」 ──
    try:
        train_ds: Dataset | DatasetDict = load_from_disk(args.dataset_path)

        # サブセットはここで実施
        train_ds = _apply_debug_subset(train_ds, args.debug_n, args.debug_fraction)

        # トークン辞書
        with open(args.token_dictionary_path, "rb") as f:
            token_dict = pickle.load(f)

        # xspec 初期化（projector）
        projector = None
        if args.xspec_enable:
            if not args.correspondence_tsv:
                raise ValueError("--xspec_enable のときは --correspondence_tsv を指定してください。")
            gene2id_human, gene2id_mouse, id2species, pad_id = _build_gene_id_tables(token_dict)
            m2h, h2m, stats = load_ortholog_maps_by_token(
                args.correspondence_tsv, gene2id_mouse, gene2id_human, len(token_dict)
            )
            if is_main_process():
                print(f"[xspec] ortholog kept={stats['pairs_kept']:,}, miss_mouse={stats['miss_mouse']:,}, miss_human={stats['miss_human']:,}")
            projector = CrossSpeciesProjector(m2h, h2m, id2species=id2species, pad_token_id=pad_id)

        # ラベル正規化
        train_ds = _maybe_normalize_labels(train_ds, args.label_field)

        # lengths を決定（debug 指定時は必ず再計算）
        if args.debug_n > 0 or (args.debug_fraction and args.debug_fraction > 0.0):
            lengths_path = None  # 強制再計算
        else:
            lengths_path = args.example_lengths_file

        if lengths_path and os.path.exists(lengths_path):
            with open(lengths_path, "rb") as f:
                lengths = pickle.load(f)
            # DatasetDict の場合は train 優先で件数を比較
            ds_for_len = train_ds["train"] if isinstance(train_ds, DatasetDict) and "train" in train_ds else train_ds
            if len(lengths) != len(ds_for_len):
                if is_main_process():
                    print(f"[warn] lengths({len(lengths):,}) とデータ({len(ds_for_len):,}) が不一致のため再計算します。")
                lengths_path = None

        if not lengths_path:
            # サブセット後の順序に一致する lengths を算出して保存
            ds_for_len = train_ds["train"] if isinstance(train_ds, DatasetDict) and "train" in train_ds else train_ds
            lengths = _compute_lengths(ds_for_len)
            lengths_path = os.path.join(training_output_dir, "computed_lengths.pkl")
            if is_main_process():
                with open(lengths_path, "wb") as f:
                    pickle.dump(lengths, f)
                print(f"[lengths] wrote: {lengths_path} ({len(lengths):,} samples)")

        prep_ok = True
    except Exception as e:
        prep_ok = False
        if is_main_process():
            print("[fatal] preprocessing failed:", repr(e))
    # 全 rank で結果を揃える（backend に応じて CPU/CUDA を切替）
    if is_dist_avail_and_initialized():
        use_cuda_for_ddp = (dist.get_backend() == "nccl") and torch.cuda.is_available()
        dev = torch.device("cuda", torch.cuda.current_device()) if use_cuda_for_ddp else torch.device("cpu")
        prep_tensor = torch.tensor([1 if prep_ok else 0], device=dev)
        dist.all_reduce(prep_tensor, op=dist.ReduceOp.MIN)
        prep_ok = bool(prep_tensor.item() == 1)
    if not prep_ok:
        raise SystemExit(1)

    # モデル設定
    task = "MLM-SimCSE"  # ← 小文字。compute_loss 側は "simcse" を含むかどうかで判定
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

    cfg_hf = dict(
        hidden_size=num_embed_dim,
        num_hidden_layers=num_layers,
        initializer_range=initializer_range,
        layer_norm_eps=layer_norm_eps,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        hidden_dropout_prob=hidden_dropout_prob,
        intermediate_size=intermed_size,
        hidden_act=activ_fn,
        max_position_embeddings=max_input_size,
        model_type="bert",
        num_attention_heads=num_attn_heads,
        pad_token_id=token_dict.get("<pad>", 0),
        vocab_size=len(token_dict),
    )

    if not args.use_pretrained:
        model = BertForMaskedMLandSimCSE(BertConfig(**cfg_hf))
    else:
        if not args.pretrained_path:
            raise ValueError("--use_pretrained の場合は --pretrained_path を指定してください。")
        model = BertForMaskedMLandSimCSE.from_pretrained(args.pretrained_path)

    # DeepSpeed 設定
    ds_cfg = None if args.no_deepspeed else _resolve_deepspeed_cfg(args.deepspeed)

    training_args = TrainingArguments(
        output_dir=training_output_dir,
        logging_dir=logging_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        group_by_length=True,
        length_column_name="length",
        disable_tqdm=True,               # 進捗バー抑制
        save_strategy="epoch",
        logging_steps=500,               # ログ頻度を下げる
        logging_strategy="steps",
        dataloader_drop_last=False,
        optim="adamw_torch",
        report_to="none",
        deepspeed=ds_cfg,
        remove_unused_columns=True,
        label_names=[args.label_field],
    )

    # 1) DatasetDict -> Dataset（"train" を優先）
    if isinstance(train_ds, DatasetDict):
        if "train" in train_ds:
            train_ds = train_ds["train"]
        else:
            first_key = next(iter(train_ds.keys()))
            train_ds = train_ds[first_key]

    # 2) group_by_length 用に 'length' カラムを必ず付与
    if "length" not in train_ds.column_names:
        lengths = _compute_lengths(train_ds)
        train_ds = train_ds.add_column("length", lengths)

    trainer_cls = GeneformerPretrainerXSpec if args.xspec_enable else GeneformerPretrainer
    # Trainer 作成
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        example_lengths_file=lengths_path,
        token_dictionary=token_dict,
        pretext_task=task,    # ← 小文字 "mlm" を渡す
        # xspec 追加
        projector=projector,
        label_field=args.label_field,
        xspec_enable=args.xspec_enable,
        xspec_rate=args.xspec_rate,
        tau_cross=args.tau_cross,
        lambda_xspec=args.lambda_xspec,
    )
    trainer.add_callback(GradientNormCallback())

    # 健全性ログ（rank0 のみ）
    if is_main_process():
        ds_for_len = train_ds["train"] if isinstance(train_ds, DatasetDict) and "train" in train_ds else train_ds
        print(f"[sanity] world_size={world_size}, dataset_len={len(ds_for_len):,}, batch_size(per_device)={args.batch_size}")
        print(f"[sanity] group_by_length={training_args.group_by_length}, drop_last={training_args.dataloader_drop_last}")

    # 学習
    last_ckpt = get_last_checkpoint(training_output_dir)
    ckpt = args.resume_from_checkpoint or last_ckpt
    try:
        if ckpt:
            if is_main_process():
                print(f"Resuming from checkpoint: {ckpt}")
            trainer.train(resume_from_checkpoint=ckpt)
        else:
            if is_main_process():
                print("Starting training from scratch")
            trainer.train()
    finally:
        if is_dist_avail_and_initialized():
            try:
                barrier()
                dist.destroy_process_group()
            except Exception:
                pass

    # 保存
    trainer.save_model(model_output_dir)
    if is_main_process():
        print(f"Saved model to {model_output_dir}")

if __name__ == "__main__":
    main()

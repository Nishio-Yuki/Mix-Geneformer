#!/usr/bin/env python
# coding: utf-8
"""
pretrain_geneformer_xspec_debug.py

目的：
- MLM（および必要に応じて SimCSE）＋ xspec を同時に学習し，
  「両損失が効いているか」をステップ毎ログで可視化する．
- 投影（ヒト↔マウス置換）の前後をコンソールに数サンプル表示できる．

主な追加：
- loss の内訳ログ：loss_base（= 親の compute_loss，MLM±SimCSE），loss_xspec，loss_total
- xspec 実行率・有効ペア統計のロギング
- --xspec_dump_examples で前後のトークン列ダンプ
"""

import os
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

import argparse
import datetime
import random
import pickle
from typing import Optional, Dict, Any, Union, List, Type
import torch.nn as nn

import numpy as np
import pytz
import torch
import torch.nn.functional as F
import torch.distributed as dist
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    BertConfig,
    BertForMaskedLM,
    logging as hf_logging,
)
from transformers.trainer_utils import get_last_checkpoint

# 既存依存
from geneformer import GeneformerPretrainer
from SimCSE_util import BertForMaskedMLandSimCSE, BertForSimCSE


# ─────────────────────────────────────────────────────────────
# プレテキストタスクごとのモデル選択
# ─────────────────────────────────────────────────────────────

PRETEXT_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "MLM": BertForMaskedLM,
    "SimCSE": BertForSimCSE,
    "MLM-SimCSE": BertForMaskedMLandSimCSE,
}

def _load_model_for_pretext(task: str, config: BertConfig, use_pretrained: bool,
                            pretrained_path: Optional[str]) -> nn.Module:
    if task not in PRETEXT_MODEL_REGISTRY:
        raise ValueError(f"Unsupported pretext_task: {task}")
    model_cls = PRETEXT_MODEL_REGISTRY[task]
    if use_pretrained:
        if not pretrained_path:
            raise ValueError("--use_pretrained のときは --pretrained_path を指定してください．")
        model = model_cls.from_pretrained(pretrained_path)
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = config.pad_token_id
        if config.vocab_size != getattr(model.config, "vocab_size", config.vocab_size):
            model.resize_token_embeddings(config.vocab_size)
        return model
    return model_cls(config)


# ─────────────────────────────────────────────────────────────
# xspec ユーティリティ
# ─────────────────────────────────────────────────────────────

def _unwrap_model(m):
    try:
        from accelerate.utils import extract_model_from_parallel
        return extract_model_from_parallel(m)
    except Exception:
        pass
    seen = set()
    while hasattr(m, "module") and m not in seen:
        seen.add(m); m = m.module
    return m

def _get_base_encoder(unwrapped):
    if hasattr(unwrapped, "bert"):
        obj = getattr(unwrapped, "bert")
        if obj is not None:
            return obj
    for attr in ("backbone", "encoder", "transformer", "model"):
        if hasattr(unwrapped, attr):
            obj = getattr(unwrapped, attr)
            if obj is not None:
                return obj
    return unwrapped

def _build_gene_id_tables(token_dict: dict):
    pad_id = int(token_dict.get("<pad>", 0))
    gene2id_human: Dict[str, int] = {}
    gene2id_mouse: Dict[str, int] = {}
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
            gene2id_human[tok] = tid_int; id2species[tid_int] = 1
        elif tok.startswith("ENSMUSG"):
            gene2id_mouse[tok] = tid_int; id2species[tid_int] = 2
    return gene2id_human, gene2id_mouse, id2species, pad_id

def load_ortholog_maps_by_token(path_tsv: str, gene2id_mouse: dict, gene2id_human: dict, vocab_size: int):
    import pandas as pd
    df = pd.read_csv(path_tsv, sep="\t")
    if not {"mouse_gene", "human_gene"}.issubset(df.columns):
        raise ValueError("correspondence.tsv は 'mouse_gene' と 'human_gene' 列が必要です．")
    df = df.dropna(subset=["mouse_gene", "human_gene"]).astype(str)
    m_counts = df["mouse_gene"].value_counts()
    h_counts = df["human_gene"].value_counts()
    df = df[df["mouse_gene"].map(m_counts).eq(1) & df["human_gene"].map(h_counts).eq(1)]
    m2h = torch.full((vocab_size,), -1, dtype=torch.long)
    h2m = torch.full((vocab_size,), -1, dtype=torch.long)
    kept = miss_m = miss_h = 0
    for m_tok, h_tok in zip(df["mouse_gene"], df["human_gene"]):
        mid = gene2id_mouse.get(m_tok); hid = gene2id_human.get(h_tok)
        if mid is None: miss_m += 1; continue
        if hid is None: miss_h += 1; continue
        m2h[mid] = hid; h2m[hid] = mid; kept += 1
    stats = {"pairs_kept": kept, "miss_mouse": miss_m, "miss_human": miss_h}
    return m2h, h2m, stats

class CrossSpeciesProjector:
    def __init__(self, m2h: torch.Tensor, h2m: torch.Tensor, id2species: torch.Tensor, pad_token_id: int):
        self.m2h = m2h.cpu(); self.h2m = h2m.cpu(); self.id2sp = id2species.cpu()
        self.pad = int(pad_token_id)

    @torch.no_grad()
    def project_one(self, input_ids: torch.Tensor, species: str):
        ids = input_ids.detach().clone().cpu()
        spv = self.id2sp[ids]
        if species == "mouse":
            gene_mask = spv.eq(2); mapped = self.m2h[ids[gene_mask]]
        else:
            gene_mask = spv.eq(1); mapped = self.h2m[ids[gene_mask]]
        valid = mapped.ge(0)
        num_pad = int((~valid).sum().item())
        ids[gene_mask] = torch.where(valid, mapped, torch.full_like(mapped, self.pad))
        use = bool(valid.any().item())
        return ids, use, num_pad

def info_nce(a: torch.Tensor, p: torch.Tensor, temperature: float = 0.05):
    a = F.normalize(a, dim=-1); p = F.normalize(p, dim=-1)
    logits = a @ p.t() / temperature
    targets = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, targets)


# ─────────────────────────────────────────────────────────────
# 可視化ユーティリティ（投影の前後ダンプ）
# ─────────────────────────────────────────────────────────────

def _invert_token_dict(token_dict: Dict[Union[int, str], Union[int, str]]) -> List[str]:
    max_id = 0
    for v in token_dict.values():
        if isinstance(v, (int, np.integer)): max_id = max(max_id, int(v))
        elif isinstance(v, str) and v.isdigit(): max_id = max(max_id, int(v))
    id2tok = ["UNK"] * (max_id + 1)
    for tok, tid in token_dict.items():
        if isinstance(tid, (int, np.integer)): i = int(tid)
        elif isinstance(tid, str) and tid.isdigit(): i = int(tid)
        else: continue
        if 0 <= i <= max_id: id2tok[i] = str(tok)
    return id2tok

def _ids_to_tokens(ids: torch.Tensor, id2tok: List[str], pad_id: int) -> List[str]:
    toks = []
    for t in ids.tolist():
        toks.append("<pad>" if t == pad_id else (id2tok[t] if 0 <= t < len(id2tok) else f"<id:{t}>"))
    return toks

def _guess_species_from_label(v: Union[int, str, bytes]) -> str:
    if isinstance(v, (bytes, bytearray)): v = v.decode("utf-8")
    if isinstance(v, str):
        s = v.strip().lower(); return "mouse" if ("mouse" in s or s == "m") else "human"
    return "mouse" if int(v) == 1 else "human"

def dump_xspec_examples(ds: Union[Dataset, DatasetDict], projector: CrossSpeciesProjector,
                        token_dict: dict, label_field: str, n_samples: int,
                        max_len: int, seed: int):
    import random as pyrand
    id2tok = _invert_token_dict(token_dict)
    pad_id = int(token_dict.get("<pad>", 0))
    base = ds["train"] if isinstance(ds, DatasetDict) and "train" in ds else (ds[next(iter(ds))] if isinstance(ds, DatasetDict) else ds)
    if len(base) == 0:
        print("[xspec-dump] dataset is empty．"); return
    pyrand.seed(seed)
    idxs = list(range(len(base))); pyrand.shuffle(idxs)
    shown = 0
    print(f"[xspec-dump] show up to {n_samples:,} samples (max_len={max_len})")
    for i in idxs:
        ex = base[i]
        input_ids = torch.tensor(ex["input_ids"], dtype=torch.long)
        species = _guess_species_from_label(ex.get(label_field, 0))
        proj_ids, use, num_pad_from_map = projector.project_one(input_ids, species)
        if not use: continue
        opposite = "human" if species == "mouse" else "mouse"
        back_ids, use_back, _ = projector.project_one(proj_ids, opposite)
        src_toks  = _ids_to_tokens(input_ids[:max_len], id2tok, pad_id)
        proj_toks = _ids_to_tokens(proj_ids[:max_len],  id2tok, pad_id)
        back_toks = _ids_to_tokens(back_ids[:max_len],  id2tok, pad_id)
        print("─" * 80)
        print(f"[{shown+1}] label={species}  valid_mapped_positions≈{(proj_ids != pad_id).sum().item():,}"
              f"  (padded_by_mapping={num_pad_from_map:,})")
        print("src :", " ".join(src_toks))
        print("proj:", " ".join(proj_toks))
        if use_back: print("back:", " ".join(back_toks))
        shown += 1
        if shown >= n_samples: break
    if shown == 0:
        print("[xspec-dump] no mappable examples were found in the sampled subset．")


# ─────────────────────────────────────────────────────────────
# 分散ユーティリティ
# ─────────────────────────────────────────────────────────────

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return (not is_dist_avail_and_initialized()) or dist.get_rank() == 0

def barrier():
    if is_dist_avail_and_initialized(): dist.barrier()

def maybe_init_distributed(backend: str = "nccl", timeout_sec: int = 900):
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "-1")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1: return -1, 1
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    from datetime import timedelta
    dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(seconds=timeout_sec))
    return local_rank, world_size

def _normalize_label_fn(batch: Dict[str, Any], col: str):
    vals = []
    for v in batch[col]:
        if isinstance(v, (bytes, bytearray)): v = v.decode("utf-8")
        if isinstance(v, str): s = v.strip().lower(); vals.append(1 if ("mouse" in s or s == "m") else 0)
        else: vals.append(int(v))
    return {col: vals}

def _maybe_normalize_labels(ds: Union[Dataset, DatasetDict], label_col: str) -> Union[Dataset, DatasetDict]:
    def _needs_map(one: Dataset) -> bool:
        if label_col not in one.column_names or len(one) == 0: return False
        sample = one[0][label_col]
        try: int(sample); return False
        except Exception: return True
    if isinstance(ds, DatasetDict):
        out = {}
        for k, split in ds.items():
            out[k] = split.map(_normalize_label_fn, batched=True, fn_kwargs={"col": label_col},
                               desc=f"normalize '{label_col}' in split '{k}'") if _needs_map(split) else split
        return DatasetDict(out)
    else:
        return ds.map(_normalize_label_fn, batched=True, fn_kwargs={"col": label_col},
                      desc=f"normalize '{label_col}'") if _needs_map(ds) else ds

def _apply_debug_subset(ds: Union[Dataset, DatasetDict], debug_n: int = 0, debug_fraction: float = 0.0):
    if (not debug_n or debug_n <= 0) and (not debug_fraction or debug_fraction <= 0.0): return ds
    def _subset(one: Dataset) -> Dataset:
        n = len(one)
        if debug_n and debug_n > 0: return one.select(range(min(debug_n, n)))
        if debug_fraction and 0.0 < debug_fraction < 1.0: return one.select(range(max(1, int(n * debug_fraction))))
        return one
    return DatasetDict({k: _subset(v) for k, v in ds.items()}) if isinstance(ds, DatasetDict) else _subset(ds)

def _compute_lengths(ds: Dataset) -> List[int]:
    if "length" in ds.column_names: return [int(x) for x in ds["length"]]
    return [len(ex["input_ids"]) for ex in ds]

# Trainer.save_model を safe_serialization=False に統一
def _save_model_no_safe(self, output_dir=None, _internal_call=False):
    output_dir = output_dir or self.args.output_dir
    self.model.save_pretrained(output_dir, safe_serialization=False)
    if getattr(self, "tokenizer", None): self.tokenizer.save_pretrained(output_dir)
Trainer.save_model = _save_model_no_safe


# ─────────────────────────────────────────────────────────────
# Pretrainer 拡張（MLM±SimCSE の上に xspec を加算＆ロギング）
# ─────────────────────────────────────────────────────────────

class GeneformerPretrainerXSpec(GeneformerPretrainer):
    def __init__(self, *args, projector=None, label_field="label",
        xspec_enable=False, xspec_rate=1.0, tau_cross=0.05, lambda_xspec=0.3,
        xspec_requires_grad=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._xspec_enable = bool(xspec_enable)
        self._projector = projector
        self._label_field = label_field
        self._xspec_rate = float(xspec_rate)
        self._tau_cross = float(tau_cross)
        self._lambda_xspec = float(lambda_xspec)
        self._xspec_requires_grad = bool(xspec_requires_grad)
        # ログ用カウンタ
        self._xspec_applied = 0
        self._xspec_steps = 0
        self._xspec_valid_positions_ema = 0.0  # 有効位置の指数平均
        self._nan_guard_counter = 0

    def _sanitize_loss(self, loss_tensor: Optional[torch.Tensor], name: str):
        if torch.isfinite(loss_tensor).all():
            return loss_tensor, False
        self._nan_guard_counter += 1
        step = getattr(getattr(self, "state", None), "global_step", -1)
        if is_main_process():
            print(f"[warn] loss guard activated: {name} produced non-finite values at step {step} "
                  f"(count={self._nan_guard_counter}). Zeroing this contribution.")
        sanitized = torch.nan_to_num(loss_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return sanitized, True

    def _labels_to_species(self, labels):
        species_list = []
        if isinstance(labels, (list, tuple)):
            for v in labels:
                if isinstance(v, (bytes, bytearray)): v = v.decode("utf-8")
                if isinstance(v, str):
                    s = v.strip().lower()
                    species_list.append("mouse" if ("mouse" in s or s == "m") else "human")
                else:
                    species_list.append("mouse" if int(v) == 1 else "human")
        elif torch.is_tensor(labels):
            for v in labels.view(-1).tolist():
                species_list.append("mouse" if int(v) == 1 else "human")
        else:
            raise ValueError("未知の label フォーマットです．")
        return species_list

    @staticmethod
    def _cls(output_obj):
        po = getattr(output_obj, "pooler_output", None)
        if po is not None: return po
        lho = getattr(output_obj, "last_hidden_state", None)
        if lho is not None: return lho[:, 0]
        hs = getattr(output_obj, "hidden_states", None)
        if hs is not None and len(hs) > 0: return hs[-1][:, 0]
        if isinstance(output_obj, dict):
            if "pooler_output" in output_obj: return output_obj["pooler_output"]
            if "last_hidden_state" in output_obj: return output_obj["last_hidden_state"][:, 0]
            if "hidden_states" in output_obj and len(output_obj["hidden_states"]) > 0:
                return output_obj["hidden_states"][-1][:, 0]
        raise AttributeError("CLS representation not found in model output")

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        # 親（= MLM ± SimCSE）を先に計算
        inputs_wo_label = {}
        for k, v in inputs.items():
            if k == self._label_field: continue
            inputs_wo_label[k] = v.detach().clone() if torch.is_tensor(v) else v
        inputs_wo_label.setdefault("return_dict", True)

        loss_base, base_outputs = super().compute_loss(model, inputs_wo_label, return_outputs=True)
        loss_base, base_guarded = self._sanitize_loss(loss_base, "loss_base")
        guard_triggered = base_guarded

        loss_components: Dict[str, float] = {}

        def _extract_output(outputs_obj, key: str):
            if outputs_obj is None:
                return None
            if isinstance(outputs_obj, dict):
                return outputs_obj.get(key)
            return getattr(outputs_obj, key, None)

        def _record_component(name: str, value):
            nonlocal guard_triggered
            if value is None:
                return
            if isinstance(value, torch.Tensor):
                value, guarded = self._sanitize_loss(value, name)
                guard_triggered = guard_triggered or guarded
                scalar = float(value.detach().cpu())
            else:
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    return
            loss_components[name] = scalar

        _record_component("loss_mlm", _extract_output(base_outputs, "masked_lm_loss"))
        _record_component("loss_simcse", _extract_output(base_outputs, "contrastive_learning_loss"))

        # xspec のためのスナップショット
        orig_input_ids = inputs["input_ids"].detach().clone()
        orig_attn = inputs.get("attention_mask", None)
        if orig_attn is not None: orig_attn = orig_attn.detach().clone()

        device = orig_input_ids.device
        base_model = _unwrap_model(model)
        pad_id = getattr(base_model.config, "pad_token_id", 0)
        base_encoder = _get_base_encoder(base_model)
        if not getattr(base_model.config, "output_hidden_states", False):
            base_model.config.output_hidden_states = True

        loss_cross = None
        valid_positions_count = 0

        lbl = inputs.get(self._label_field, None)
        if self._xspec_enable and (self._projector is not None) and (self._lambda_xspec > 0.0) and (lbl is not None):
            species = self._labels_to_species(lbl)
            self._xspec_steps += 1
            if random.random() <= self._xspec_rate:
                # 投影 ID を構築
                proj_ids_list, valid_mask_list = [], []
                B = orig_input_ids.size(0)
                for i in range(B):
                    p_ids, use, _ = self._projector.project_one(orig_input_ids[i], species[i])
                    proj_ids_list.append(p_ids.unsqueeze(0))
                    valid_mask_list.append(use)
                proj_ids = torch.cat(proj_ids_list, dim=0).to(device).contiguous()
                valid_mask_batch = torch.tensor(valid_mask_list, dtype=torch.bool, device=device)

                exec_ctx = torch.enable_grad if self._xspec_requires_grad else torch.no_grad
                with exec_ctx():
                    attn_anchor = (orig_attn if orig_attn is not None else (orig_input_ids != pad_id)).to(device).clone()
                    out_anchor = base_encoder(input_ids=orig_input_ids,
                                             attention_mask=attn_anchor,
                                             output_hidden_states=True, return_dict=True)
                    anchor_rep = self._cls(out_anchor)

                    attn_proj = (proj_ids != pad_id).to(device).clone()
                    out_proj = base_encoder(input_ids=proj_ids,
                                            attention_mask=attn_proj,
                                            output_hidden_states=True, return_dict=True)
                    proj_rep = self._cls(out_proj)

                idx = valid_mask_batch.nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    raw_loss_cross = info_nce(anchor_rep[idx], proj_rep[idx], temperature=self._tau_cross)
                    loss_cross, cross_guarded = self._sanitize_loss(raw_loss_cross, "loss_xspec")
                    guard_triggered = guard_triggered or cross_guarded
                    if not cross_guarded:
                        self._xspec_applied += 1
                        valid_positions_count = int((proj_ids != pad_id).sum().item())
                    else:
                        loss_cross = None

        if loss_cross is not None:
            loss_xspec_term = self._lambda_xspec * loss_cross
        else:
            loss_xspec_term = torch.zeros_like(loss_base)

        loss_total = loss_base + loss_xspec_term
        loss_total, total_guarded = self._sanitize_loss(loss_total, "loss_total")
        guard_triggered = guard_triggered or total_guarded

        # ログ（steps ごと）
        if hasattr(self, "state") and (self.state.global_step % max(1, self.args.logging_steps) == 0):
            # 有効位置の指数移動平均（粗めの健全性指標）
            if valid_positions_count > 0:
                alpha = 0.1
                self._xspec_valid_positions_ema = (1 - alpha) * self._xspec_valid_positions_ema + alpha * valid_positions_count
            xspec_rate_obs = (self._xspec_applied / self._xspec_steps) if self._xspec_steps > 0 else 0.0
            payload = {
                "loss_total": float(loss_total.detach().cpu()),
                "loss_base": float(loss_base.detach().cpu()),
                "loss_xspec": float(loss_xspec_term.detach().cpu()) if loss_cross is not None else 0.0,
                "xspec_applied_rate": xspec_rate_obs,
                "xspec_valid_positions_ema": float(self._xspec_valid_positions_ema),
            }
            if loss_components:
                payload.update(loss_components)
            if guard_triggered:
                payload["loss_nan_guard_count"] = int(self._nan_guard_counter)
            self.log(payload)

        return (loss_total, None) if return_outputs else loss_total


# ─────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────

def _resolve_deepspeed_cfg(path: Optional[str]) -> Optional[str]:
    if not path: return None
    maybe = os.path.abspath(path)
    if os.path.isfile(maybe): return maybe
    print(f"[warn] DeepSpeed config not found: {path} → {maybe}．DeepSpeed を無効化します．"); return None

def main():
    ap = argparse.ArgumentParser(description="Pretrain Mix-Geneformer w/ cross-species loss (label=mouse/human)")

    # データ
    ap.add_argument("--run_dir", type=str, default=None)
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--token_dictionary_path", type=str, required=True)
    ap.add_argument("--resume_from_checkpoint", type=str, default=None)
    ap.add_argument("--example_lengths_file", type=str, default=None)
    ap.add_argument("--pretext_task", type=str, default="MLM-SimCSE",
                    choices=sorted(PRETEXT_MODEL_REGISTRY.keys()))

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
    ap.add_argument("--xspec_requires_grad", action="store_true")
    ap.add_argument("--label_field", type=str, default="label")

    # ダンプ（投影の前後）
    ap.add_argument("--xspec_dump_examples", type=int, default=0,
                    help="投影の前後を N サンプル表示（rank0のみ）．0 で無効")
    ap.add_argument("--xspec_dump_max_len", type=int, default=64,
                    help="表示トークン列の最大長")
    ap.add_argument("--xspec_dump_seed", type=int, default=42,
                    help="ダンプ時の乱数シード")

    # 分散/DeepSpeed
    ap.add_argument("--deepspeed", type=str, default=None)
    ap.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    ap.add_argument("--no_deepspeed", action="store_true")
    ap.add_argument("--local_rank", type=int, default=-1)

    # デバッグ用サブセット
    ap.add_argument("--debug_n", type=int, default=0)
    ap.add_argument("--debug_fraction", type=float, default=0.0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--tf32", action="store_true")

    # 未知引数も許容
    args, _ = ap.parse_known_args()

    # 分散初期化
    local_rank, world_size = maybe_init_distributed(backend=args.backend)
    _dist_inited = (world_size > 1)

    # 乱数初期化
    base_seed = 1234
    torch.manual_seed(base_seed + (dist.get_rank() if is_dist_avail_and_initialized() else 0))
    np.random.seed(base_seed); random.seed(base_seed)

    if args.tf32:
        try:
            import torch.backends.cuda as cuda_backends
            cuda_backends.matmul.allow_tf32 = True
            cuda_backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    try:
        # 出力ディレクトリ
        if args.run_dir:
            run_root = args.run_dir
        else:
            tz = pytz.timezone("Asia/Tokyo"); now = datetime.datetime.now(tz)
            run_name = now.strftime("%y%m%d_%H%M%S") + "_mix-geneformer_xspec_label"
            if is_dist_avail_and_initialized():
                obj = [run_name] if is_main_process() else [None]
                dist.broadcast_object_list(obj, src=0); run_name = obj[0]
            run_root = os.path.join("/work/mouse-Geneformer++", "models", run_name)

        training_output_dir = os.path.join(run_root, "training")
        logging_dir = os.path.join(run_root, "runs")
        model_output_dir = os.path.join(training_output_dir, "models")

        if is_main_process():
            os.makedirs(training_output_dir, exist_ok=True)
            os.makedirs(logging_dir, exist_ok=True)
            os.makedirs(model_output_dir, exist_ok=True)
        barrier()

        # ── 前処理 ──
        try:
            train_ds: Union[Dataset, DatasetDict] = load_from_disk(args.dataset_path)
            train_ds = _apply_debug_subset(train_ds, args.debug_n, args.debug_fraction)

            with open(args.token_dictionary_path, "rb") as f:
                token_dict = pickle.load(f)

            projector = None
            if args.xspec_enable:
                if not args.correspondence_tsv:
                    raise ValueError("--xspec_enable のときは --correspondence_tsv を指定してください．")
                gene2id_human, gene2id_mouse, id2species, pad_id = _build_gene_id_tables(token_dict)
                m2h, h2m, stats = load_ortholog_maps_by_token(
                    args.correspondence_tsv, gene2id_mouse, gene2id_human, len(token_dict)
                )
                if is_main_process():
                    print(f"[xspec] ortholog kept={stats['pairs_kept']:,}, miss_mouse={stats['miss_mouse']:,}, miss_human={stats['miss_human']:,}")
                projector = CrossSpeciesProjector(m2h, h2m, id2species=id2species, pad_token_id=pad_id)

            # ラベル正規化
            train_ds = _maybe_normalize_labels(train_ds, args.label_field)

            # lengths 保存
            if args.debug_n > 0 or (args.debug_fraction and args.debug_fraction > 0.0):
                lengths_path = None
            else:
                lengths_path = args.example_lengths_file
            if lengths_path and os.path.exists(lengths_path):
                try:
                    with open(lengths_path, "rb") as f: lengths = pickle.load(f)
                    ds_for_len = train_ds["train"] if isinstance(train_ds, DatasetDict) and "train" in train_ds else train_ds
                    if len(lengths) != len(ds_for_len):
                        if is_main_process(): print(f"[warn] lengths({len(lengths):,}) とデータ({len(ds_for_len):,}) が不一致のため再計算します．")
                        lengths_path = None
                except Exception:
                    lengths_path = None
            if not lengths_path:
                ds_for_len = train_ds["train"] if isinstance(train_ds, DatasetDict) and "train" in train_ds else train_ds
                lengths = _compute_lengths(ds_for_len)
                lengths_path = os.path.join(training_output_dir, "computed_lengths.pkl")
                if is_main_process():
                    with open(lengths_path, "wb") as f: pickle.dump(lengths, f)
                    print(f"[lengths] wrote: {lengths_path} ({len(lengths):,} samples)")
                barrier()

            # 投影の前後ダンプ
            if is_main_process() and args.xspec_enable and projector is not None and args.xspec_dump_examples > 0:
                try:
                    dump_xspec_examples(train_ds, projector, token_dict, args.label_field,
                                        n_samples=args.xspec_dump_examples,
                                        max_len=args.xspec_dump_max_len,
                                        seed=args.xspec_dump_seed)
                except Exception as e:
                    print("[xspec-dump] failed:", repr(e))

            prep_ok = True
        except Exception as e:
            prep_ok = False
            if is_main_process(): print("[fatal] preprocessing failed:", repr(e))

        if is_dist_avail_and_initialized():
            prep_tensor = torch.tensor([1 if prep_ok else 0], device=torch.device("cuda", torch.cuda.current_device()))
            dist.all_reduce(prep_tensor, op=dist.ReduceOp.MIN); prep_ok = bool(prep_tensor.item() == 1)
        if not prep_ok: raise SystemExit(1)

        # モデル設定（軽量デフォルト）
        task = args.pretext_task
        num_layers = 6; num_embed_dim = 256
        intermed_size = num_embed_dim * 2
        cfg_hf = dict(
            hidden_size=num_embed_dim,
            num_hidden_layers=num_layers,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1,
            intermediate_size=intermed_size,
            hidden_act="silu",
            max_position_embeddings=2**12,
            model_type="bert",
            num_attention_heads=4,
            pad_token_id=token_dict.get("<pad>", 0),
            vocab_size=len(token_dict),
        )
        model_config = BertConfig(**cfg_hf)
        model = _load_model_for_pretext(task, model_config, args.use_pretrained, args.pretrained_path)
        if is_main_process():
            print(f"[pretext] task={task}, model={model.__class__.__name__}")

        # DeepSpeed／AMP
        ds_cfg = None if args.no_deepspeed else _resolve_deepspeed_cfg(args.deepspeed)
        if args.fp16 and args.bf16: raise ValueError("fp16 と bf16 は同時に有効化できません．")

        training_args = TrainingArguments(
            output_dir=training_output_dir,
            logging_dir=logging_dir,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.num_train_epochs,
            lr_scheduler_type="cosine",
            warmup_steps=int(args.warmup_steps),
            weight_decay=args.weight_decay,
            group_by_length=True,
            length_column_name="length",
            disable_tqdm=False,
            save_strategy="epoch",
            logging_steps=5,
            dataloader_drop_last=False,
            optim="adamw_torch",
            max_grad_norm=1.0,
            report_to="none",
            deepspeed=ds_cfg,
            remove_unused_columns=True,
            label_names=[args.label_field],
            fp16=bool(args.fp16) if ds_cfg is None else False,
            bf16=bool(args.bf16) if ds_cfg is None else False,
        )

        # Trainer 選択
        trainer_cls = GeneformerPretrainerXSpec if args.xspec_enable else GeneformerPretrainer
        common_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            example_lengths_file=lengths_path,
            token_dictionary=token_dict,
            pretext_task=task,
        )
        xspec_kwargs = dict(
            projector=projector,
            label_field=args.label_field,
            xspec_enable=True,
            xspec_rate=args.xspec_rate,
            tau_cross=args.tau_cross,
            lambda_xspec=args.lambda_xspec,
            xspec_requires_grad=args.xspec_requires_grad,
        ) if args.xspec_enable else {}
        trainer = trainer_cls(**common_kwargs, **xspec_kwargs)

        if is_main_process():
            ds_for_len = train_ds["train"] if isinstance(train_ds, DatasetDict) and "train" in train_ds else train_ds
            print(f"[sanity] world_size={world_size}, dataset_len={len(ds_for_len):,}, batch_size(per_device)={args.batch_size}")
            print(f"[sanity] group_by_length={training_args.group_by_length}, drop_last={training_args.dataloader_drop_last}")

        last_ckpt = get_last_checkpoint(training_output_dir)
        ckpt = args.resume_from_checkpoint or last_ckpt

        if ckpt:
            if is_main_process(): print(f"Resuming from checkpoint: {ckpt}")
            trainer.train(resume_from_checkpoint=ckpt)
        else:
            if is_main_process(): print("Starting training from scratch")
            trainer.train()

        trainer.save_model(os.path.join(training_output_dir, "models"))
        if is_main_process(): print(f"Saved model to {os.path.join(training_output_dir, 'models')}")
    finally:
        if _dist_inited:
            try:
                barrier(); dist.destroy_process_group()
            except Exception:
                pass

if __name__ == "__main__":
    hf_logging.set_verbosity_info()
    main()

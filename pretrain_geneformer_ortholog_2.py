#!/usr/bin/env python
# coding: utf-8
"""
pretrain_geneformer_xspec.py (stable debug-ready)

✅ 追加 forward は「純粋なエンコード」のみ（labels も loss 分岐も通さない）
✅ 入力は必ず clone() 済みスナップショットを使用（親の in-place と分離）
✅ model.config 参照は unwrap 後に実施（DDP/DeepSpeed を剥いでから）
✅ computed_lengths.pkl は run_dir/training/ に固定保存し、barrier 同期
✅ length 列は remove_unused_columns=True でモデルに渡さない
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
from typing import Optional, Dict, Any

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
    logging as hf_logging,
)
from transformers.trainer_utils import get_last_checkpoint

# 既存依存
from geneformer import GeneformerPretrainer
from SimCSE_util import BertForMaskedMLandSimCSE


# ─────────────────────────────────────────────────────────────
# xspec ユーティリティ
# ─────────────────────────────────────────────────────────────

def _unwrap_model(m):
    """DDP / DeepSpeed / FSDP などのラッパから実モデルを取り出す簡易版．"""
    try:
        from accelerate.utils import extract_model_from_parallel
        return extract_model_from_parallel(m)
    except Exception:
        pass
    seen = set()
    while hasattr(m, "module") and m not in seen:
        seen.add(m)
        m = m.module
    return m

def _get_base_encoder(unwrapped):
    """
    追加 forward を純エンコードだけで通すためのエンコーダ取り出し．
    - HF BERT系: `unwrapped.bert` が BertModel
    - その他: backbone/encoder/transformer/model の順に探索
    - 無ければ unwrapped 自身（※labels は渡さない）
    """
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
            gene2id_human[tok] = tid_int
            id2species[tid_int] = 1
        elif tok.startswith("ENSMUSG"):
            gene2id_mouse[tok] = tid_int
            id2species[tid_int] = 2
    return gene2id_human, gene2id_mouse, id2species, pad_id

def load_ortholog_maps_by_token(path_tsv: str, gene2id_mouse: dict, gene2id_human: dict, vocab_size: int):
    import pandas as pd
    df = pd.read_csv(path_tsv, sep="\t")
    if not {"mouse_gene", "human_gene"}.issubset(df.columns):
        raise ValueError("correspondence.tsv は 'mouse_gene' と 'human_gene' 列が必要です．")
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
# 分散 & 前処理ユーティリティ
# ─────────────────────────────────────────────────────────────

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return (not is_dist_avail_and_initialized()) or dist.get_rank() == 0

def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()

def maybe_init_distributed(backend: str = "nccl", timeout_sec: int = 900):
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

# Trainer.save_model を safe_serialization=False に統一
def _save_model_no_safe(self, output_dir=None, _internal_call=False):
    output_dir = output_dir or self.args.output_dir
    self.model.save_pretrained(output_dir, safe_serialization=False)
    if getattr(self, "tokenizer", None):
        self.tokenizer.save_pretrained(output_dir)
Trainer.save_model = _save_model_no_safe


# ─────────────────────────────────────────────────────────────
# Pretrainer 拡張（追加 forward は純エンコードのみ）
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
            raise ValueError("未知の label フォーマットです．")
        return species_list

    @staticmethod
    def _cls(output_obj):
        """モデル出力から CLS 表現を取り出す（pooler/last_hidden_state/hidden_states の順で探索）"""
        po = getattr(output_obj, "pooler_output", None)
        if po is not None:
            return po
        lho = getattr(output_obj, "last_hidden_state", None)
        if lho is not None:
            return lho[:, 0]
        hs = getattr(output_obj, "hidden_states", None)
        if hs is not None and len(hs) > 0:
            return hs[-1][:, 0]
        if isinstance(output_obj, dict):
            if "pooler_output" in output_obj: return output_obj["pooler_output"]
            if "last_hidden_state" in output_obj: return output_obj["last_hidden_state"][:, 0]
            if "hidden_states" in output_obj and len(output_obj["hidden_states"]) > 0:
                return output_obj["hidden_states"][-1][:, 0]
        raise AttributeError("CLS representation not found in model output")

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        # 0) ラベル退避（追加 forward には渡さない）
        lbl = inputs.get(self._label_field, None)

        # 親に渡す引数は shallow copy ではなくテンソルも分離
        inputs_wo_label = {}
        for k, v in inputs.items():
            if k == self._label_field:
                continue
            if torch.is_tensor(v):
                inputs_wo_label[k] = v.detach().clone()
            else:
                inputs_wo_label[k] = v

        # 1) まず親（MLM+SimCSE）の損失を先に計算（親側の in-place を先に消化）
        base_loss = super().compute_loss(model, inputs_wo_label, return_outputs=False)

        # 2) 種間ビューを作るためのスナップショット
        orig_input_ids = inputs["input_ids"].detach().clone()
        orig_attn = inputs.get("attention_mask", None)
        if orig_attn is not None:
            orig_attn = orig_attn.detach().clone()

        device = orig_input_ids.device
        base_model = _unwrap_model(model)
        pad_id = getattr(base_model.config, "pad_token_id", 0)
        base_encoder = _get_base_encoder(base_model)
        if not getattr(base_model.config, "output_hidden_states", False):
            base_model.config.output_hidden_states = True

        # 3) cross-species の純エンコード（必要時のみ）
        loss_cross = None
        if self._xspec_enable and (self._projector is not None) and (self._lambda_xspec > 0.0) and (lbl is not None):
            species = self._labels_to_species(lbl)
            if random.random() <= self._xspec_rate:
                # 投影 ID を構築
                proj_ids_list, valid_mask_list = [], []
                B = orig_input_ids.size(0)
                for i in range(B):
                    p_ids, use, _ = self._projector.project_one(orig_input_ids[i], species[i])
                    proj_ids_list.append(p_ids.unsqueeze(0))
                    valid_mask_list.append(use)
                proj_ids = torch.cat(proj_ids_list, dim=0).to(device).contiguous()
                valid_mask = torch.tensor(valid_mask_list, dtype=torch.bool, device=device)

                # 勾配を通す/通さないのスイッチ
                exec_ctx = torch.enable_grad if self._xspec_requires_grad else torch.no_grad
                with exec_ctx():
                    attn_anchor = (orig_attn if orig_attn is not None else (orig_input_ids != pad_id)).to(device).clone()
                    out_anchor = base_encoder(
                        input_ids=orig_input_ids,
                        attention_mask=attn_anchor,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    anchor_rep = self._cls(out_anchor)

                    attn_proj = (proj_ids != pad_id).to(device).clone()
                    out_proj = base_encoder(
                        input_ids=proj_ids,
                        attention_mask=attn_proj,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    proj_rep = self._cls(out_proj)

                idx = valid_mask.nonzero(as_tuple=True)[0]
                if idx.numel() > 0:
                    loss_cross = info_nce(anchor_rep[idx], proj_rep[idx], temperature=self._tau_cross)

        # 4) 合算
        if loss_cross is not None:
            base_loss = base_loss + self._lambda_xspec * loss_cross

        return base_loss


# ─────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────

def _resolve_deepspeed_cfg(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    maybe = os.path.abspath(path)
    if os.path.isfile(maybe):
        return maybe
    print(f"[warn] DeepSpeed config not found: {path} → {maybe}．DeepSpeed を無効化します．")
    return None

def main():
    ap = argparse.ArgumentParser(description="Pretrain Mix-Geneformer w/ cross-species loss (label=mouse/human)")
    # データ
    ap.add_argument("--run_dir", type=str, default=None, help="出力ルート（全rank共通）．未指定なら時刻で作成")
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--token_dictionary_path", type=str, required=True)
    ap.add_argument("--resume_from_checkpoint", type=str, default=None)
    ap.add_argument("--example_lengths_file", type=str, default=None)

    # 学習設定
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--warmup_steps", type=int, default=1_000)
    ap.add_argument("--weight_decay", type=float, default=0.001)

    # モデルロード
    ap.add_argument("--use_pretrained", action="store_true")
    ap.add_argument("--pretrained_path", type=str, default="")

    # xspec（※このブロックは一意）
    ap.add_argument("--xspec_enable", action="store_true")
    ap.add_argument("--correspondence_tsv", type=str, default="")
    ap.add_argument("--xspec_rate", type=float, default=1.0)
    ap.add_argument("--lambda_xspec", type=float, default=0.3)
    ap.add_argument("--tau_cross", type=float, default=0.05)
    ap.add_argument("--xspec_requires_grad", action="store_true",
                    help="cross-species InfoNCE を encoder へ勾配伝播させる（デフォルトは値のみ合算）")
    ap.add_argument("--label_field", type=str, default="label")

    # 分散/DeepSpeed
    ap.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed 設定ファイル（絶対パス推奨）")
    ap.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    ap.add_argument("--no_deepspeed", action="store_true", help="単機・小規模デバッグ用に DeepSpeed を使わない")
    ap.add_argument("--local_rank", type=int, default=-1, help="Injected by DeepSpeed/torchrun; safe to ignore.")

    # デバッグ用サブセット
    ap.add_argument("--debug_n", type=int, default=0, help="各 split から最大 N サンプルのみ使用")
    ap.add_argument("--debug_fraction", type=float, default=0.0, help="各 split の先頭 frac を使用（0-1）")
    ap.add_argument("--fp16", action="store_true", help="AMP fp16 を有効化")
    ap.add_argument("--bf16", action="store_true", help="AMP bf16 を有効化（対応GPUのみ）")
    ap.add_argument("--tf32", action="store_true", help="TF32 を有効化（Ampere+）")

    # 未知引数が混ざっても落ちないように
    args, _ = ap.parse_known_args()

    # 分散初期化（必要時のみ）
    local_rank, world_size = maybe_init_distributed(backend=args.backend)
    _dist_inited = (world_size > 1)

    # 乱数初期化
    base_seed = 1234
    torch.manual_seed(base_seed + (dist.get_rank() if is_dist_avail_and_initialized() else 0))
    np.random.seed(base_seed)
    random.seed(base_seed)
    # TF32（行列演算の高速化）
    if args.tf32:
        try:
            import torch.backends.cuda as cuda_backends
            cuda_backends.matmul.allow_tf32 = True
            cuda_backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    try:
        # 出力ディレクトリ（全rank統一）
        if args.run_dir:
            run_root = args.run_dir
        else:
            tz = pytz.timezone("Asia/Tokyo")
            now = datetime.datetime.now(tz)
            run_name = now.strftime("%y%m%d_%H%M%S") + "_mix-geneformer_xspec_label"
            if is_dist_avail_and_initialized():
                obj = [run_name] if is_main_process() else [None]
                dist.broadcast_object_list(obj, src=0)
                run_name = obj[0]
            run_root = os.path.join("/work/mouse-Geneformer++", "models", run_name)

        training_output_dir = os.path.join(run_root, "training")
        logging_dir = os.path.join(run_root, "runs")
        model_output_dir = os.path.join(training_output_dir, "models")

        if is_main_process():
            os.makedirs(training_output_dir, exist_ok=True)
            os.makedirs(logging_dir, exist_ok=True)
            os.makedirs(model_output_dir, exist_ok=True)
        barrier()

        # ── 前処理（全 rank 同期） ──
        try:
            train_ds: Dataset | DatasetDict = load_from_disk(args.dataset_path)
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

            # ラベルを 0/1 に正規化（human=0, mouse=1）
            train_ds = _maybe_normalize_labels(train_ds, args.label_field)

            # lengths（computed_lengths.pkl を run_dir/training 下に固定保存）
            if args.debug_n > 0 or (args.debug_fraction and args.debug_fraction > 0.0):
                lengths_path = None
            else:
                lengths_path = args.example_lengths_file

            if lengths_path and os.path.exists(lengths_path):
                try:
                    with open(lengths_path, "rb") as f:
                        lengths = pickle.load(f)
                    ds_for_len = train_ds["train"] if isinstance(train_ds, DatasetDict) and "train" in train_ds else train_ds
                    if len(lengths) != len(ds_for_len):
                        if is_main_process():
                            print(f"[warn] lengths({len(lengths):,}) とデータ({len(ds_for_len):,}) が不一致のため再計算します．")
                        lengths_path = None
                except Exception:
                    lengths_path = None

            if not lengths_path:
                ds_for_len = train_ds["train"] if isinstance(train_ds, DatasetDict) and "train" in train_ds else train_ds
                lengths = _compute_lengths(ds_for_len)
                lengths_path = os.path.join(training_output_dir, "computed_lengths.pkl")
                if is_main_process():
                    with open(lengths_path, "wb") as f:
                        pickle.dump(lengths, f)
                    print(f"[lengths] wrote: {lengths_path} ({len(lengths):,} samples)")
                barrier()  # 書き出し直後に同期

            prep_ok = True
        except Exception as e:
            prep_ok = False
            if is_main_process():
                print("[fatal] preprocessing failed:", repr(e))

        if is_dist_avail_and_initialized():
            prep_tensor = torch.tensor([1 if prep_ok else 0], device=torch.device("cuda", torch.cuda.current_device()))
            dist.all_reduce(prep_tensor, op=dist.ReduceOp.MIN)
            prep_ok = bool(prep_tensor.item() == 1)
        if not prep_ok:
            raise SystemExit(1)

        # モデル設定（デフォルトは軽量 6layer/emb256）
        task = "MLM-SimCSE"
        num_layers = 6
        num_embed_dim = 256
        intermed_size = num_embed_dim * 2
        activ_fn = "silu"
        initializer_range = 0.02
        layer_norm_eps = 1e-12
        attention_probs_dropout_prob = 0.1
        hidden_dropout_prob = 0.1
        max_input_size = 2**12
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
                raise ValueError("--use_pretrained の場合は --pretrained_path を指定してください．")
            model = BertForMaskedMLandSimCSE.from_pretrained(args.pretrained_path)

        # DeepSpeed 設定
        ds_cfg = None if args.no_deepspeed else _resolve_deepspeed_cfg(args.deepspeed)
        if args.fp16 and args.bf16:
            raise ValueError("fp16 と bf16 は同時に有効化できません。どちらか一方にしてください。")
        # Trainer 設定（length 列は remove_unused_columns=True で自動除去）
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
            report_to="none",
            deepspeed=ds_cfg,
            remove_unused_columns=True,
            label_names=[args.label_field],
            fp16=bool(args.fp16) if ds_cfg is None else False,   # DeepSpeed cfg がある時はそちらに従う
            bf16=bool(args.bf16) if ds_cfg is None else False,
        )

        # Trainer クラスと引数
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
            if is_main_process():
                print(f"Resuming from checkpoint: {ckpt}")
            trainer.train(resume_from_checkpoint=ckpt)
        else:
            if is_main_process():
                print("Starting training from scratch")
            trainer.train()

        trainer.save_model(os.path.join(training_output_dir, "models"))
        if is_main_process():
            print(f"Saved model to {os.path.join(training_output_dir, 'models')}")
    finally:
        if _dist_inited:
            try:
                barrier()
                dist.destroy_process_group()
            except Exception:
                pass

if __name__ == "__main__":
    hf_logging.set_verbosity_info()
    main()

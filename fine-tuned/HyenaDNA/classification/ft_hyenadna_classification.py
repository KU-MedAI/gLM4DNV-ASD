import os
import json
import argparse
import random
import sys
import logging
import time
import traceback
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from peft import LoraConfig, get_peft_model, PeftModel

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_chrom(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("^chr", "", regex=True).str.strip()

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def infer_d_model(backbone: nn.Module) -> int:
    cfg = getattr(backbone, "config", None)
    for k in ["d_model", "hidden_size", "n_embd"]:
        if cfg is not None and hasattr(cfg, k):
            return int(getattr(cfg, k))
    emb_fn = getattr(backbone, "get_input_embeddings", None)
    if callable(emb_fn):
        w = emb_fn().weight
        return int(w.shape[1])
    raise RuntimeError("Cannot infer model hidden dimension.")

def str2bool(x: str) -> bool:
    s = str(x).strip().lower()
    if s in ["1", "true", "t", "yes", "y"]:
        return True
    if s in ["0", "false", "f", "no", "n"]:
        return False
    raise ValueError(f"Invalid boolean string: {x}")

def count_params(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)

def patch_model_forward_to_ignore_hf_kwargs(model: nn.Module) -> None:
    pop_keys = {
        "attention_mask",
        "token_type_ids",
        "output_attentions",
        "output_hidden_states",
        "return_dict",
        "head_mask",
        "inputs_embeds",
        "use_cache",
        "past_key_values",
        "labels",
        "num_items_in_batch",
    }

    def wrap(fn):
        def new_forward(*args, **kwargs):
            for k in list(kwargs.keys()):
                if k in pop_keys:
                    kwargs.pop(k, None)
            return fn(*args, **kwargs)
        return new_forward

    if hasattr(model, "forward"):
        model.forward = wrap(model.forward)

    for attr in ["base_model", "model"]:
        if hasattr(model, attr):
            try:
                inner = getattr(model, attr)
                if hasattr(inner, "forward"):
                    inner.forward = wrap(inner.forward)
            except Exception:
                pass

def save_pth(model: nn.Module, path: str, extra: Optional[Dict] = None) -> None:
    ensure_dir(os.path.dirname(path))
    payload = {"state_dict": model.state_dict()}
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)

def save_lora_adapter_if_any(backbone: nn.Module, out_dir: str) -> bool:
    ensure_dir(out_dir)
    if isinstance(backbone, PeftModel):
        backbone.save_pretrained(out_dir)
        return True
    return False

def log_mem(prefix: str = "") -> None:
    try:
        import psutil
        p = psutil.Process(os.getpid())
        rss_gb = p.memory_info().rss / (1024**3)
        logging.info("%s[MEM] RSS=%.3f GB", prefix, rss_gb)
    except Exception:
        pass

class TeeStdout:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

def setup_logging(log_dir: str) -> str:
    ensure_dir(log_dir)
    log_file = os.path.join(log_dir, "train.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    f = open(log_file, "a", buffering=1)
    sys.stdout = TeeStdout(sys.stdout, f)
    sys.stderr = TeeStdout(sys.stderr, f)
    logging.info("logging to: %s", log_file)
    return log_file

def compute_metrics_binary_from_logits(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    logits = logits.reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_true = labels.astype(np.int64, copy=False)
    y_prob = probs.astype(np.float64, copy=False)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))
    denom_val = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    denom = np.sqrt(denom_val) if denom_val > 0 else 0.0
    mcc = (((tp * tn) - (fp * fn)) / (denom + 1e-12)) if denom > 0 else 0.0
    auroc = float("nan")
    auprc = float("nan")
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if len(np.unique(y_true)) > 1:
            auroc = float(roc_auc_score(y_true, y_prob))
            auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pass
    return {
        "Accuracy": float(acc),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "F1": float(f1),
        "Precision": float(prec),
        "Recall": float(rec),
        "Specificity": float(spec),
        "MCC": float(mcc),
    }

class VariantDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, seq_col: str, label_col: str, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.seq_col = seq_col
        self.label_col = label_col
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq = str(row[self.seq_col])
        y = float(row[self.label_col])
        out = self.tokenizer(
            seq,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return {
            "input_ids": torch.tensor(out["input_ids"], dtype=torch.long),
            "labels": torch.tensor(y, dtype=torch.float32),
        }

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, w_pos: float, w_neg: float):
        super().__init__()
        self.w_pos = float(w_pos)
        self.w_neg = float(w_neg)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float()
        loss_raw = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none")
        sample_weight = labels * self.w_pos + (1.0 - labels) * self.w_neg
        return (loss_raw * sample_weight).mean()

class HyenaBinaryCLS(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        pooling: str = "mean",
        w_pos: float = 1.0,
        w_neg: float = 1.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling = str(pooling).lower().strip()
        if self.pooling not in {"mean", "last"}:
            raise ValueError(f"Invalid pooling: {pooling}")
        self.classifier = nn.Linear(int(d_model), 1)
        self.register_buffer("w_pos", torch.tensor(float(w_pos), dtype=torch.float32))
        self.register_buffer("w_neg", torch.tensor(float(w_neg), dtype=torch.float32))
        self.loss_fn = WeightedBCEWithLogitsLoss(w_pos=float(w_pos), w_neg=float(w_neg))
        self._hf_pop_keys = {
            "attention_mask", "token_type_ids", "output_attentions",
            "output_hidden_states", "return_dict", "head_mask",
            "inputs_embeds", "use_cache", "past_key_values",
            "labels", "num_items_in_batch",
        }

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs):
        for k in list(kwargs.keys()):
            if k in self._hf_pop_keys:
                kwargs.pop(k, None)
        out = self.backbone(input_ids=input_ids, **kwargs)
        hidden = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
        if self.pooling == "last":
            pooled = hidden[:, -1, :]
        else:
            pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        if labels is not None:
            labels = labels.view(-1).float()
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        fn = getattr(self.backbone, "gradient_checkpointing_enable", None)
        if callable(fn):
            try:
                if gradient_checkpointing_kwargs is not None:
                    fn(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                else:
                    fn()
            except TypeError:
                fn()

    def gradient_checkpointing_disable(self):
        fn = getattr(self.backbone, "gradient_checkpointing_disable", None)
        if callable(fn):
            fn()

def freeze_backbone_except_lora(backbone: nn.Module, keep_ln: bool = False) -> None:
    for name, p in backbone.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        elif keep_ln and any(k in name.lower() for k in ["ln", "layernorm", "norm"]):
            p.requires_grad = True
        else:
            p.requires_grad = False

def _resolve_seq_col(df: pd.DataFrame, span_bp: int) -> str:
    want = f"var_seq_{span_bp}bp"
    if want in df.columns:
        return want
    candidates = sorted([c for c in df.columns if c.startswith("var_seq_")])
    raise KeyError(f"Missing seq column: {want}")

def _standardize_split(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    if "split" not in df.columns:
        cand = None
        for c in df.columns:
            if "split" in c.strip().lower():
                cand = c
                break
        if cand is not None:
            df = df.rename(columns={cand: "split"})
    if "split" not in df.columns:
        raise KeyError(f"Missing column: split")
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annot_path", required=True)
    ap.add_argument("--span_bp", type=int, required=True)
    ap.add_argument("--label_col", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--output_subdir", required=True)
    ap.add_argument("--log_root", default="")
    ap.add_argument("--log_subdir", default="")
    ap.add_argument("--num_train_epochs", type=int, default=100)
    ap.add_argument("--per_device_train_batch_size", type=int, default=256)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=256)
    ap.add_argument("--learning_rate", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--eval_strategy", default="epoch", choices=["no", "steps", "epoch"])
    ap.add_argument("--save_strategy", default="epoch", choices=["no", "steps", "epoch"])
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--load_best_model_at_end", action="store_true")
    ap.add_argument("--metric_for_best_model", default="eval_loss")
    ap.add_argument("--greater_is_better", action="store_true")
    ap.add_argument("--overwrite_output_dir", action="store_true")
    ap.add_argument("--pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--max_length", type=int, default=0)
    ap.add_argument("--max_length_cap", type=int, default=0)
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--early_stopping_patience", type=int, default=10)
    ap.add_argument("--early_stopping_threshold", type=float, default=0.0)
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--remove_unused_columns", type=str, default="False")
    ap.add_argument("--dataloader_pin_memory", type=int, default=1)
    ap.add_argument("--dataloader_num_workers", type=int, default=4)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, default="")
    ap.add_argument("--keep_layernorm_trainable", type=int, default=0)
    ap.add_argument("--w_pos", type=float, default=0.0)
    ap.add_argument("--w_neg", type=float, default=0.0)
    ap.add_argument("--keep_minimal_columns", type=int, default=1)

    args = ap.parse_args()
    seed_everything(args.seed)

    if args.bf16 and args.fp16:
        raise ValueError("bf16 and fp16 cannot both be enabled.")

    out_dir = os.path.join(args.output_root, args.output_subdir)
    ensure_dir(out_dir)

    if args.log_root and args.log_subdir:
        log_dir = os.path.join(args.log_root, args.log_subdir)
    elif args.log_root and (not args.log_subdir):
        log_dir = os.path.join(args.log_root, args.output_subdir)
    else:
        log_dir = os.path.join(out_dir, "logs")

    ensure_dir(log_dir)
    log_file = setup_logging(log_dir)

    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    logging.info("out_dir = %s", out_dir)
    logging.info("log_dir = %s", log_dir)

    t0 = time.time()
    logging.info("loading annot_df: %s", args.annot_path)
    try:
        df = pd.read_parquet(args.annot_path)
        logging.info("df loaded: shape=%s", df.shape)
    except Exception as e:
        logging.error("read_parquet fail: %r", e)
        raise

    try:
        df.columns = [c.strip() for c in df.columns]
        df = _standardize_split(df)
        df["chrom"] = normalize_chrom(df["chrom"])
        df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
        df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
        seq_col = _resolve_seq_col(df, args.span_bp)
        df = df.dropna(subset=["chrom", "pos", "split", args.label_col, seq_col]).copy()
        df["pos"] = df["pos"].astype("int64")
        df[args.label_col] = df[args.label_col].astype(int)
        if int(args.keep_minimal_columns) == 1:
            keep_cols = [c for c in ["chrom", "pos", "ref", "alt", "split", args.label_col, seq_col] if c in df.columns]
            df = df[keep_cols].copy()
    except Exception as e:
        logging.error("normalize fail: %r", e)
        raise

    s = df["split"]
    train_df = df[s == "train"].copy()
    val_df = df[s == "val"].copy()
    test_df = df[s == "test"].copy()

    y = train_df[args.label_col].astype(int).values
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    N = pos + neg

    if args.w_pos > 0 and args.w_neg > 0:
        w_pos, w_neg = float(args.w_pos), float(args.w_neg)
    else:
        if pos == 0 or neg == 0:
            w_pos, w_neg = 1.0, 1.0
        else:
            w_pos, w_neg = float(N / (2.0 * pos)), float(N / (2.0 * neg))

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.max_length and args.max_length > 0:
        max_len = int(args.max_length)
    else:
        max_len = int(train_df[seq_col].astype(str).str.len().max())
    if args.max_length_cap and args.max_length_cap > 0:
        max_len = min(max_len, int(args.max_length_cap))

    train_ds = VariantDataset(train_df, tokenizer, seq_col, args.label_col, max_len)
    val_ds = VariantDataset(val_df, tokenizer, seq_col, args.label_col, max_len)
    test_ds = VariantDataset(test_df, tokenizer, seq_col, args.label_col, max_len)

    backbone = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True)

    if args.use_lora:
        targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
        alpha = int(args.lora_alpha) if args.lora_alpha > 0 else 2 * int(args.lora_r)
        lora_cfg = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=alpha,
            lora_dropout=float(args.lora_dropout),
            target_modules=targets,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        backbone = get_peft_model(backbone, lora_cfg)

    patch_model_forward_to_ignore_hf_kwargs(backbone)
    d_model = infer_d_model(backbone)
    model = HyenaBinaryCLS(backbone, d_model, pooling=args.pooling, w_pos=w_pos, w_neg=w_neg)

    if args.use_lora:
        keep_ln = bool(int(args.keep_layernorm_trainable))
        freeze_backbone_except_lora(model.backbone, keep_ln=keep_ln)

    callbacks = []
    if args.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=args.report_to if args.report_to != "none" else "none",
        run_name=args.run_name,
        remove_unused_columns=str2bool(args.remove_unused_columns),
        dataloader_pin_memory=bool(int(args.dataloader_pin_memory)),
        dataloader_num_workers=int(args.dataloader_num_workers),
        logging_dir=log_dir,
        save_safetensors=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_binary_from_logits,
        callbacks=callbacks,
    )

    trainer.train()

    best_pth = os.path.join(out_dir, "best.pth")
    save_pth(model, best_pth, extra={"best": True, "w_pos": float(w_pos), "w_neg": float(w_neg)})

    if args.use_lora:
        adir = os.path.join(out_dir, "adapter_best")
        save_lora_adapter_if_any(model.backbone, adir)

    tokenizer.save_pretrained(out_dir)
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    logging.info("Done: %s", out_dir)

if __name__ == "__main__":
    main()
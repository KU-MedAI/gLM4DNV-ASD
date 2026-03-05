#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, PeftModel

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def infer_d_model(backbone: nn.Module) -> int:
    cfg = getattr(backbone, "config", None)
    for k in ["d_model", "hidden_size", "n_embd"]:
        if cfg is not None and hasattr(cfg, k):
            return int(getattr(cfg, k))
    emb_fn = getattr(backbone, "get_input_embeddings", None)
    if callable(emb_fn):
        return int(emb_fn().weight.shape[1])
    raise RuntimeError("Cannot infer model hidden dimension.")

def str2bool(x: str) -> bool:
    s = str(x).strip().lower()
    if s in ["1", "true", "t", "yes", "y"]: return True
    if s in ["0", "false", "f", "no", "n"]: return False
    raise ValueError(f"Invalid boolean string: {x}")

def count_params(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)

def patch_model_forward_to_ignore_hf_kwargs(model: nn.Module) -> None:
    pop_keys = {
        "attention_mask", "token_type_ids", "output_attentions", "output_hidden_states",
        "return_dict", "head_mask", "inputs_embeds", "use_cache", "past_key_values",
        "labels", "num_items_in_batch",
    }
    def wrap(fn):
        def new_forward(*args, **kwargs):
            for k in list(kwargs.keys()):
                if k in pop_keys: kwargs.pop(k, None)
            return fn(*args, **kwargs)
        return new_forward
    if hasattr(model, "forward"): model.forward = wrap(model.forward)
    for attr in ["base_model", "model"]:
        if hasattr(model, attr):
            try:
                inner = getattr(model, attr)
                if hasattr(inner, "forward"): inner.forward = wrap(inner.forward)
            except Exception: pass

def save_pth(model: nn.Module, path: str, extra: Optional[Dict] = None) -> None:
    ensure_dir(os.path.dirname(path))
    payload = {"state_dict": model.state_dict()}
    if extra is not None: payload["extra"] = extra
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
    except Exception: pass

class TeeStdout:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass
    def isatty(self):
        for s in self.streams:
            try:
                if hasattr(s, "isatty") and s.isatty(): return True
            except Exception: pass
        return False
    def fileno(self):
        for s in self.streams:
            try:
                if hasattr(s, "fileno"): return s.fileno()
            except Exception: pass
        raise OSError("No fileno available")

def setup_logging(log_dir: str) -> str:
    ensure_dir(log_dir)
    log_file = os.path.join(log_dir, "train.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    f = open(log_file, "a", buffering=1)
    sys.stdout = TeeStdout(sys.stdout, f)
    sys.stderr = TeeStdout(sys.stderr, f)
    return log_file

def compute_metrics_regression_from_logits(eval_pred) -> Dict[str, float]:
    preds, labels = eval_pred
    if isinstance(preds, (tuple, list)): preds = preds[0]
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if len(preds) == 0: return {"MSE": float("nan")}
    diff = preds - labels
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    r2, pcc = float("nan"), float("nan")
    try:
        from sklearn.metrics import r2_score
        if len(labels) >= 2: r2 = float(r2_score(labels, preds))
        if len(labels) >= 2 and (np.std(labels) > 0) and (np.std(preds) > 0):
            pcc = float(np.corrcoef(labels, preds)[0, 1])
    except Exception: pass
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "PCC": pcc}

class NCRERegressionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, seq_col: str, label_col: str, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.seq_col = seq_col
        self.label_col = label_col
        self.max_len = int(max_len)
    def __len__(self) -> int: return len(self.df)
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        out = self.tokenizer(str(row[self.seq_col]), truncation=True, padding="max_length", 
                             max_length=self.max_len, return_attention_mask=False, return_token_type_ids=False)
        return {"input_ids": torch.tensor(out["input_ids"], dtype=torch.long),
                "labels": torch.tensor(float(row[self.label_col]), dtype=torch.float32)}

class HyenaRegression(nn.Module):
    def __init__(self, backbone: nn.Module, d_model: int, pooling: str = "mean"):
        super().__init__()
        self.backbone = backbone
        self.pooling = str(pooling).lower().strip()
        self.regressor = nn.Linear(int(d_model), 1)
        self.loss_fn = nn.MSELoss()
        self._hf_pop_keys = {"attention_mask", "token_type_ids", "output_attentions", "output_hidden_states",
                             "return_dict", "head_mask", "inputs_embeds", "use_cache", "past_key_values", "labels", "num_items_in_batch"}

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs):
        for k in list(kwargs.keys()):
            if k in self._hf_pop_keys: kwargs.pop(k, None)
        out = self.backbone(input_ids=input_ids, **kwargs)
        hidden = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
        pooled = hidden[:, -1, :] if self.pooling == "last" else hidden.mean(dim=1)
        pred = self.regressor(pooled).squeeze(-1)
        if labels is not None:
            loss = self.loss_fn(pred, labels.view(-1).float())
            return {"loss": loss, "logits": pred}
        return {"logits": pred}

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        fn = getattr(self.backbone, "gradient_checkpointing_enable", None)
        if callable(fn):
            try: fn(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            except TypeError: fn()

    def gradient_checkpointing_disable(self):
        fn = getattr(self.backbone, "gradient_checkpointing_disable", None)
        if callable(fn): fn()

def freeze_backbone_except_lora(backbone: nn.Module, keep_ln: bool = True) -> None:
    for name, p in backbone.named_parameters():
        if "lora_" in name: p.requires_grad = True
        elif keep_ln and any(k in name.lower() for k in ["ln", "layernorm", "norm"]): p.requires_grad = True
        else: p.requires_grad = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--seq_col", default="NCREs_seq")
    ap.add_argument("--label_col", default="activity_score")
    ap.add_argument("--split_col", default="split")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--output_subdir", required=True)
    ap.add_argument("--log_root", default="")
    ap.add_argument("--log_subdir", default="")
    ap.add_argument("--num_train_epochs", type=int, default=100)
    ap.add_argument("--per_device_train_batch_size", type=int, default=64)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--eval_strategy", default="epoch")
    ap.add_argument("--save_strategy", default="epoch")
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--load_best_model_at_end", action="store_true")
    ap.add_argument("--metric_for_best_model", default="eval_MSE")
    ap.add_argument("--greater_is_better", action="store_true")
    ap.add_argument("--overwrite_output_dir", action="store_true")
    ap.add_argument("--pooling", default="mean")
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
    ap.add_argument("--rmse_eps", type=float, default=1e-8)
    args = ap.parse_args()

    seed_everything(args.seed)
    out_dir = os.path.join(args.output_root, args.output_subdir)
    ensure_dir(out_dir)
    log_dir = os.path.join(args.log_root or out_dir, args.log_subdir or ("logs" if not args.log_root else ""))
    log_file = setup_logging(log_dir)

    with open(os.path.join(out_dir, "args.json"), "w") as f: json.dump(vars(args), f, indent=2)

    df = pd.read_parquet(args.data_path)
    df[args.split_col] = df[args.split_col].astype(str).str.lower().str.strip()
    df = df.dropna(subset=[args.seq_col, args.label_col, args.split_col]).copy()

    train_df, val_df, test_df = df[df[args.split_col] == "train"], df[df[args.split_col] == "val"], df[df[args.split_col] == "test"]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

    max_len = args.max_length if args.max_length > 0 else int(train_df[args.seq_col].astype(str).str.len().max())
    if args.max_length_cap > 0: max_len = min(max_len, args.max_length_cap)

    train_ds = NCRERegressionDataset(train_df, tokenizer, args.seq_col, args.label_col, max_len)
    val_ds = NCRERegressionDataset(val_df, tokenizer, args.seq_col, args.label_col, max_len)
    test_ds = NCRERegressionDataset(test_df, tokenizer, args.seq_col, args.label_col, max_len)

    backbone = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True)
    if args.use_lora:
        targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
        alpha = args.lora_alpha if args.lora_alpha > 0 else 2 * args.lora_r
        backbone = get_peft_model(backbone, LoraConfig(r=args.lora_r, lora_alpha=alpha, lora_dropout=args.lora_dropout,
                                                       target_modules=targets, bias="none", task_type="FEATURE_EXTRACTION"))
    
    patch_model_forward_to_ignore_hf_kwargs(backbone)
    model = HyenaRegression(backbone=backbone, d_model=infer_d_model(backbone), pooling=args.pooling)

    if args.use_lora:
        freeze_backbone_except_lora(model.backbone, keep_ln=bool(args.keep_layernorm_trainable))

    mfbm = str(args.metric_for_best_model).strip()
    greater_is_better = args.greater_is_better or any(x in mfbm.lower() for x in ["pcc", "r2"])

    training_args = TrainingArguments(
        output_dir=out_dir, overwrite_output_dir=args.overwrite_output_dir, num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size, per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps, eval_strategy=args.eval_strategy, save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit, load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=mfbm, greater_is_better=greater_is_better, bf16=args.bf16, fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing, report_to=args.report_to if args.report_to != "none" else "none",
        run_name=args.run_name, remove_unused_columns=str2bool(args.remove_unused_columns),
        dataloader_pin_memory=bool(args.dataloader_pin_memory), dataloader_num_workers=args.dataloader_num_workers, save_safetensors=False
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience, early_stopping_threshold=args.early_stopping_threshold)] if args.early_stopping else []
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics_regression_from_logits, callbacks=callbacks)

    trainer.train()
    save_pth(model, os.path.join(out_dir, "best.pth"), extra={"metric": mfbm})
    if args.use_lora: save_lora_adapter_if_any(model.backbone, os.path.join(out_dir, "adapter_best"))
    tokenizer.save_pretrained(out_dir)

    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    pd.DataFrame([{k: test_metrics.get(k) for k in ["test_MSE", "test_RMSE", "test_MAE", "test_R2", "test_PCC", "test_loss"]}]).to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)

    logging.info("Done: %s", out_dir)

if __name__ == "__main__":
    main()
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
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from peft import LoraConfig, get_peft_model, PeftModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_chrom(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("^chr", "", regex=True).str.strip()


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def str2bool(x: str) -> bool:
    s = str(x).strip().lower()
    if s in ["1", "true", "t", "yes", "y"]:
        return True
    if s in ["0", "false", "f", "no", "n"]:
        return False
    raise ValueError(f"Invalid boolean string: {x} (use True/False or 1/0)")


def count_params(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)


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


class PhyloGPNForSequenceClassification(nn.Module):
    def __init__(self, base_model, num_labels, w_pos=1.0, w_neg=1.0):
        super().__init__()
        self.phylogpn = base_model._model
        self.num_labels = num_labels
        self.config = base_model.config
        embedding_dim = base_model.config.outer_dim
        self.classifier = nn.Linear(embedding_dim, num_labels)
        self._use_gradient_checkpointing = False
        self.loss_fn = WeightedBCEWithLogitsLoss(w_pos, w_neg)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self._use_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._use_gradient_checkpointing = False

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.phylogpn.embedding(input_ids)
        x = x.transpose(1, 2)

        if self._use_gradient_checkpointing and self.training:
            x = self._forward_with_checkpointing(x)
        else:
            x = self.phylogpn.blocks(x)

        x = self.phylogpn.output_layers[0](x)
        x = x.transpose(1, 2)
        pooled_output = x.mean(dim=1)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.squeeze(), labels.squeeze())

        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(loss=loss, logits=logits)

    def _forward_with_checkpointing(self, x):
        import torch.utils.checkpoint as checkpoint
        num_blocks = len(self.phylogpn.blocks)
        chunk_size = 8

        for i in range(0, num_blocks, chunk_size):
            end_idx = min(i + chunk_size, num_blocks)
            chunk_blocks = nn.Sequential(*self.phylogpn.blocks[i:end_idx])
            x = checkpoint.checkpoint(chunk_blocks, x, use_reentrant=False)

        return x
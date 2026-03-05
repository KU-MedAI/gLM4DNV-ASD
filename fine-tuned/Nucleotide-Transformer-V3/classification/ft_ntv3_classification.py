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
    AutoConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)

from peft import LoraConfig, get_peft_model, PeftModel


# ============================================================
# Utils
# ============================================================
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

# ============================================================
# Pinpoint helpers (hang/OOM diagnosis)
# ============================================================
def log_mem(prefix: str = "") -> None:
    try:
        import psutil

        p = psutil.Process(os.getpid())
        rss_gb = p.memory_info().rss / (1024**3)
        logging.info("%s[MEM] RSS=%.3f GB", prefix, rss_gb)
    except Exception:
        pass

# ============================================================
# Logging (stdout + file)
# ============================================================
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

    # tee print() -> train.log
    f = open(log_file, "a", buffering=1)
    sys.stdout = TeeStdout(sys.stdout, f)
    sys.stderr = TeeStdout(sys.stderr, f)

    logging.info("logging to: %s", log_file)
    return log_file


# ============================================================
# Metrics - Binary Classification
# ============================================================
def compute_metrics_binary_from_logits(eval_pred) -> Dict[str, float]:
    """
    Binary classification metrics from logits
    - AUROC, AUPRC, Accuracy, F1, Precision, Recall, Specificity, MCC
    """
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    # Robust dtype/shape
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)

    # Flatten labels
    labels = labels.reshape(-1)

    # Flatten logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    logits = logits.reshape(-1)

    # Convert to probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))

    # Convert to int
    y_true = labels.astype(np.int64, copy=False)
    y_prob = probs.astype(np.float64, copy=False)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    # Confusion matrix
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    # Metrics
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))

    # MCC
    denom_val = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    denom = np.sqrt(denom_val) if denom_val > 0 else 0.0
    mcc = (((tp * tn) - (fp * fn)) / (denom + 1e-12)) if denom > 0 else 0.0

    # AUROC, AUPRC
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


# ============================================================
# Center Crop Helper
# ============================================================
def crop_center(x: torch.Tensor, keep_target_center_fraction: float = 0.375) -> torch.Tensor:
    """
    Crop the central sequence-length fraction
    
    Args:
        x: Input tensor of shape (..., seq_len, embed_dim)
        keep_target_center_fraction: Fraction of sequence to keep (default: 0.375)
    
    Returns:
        Cropped tensor of shape (..., cropped_seq_len, embed_dim)
    """
    seq_len = x.shape[-2]
    target_offset = int(seq_len * (1 - keep_target_center_fraction) // 2)
    target_length = seq_len - 2 * target_offset
    return x[..., target_offset:target_offset + target_length, :]


# ============================================================
# NTv3 Model Wrapper - Binary Classification with Center Crop
# ============================================================

class NTv3ForBinaryClassification(nn.Module):
    """
    NTv3 core model + center crop + max pooling + binary classification head
    """
    def __init__(
        self, 
        model_name: str, 
        species_str: str = "human",
        keep_target_center_fraction: float = 0.375,
    ):
        super().__init__()
        
        # Load base model config and model (exactly following fine-tuning example)
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        ntv3_base_model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            config=self.config,
        )

        # Extract the discrete conditioned model (i.e. remove the heads) for finetuning
        discrete_conditioned_model = type(ntv3_base_model.core).__bases__[0]
        self.core = discrete_conditioned_model(self.config)
        # Load pre-trained weights (strict=False because we don't load the heads)
        self.load_state_dict(ntv3_base_model.state_dict(), strict=False) 

        # Species setup (exactly following fine-tuning example)
        if species_str in self.config.species_to_token_id:
            species_ids = self.config.species_to_token_id[species_str]
            self.species_ids = torch.LongTensor([species_ids])
            print(f"Using species: {species_str} with ids: {self.species_ids}")
        else:
            # Mask token id
            print(f"{species_str} not in supported species, using mask token id")
            self.species_ids = torch.LongTensor([2])

        # Center crop fraction
        self.keep_target_center_fraction = float(keep_target_center_fraction)
        
        # Binary classification head
        self.layer_norm = nn.LayerNorm(self.config.embed_dim)
        self.head = nn.Linear(self.config.embed_dim, 1)  # Single logit output
        
        # Initialize head
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
        
        self.model_name = model_name
        
        logging.info(f"Center crop fraction: {self.keep_target_center_fraction}")
    
    def center_crop_max_pooling(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Center crop + max pooling
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
        
        Returns:
            pooled: (batch, hidden_size)
        """
        # 1. Crop to center fraction
        if self.keep_target_center_fraction < 1.0:
            hidden_states = crop_center(hidden_states, self.keep_target_center_fraction)
        
        # 2. Max pooling over cropped sequence dimension
        pooled, _ = torch.max(hidden_states, dim=1)  # (batch, hidden_size)
        
        return pooled
    
    def forward(self, input_ids, labels=None, **kwargs):
        # Prepare the species tokens (exactly following fine-tuning example)
        species_tokens = torch.repeat_interleave(self.species_ids, input_ids.shape[0])
        species_tokens = species_tokens.to(input_ids.device)

        # Forward through core (exactly following fine-tuning example)
        outputs = self.core(input_ids, [species_tokens], output_hidden_states=True)
        hidden_states = outputs["hidden_states"][-1]  # (batch, seq_len, hidden_size)
        
        # Center crop + max pooling
        pooled_output = self.center_crop_max_pooling(hidden_states)
        
        # Apply layer norm and classification head
        pooled_output = self.layer_norm(pooled_output)
        logits = self.head(pooled_output)  # (batch, 1)
        
        outputs_dict = {"logits": logits}
        
        if labels is not None:
            # Binary classification loss will be handled by WeightedTrainer
            # Here we just store logits for loss computation
            outputs_dict["labels"] = labels
        
        return outputs_dict


# ============================================================
# Dataset - No Padding, 128-Multiple Validation
# ============================================================

class NCREBinaryDataset(Dataset):
    """
    Binary classification dataset - NTv3 (No padding, 128-multiple validation)
    """
    def __init__(self, df, seq_col="NCREs_seq", label_col="label"):
        self.df = df.reset_index(drop=True)
        self.seq_col = seq_col
        self.label_col = label_col
        
        # Character-level tokenization mapping (following NTv3 vocab - huggingface / vocab.json)
        self.char_to_id = {
            "A": 6, "T": 7, "C": 8, "G": 9, "N": 10,
            # Special tokens
            "<unk>": 0, "<pad>": 1, "<mask>": 2, "<cls>": 3, "<eos>": 4, "<bos>": 5
        }
        
        # Validate all sequences are 128-multiple BEFORE training
        self._validate_sequence_lengths()
    
    def _validate_sequence_lengths(self):
        """Validate that ALL sequences are multiples of 128"""
        seq_lengths = self.df[self.seq_col].astype(str).str.len()
        
        # Check if all are multiples of 128
        non_128_multiple = seq_lengths[seq_lengths % 128 != 0]
        
        if len(non_128_multiple) > 0:
            # Get some examples of bad lengths
            bad_examples = non_128_multiple.head(10).tolist()
            raise ValueError(
                f"Found {len(non_128_multiple)} sequences that are NOT multiples of 128!\n"
                f"   Examples of invalid lengths: {bad_examples}\n"
                f"   All sequences must have length = 128 * N (e.g., 128, 256, 384, 512, ...)\n"
                f"   Please preprocess your data to pad/truncate to 128-multiple lengths."
            )
        
        # Log length distribution
        unique_lengths = seq_lengths.unique()
        logging.info(f"All sequences are 128-multiples. Unique lengths: {sorted(unique_lengths.tolist())}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seq = str(row[self.seq_col]).upper()
        y = int(row[self.label_col])  # 0 or 1
        
        # Character-level tokenization
        tokens = [self.char_to_id.get(char, 10) for char in seq]  # Unknown chars → N token (10)
        
        # Final safety check (should never fail if _validate_sequence_lengths passed)
        if len(tokens) % 128 != 0:
            raise ValueError(
                f"Sequence at index {idx} has length {len(tokens)}, which is not a multiple of 128!"
            )
        
        return {
            "input_ids": tokens,  # List of token ids (128-multiple length, no padding needed)
            "labels": y,          # Binary label (0 or 1)
        }


def static_collate_fn(batch):
    """
    Static collate function - NO PADDING
    All sequences must already be the same length (128-multiple)
    """
    # Extract sequences and labels
    input_ids_list = [item["input_ids"] for item in batch]
    labels_list = [item["labels"] for item in batch]
    
    # Verify all sequences have the same length
    seq_lengths = [len(seq) for seq in input_ids_list]
    if len(set(seq_lengths)) > 1:
        raise ValueError(
            f"Batch contains sequences of different lengths: {set(seq_lengths)}\n"
            f"   All sequences in a batch must have the same length.\n"
            f"   Please ensure your data is properly preprocessed to fixed lengths."
        )
    
    # Verify length is 128-multiple
    batch_length = seq_lengths[0]
    if batch_length % 128 != 0:
        raise ValueError(
            f"Batch sequence length {batch_length} is not a multiple of 128!"
        )
    
    # Convert to tensors (NO PADDING!)
    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.float32),
    }


# ============================================================
# Weighted BCEWithLogitsLoss
# ============================================================
class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, w_pos: float, w_neg: float):
        super().__init__()
        self.w_pos = float(w_pos)
        self.w_neg = float(w_neg)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.float()
        loss_raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        sample_weight = labels * self.w_pos + (1.0 - labels) * self.w_neg
        return (loss_raw * sample_weight).mean()

class WeightedTrainer(Trainer):
    def __init__(self, w_pos=1.0, w_neg=1.0, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = WeightedBCEWithLogitsLoss(w_pos, w_neg)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels").squeeze(-1)
        outputs = model(**inputs)
        logits = outputs.get("logits").squeeze(-1)
        
        loss = self.loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


# ============================================================
# Freeze helpers
# ============================================================
def freeze_backbone_except_lora(backbone: nn.Module, keep_ln: bool = False) -> None:
    """Freeze all parameters except LoRA layers"""
    for name, p in backbone.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        elif keep_ln and any(k in name.lower() for k in ["ln", "layernorm", "norm"]):
            p.requires_grad = True
        else:
            p.requires_grad = False


# ============================================================
# Split mapping helpers (NO MERGE, OOM-safe)
# ============================================================
def _find_col_case_insensitive(df: pd.DataFrame, name: str) -> Optional[str]:
    m = {c.strip().lower(): c for c in df.columns}
    return m.get(name.strip().lower(), None)


def _standardize_split_column(annot_df: pd.DataFrame, seq_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cand = _find_col_case_insensitive(annot_df, "split")
    if cand is None:
        for c in annot_df.columns:
            if "split" in c.strip().lower():
                cand = c
                break
    if cand is not None and cand != "split":
        annot_df = annot_df.rename(columns={cand: "split"})

    cand2 = _find_col_case_insensitive(seq_df, "split")
    if cand2 is None:
        for c in seq_df.columns:
            if "split" in c.strip().lower():
                cand2 = c
                break
    if cand2 is not None and cand2 != "split":
        seq_df = seq_df.rename(columns={cand2: "split"})

    return annot_df, seq_df


def _resolve_seq_col(df: pd.DataFrame, span_bp: int) -> str:
    want = f"var_seq_{span_bp}bp"
    if want in df.columns:
        return want
    candidates = sorted([c for c in df.columns if c.startswith("var_seq_")])
    raise KeyError(f"Missing seq column: {want} | candidates={candidates}")

# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    # Data arguments
    ap.add_argument("--annot_path", required=True)
    ap.add_argument("--seq_path", required=True)
    ap.add_argument("--span_bp", type=int, required=True)
    ap.add_argument("--label_col", required=True, help="Column name for binary label (0/1)")

    # Model arguments
    ap.add_argument("--checkpoint", required=True, help="Pretrained model checkpoint")
    ap.add_argument("--species", default="human", help="Species for conditioning (default: human)")
    ap.add_argument("--output_root", required=True, help="Root directory for outputs")
    ap.add_argument("--output_subdir", required=True, help="Subdirectory for this run")

    # Logging arguments
    ap.add_argument("--log_root", default="", help="Root directory for logs")
    ap.add_argument("--log_subdir", default="", help="Subdirectory for logs")

    # Training arguments
    ap.add_argument("--num_train_epochs", type=int, default=100)
    ap.add_argument("--per_device_train_batch_size", type=int, default=64)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=64)
    ap.add_argument("--learning_rate", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)

    # Mixed precision & optimization
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision")
    ap.add_argument("--fp16", action="store_true", help="Use float16 mixed precision")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")

    # Evaluation & saving
    ap.add_argument("--eval_strategy", default="epoch", choices=["no", "steps", "epoch"])
    ap.add_argument("--save_strategy", default="epoch", choices=["no", "steps", "epoch"])
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--load_best_model_at_end", action="store_true")
    ap.add_argument("--metric_for_best_model", default="eval_loss")
    ap.add_argument("--greater_is_better", action="store_true")
    ap.add_argument("--overwrite_output_dir", action="store_true")

    # Pooling
    ap.add_argument("--keep_target_center_fraction", type=float, default=0.375, 
                   help="Fraction of center sequence to use for pooling (default: 0.375)")

    # Early stopping
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--early_stopping_patience", type=int, default=10)
    ap.add_argument("--early_stopping_threshold", type=float, default=0.0)

    # Logging & monitoring
    ap.add_argument("--report_to", default="none", help="Reporting destination (wandb, tensorboard, none)")
    ap.add_argument("--run_name", default=None, help="Run name for logging")

    # Data loading
    ap.add_argument("--remove_unused_columns", type=str, default="False")
    ap.add_argument("--dataloader_pin_memory", type=int, default=1)
    ap.add_argument("--dataloader_num_workers", type=int, default=4)

    # LoRA arguments
    ap.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
    ap.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    ap.add_argument("--lora_alpha", type=int, default=0, help="LoRA alpha (0=auto:2*r)")
    ap.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    ap.add_argument("--lora_target_modules", type=str, default="", help="Comma-separated target modules")
    ap.add_argument("--keep_layernorm_trainable", type=int, default=0, help="Keep LayerNorm trainable with LoRA")

    # Loss weights
    ap.add_argument("--w_pos", type=float, default=0.0, help="Positive class weight (0=auto)")
    ap.add_argument("--w_neg", type=float, default=0.0, help="Negative class weight (0=auto)")

    args = ap.parse_args()
    seed_everything(args.seed)

    # Setup directories
    out_dir = os.path.join(args.output_root, args.output_subdir)
    ensure_dir(out_dir)

    # Determine log directory
    if args.log_root and args.log_subdir:
        log_dir = os.path.join(args.log_root, args.log_subdir)
    elif args.log_root and (not args.log_subdir):
        log_dir = os.path.join(args.log_root, args.output_subdir)
    else:
        log_dir = os.path.join(out_dir, "logs")

    ensure_dir(log_dir)
    log_file = setup_logging(log_dir)

    # Save arguments
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    logging.info("out_dir = %s", out_dir)
    logging.info("log_dir = %s", log_dir)

    # ============================================================
    # Data loading
    # ============================================================
    t0 = time.time()
    logging.info("=== [STEP] loading seq_df: %s", args.seq_path)
    log_mem("[BEFORE] ")
    try:
        seq_df = pd.read_parquet(args.seq_path)
        logging.info("=== [OK] seq_df loaded: shape=%s | elapsed=%.2fs", seq_df.shape, time.time() - t0)
        log_mem("[AFTER ] ")
    except Exception as e:
        logging.error("=== [FAIL] read_parquet(seq_path): %r", e)
        logging.error(traceback.format_exc())
        raise

    t1 = time.time()
    logging.info("=== [STEP] loading annot_df: %s", args.annot_path)
    log_mem("[BEFORE] ")
    try:
        annot_df = pd.read_feather(args.annot_path)
        logging.info("=== [OK] annot_df loaded: shape=%s | elapsed=%.2fs", annot_df.shape, time.time() - t1)
        log_mem("[AFTER ] ")
    except Exception as e:
        logging.error("=== [FAIL] read_feather(annot_path): %r", e)
        logging.error(traceback.format_exc())
        raise

    logging.info("=== [STEP] normalize/dtype optimize start")
    t2 = time.time()
    log_mem("[BEFORE] ")
    try:
        seq_df.columns = [c.strip() for c in seq_df.columns]
        annot_df.columns = [c.strip() for c in annot_df.columns]

        annot_df, seq_df = _standardize_split_column(annot_df, seq_df)

        seq_df["chrom"] = normalize_chrom(seq_df["chrom"])
        annot_df["chrom"] = normalize_chrom(annot_df["chrom"])

        seq_df["pos"] = pd.to_numeric(seq_df["pos"], errors="coerce")
        annot_df["pos"] = pd.to_numeric(annot_df["pos"], errors="coerce")

        seq_df = seq_df.dropna(subset=["chrom", "pos"]).copy()
        annot_df = annot_df.dropna(subset=["chrom", "pos", args.label_col]).copy()

        seq_df["chrom"] = seq_df["chrom"].astype("category")
        annot_df["chrom"] = annot_df["chrom"].astype("category")

        seq_df["pos"] = seq_df["pos"].astype("int64")
        annot_df["pos"] = annot_df["pos"].astype("int64")

        logging.info("=== [OK] normalize/dtype optimize done | elapsed=%.2fs", time.time() - t2)
        logging.info("    seq_df cols: %s", list(seq_df.columns))
        logging.info("    annot_df cols: %s", list(annot_df.columns))
        log_mem("[AFTER ] ")
    except Exception as e:
        logging.error("=== [FAIL] normalize/dtype optimize: %r", e)
        logging.error(traceback.format_exc())
        raise

    logging.info("=== [STEP] build label+split map start")
    t3 = time.time()
    log_mem("[BEFORE] ")
    try:
        has_ref_alt = ("ref" in annot_df.columns and "alt" in annot_df.columns and 
                      "ref" in seq_df.columns and "alt" in seq_df.columns)
        
        if has_ref_alt:
            # Use (chrom, pos, ref, alt) for variant identification
            variant_key_cols = ["chrom", "pos", "ref", "alt"]
            logging.info("Using variant key: (chrom, pos, ref, alt)")
        else:
            # Fallback to (chrom, pos) only
            variant_key_cols = ["chrom", "pos"]
            logging.info("Using variant key: (chrom, pos) only - ref/alt columns missing")
        
        # label map with duplicate checking
        annot_small = annot_df[variant_key_cols + [args.label_col]].copy()
        
        # Check for true duplicates (same variant key but different labels)
        dup_check = annot_small.groupby(variant_key_cols)[args.label_col].nunique()
        conflicting_variants = dup_check[dup_check > 1]
        
        if len(conflicting_variants) > 0:
            logging.error("Found %d variants with conflicting labels!", len(conflicting_variants))
            for variant_key, _ in conflicting_variants.head(5).items():
                if has_ref_alt:
                    chrom, pos, ref, alt = variant_key
                    sample_rows = annot_df[(annot_df['chrom'] == chrom) & 
                                         (annot_df['pos'] == pos) & 
                                         (annot_df['ref'] == ref) & 
                                         (annot_df['alt'] == alt)]
                    logging.error(f"  Conflicting variant: chr{chrom}:{pos} {ref}>{alt} -> labels: {sample_rows[args.label_col].unique().tolist()}")
                else:
                    chrom, pos = variant_key
                    sample_rows = annot_df[(annot_df['chrom'] == chrom) & (annot_df['pos'] == pos)]
                    logging.error(f"  Conflicting position: chr{chrom}:{pos} -> labels: {sample_rows[args.label_col].unique().tolist()}")
            raise ValueError(f"Found {len(conflicting_variants)} variants with conflicting labels. Please clean the annotation file first.")
        
        # Remove duplicates (same variant key AND same label)
        annot_small = annot_small.drop_duplicates()
        
        try:
            annot_small[args.label_col] = annot_small[args.label_col].astype("int8")
        except Exception:
            pass
        lab_map = annot_small.set_index(variant_key_cols)[args.label_col]

        split_map = None
        if "split" in annot_df.columns:
            annot_split_small = annot_df[variant_key_cols + ["split"]].drop_duplicates()
            annot_split_small["split"] = annot_split_small["split"].astype(str).str.lower().str.strip()
            split_map = annot_split_small.set_index(variant_key_cols)["split"]

        logging.info(
            "=== [OK] maps built: label_len=%d | split_map=%s | elapsed=%.2fs",
            int(len(lab_map)),
            ("YES(len=%d)" % len(split_map) if split_map is not None else "NO"),
            time.time() - t3,
        )
        log_mem("[AFTER ] ")
    except Exception as e:
        logging.error("=== [FAIL] build maps: %r", e)
        logging.error(traceback.format_exc())
        raise

    logging.info("=== [STEP] apply label/split map to seq_df start")
    t4 = time.time()
    log_mem("[BEFORE] ")
    try:
        seq_key = pd.MultiIndex.from_frame(seq_df[variant_key_cols])

        seq_df[args.label_col] = lab_map.reindex(seq_key).to_numpy()

        if "split" not in seq_df.columns:
            if split_map is None:
                raise KeyError(
                    "Missing column: split (train/val/test). Not found in seq_df nor annot_df.\n"
                    f"seq_df cols={seq_df.columns.tolist()}\n"
                    f"annot_df cols={annot_df.columns.tolist()}"
                )
            seq_df["split"] = split_map.reindex(seq_key).to_numpy()

        logging.info("=== [OK] maps applied | elapsed=%.2fs", time.time() - t4)
        log_mem("[AFTER ] ")
    except Exception as e:
        logging.error("=== [FAIL] apply maps: %r", e)
        logging.error(traceback.format_exc())
        raise

    logging.info("=== [STEP] filter labeled rows start")
    t5 = time.time()
    log_mem("[BEFORE] ")
    try:
        df = seq_df.dropna(subset=[args.label_col]).copy()
        df[args.label_col] = df[args.label_col].astype(int)

        # split normalize + missing drop
        if "split" not in df.columns:
            raise KeyError("Missing column: split (train/val/test)")
        df["split"] = df["split"].astype(str).str.lower().str.strip()

        logging.info("=== [OK] filter labeled rows done: shape=%s | elapsed=%.2fs", df.shape, time.time() - t5)
        log_mem("[AFTER ] ")
    except Exception as e:
        logging.error("=== [FAIL] filter/label cast: %r", e)
        logging.error(traceback.format_exc())
        raise

    seq_col = _resolve_seq_col(df, args.span_bp)
    df = df.dropna(subset=[seq_col]).copy()

    s = df["split"]
    train_df = df[s == "train"].copy()
    val_df = df[s == "val"].copy()
    test_df = df[s == "test"].copy()
    logging.info("Train=%d  Val=%d  Test=%d", len(train_df), len(val_df), len(test_df))

    # ============================================================
    # Calculate class weights (auto-balanced)
    # ============================================================
    y = train_df[args.label_col].values
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    N = pos + neg

    if args.w_pos > 0 and args.w_neg > 0:
        w_pos = float(args.w_pos)
        w_neg = float(args.w_neg)
        logging.info("Loss weights: user-specified | w_pos=%.6f w_neg=%.6f", w_pos, w_neg)
    else:
        if pos == 0 or neg == 0:
            w_pos = 1.0
            w_neg = 1.0
            logging.warning(
                "Train split has pos=%d, neg=%d (one class missing). Fallback w_pos=w_neg=1.0",
                pos, neg,
            )
        else:
            w_pos = float(N / (2.0 * pos))
            w_neg = float(N / (2.0 * neg))
        logging.info(
            "Loss weights: AUTO balanced | pos=%d neg=%d N=%d -> w_pos=N/(2*pos)=%.6f, w_neg=N/(2*neg)=%.6f",
            pos, neg, N, w_pos, w_neg,
        )

    # Log sequence length distribution
    train_lengths = train_df[seq_col].astype(str).str.len()
    logging.info("📏 Sequence length statistics:")
    logging.info("   Min: %d | Max: %d | Mean: %.1f | Median: %.1f", 
                 train_lengths.min(), train_lengths.max(), 
                 train_lengths.mean(), train_lengths.median())
    logging.info("   Unique lengths: %s", sorted(train_lengths.unique().tolist()))

    # ============================================================
    # Tokenizer setup (for compatibility - actual tokenization is manual)
    # ============================================================
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    logging.info("Tokenizer loaded (manual character-level tokenization will be used)")
    logging.info("NO PADDING - all sequences must be 128-multiples")

    # Create datasets (with 128-multiple validation)
    logging.info("Creating datasets with 128-multiple validation...")
    
    train_ds = NCREBinaryDataset(
        train_df,
        seq_col=seq_col,
        label_col=args.label_col,
    )

    val_ds = NCREBinaryDataset(
        val_df,
        seq_col=seq_col,
        label_col=args.label_col,
    )

    test_ds = NCREBinaryDataset(
        test_df,
        seq_col=seq_col,
        label_col=args.label_col,
    )

    # ============================================================
    # Model setup - NTv3 Binary Classification
    # ============================================================
    logging.info("Loading NTv3 model for binary classification: %s", args.checkpoint)
    model = NTv3ForBinaryClassification(
        model_name=args.checkpoint, 
        species_str=args.species,
        keep_target_center_fraction=args.keep_target_center_fraction,
    )
    
    logging.info(f"Model loaded: {args.checkpoint}")
    logging.info(f"Species: {args.species}")
    logging.info(f"Center crop fraction: {args.keep_target_center_fraction}")
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # LoRA setup
    if args.use_lora:
        targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
        if len(targets) == 0:
            # Default NTv3 transformer attention targets
            targets = ["sa_layer.query_head.linear", "sa_layer.value_head.linear"]
            logging.info(f"Using default NTv3 LoRA targets: {targets}")

        alpha = int(args.lora_alpha)
        if alpha <= 0:
            alpha = 2 * int(args.lora_r)

        lora_cfg = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules=targets,
            bias="none",
            task_type="FEATURE_EXTRACTION",  # Not SEQ_CLS since we have custom head
        )
        model = get_peft_model(model, lora_cfg)
        logging.info(
            f"LoRA enabled. targets={targets} | r={args.lora_r} | alpha={alpha} | drop={args.lora_dropout:.3f}"
        )

    if args.use_lora:
        model.print_trainable_parameters()
    else:
        tr, tot = count_params(model)
        logging.info(
            "trainable: %d || total: %d || trainable%%: %.2f%%",
            tr, tot, 100.0 * tr / max(1, tot)
        )


    # ============================================================
    # Training setup
    # ============================================================
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

    # Validation checks
    if args.early_stopping and (not args.load_best_model_at_end):
        raise ValueError("early_stopping requires --load_best_model_at_end")
    if args.load_best_model_at_end and (args.eval_strategy != args.save_strategy):
        raise ValueError(
            f"--load_best_model_at_end requires eval/save strategy match. "
            f"Got eval={args.eval_strategy}, save={args.save_strategy}"
        )

    # Trainer with weighted loss
    trainer = WeightedTrainer(
        w_pos=w_pos,
        w_neg=w_neg,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=static_collate_fn,  # No padding!
        compute_metrics=compute_metrics_binary_from_logits,
        callbacks=callbacks,
    )

    # ============================================================
    # Training
    # ============================================================
    logging.info("Starting training...")
    trainer.train()

    # Save best model
    best_pth = os.path.join(out_dir, "best.pth")
    save_pth(
        model, 
        best_pth, 
        extra={
            "best": True, 
            "w_pos": float(w_pos), 
            "w_neg": float(w_neg),
            "keep_target_center_fraction": float(args.keep_target_center_fraction),
        }
    )
    logging.info("Saved best.pth -> %s", best_pth)

    if args.use_lora:
        adir = os.path.join(out_dir, "adapter_best")
        if save_lora_adapter_if_any(model, adir):
            logging.info("Saved adapter_best -> %s", adir)

    # ============================================================
    # Test evaluation
    # ============================================================
    logging.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    # Save test metrics
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save test metrics CSV
    keys = [
        "test_Accuracy", "test_AUROC", "test_AUPRC", "test_F1",
        "test_Precision", "test_Recall", "test_Specificity", "test_MCC", "test_loss"
    ]
    row = {k: test_metrics.get(k, None) for k in keys}

    pd.DataFrame([row]).to_csv(
        os.path.join(out_dir, "test_metrics.csv"),
        index=False
    )

    # Log all test metrics
    logging.info("=" * 60)
    logging.info("FINAL TEST RESULTS")
    logging.info("=" * 60)
    for key in keys:
        value = test_metrics.get(key, None)
        if value is not None:
            if isinstance(value, float) and not np.isnan(value):
                logging.info(f"  {key:<20} = {value:.6f}")
            else:
                logging.info(f"  {key:<20} = {value}")
        else:
            logging.info(f"  {key:<20} = N/A")
    logging.info("=" * 60)

    # WandB logging
    try:
        import wandb
        if wandb.run is not None:
            # Hyperparameters
            wandb.log({
                "hp/learning_rate": args.learning_rate,
                "hp/weight_decay": args.weight_decay,
                "hp/bs_per_device": args.per_device_train_batch_size,
                "hp/warmup_steps": args.warmup_steps,
                "hp/lora_r": args.lora_r if args.use_lora else 0,
                "hp/lora_alpha_effective": (2 * args.lora_r if args.lora_alpha <= 0 else args.lora_alpha) if args.use_lora else 0,
                "hp/lora_dropout": args.lora_dropout if args.use_lora else 0,
                "hp/keep_ln": int(args.keep_layernorm_trainable),
                "hp/w_pos": float(w_pos),
                "hp/w_neg": float(w_neg),
                "hp/center_crop": float(args.keep_target_center_fraction),
                "hp/pin_memory": int(args.dataloader_pin_memory),
                "hp/num_workers": int(args.dataloader_num_workers),
            })
            # Test metrics
            wandb.log({
                "test/Accuracy": test_metrics.get("test_Accuracy", None),
                "test/AUROC": test_metrics.get("test_AUROC", None),
                "test/AUPRC": test_metrics.get("test_AUPRC", None),
                "test/F1": test_metrics.get("test_F1", None),
                "test/Precision": test_metrics.get("test_Precision", None),
                "test/Recall": test_metrics.get("test_Recall", None),
                "test/Specificity": test_metrics.get("test_Specificity", None),
                "test/MCC": test_metrics.get("test_MCC", None),
                "test/loss": test_metrics.get("test_loss", None),
            })
    except Exception:
        pass

    logging.info("Training completed!")
    logging.info("Output directory: %s", out_dir)
    logging.info("Test metrics saved: %s", os.path.join(out_dir, "test_metrics.csv"))
    logging.info("Log file: %s", log_file)


if __name__ == "__main__":
    main()
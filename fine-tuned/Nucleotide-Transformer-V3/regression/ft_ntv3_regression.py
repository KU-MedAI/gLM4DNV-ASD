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
# Metrics 
# ============================================================

def compute_metrics_regression(eval_pred) -> Dict[str, float]:
    """
    - MSE, RMSE, MAE, R², PCC, SCC
    """
    preds, labels = eval_pred
    
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    
    # Basic regression metrics
    mse = float(np.mean((preds - labels) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - labels)))
    
    # R² (coefficient of determination)
    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else float("nan")
    
    # Correlation metrics
    pcc = float("nan")
    scc = float("nan")
    try:
        from scipy.stats import pearsonr, spearmanr
        if len(np.unique(labels)) > 1:
            pcc = float(pearsonr(labels, preds)[0])
            scc = float(spearmanr(labels, preds)[0])
    except Exception:
        pass
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "PCC": pcc,
        "SCC": scc,
    }


# ============================================================
# NTv3 Model Wrapper
# ============================================================

class NTv3ForRegression(nn.Module):
    """
    NTv3 core model + masked mean pooling + regression head
    """
    def __init__(self, model_name: str, species_str: str = "human"):
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
        self.core = discrete_conditioned_model(self.config) # follows name convention
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

        # Regression head (NTv3 outputs at single-nucleotide resolution, then pooled)
        self.layer_norm = nn.LayerNorm(self.config.embed_dim)
        self.head = nn.Linear(self.config.embed_dim, 1)
        
        # Initialize head
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
        
        self.model_name = model_name
    
    def masked_max_pooling(self, hidden_states, attention_mask):
        """
        Masked max pooling: max over non-padded tokens only
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
        """
        expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        masked_hidden_states = hidden_states.masked_fill(expanded_mask == 0, -1e9)
        pooled, _ = torch.max(masked_hidden_states, dim=1)
        
        return pooled
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Prepare the species tokens (exactly following fine-tuning example)
        species_tokens = torch.repeat_interleave(self.species_ids, input_ids.shape[0])
        species_tokens = species_tokens.to(input_ids.device)

        # Forward through core (exactly following fine-tuning example)
        outputs = self.core(input_ids, [species_tokens], output_hidden_states=True)
        hidden_states = outputs["hidden_states"][-1]  # (batch, seq_len, hidden_size)
        
        # Create attention mask if not provided
        if attention_mask is None:
            # N token (10) is used for padding, so mask it out
            attention_mask = (input_ids != 10).long()
        
        # Masked mean pooling (our addition for sequence-level regression)
        pooled_output = self.masked_max_pooling(hidden_states, attention_mask)
        
        # Apply layer norm and regression head (following LinearHead pattern)
        pooled_output = self.layer_norm(pooled_output)
        logits = self.head(pooled_output)  # (batch, 1)
        # Note: No softplus for regression (unlike BigWig tracks which need positive values)
        
        outputs_dict = {"logits": logits}
        
        if labels is not None:
            # Regression loss (MSE)
            loss_fct = nn.MSELoss()
            labels = labels.view(-1, 1).float()
            logits = logits.view(-1, 1)
            loss = loss_fct(logits, labels)
            outputs_dict["loss"] = loss
        
        return outputs_dict


# ============================================================
# Dataset
# ============================================================

class NCRERegressionDataset(Dataset):
    def __init__(self, df, seq_col="NCREs_seq", label_col="activity_score", max_length_cap=None):
        self.df = df.reset_index(drop=True)
        self.seq_col = seq_col
        self.label_col = label_col
        self.max_length_cap = max_length_cap
        
        # Character-level tokenization mapping (following NTv3 vocab from fine-tuning example)
        self.char_to_id = {
            "A": 6, "T": 7, "C": 8, "G": 9, "N": 10,
            # Special tokens (not used but following vocab)
            "<unk>": 0, "<pad>": 1, "<mask>": 2, "<cls>": 3, "<eos>": 4, "<bos>": 5
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seq = str(row[self.seq_col]).upper()
        y = float(row[self.label_col])
        
        # Character-level tokenization (following fine-tuning example)
        tokens = [self.char_to_id.get(char, 10) for char in seq]  # Unknown chars → N token (10)
        
        # Apply max length cap if specified
        if self.max_length_cap and len(tokens) > self.max_length_cap:
            tokens = tokens[:self.max_length_cap]
        
        # Return raw tokens and labels - padding will be done in collate_fn
        return {
            "input_ids": tokens,  # List of token ids (variable length)
            "labels": y,          # Single float value
        }


def dynamic_collate_fn(batch):
    # Extract sequences and labels
    input_ids_list = [item["input_ids"] for item in batch]
    labels_list = [item["labels"] for item in batch]
    
    # Find max length in this batch
    max_len_in_batch = max(len(seq) for seq in input_ids_list)
    
    # Round up to nearest multiple of 128
    padded_length = ((max_len_in_batch + 127) // 128) * 128
    # print(f"[Batch] padded_length={padded_length}", flush=True)
    import wandb
    if wandb.run is not None:
        wandb.log({"batch/padded_length": padded_length})
    
    n_token_id = 10  # N token for padding
    
    # Pad sequences and create attention masks
    padded_input_ids = []
    attention_masks = []
    
    for tokens in input_ids_list:
        original_length = len(tokens)
        
        # Pad with N tokens
        padding_length = padded_length - original_length
        padded_tokens = tokens + [n_token_id] * padding_length
        
        # Create attention mask (1 for real tokens, 0 for N padding)
        attention_mask = [1] * original_length + [0] * padding_length
        
        padded_input_ids.append(padded_tokens)
        attention_masks.append(attention_mask)
    
    # Convert to tensors
    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.float32),
    }


# ============================================================
# Freeze helpers
# ============================================================
def freeze_backbone_except_lora(backbone: nn.Module, keep_ln: bool = False) -> None:
    for name, p in backbone.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
        elif keep_ln and any(k in name.lower() for k in ["ln", "layernorm", "norm"]):
            p.requires_grad = True
        else:
            p.requires_grad = False


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    # Data arguments
    ap.add_argument("--data_path", required=True, help="Path to parquet file with train/val/test splits")
    ap.add_argument("--seq_col", required=True, help="Column name for sequence data")
    ap.add_argument("--label_col", required=True, help="Column name for regression target")

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

    # Sequence length
    ap.add_argument("--max_length", type=int, default=0, help="Max sequence length (0=auto)")
    ap.add_argument("--max_length_cap", type=int, default=0, help="Cap for max length")

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
    logging.info("=== Loading NCRE dataset: %s", args.data_path)
    
    try:
        df = pd.read_parquet(args.data_path)
        logging.info("Data loaded: shape=%s", df.shape)
        logging.info("Columns: %s", list(df.columns))
    except Exception as e:
        logging.error("Failed to load data: %r", e)
        raise

    # Validate required columns
    seq_col = args.seq_col
    label_col = args.label_col
    required_cols = [seq_col, label_col, "split"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Normalize split column
    df["split"] = df["split"].astype(str).str.lower().str.strip()

    # Remove NaN rows
    df = df.dropna(subset=[seq_col, label_col, "split"]).copy()

    # Split data
    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()

    logging.info("Data splits: Train=%d  Val=%d  Test=%d", len(train_df), len(val_df), len(test_df))

    # Log sequence length distribution
    train_lengths = train_df[seq_col].astype(str).str.len()
    logging.info("📏 Sequence length statistics:")
    logging.info("   Min: %d | Max: %d | Mean: %.1f | Median: %.1f", 
                 train_lengths.min(), train_lengths.max(), 
                 train_lengths.mean(), train_lengths.median())
    logging.info("   25th percentile: %.1f | 75th percentile: %.1f | 95th percentile: %.1f",
                 train_lengths.quantile(0.25), train_lengths.quantile(0.75), train_lengths.quantile(0.95))

    # ============================================================
    # Tokenizer setup (for compatibility - actual tokenization is manual)
    # ============================================================
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    logging.info("Tokenizer loaded (manual character-level tokenization will be used)")
    logging.info("Tokenizer vocabulary size: %d", len(tokenizer))

    # Determine max length cap (for truncation only)
    max_length_cap = None
    if args.max_length and args.max_length > 0:
        max_length_cap = int(args.max_length)
    elif args.max_length_cap and args.max_length_cap > 0:
        max_length_cap = int(args.max_length_cap)
    else:
        # Use 95th percentile as default cap to avoid extreme outliers
        max_length_cap = int(train_df[seq_col].astype(str).str.len().quantile(0.95))
        max_length_cap = ((max_length_cap + 127) // 128) * 128
    
    logging.info("Max length cap (for truncation): %s", max_length_cap)
    logging.info("Using dynamic padding - batch size will vary per batch")

    # Create datasets
    train_ds = NCRERegressionDataset(
        train_df,
        seq_col=seq_col,
        label_col=label_col,
        max_length_cap=max_length_cap,
    )

    val_ds = NCRERegressionDataset(
        val_df,
        seq_col=seq_col,
        label_col=label_col,
        max_length_cap=max_length_cap,
    )

    test_ds = NCRERegressionDataset(
        test_df,
        seq_col=seq_col,
        label_col=label_col,
        max_length_cap=max_length_cap,
    )

    # ============================================================
    # Model setup - NTv3 Regression
    # ============================================================
    logging.info("Loading NTv3 model for regression: %s", args.checkpoint)
    model = NTv3ForRegression(model_name=args.checkpoint, species_str=args.species)
    
    logging.info(f"Model loaded: {args.checkpoint}")
    logging.info(f"Species: {args.species}")
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # LoRA setup
    if args.use_lora:
        targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
        if len(targets) == 0:
            # Default NTv3 transformer attention targets (following fine-tuning example)
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

        # Freeze non-LoRA parameters
        keep_ln = bool(int(args.keep_layernorm_trainable))
        freeze_backbone_except_lora(model, keep_ln=keep_ln)
        logging.info("Freeze backbone except LoRA (+LN trainable=%s)", str(keep_ln))

    tr, tot = count_params(model)
    logging.info(
        "Trainable params: %d || All params: %d || Trainable%%: %.6f",
        tr, tot, 100.0 * tr / max(1, tot),
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

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=dynamic_collate_fn,  # Dynamic padding collate function
        compute_metrics=compute_metrics_regression,
        callbacks=callbacks,
    )

    # ============================================================
    # Training
    # ============================================================
    logging.info("Starting training...")
    trainer.train()

    # Save best model
    best_pth = os.path.join(out_dir, "best.pth")
    save_pth(model, best_pth, extra={"best": True})
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

    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    keys = ["test_MSE", "test_RMSE", "test_MAE", "test_R2", "test_PCC", "test_SCC", "test_loss"]
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

    # WandB logging (regression metrics only)
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
                "hp/pin_memory": int(args.dataloader_pin_memory),
                "hp/num_workers": int(args.dataloader_num_workers),
            })
            # Test metrics
            wandb.log({
                "test/MSE": test_metrics.get("test_MSE", None),
                "test/RMSE": test_metrics.get("test_RMSE", None),
                "test/MAE": test_metrics.get("test_MAE", None),
                "test/R2": test_metrics.get("test_R2", None),
                "test/PCC": test_metrics.get("test_PCC", None),
                "test/SCC": test_metrics.get("test_SCC", None),
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
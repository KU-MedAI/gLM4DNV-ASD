import os
import re
import json
import math
import time
import argparse
import random
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_pth(payload: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    torch.save(payload, path)

def parse_lora_pair(s: str) -> Tuple[int, int]:
    if s is None:
        raise ValueError("lora_pair is None")
    m = re.match(r"^\s*r(\d+)_a(\d+)\s*$", str(s))
    if not m:
        raise ValueError(f"Invalid --lora_pair format: {s} (expected like r16_a32)")
    r = int(m.group(1))
    a = int(m.group(2))
    return r, a

def count_params(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return int(trainable), int(total)

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("hf_lora")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def safe_read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as e:
            raise ImportError(
                f"pandas.read_parquet requires pyarrow.\n"
                f"pip install pyarrow\n"
                f"Original: {e}"
            )
    if ext == ".feather":
        try:
            return pd.read_feather(path)
        except ImportError as e:
            raise ImportError(
                f"pandas.read_feather requires pyarrow.\n"
                f"pip install pyarrow\n"
                f"Original: {e}"
            )
    if ext in [".csv", ".tsv"]:
        sep = "\t" if ext == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(f"Unsupported file type: {path}")

def fmt_float(x: Any) -> str:
    try:
        return f"{float(x):.4g}"
    except Exception:
        return str(x)

def looks_like_na_subdir(s: str) -> bool:
    if s is None:
        return True
    s = str(s).strip()
    if s == "" or s.lower() in {"na", "none", "null"}:
        return True
    bad_tokens = ["lrNA", "rNA", "bsNA", "wdNA", "dpNA", "rNone", "bsNone", "dpNone", "wdNone", "lrNone"]
    if any(t in s for t in bad_tokens):
        return True
    if s.startswith("lr") and ("_r" in s) and ("_bs" in s) and ("_wd" in s) and ("_dp" in s):
        m = re.match(r"^lr([0-9eE\.\-\+]+)_r", s)
        if m is None:
            return True
    return False

def build_canonical_subdir(args: argparse.Namespace) -> str:
    rid = os.environ.get("WANDB_RUN_ID") or os.environ.get("WANDB_RUN_NAME") or "local"
    seed = getattr(args, "seed", 42)
    wu = getattr(args, "warmup_steps", 0)
    lr = fmt_float(getattr(args, "learning_rate", "NA"))
    wd = fmt_float(getattr(args, "weight_decay", "NA"))
    bs = getattr(args, "per_device_train_batch_size", "NA")
    r = getattr(args, "lora_r", "NA")
    dp = getattr(args, "lora_dropout", "NA")
    return f"lr{lr}_r{r}_bs{bs}_wd{wd}_dp{dp}_seed{seed}_wu{wu}_{rid}"

@torch.no_grad()
def compute_metrics_binary_from_logits_tensor(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy().astype(int)
    y_prob = np.asarray(probs)
    y_pred = (y_prob >= 0.5).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    f1 = 2 * prec * rec / max(1e-12, (prec + rec))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / (denom + 1e-12)
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

def load_evo2(model_name: str):
    try:
        import evo2
    except Exception as e:
        raise RuntimeError(f"Failed to import evo2 package: {e}")
    for fn_name in ["load_model", "load", "get_model", "create_model"]:
        try:
            fn = getattr(__import__("evo2"), fn_name)
            evo2_obj = fn(model_name)
            return evo2_obj
        except Exception:
            pass
    try:
        from evo2 import models as evo2_models
        for fn_name in ["load_model", "load", "get_model", "create_model"]:
            if hasattr(evo2_models, fn_name):
                evo2_obj = getattr(evo2_models, fn_name)(model_name)
                return evo2_obj
    except Exception:
        pass
    try:
        from evo2.models import Evo2
        return Evo2(model_name)
    except Exception:
        pass
    raise RuntimeError("Could not load evo2 model with known patterns.")

class SimpleDNATokenizer:
    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id
        self.map = {
            "A": 1, "C": 2, "G": 3, "T": 4, "N": 5,
            "a": 1, "c": 2, "g": 3, "t": 4, "n": 5,
        }
    def encode(self, seq: str, max_len: int) -> List[int]:
        ids = [self.map.get(ch, 5) for ch in seq]
        if len(ids) >= max_len:
            ids = ids[:max_len]
        else:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids
    def make_attention_mask(self, ids: List[int]) -> List[int]:
        return [0 if t == self.pad_id else 1 for t in ids]

def get_evo2_tokenizer(evo2_obj) -> Any:
    for attr in ["tokenizer", "tok", "encoder"]:
        if hasattr(evo2_obj, attr):
            return getattr(evo2_obj, attr)
    return SimpleDNATokenizer(pad_id=0)

def encode_batch(tokenizer, seqs: List[str], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if callable(tokenizer):
        try:
            out = tokenizer(
                seqs,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_attention_mask=True,
                return_token_type_ids=False,
            )
            input_ids = torch.tensor(out["input_ids"], dtype=torch.long)
            attn = out.get("attention_mask", None)
            if attn is None:
                attention_mask = (input_ids != 0).long()
            else:
                attention_mask = torch.tensor(attn, dtype=torch.long)
            return input_ids, attention_mask
        except Exception:
            pass
    if hasattr(tokenizer, "encode"):
        ids = []
        masks = []
        fb = SimpleDNATokenizer(pad_id=0)
        for s in seqs:
            try:
                if hasattr(tokenizer.encode, "__code__") and ("max_len" in tokenizer.encode.__code__.co_varnames):
                    x = tokenizer.encode(s, max_len=max_len)
                else:
                    x = tokenizer.encode(s)
            except Exception:
                x = None
            if x is None:
                x = fb.encode(s, max_len=max_len)
            x = list(x)
            if len(x) >= max_len:
                x = x[:max_len]
            else:
                x = x + [0] * (max_len - len(x))
            ids.append(x)
            if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
                pad_id = int(tokenizer.pad_token_id)
                masks.append([0 if t == pad_id else 1 for t in x])
            else:
                masks.append([0 if t == 0 else 1 for t in x])
        return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long)
    fb = SimpleDNATokenizer(pad_id=0)
    ids = [fb.encode(s, max_len=max_len) for s in seqs]
    masks = [fb.make_attention_mask(x) for x in ids]
    return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

class VariantDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_col: str, label_col: str):
        self.df = df.reset_index(drop=True)
        self.seq_col = seq_col
        self.label_col = label_col
    def __len__(self) -> int:
        return len(self.df)
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq = str(row[self.seq_col])
        y = int(row[self.label_col])
        return {"seq": seq, "label": y}

@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

def collate_fn_factory(tokenizer, max_len: int):
    def _collate(examples: List[Dict[str, Any]]) -> Batch:
        seqs = [ex["seq"] for ex in examples]
        labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
        input_ids, attention_mask = encode_batch(tokenizer, seqs, max_len=max_len)
        return Batch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return _collate

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear requires nn.Linear")
        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / max(1, int(r))
        self.dropout = nn.Dropout(float(dropout))

        for p in self.base.parameters():
            p.requires_grad = False

        in_f = base.in_features
        out_f = base.out_features
        dev = base.weight.device
        dt = base.weight.dtype

        self.lora_A = nn.Linear(in_f, self.r, bias=False, device=dev, dtype=dt)
        self.lora_B = nn.Linear(self.r, out_f, bias=False, device=dev, dtype=dt)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.active = True

    def set_active(self, flag: bool) -> None:
        self.active = bool(flag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if (not self.active) or (self.r <= 0):
            return y

        if x.dtype != self.base.weight.dtype:
            x = x.to(dtype=self.base.weight.dtype)

        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return y + lora_out

def should_wrap_lora(name: str, target_keywords: List[str]) -> bool:
    return any(k in name for k in target_keywords)

def apply_lora_linear_only(
    model: nn.Module,
    target_keywords: List[str],
    r: int,
    alpha: int,
    dropout: float
) -> Tuple[int, List[str]]:
    wrapped = 0
    samples: List[str] = []
    named_modules = dict(model.named_modules())

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not should_wrap_lora(name, target_keywords):
            continue

        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = named_modules.get(parent_name, None)
        else:
            parent = model
            child_name = name

        if parent is None or not hasattr(parent, child_name):
            continue

        base_linear = getattr(parent, child_name)
        if not isinstance(base_linear, nn.Linear):
            continue

        setattr(parent, child_name, LoRALinear(base_linear, r=r, alpha=alpha, dropout=dropout))
        wrapped += 1
        if len(samples) < 20:
            samples.append(name)

    return wrapped, samples

def iter_lora_modules(model: nn.Module) -> List[LoRALinear]:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]

def freeze_all_except_lora_and_classifier(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False

    for m in model.modules():
        if isinstance(m, LoRALinear):
            for p in m.lora_A.parameters():
                p.requires_grad = True
            for p in m.lora_B.parameters():
                p.requires_grad = True

    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    if hasattr(model, "pre_ln"):
        for p in model.pre_ln.parameters():
            p.requires_grad = True

def set_lora_gradual_active(model: nn.Module, active_k: int) -> Tuple[int, int]:
    loras = iter_lora_modules(model)
    total = len(loras)
    active_k = int(max(0, min(active_k, total)))
    for i, m in enumerate(loras):
        m.set_active(i < active_k)
    return active_k, total

def print_trainable_params(model: nn.Module, logger: logging.Logger) -> None:
    trainable = []
    total = 0
    trainable_total = 0

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable.append((name, tuple(p.shape), p.dtype, p.device, n))
            trainable_total += n

    logger.info("========== TRAINABLE PARAMS ==========")
    logger.info(f"Total params     : {total:,}")
    logger.info(f"Trainable params : {trainable_total:,}")
    logger.info(f"Trainable %      : {100.0 * trainable_total / max(1, total):.6f}%")
    logger.info(f"#Trainable tensors: {len(trainable)}")
    logger.info("=====================================")
    for name, shape, dtype, device, n in trainable[:60]:
        logger.info(f"{name:90s}  shape={shape}  dtype={dtype}  device={device}  numel={n:,}")

class Evo2CLS(nn.Module):
    def __init__(
        self,
        evo2_obj,
        emb_layer: str,
        pooling: str = "mean",
        num_labels: int = 2,
        d_model_hint: int = 4096,
        debug_first_forward: bool = True,
    ):
        super().__init__()
        self.evo2_obj = evo2_obj

        if not hasattr(evo2_obj, "model") or not isinstance(evo2_obj.model, nn.Module):
            raise RuntimeError("evo2_obj.model must exist and be an nn.Module")
        self.backbone: nn.Module = evo2_obj.model

        self.emb_layer = str(emb_layer)
        self.pooling = pooling
        self.num_labels = int(num_labels)
        self.debug_first_forward = bool(debug_first_forward)
        self._printed_out_keys = False

        self.pre_ln = nn.LayerNorm(int(d_model_hint), elementwise_affine=True)
        self.classifier = nn.Linear(int(d_model_hint), self.num_labels)
        nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)
        nn.init.zeros_(self.classifier.bias)

        self.loss_fn = nn.CrossEntropyLoss()

    def reset_classifier(self, d_model: int) -> None:
        self.pre_ln = nn.LayerNorm(int(d_model), elementwise_affine=True)
        self.classifier = nn.Linear(int(d_model), self.num_labels)
        nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)
        nn.init.zeros_(self.classifier.bias)

    def _extract_hidden(self, out) -> torch.Tensor:
        if isinstance(out, dict):
            if (not self._printed_out_keys) and self.debug_first_forward:
                self._printed_out_keys = True
                try:
                    keys = list(out.keys())
                    print(f"[DEBUG] evo2 out keys: {keys[:50]}")
                    if "embeddings" in out and isinstance(out["embeddings"], dict):
                        print(f"[DEBUG] out['embeddings'] keys: {list(out['embeddings'].keys())[:50]}")
                except Exception:
                    pass

            if "embeddings" in out and isinstance(out["embeddings"], dict):
                emb_dict = out["embeddings"]
                if self.emb_layer in emb_dict:
                    hidden = emb_dict[self.emb_layer]
                    if torch.is_tensor(hidden):
                        return hidden
                for v in emb_dict.values():
                    if torch.is_tensor(v):
                        return v

            if self.emb_layer in out and torch.is_tensor(out[self.emb_layer]):
                return out[self.emb_layer]

        embeddings = None
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            embeddings = out[1]
        elif isinstance(out, dict) and "embeddings" in out:
            embeddings = out["embeddings"]
        elif isinstance(out, dict):
            embeddings = out

        if embeddings is None:
            raise RuntimeError("Could not find embeddings from evo2 output.")

        if isinstance(embeddings, dict):
            if self.emb_layer in embeddings:
                hidden = embeddings[self.emb_layer]
            else:
                hidden = next(iter(embeddings.values()))
        else:
            hidden = embeddings[0] if isinstance(embeddings, (list, tuple)) else embeddings

        if not torch.is_tensor(hidden):
            raise RuntimeError(f"Extracted hidden is not a tensor: type={type(hidden)}")
        return hidden

    @staticmethod
    def _mask_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        m = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        return (hidden * m).sum(dim=1) / denom

    @staticmethod
    def _mask_last(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        idx = attention_mask.long().sum(dim=1) - 1
        idx = idx.clamp(min=0)
        B = hidden.size(0)
        return hidden[torch.arange(B, device=hidden.device), idx, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        out = self.evo2_obj(input_ids, return_embeddings=True, layer_names=[self.emb_layer])
        hidden = self._extract_hidden(out)

        if hidden.dim() != 3:
            raise RuntimeError(f"Hidden must be (B,L,D). Got shape={tuple(hidden.shape)}")

        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        if self.pooling == "last":
            pooled = self._mask_last(hidden, attention_mask)
        else:
            pooled = self._mask_mean(hidden, attention_mask)

        if not torch.isfinite(pooled).all():
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1e4, neginf=-1e4)

        pooled_fp32 = pooled.float()
        pooled_fp32 = self.pre_ln(pooled_fp32)
        logits = self.classifier(pooled_fp32)

        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            return {"loss": loss, "logits": logits}

        return {"logits": logits}
    
@torch.no_grad()
def infer_d_model_from_one_forward(evo2_obj, emb_layer: str, device: torch.device) -> int:
    x = torch.zeros((1, 16), dtype=torch.long, device=device)
    out = evo2_obj(x, return_embeddings=True, layer_names=[emb_layer])
    if isinstance(out, dict) and "embeddings" in out and isinstance(out["embeddings"], dict):
        emb = out["embeddings"]
        if emb_layer in emb and torch.is_tensor(emb[emb_layer]):
            return int(emb[emb_layer].shape[-1])
        for v in emb.values():
            if torch.is_tensor(v):
                return int(v.shape[-1])
    embeddings = None
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        embeddings = out[1]
    elif isinstance(out, dict) and "embeddings" in out:
        embeddings = out["embeddings"]
    elif isinstance(out, dict):
        embeddings = out
    if embeddings is None:
        raise RuntimeError("Could not find embeddings from evo2 output.")
    if isinstance(embeddings, dict):
        hidden = embeddings[emb_layer] if emb_layer in embeddings else next(iter(embeddings.values()))
    else:
        hidden = embeddings[0] if isinstance(embeddings, (list, tuple)) else embeddings
    return int(hidden.shape[-1])

class WarmupLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, mode: str):
        self.optimizer = optimizer
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(0, int(warmup_steps))
        self.mode = str(mode).lower().strip()
        if self.mode not in ["linear", "cosine"]:
            raise ValueError(f"Unsupported lr_scheduler_type: {mode}")
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_i = 0
    def _lr_mult(self, step_i: int) -> float:
        if step_i <= 0:
            return 0.0 if self.warmup_steps > 0 else 1.0
        if self.warmup_steps > 0 and step_i < self.warmup_steps:
            return float(step_i) / float(max(1, self.warmup_steps))
        denom = max(1, self.total_steps - self.warmup_steps)
        t = float(step_i - self.warmup_steps) / float(denom)
        t = min(max(t, 0.0), 1.0)
        if self.mode == "linear":
            return 1.0 - t
        return 0.5 * (1.0 + math.cos(math.pi * t))
    def step(self):
        self.step_i += 1
        mult = self._lr_mult(self.step_i)
        for lr0, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = lr0 * mult

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_bf16_autocast: bool):
    model.eval()
    losses: List[float] = []
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)
        if use_bf16_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out["loss"]
        logits = out["logits"]
        losses.append(float(loss.detach().cpu().item()))
        all_logits.append(logits.detach().float().cpu())
        all_labels.append(labels.detach().cpu())
    mean_loss = float(np.mean(losses)) if len(losses) > 0 else float("nan")
    logits_cat = torch.cat(all_logits, dim=0) if all_logits else torch.zeros((0, 2))
    labels_cat = torch.cat(all_labels, dim=0) if all_labels else torch.zeros((0,), dtype=torch.long)
    metrics = compute_metrics_binary_from_logits_tensor(logits_cat, labels_cat)
    return mean_loss, metrics

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[WarmupLRScheduler],
    device: torch.device,
    grad_clip: float,
    use_bf16_autocast: bool,
    epoch: int = 1,
    logger: Optional[logging.Logger] = None,
) -> float:
    model.train()
    losses: List[float] = []
    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_bf16_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out["loss"]
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out["loss"]
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if len(losses) > 0 else float("nan")

def save_epoch_payload(model: nn.Module, out_dir: str, save_adapter_only: bool, tag: str) -> None:
    ensure_dir(out_dir)
    payload: Dict[str, Any] = {"tag": tag}
    if save_adapter_only:
        sd = model.state_dict()
        keep = {}
        for k, v in sd.items():
            if ("lora_A" in k) or ("lora_B" in k) or k.startswith("classifier.") or k.startswith("pre_ln."):
                keep[k] = v.detach().cpu()
        payload["state_dict"] = keep
        payload["adapter_only"] = True
    else:
        payload["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        payload["adapter_only"] = False
    fname = "epoch1.pth" if tag == "epoch1" else ("best.pth" if tag == "best" else f"{tag}.pth")
    save_pth(payload, os.path.join(out_dir, fname))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--span_bp", type=int, required=True)
    ap.add_argument("--seq_col", type=str, default="")
    ap.add_argument("--label_col", required=True)
    ap.add_argument("--split_col", type=str, default="split")
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--emb_layer", required=True)
    ap.add_argument("--pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--output_subdir", required=True)
    ap.add_argument("--log_root", type=str, default="")
    ap.add_argument("--log_subdir", type=str, default="")
    ap.add_argument("--num_train_epochs", type=int, default=100)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "cosine"])
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--num_warmup_steps", type=int, default=-1)
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--early_stopping", action="store_true")
    ap.add_argument("--early_stopping_patience", type=int, default=10)
    ap.add_argument("--early_stopping_threshold", type=float, default=0.0)
    ap.add_argument("--load_best_model_at_end", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_length", type=int, default=0)
    ap.add_argument("--max_length_cap", type=int, default=0)
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_pair", type=str, default="")
    ap.add_argument("--lora_targets", type=str, default="Wqkv,out_proj,out_filter_dense")
    ap.add_argument("--lora_gradual", action="store_true")
    ap.add_argument("--lora_gradual_init_k", type=int, default=1)
    ap.add_argument("--lora_gradual_step", type=int, default=1)
    ap.add_argument("--lora_gradual_every", type=int, default=1)
    ap.add_argument("--save_adapter_only", action="store_true")
    ap.add_argument("--report_to", type=str, default="")
    ap.add_argument("--run_name", type=str, default="")

    args, _unknown = ap.parse_known_args()

    model_name = (str(args.model).strip() or str(args.checkpoint).strip())
    if not model_name:
        raise ValueError("Need --model or --checkpoint")
    args.model = model_name

    if not str(args.seq_col).strip():
        args.seq_col = f"var_seq_{int(args.span_bp)}bp"

    resolved_warmup_steps = int(args.warmup_steps)
    if int(args.num_warmup_steps) >= 0:
        resolved_warmup_steps = int(args.num_warmup_steps)

    if args.use_lora:
        if args.lora_pair and str(args.lora_pair).strip():
            r, a = parse_lora_pair(args.lora_pair)
            args.lora_r = r
            args.lora_alpha = a
        elif int(args.lora_alpha) <= 0:
            args.lora_alpha = int(args.lora_r) * 2

    if looks_like_na_subdir(args.output_subdir):
        args.output_subdir = build_canonical_subdir(args)

    if (not str(args.log_subdir).strip()) or looks_like_na_subdir(args.log_subdir):
        args.log_subdir = str(args.output_subdir)

    seed_everything(int(args.seed))

    out_dir = os.path.join(args.output_root, args.output_subdir)
    ensure_dir(out_dir)

    log_file = None
    if str(args.log_root).strip():
        log_dir = os.path.join(args.log_root, args.log_subdir or args.output_subdir)
        ensure_dir(log_dir)
        log_file = os.path.join(log_dir, "train.log")

    logger = setup_logger(log_file=log_file)

    dump = vars(args).copy()
    dump["resolved_warmup_steps"] = resolved_warmup_steps
    dump["out_dir"] = out_dir
    save_json(dump, os.path.join(out_dir, "args.json"))

    use_wandb = (str(args.report_to).strip().lower() == "wandb")
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", None),
                entity=os.environ.get("WANDB_ENTITY", None),
                name=(args.run_name or None),
                config=dump
            )
        except Exception as e:
            logger.info(f"wandb init failed: {e}")
            use_wandb = False
            wandb_run = None

    device = torch.device(args.device)
    use_bf16 = bool(args.bf16) and ("cuda" in str(device))

    logger.info(f"out_dir   = {out_dir}")
    logger.info(f"model     = {args.model}")
    logger.info(f"device    = {device} | bf16={use_bf16}")
    logger.info(f"pooling   = {args.pooling}")

    t0 = time.time()
    df = safe_read_table(args.data_path)
    logger.info(f"loaded data: shape={df.shape} | elapsed={time.time()-t0:.2f}s")

    need_cols = [args.seq_col, args.label_col, args.split_col]
    for c in need_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    df = df.dropna(subset=[args.seq_col, args.label_col, args.split_col]).copy()
    df[args.label_col] = df[args.label_col].astype(int)

    uniq = sorted(df[args.label_col].unique().tolist())
    if not set(uniq).issubset({0, 1}):
        raise ValueError(f"label_col must be binary 0/1. got unique={uniq}")

    s = df[args.split_col].astype(str).str.lower().str.strip()
    train_df = df[s == "train"].copy()
    val_df = df[s == "val"].copy()
    test_df = df[s == "test"].copy()
    logger.info(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    if args.max_length and args.max_length > 0:
        max_len = int(args.max_length)
    else:
        max_len = int(train_df[args.seq_col].astype(str).str.len().max())
    if args.max_length_cap and args.max_length_cap > 0:
        max_len = min(max_len, int(args.max_length_cap))
    logger.info(f"max_length = {max_len}")

    evo2_obj = load_evo2(args.model)
    tokenizer = get_evo2_tokenizer(evo2_obj)

    if hasattr(evo2_obj, "model") and isinstance(evo2_obj.model, nn.Module):
        evo2_obj.model.to(device)
        if use_bf16:
            evo2_obj.model.to(dtype=torch.bfloat16)

    try:
        d_model = infer_d_model_from_one_forward(evo2_obj, emb_layer=args.emb_layer, device=device)
    except Exception as e:
        logger.info(f"infer_d_model failed -> fallback 4096. err={e}")
        d_model = 4096
    logger.info(f"inferred d_model = {d_model}")

    model = Evo2CLS(
        evo2_obj=evo2_obj,
        emb_layer=args.emb_layer,
        pooling=args.pooling,
        num_labels=2,
        d_model_hint=d_model,
        debug_first_forward=True,
    )
    model.reset_classifier(d_model)
    model.to(device)

    if use_bf16:
        model.backbone.to(dtype=torch.bfloat16)
    model.pre_ln.to(device=device, dtype=torch.float32)
    model.classifier.to(device=device, dtype=torch.float32)

    if args.use_lora:
        target_keywords = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        if len(target_keywords) == 0:
            raise ValueError("--use_lora requires --lora_targets")

        wrapped, samples = apply_lora_linear_only(
            model.backbone,
            target_keywords=target_keywords,
            r=int(args.lora_r),
            alpha=int(args.lora_alpha),
            dropout=float(args.lora_dropout),
        )
        logger.info(f"LoRA wrapped={wrapped}")
        if wrapped > 0:
            logger.info(f"   sample wrapped: {samples[:20]}")

        freeze_all_except_lora_and_classifier(model)

    logger.info(f"backbone: {next(model.backbone.parameters()).dtype}")
    logger.info(f"pre_ln  : {next(model.pre_ln.parameters()).dtype}")
    logger.info(f"classifier: {model.classifier.weight.dtype}")

    tr, tot = count_params(model)
    logger.info(f"trainable: {tr:,} | total: {tot:,} | {100.0*tr/max(1,tot):.4f}%")
    print_trainable_params(model, logger)

    train_ds = VariantDataset(train_df, seq_col=args.seq_col, label_col=args.label_col)
    val_ds = VariantDataset(val_df, seq_col=args.seq_col, label_col=args.label_col)
    test_ds = VariantDataset(test_df, seq_col=args.seq_col, label_col=args.label_col)

    train_loader = DataLoader(
        train_ds, batch_size=int(args.per_device_train_batch_size), shuffle=True, num_workers=0,
        collate_fn=collate_fn_factory(tokenizer, max_len=max_len), pin_memory=("cuda" in str(device))
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(args.per_device_eval_batch_size), shuffle=False, num_workers=0,
        collate_fn=collate_fn_factory(tokenizer, max_len=max_len), pin_memory=("cuda" in str(device))
    )
    test_loader = DataLoader(
        test_ds, batch_size=int(args.per_device_eval_batch_size), shuffle=False, num_workers=0,
        collate_fn=collate_fn_factory(tokenizer, max_len=max_len), pin_memory=("cuda" in str(device))
    )

    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(opt_params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * int(args.num_train_epochs)
    warmup = int(resolved_warmup_steps)
    logger.info(f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} warmup={warmup}")

    scheduler = WarmupLRScheduler(optimizer, total_steps, warmup, args.lr_scheduler_type)

    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0
    epoch1_saved = False
    best_payload_in_mem = None

    for epoch in range(1, int(args.num_train_epochs) + 1):
        if args.use_lora and args.lora_gradual:
            prog = (epoch - 1) // max(1, int(args.lora_gradual_every))
            target_k = int(args.lora_gradual_init_k) + prog * int(args.lora_gradual_step)
            active_k, total_lora = set_lora_gradual_active(model, target_k)
            logger.info(f"epoch={epoch} active={active_k}/{total_lora}")

        train_loss = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, scheduler=scheduler,
            device=device, grad_clip=float(args.grad_clip), use_bf16_autocast=use_bf16,
            epoch=epoch, logger=logger
        )

        val_loss, val_metrics = evaluate(model, val_loader, device, use_bf16_autocast=use_bf16)

        logger.info(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} | val_metrics={val_metrics}")

        if use_wandb and wandb_run is not None:
            try:
                import wandb
                wandb.log({
                    "epoch": epoch, "train/loss": train_loss, "val/loss": val_loss,
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=epoch)
            except Exception:
                pass

        if (not epoch1_saved) and epoch >= 1:
            save_epoch_payload(model, out_dir, bool(args.save_adapter_only), "epoch1")
            logger.info("Saved epoch1.pth")
            epoch1_saved = True

        improved = (val_loss < (best_val_loss - float(args.early_stopping_threshold)))
        if improved:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            bad_epochs = 0
            save_epoch_payload(model, out_dir, bool(args.save_adapter_only), "best")
            logger.info(f"Saved best.pth (epoch={best_epoch})")

            if args.load_best_model_at_end:
                if args.save_adapter_only:
                    sd = model.state_dict()
                    keep = {}
                    for k, v in sd.items():
                        if ("lora_A" in k) or ("lora_B" in k) or k.startswith("classifier.") or k.startswith("pre_ln."):
                            keep[k] = v.detach().cpu().clone()
                    best_payload_in_mem = keep
                else:
                    best_payload_in_mem = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if args.early_stopping and (bad_epochs >= int(args.early_stopping_patience)):
            logger.info(f"EarlyStopping at epoch={epoch} (best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f})")
            break

    if args.load_best_model_at_end and (best_payload_in_mem is not None) and (best_epoch > 0):
        logger.info(f"Loading best model (epoch={best_epoch})")
        sd = model.state_dict()
        for k, v in best_payload_in_mem.items():
            if k in sd:
                sd[k] = v.to(sd[k].device)
        model.load_state_dict(sd, strict=False)

    test_loss, test_metrics = evaluate(model, test_loader, device, use_bf16_autocast=use_bf16)

    test_out = {"test_loss": float(test_loss)}
    for k, v in test_metrics.items():
        test_out[f"test_{k}"] = float(v)

    save_json(test_out, os.path.join(out_dir, "test_metrics.json"))

    keys = ["test_Accuracy", "test_AUROC", "test_AUPRC", "test_F1",
            "test_Precision", "test_Recall", "test_Specificity", "test_MCC", "test_loss"]
    row = {k: test_out.get(k, None) for k in keys}
    pd.DataFrame([row]).to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)

    save_epoch_payload(model, out_dir, bool(args.save_adapter_only), "best_final")

    logger.info(f"Done: {out_dir}")
    logger.info(f"Test metrics: {os.path.join(out_dir, 'test_metrics.csv')}")

    if use_wandb and wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass

if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import json
import argparse
import ast
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

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
    }

    def _patch_one(m: nn.Module):
        if not hasattr(m, "forward"):
            return
        orig = m.forward
        if getattr(orig, "_hyena_patched", False):
            return

        def fwd(*args, **kwargs):
            for k in pop_keys:
                kwargs.pop(k, None)
            return orig(*args, **kwargs)

        fwd._hyena_patched = True
        m.forward = fwd

    _patch_one(model)
    for attr in ["base_model", "model"]:
        if hasattr(model, attr):
            try:
                _patch_one(getattr(model, attr))
            except Exception:
                pass

def get_last_hidden(output) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output[0]

def extract_state_dict_from_pth(pth_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(pth_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_state_dict", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        return {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
    raise ValueError(f"Unrecognized checkpoint format: {pth_path}")

def normalize_state_dict_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module."):]
        for pref in ["backbone.", "base_model.", "model."]:
            if k2.startswith(pref):
                k2 = k2[len(pref):]
        out[k2] = v
    return out

def ensure_int_list(x) -> List[int]:
    if isinstance(x, (list, tuple, np.ndarray)):
        return [int(v) for v in list(x)]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() == "nan":
            return []
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple, np.ndarray)):
                return [int(z) for z in list(v)]
            return [int(v)]
        except Exception:
            try:
                return [int(p) for p in s.split(",") if p.strip() != ""]
            except Exception:
                return []
    try:
        return [int(x)]
    except Exception:
        return []

def _count_lora_modules(model: nn.Module) -> Tuple[int, int]:
    na = 0
    nb = 0
    for name, _ in model.named_parameters():
        if ".lora_A." in name or "lora_A" in name:
            na += 1
        if ".lora_B." in name or "lora_B" in name:
            nb += 1
    return na, nb

def _print_adapter_config_summary(adir: str) -> None:
    p = os.path.join(adir, "adapter_config.json")
    if not os.path.exists(p):
        print(f"[LORA][WARN] adapter_config.json not found: {p}")
        return
    try:
        cfg = json.load(open(p))
        print("[LORA] adapter_config.json summary:")
        print("   - peft_type     :", cfg.get("peft_type"))
        print("   - task_type     :", cfg.get("task_type"))
        print("   - r             :", cfg.get("r"))
        print("   - lora_alpha    :", cfg.get("lora_alpha"))
        print("   - lora_dropout  :", cfg.get("lora_dropout"))
        print("   - target_modules:", cfg.get("target_modules"))
        print("   - base_model    :", cfg.get("base_model_name_or_path"))
    except Exception as e:
        print(f"[LORA][WARN] failed to read adapter_config.json: {e}")

@torch.inference_mode()
def _quick_forward_summary(model: nn.Module, tokenizer, seqs: List[str], length: int, device: torch.device) -> torch.Tensor:
    toks = tokenizer(
        seqs,
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
        max_length=length,
        return_tensors="pt",
    )
    input_ids = toks["input_ids"].to(device)
    out = model(input_ids=input_ids)
    hidden = get_last_hidden(out)
    vec = hidden.mean(dim=1)
    return vec.detach().float().cpu()

def _compare_vecs(tag_a: str, a: torch.Tensor, tag_b: str, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        print(f"[VERIFY][WARN] shape mismatch {tag_a}={tuple(a.shape)} vs {tag_b}={tuple(b.shape)}")
        return
    diff = (a - b)
    max_abs = diff.abs().max().item()
    l2 = diff.pow(2).sum(dim=1).sqrt().mean().item()
    eps = 1e-12
    an = a / (a.norm(dim=1, keepdim=True) + eps)
    bn = b / (b.norm(dim=1, keepdim=True) + eps)
    cos = (an * bn).sum(dim=1).mean().item()
    print(f"[VERIFY] {tag_a} vs {tag_b}: max_abs_diff={max_abs:.6g} | mean_L2_diff={l2:.6g} | mean_cos={cos:.6g}")

class HyenaDNAEmbedderMutMaxConcatFR:
    def __init__(
        self,
        base_ckpt: str,
        length: int,
        device: str = "cuda:0",
        batch_size: int = 256,
        lora_adapter_dir: Optional[str] = None,
        ft_pth_path: Optional[str] = None,
        merge_lora: bool = False,
        pad_idx: Optional[int] = None,
        reverse_mode: str = "reverse",
        verify_weights: bool = True,
        verify_seq: str = "",
        verify_n: int = 2,
    ):
        self.length = int(length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.reverse_mode = str(reverse_mode).lower().strip()
        if self.reverse_mode not in ["forward", "reverse"]:
            raise ValueError(f"--reverse_mode must be one of [forward, reverse], got: {reverse_mode}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_ckpt, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.pad_idx = int(self.tokenizer.pad_token_id) if pad_idx is None else int(pad_idx)

        base_model = AutoModel.from_pretrained(base_ckpt, trust_remote_code=True)
        patch_model_forward_to_ignore_hf_kwargs(base_model)
        base_model = base_model.to(self.device).eval()

        verify_n = max(1, int(verify_n))
        if not verify_seq.strip():
            verify_seq = ("ACGT" * ((min(self.length, 256) + 3) // 4))[: min(self.length, 256)]
        verify_seqs = [verify_seq for _ in range(verify_n)]

        vec_base = None
        vec_after_ft = None
        vec_after_lora = None

        if verify_weights:
            try:
                vec_base = _quick_forward_summary(base_model, self.tokenizer, verify_seqs, length=min(self.length, 256), device=self.device)
                print(f"[VERIFY] base_ckpt forward OK | vec shape={tuple(vec_base.shape)}")
            except Exception as e:
                print(f"[VERIFY][WARN] base forward failed: {e}")
                vec_base = None

        if ft_pth_path and str(ft_pth_path).strip():
            sd_raw = extract_state_dict_from_pth(ft_pth_path)
            sd = normalize_state_dict_keys(sd_raw)

            missing, unexpected = base_model.load_state_dict(sd, strict=False)
            print(f"loaded non-LoRA FT weights: {ft_pth_path}")
            print(f"   - missing keys   : {len(missing)}")
            print(f"   - unexpected keys: {len(unexpected)}")

            if verify_weights and vec_base is not None:
                try:
                    vec_after_ft = _quick_forward_summary(base_model, self.tokenizer, verify_seqs, length=min(self.length, 256), device=self.device)
                    _compare_vecs("BASE", vec_base, "AFTER_FT_PTH", vec_after_ft)
                except Exception as e:
                    print(f"[VERIFY][WARN] after-FT forward failed: {e}")

        if lora_adapter_dir and str(lora_adapter_dir).strip():
            if os.path.isfile(lora_adapter_dir):
                raise ValueError(f"lora_adapter_dir must be a directory, got file: {lora_adapter_dir}")

            print(f"loading LoRA adapter: {lora_adapter_dir}")
            _print_adapter_config_summary(lora_adapter_dir)

            vec_before_lora = None
            if verify_weights:
                try:
                    vec_before_lora = _quick_forward_summary(base_model, self.tokenizer, verify_seqs, length=min(self.length, 256), device=self.device)
                    print("[VERIFY] pre-LoRA forward OK")
                except Exception as e:
                    print(f"[VERIFY][WARN] pre-LoRA forward failed: {e}")
                    vec_before_lora = None

            model = PeftModel.from_pretrained(base_model, lora_adapter_dir)
            na, nb = _count_lora_modules(model)
            print(f"[LORA] detected lora modules by param name: lora_A={na} | lora_B={nb}")
            if na == 0 and nb == 0:
                print("[LORA][WARN] No lora_A/lora_B params detected. Adapter may NOT be attached correctly.")

            if merge_lora:
                print("merge_lora=1 -> merge_and_unload()")
                model = model.merge_and_unload()

            patch_model_forward_to_ignore_hf_kwargs(model)
            model = model.to(self.device).eval()

            if verify_weights and vec_before_lora is not None:
                try:
                    vec_after_lora = _quick_forward_summary(model, self.tokenizer, verify_seqs, length=min(self.length, 256), device=self.device)
                    _compare_vecs("PRE_LORA", vec_before_lora, "AFTER_LORA", vec_after_lora)
                except Exception as e:
                    print(f"[VERIFY][WARN] after-LoRA forward failed: {e}")

        else:
            model = base_model

        patch_model_forward_to_ignore_hf_kwargs(model)
        self.model = model.to(self.device).eval()

        self.dim = int(getattr(self.model.config, "d_model", getattr(self.model.config, "hidden_size", 256)))

        print(f"[MODEL] device={self.device} | length={self.length} | dim={self.dim} | reverse_mode={self.reverse_mode}")
        if lora_adapter_dir and str(lora_adapter_dir).strip():
            if merge_lora:
                print("[MODEL] LoRA mode: adapter loaded + merged (merge_and_unload)")
            else:
                print("[MODEL] LoRA mode: adapter loaded (not merged)")
        elif ft_pth_path and str(ft_pth_path).strip():
            print("[MODEL] FT mode: best.pth loaded (non-LoRA)")
        else:
            print("[MODEL] Base mode: no FT/LoRA weights applied")

    @torch.inference_mode()
    def _encode(self, seqs: List[str]) -> (torch.Tensor, torch.Tensor):
        toks = self.tokenizer(
            seqs,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.length,
            return_tensors="pt",
        )
        input_ids = toks["input_ids"].to(self.device)
        mask = (input_ids != self.pad_idx)
        hidden = get_last_hidden(self.model(input_ids=input_ids))
        hidden = hidden * mask.unsqueeze(-1)
        return hidden, mask

    def run(
        self,
        dnv: pd.DataFrame,
        seq_col: str,
        idx_col: str,
        out_col: str = "mut_emb_max",
        pooling: str = "max",
    ) -> pd.DataFrame:
        pooling = str(pooling).lower().strip()
        if pooling != "max":
            raise ValueError(f"Only pooling='max' supported now (got: {pooling})")

        n = len(dnv)
        out_rows = []

        mode = "fwd-only" if self.reverse_mode == "forward" else "fwd+rev-concat"
        for start in tqdm(range(0, n, self.batch_size), desc=f"HyenaDNA pooling ({mode}, mut max)"):
            end = min(start + self.batch_size, n)
            batch = dnv.iloc[start:end]

            seqs_fwd = batch[seq_col].astype(str).tolist()
            mut_lists = batch[idx_col].tolist()

            hidden_fwd, mask_fwd = self._encode(seqs_fwd)
            B, L, D = hidden_fwd.shape

            if self.reverse_mode == "reverse":
                seqs_rev = [s[::-1] for s in seqs_fwd]
                hidden_rev, mask_rev = self._encode(seqs_rev)
            else:
                hidden_rev, mask_rev = None, None

            for i in range(B):
                valid_pos_fwd = torch.nonzero(mask_fwd[i], as_tuple=True)[0]
                seq_len_eff = int(valid_pos_fwd.numel())

                out_dim = D if self.reverse_mode == "forward" else 2 * D
                if seq_len_eff <= 0:
                    out_rows.append({out_col: np.full((out_dim,), np.nan, dtype=np.float32)})
                    continue

                mut_idx = ensure_int_list(mut_lists[i])
                mut_idx = [j for j in mut_idx if 0 <= int(j) < seq_len_eff]

                if len(mut_idx) == 0:
                    out_rows.append({out_col: np.full((out_dim,), np.nan, dtype=np.float32)})
                    continue

                nuc_idx_fwd = torch.tensor(mut_idx, dtype=torch.long, device=self.device)
                tok_idx_fwd = valid_pos_fwd[nuc_idx_fwd]
                mut_tok_embs_fwd = hidden_fwd[i, tok_idx_fwd]
                pooled_fwd = mut_tok_embs_fwd.max(dim=0).values

                if self.reverse_mode == "forward":
                    pooled = pooled_fwd
                else:
                    valid_pos_rev = torch.nonzero(mask_rev[i], as_tuple=True)[0]
                    seq_len_eff_rev = int(valid_pos_rev.numel())
                    if seq_len_eff_rev <= 0:
                        pooled_rev = torch.full((D,), float("nan"), device=self.device, dtype=pooled_fwd.dtype)
                    else:
                        seq_len_for_rev = min(seq_len_eff, seq_len_eff_rev)
                        rev_nuc_idx = [seq_len_for_rev - 1 - j for j in mut_idx if 0 <= j < seq_len_for_rev]
                        if len(rev_nuc_idx) == 0:
                            pooled_rev = torch.full((D,), float("nan"), device=self.device, dtype=pooled_fwd.dtype)
                        else:
                            nuc_idx_rev = torch.tensor(rev_nuc_idx, dtype=torch.long, device=self.device)
                            tok_idx_rev = valid_pos_rev[nuc_idx_rev]
                            mut_tok_embs_rev = hidden_rev[i, tok_idx_rev]
                            pooled_rev = mut_tok_embs_rev.max(dim=0).values

                    pooled = torch.cat([pooled_fwd, pooled_rev], dim=0)

                out_rows.append({out_col: pooled.detach().cpu().numpy()})

            del batch, seqs_fwd, mut_lists, hidden_fwd, mask_fwd
            if self.reverse_mode == "reverse":
                del seqs_rev, hidden_rev, mask_rev
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return pd.DataFrame(out_rows)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", type=int, default=500)

    ap.add_argument("--variant_list_path", type=str, required=True)
    ap.add_argument("--dnv_path", type=str, required=True)

    ap.add_argument("--base_ckpt", type=str, default="LongSafari/hyenadna-large-1m-seqlen-hf")
    ap.add_argument("--ft_pth_path", type=str, default="")
    ap.add_argument("--lora_adapter_dir", type=str, default="")

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--merge_lora", action="store_true")

    ap.add_argument("--pad_idx", type=int, default=-1,
                    help="override pad token id. If -1, uses tokenizer.pad_token_id")

    ap.add_argument("--reverse_mode", type=str, default="reverse",
                    help="forward: fwd only / reverse: fwd+rev and concat")
    ap.add_argument("--pooling", type=str, default="max")

    ap.add_argument("--data_split", type=int, default=None,
                    help="Which split to process (0-indexed). If set, slices variant_list/dnv by rows.")
    ap.add_argument("--num_splits", type=int, default=1,
                    help="Total number of splits. Used with --data_split.")

    ap.add_argument("--verify_weights", type=int, default=1,
                    help="1: print evidence that FT/LoRA changes model output; 0: disable")
    ap.add_argument("--verify_seq", type=str, default="",
                    help="sequence used for verification (if empty, auto-generate ACGT...)")
    ap.add_argument("--verify_n", type=int, default=2,
                    help="# of verification sequences to run (same seq repeated)")

    ap.add_argument("--out_path", type=str, required=True)
    return ap.parse_args()

def main():
    args = parse_args()

    bp = int(args.bp)
    seq_col = f"var_seq_{bp}bp"
    idx_col = f"mut_idx_{bp}bp"

    print("[INFO] Loading variant_list:", args.variant_list_path)
    variant_list = pd.read_feather(args.variant_list_path)

    print("[INFO] Loading dnv         :", args.dnv_path)
    dnv = pd.read_feather(args.dnv_path)

    if seq_col not in dnv.columns or idx_col not in dnv.columns:
        raise ValueError(f"dnv must contain {seq_col} and {idx_col}")

    if args.data_split is not None and int(args.num_splits) > 1:
        ds = int(args.data_split)
        ns = int(args.num_splits)
        if ds < 0 or ds >= ns:
            raise ValueError(f"--data_split must be in [0, num_splits-1]. Got data_split={ds}, num_splits={ns}")

        total_rows = len(dnv)
        if len(variant_list) != total_rows:
            raise ValueError(
                f"variant_list and dnv must have same #rows for split. "
                f"Got len(variant_list)={len(variant_list)} vs len(dnv)={len(dnv)}"
            )

        split_size = total_rows // ns
        start_idx = ds * split_size
        end_idx = total_rows if (ds == ns - 1) else (start_idx + split_size)

        print(f"[DATA SPLIT] split {ds}/{ns} | rows {start_idx}:{end_idx} (total={total_rows})")

        variant_list = variant_list.iloc[start_idx:end_idx].reset_index(drop=True)
        dnv = dnv.iloc[start_idx:end_idx].reset_index(drop=True)

    max_len = int(dnv[seq_col].astype(str).str.len().max())
    print(f"[INFO] bp={bp} | seq_col={seq_col} | idx_col={idx_col} | max_len={max_len}")

    pad_idx = None if int(args.pad_idx) < 0 else int(args.pad_idx)

    embedder = HyenaDNAEmbedderMutMaxConcatFR(
        base_ckpt=args.base_ckpt,
        length=max_len,
        device=args.device,
        batch_size=args.batch_size,
        lora_adapter_dir=args.lora_adapter_dir if args.lora_adapter_dir.strip() else None,
        ft_pth_path=args.ft_pth_path if args.ft_pth_path.strip() else None,
        merge_lora=args.merge_lora,
        pad_idx=pad_idx,
        reverse_mode=args.reverse_mode,
        verify_weights=bool(int(args.verify_weights)),
        verify_seq=args.verify_seq,
        verify_n=int(args.verify_n),
    )

    out_df = embedder.run(
        dnv=dnv,
        seq_col=seq_col,
        idx_col=idx_col,
        out_col="mut_emb_max",
        pooling=args.pooling,
    )

    merged_df = pd.concat(
        [variant_list.reset_index(drop=True), out_df.reset_index(drop=True)],
        axis=1
    )

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    merged_df.to_feather(args.out_path)

    emb_dim = None
    try:
        emb_dim = int(len(merged_df["mut_emb_max"].iloc[0]))
    except Exception:
        pass

    print("saved:", args.out_path)
    print("   shape:", merged_df.shape)
    if emb_dim is not None:
        print("   embedding dim:", emb_dim)

if __name__ == "__main__":
    main()
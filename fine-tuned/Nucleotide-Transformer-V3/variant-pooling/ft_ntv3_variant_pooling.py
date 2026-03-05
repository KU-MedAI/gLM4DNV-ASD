#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import json
import argparse
import ast
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import PeftModel


# ============================================================
# Utils
# ============================================================
def ensure_int_list(x) -> List[int]:
    """Convert mut_idx column to [int, int, ...] list"""
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


def validate_sequence_length(seq: str, expected_len: int) -> None:
    """Validate sequence length (must be exact and multiple of 128)"""
    actual_len = len(seq)
    
    if actual_len != expected_len:
        raise ValueError(
            f"Sequence length mismatch!\n"
            f"   Expected: {expected_len}bp\n"
            f"   Got: {actual_len}bp\n"
            f"   Please preprocess data to {expected_len}bp with 'N' padding"
        )
    
    if expected_len % 128 != 0:
        raise ValueError(
            f"Sequence length must be multiple of 128!\n"
            f"   Got: {expected_len}bp\n"
            f"   Valid lengths: 128, 256, 384, 512, 640, 768, 896, 1024, ..."
        )


# ============================================================
# LoRA Verification Helper Functions
# ============================================================
def load_adapter_config(adapter_dir: str) -> Dict[str, Any]:
    """Load and parse adapter_config.json"""
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def print_adapter_config_summary(config: Dict[str, Any]):
    """Print formatted adapter config summary"""
    if not config:
        print("[LORA] No config loaded")
        return
    
    print("[LORA] adapter_config.json summary:")
    fields = [
        ("peft_type", "peft_type"),
        ("task_type", "task_type"),
        ("r", "r"),
        ("lora_alpha", "lora_alpha"),
        ("lora_dropout", "lora_dropout"),
        ("target_modules", "target_modules"),
        ("base_model_name_or_path", "base_model"),
    ]
    
    for key, label in fields:
        if key in config:
            value = config[key]
            if isinstance(value, list):
                value_str = str(value)
            elif isinstance(value, str) and len(value) > 50:
                value_str = value.split("/")[-1]
            else:
                value_str = str(value)
            print(f"  - {label:<15}: {value_str}")


def detect_lora_modules_by_name(model) -> Dict[str, Any]:
    """Detect LoRA modules by parameter names"""
    lora_a_names = []
    lora_b_names = []
    lora_params_total = 0
    
    for name, param in model.named_parameters():
        if "lora_A" in name:
            lora_a_names.append(name)
            lora_params_total += param.numel()
        elif "lora_B" in name:
            lora_b_names.append(name)
            lora_params_total += param.numel()
    
    return {
        "lora_A": len(lora_a_names),
        "lora_B": len(lora_b_names),
        "lora_A_names": lora_a_names,
        "lora_B_names": lora_b_names,
        "lora_params": lora_params_total
    }


def get_model_weight_sample(model, sample_size: int = 20) -> np.ndarray:
    """Extract sample of model weights"""
    for name, param in model.named_parameters():
        flat = param.detach().cpu().flatten()
        if len(flat) >= sample_size:
            return flat[:sample_size].numpy()
    
    for param in model.parameters():
        flat = param.detach().cpu().flatten()
        if len(flat) >= sample_size:
            return flat[:sample_size].numpy()
    
    return np.zeros(sample_size)


def generate_test_sequence(length: int = 100) -> str:
    """Generate test DNA sequence"""
    bases = "ACGT"
    return "".join([bases[i % 4] for i in range(length)])


def test_model_forward(model, tokenizer, device: torch.device, seq_length: int = 128):
    """Test forward pass with dummy sequence for NTv3"""
    try:
        model = model.to(device)
        
        # Round to 128 multiple
        test_len = ((seq_length + 127) // 128) * 128
        test_seq = generate_test_sequence(test_len)
        
        encoded = tokenizer(
            [test_seq],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"].to(device)
        
        with torch.inference_mode():
            if isinstance(model, PeftModel):
                core = model.base_model.model.core
            else:
                core = model.core
            
            # NTv3 core forward (following fine-tuning pattern)
            species_tokens = torch.full((input_ids.shape[0],), model.species_ids[0].item(), dtype=torch.long, device=device)
            outputs = core(input_ids, [species_tokens], output_hidden_states=True)
            
            hidden = outputs["hidden_states"][-1]
            return hidden.mean(dim=1).cpu()
            
    except Exception as e:
        print(f"[VERIFY] Forward test error: {e}")
        return None


def compare_outputs(output_pre, output_post) -> Dict[str, float]:
    """Compare model outputs"""
    if output_pre is None or output_post is None:
        return {}
    
    pre = output_pre.flatten().numpy()
    post = output_post.flatten().numpy()
    
    abs_diff = np.abs(pre - post)
    l2_diff = np.linalg.norm(pre - post)
    
    dot = np.dot(pre, post)
    norm_pre = np.linalg.norm(pre)
    norm_post = np.linalg.norm(post)
    cos_sim = dot / (norm_pre * norm_post + 1e-8)
    
    return {
        "max_abs_diff": abs_diff.max(),
        "mean_abs_diff": abs_diff.mean(),
        "l2_diff": l2_diff,
        "mean_l2_diff": l2_diff / len(pre),
        "cos_sim": cos_sim
    }


def verify_lora_impact(model):
    """Verify LoRA impact by calculating weight differences"""
    try:
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_A = module.lora_A['default'].weight
                lora_B = module.lora_B['default'].weight
                scaling = getattr(module, 'scaling', {}).get('default', 1.0)
                base_weight = module.base_layer.weight
                
                lora_delta = (lora_B @ lora_A) * scaling
                
                base_norm = base_weight.norm().item()
                delta_norm = lora_delta.norm().item()
                relative_impact = (delta_norm / base_norm) * 100
                
                print(f"[LORA IMPACT] {name}")
                print(f"   - Base weight norm: {base_norm:.6f}")
                print(f"   - LoRA delta norm: {delta_norm:.6f}")
                print(f"   - Relative impact: {relative_impact:.3f}%")
                
                if relative_impact > 0.1:
                    print(f"   Meaningful LoRA impact detected!")
                else:
                    print(f"   Low LoRA impact")
                
                return True
                
    except Exception as e:
        print(f"[LORA IMPACT] Verification error: {e}")
        return False
    
    return False


# ============================================================
# NTv3 Core Wrapper (Fine-tuning structure)
# ============================================================
class NTv3CoreWrapper(nn.Module):
    """
    Wrapper for NTv3 core model (following fine-tuning structure)
    - Extracts core (removes BigWig/BED heads)
    - Sets species token
    - Provides forward compatible with PeftModel
    """
    def __init__(self, model_name: str, species_str: str = "human"):
        super().__init__()
        
        print(f"[BASE MODEL] Loading NTv3 from {model_name}")
        
        # Load config and base model
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        ntv3_base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            config=self.config,
        )
        
        # Extract core (remove heads) - following fine-tuning pattern
        discrete_conditioned_model = type(ntv3_base_model.core).__bases__[0]
        self.core = discrete_conditioned_model(self.config)
        self.load_state_dict(ntv3_base_model.state_dict(), strict=False)
        
        print(f"[BASE MODEL] Core extracted (heads removed)")
        
        # Species setup (fixed, following fine-tuning)
        if species_str in self.config.species_to_token_id:
            species_ids = self.config.species_to_token_id[species_str]
            self.species_ids = torch.LongTensor([species_ids])
            print(f"[SPECIES] Using species: {species_str} with ids: {self.species_ids}")
        else:
            print(f"[SPECIES] {species_str} not in supported species, using mask token id")
            self.species_ids = torch.LongTensor([2])
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass following fine-tuning pattern
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) - unused but kept for compatibility
            
        Returns:
            dict with "hidden_states" key
        """
        # Prepare species tokens (repeat for batch)
        species_tokens = torch.repeat_interleave(self.species_ids, input_ids.shape[0])
        species_tokens = species_tokens.to(input_ids.device)
        
        # Core forward
        outputs = self.core(input_ids, [species_tokens], output_hidden_states=True)
        
        return outputs


# ============================================================
# NTv3 Fine-tuned Mutation Pooling Embedder
# ============================================================
class NTv3MutPoolFineTuned:
    """
    NTv3 fine-tuned mutation pooling:
    - Load fine-tuned NTv3 model (with optional LoRA)
    - Character-level tokenization (no upsampling needed)
    - Extract embeddings only at mutation positions
    - Apply MAX pooling over mutation tokens -> (D,)
    - NO PADDING: All sequences must be exact expected_seq_length (multiple of 128)
    """
    
    def __init__(
        self,
        base_ckpt: str,
        expected_seq_length: int,
        species_str: str = "human",
        device: str = "cuda:0",
        batch_size: int = 256,
        lora_adapter_dir: Optional[str] = None,
        merge_lora: bool = False,
        use_amp: bool = True,
    ):
        # Validate expected_seq_length
        if expected_seq_length % 128 != 0:
            raise ValueError(
                f"expected_seq_length must be multiple of 128!\n"
                f"   Got: {expected_seq_length}\n"
                f"   Valid: 128, 256, 384, 512, 640, 768, 896, 1024, ..."
            )
        
        self.expected_seq_length = int(expected_seq_length)
        self.batch_size = int(batch_size)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        
        print("=" * 60)
        print("[MODEL LOADING] Starting NTv3 Fine-tuned Model Initialization")
        print("=" * 60)
        print(f"[CONFIG] Expected sequence length: {self.expected_seq_length}bp (NO PADDING)")
        
        # Tokenizer
        print(f"[TOKENIZER] Loading from {base_ckpt}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_ckpt, trust_remote_code=True)
        print(f"   Tokenizer loaded (character-level, vocab size: {len(self.tokenizer)})")
        
        # Base model (core wrapper)
        base_model = NTv3CoreWrapper(model_name=base_ckpt, species_str=species_str)
        self.species_ids = base_model.species_ids
        
        # Apply LoRA adapter with complete verification
        current_model = base_model
        
        if lora_adapter_dir and str(lora_adapter_dir).strip():
            print("=" * 80)
            print("[LORA] Starting Complete LoRA Loading & Verification")
            print("=" * 80)
            print(f"[LORA] loading LoRA adapter: {lora_adapter_dir}")
            
            if not os.path.isdir(lora_adapter_dir):
                raise FileNotFoundError(f"Adapter dir not found: {lora_adapter_dir}")
            
            # Load config
            print(f"\n[LORA] Loading adapter_config.json...")
            adapter_config = load_adapter_config(lora_adapter_dir)
            print_adapter_config_summary(adapter_config)
            
            # PRE-LoRA test
            print(f"\n[VERIFY] pre-LoRA test...")
            output_pre = test_model_forward(base_model, self.tokenizer, self.device, 128)
            if output_pre is not None:
                print(f"[VERIFY] pre-LoRA forward OK")
            
            # Load LoRA
            print(f"\n[LORA] Loading adapter...")
            try:
                current_model = PeftModel.from_pretrained(base_model, lora_adapter_dir, is_trainable=False)
                print(f"[LORA] Loaded successfully")
            except Exception as e:
                print(f"[LORA] Failed: {e}")
                raise
            
            # Detect modules
            print(f"\n[LORA] Detecting modules...")
            lora_info = detect_lora_modules_by_name(current_model)
            print(f"[LORA] detected lora modules by param name: lora_A={lora_info['lora_A']} | lora_B={lora_info['lora_B']}")
            
            # POST-LoRA test
            print(f"\n[VERIFY] post-LoRA test...")
            output_post = test_model_forward(current_model, self.tokenizer, self.device, 128)
            if output_post is not None:
                print(f"[VERIFY] LoRA model forward OK")
            
            # Compare
            if output_pre is not None and output_post is not None:
                comp = compare_outputs(output_pre, output_post)
                print(f"[VERIFY] PRE_LORA vs AFTER_LORA: max_abs_diff={comp['max_abs_diff']:.5f} | mean_L2_diff={comp['mean_l2_diff']:.5f} | cos_sim={comp['cos_sim']:.6f}")
                
                if comp['max_abs_diff'] < 1e-6:
                    print(f"[WARNING] Outputs identical! LoRA may not be active!")
            
            # LoRA impact verification
            print(f"\n[VERIFY] Analyzing LoRA impact...")
            verify_lora_impact(current_model)
            
            if lora_info['lora_params'] == 0:
                raise RuntimeError("LoRA params = 0! Loading FAILED!")
            
            # Merge
            if merge_lora:
                print(f"\n[LORA] Merging weights...")
                current_model = current_model.merge_and_unload()
                print(f"[LORA] Merged")
            
            print(f"\n[MODEL] device={self.device} | length={self.expected_seq_length} | LoRA mode: {'merged' if merge_lora else 'adapter loaded (not merged)'}")
            print("=" * 80)
        else:
            print("[LORA] No adapter specified")
        
        # Move to device and set eval mode
        self.model = current_model.to(self.device).eval()
        print(f"[DEVICE] Model moved to {self.device}")
        
        # Extract dimension
        self.dim = int(self.model.config.embed_dim)
        
        print("=" * 60)
        print("[MODEL READY] NTv3 Fine-tuned Model Initialization Complete!")
        print(f"   - Model type: {type(self.model).__name__}")
        print(f"   - Hidden dimension: {self.dim}")
        print(f"   - Expected seq length: {self.expected_seq_length}bp (NO PADDING)")
        print(f"   - Device: {self.device}")
        print(f"   - AMP enabled: {self.use_amp}")
        print("=" * 60)
    
    def _encode_sequences(self, seqs: List[str]) -> Dict[str, torch.Tensor]:
        """
        Encode sequences WITHOUT padding
        All sequences must be exactly expected_seq_length
        """
        # Validate all sequences have same length
        expected_len = self.expected_seq_length
        
        for i, seq in enumerate(seqs):
            seq_len = len(seq)
            if seq_len != expected_len:
                raise ValueError(
                    f"Sequence length mismatch in batch!\n"
                    f"   Expected: {expected_len}bp\n"
                    f"   seq[{i}]: {seq_len}bp\n"
                    f"   All sequences must be preprocessed to {expected_len}bp"
                )
        
        # Tokenize without padding
        encoded = self.tokenizer(
            seqs,
            add_special_tokens=False,
            padding=False, 
            truncation=False, 
            return_tensors="pt",
        )
        
        # Final safety check
        seq_len = encoded["input_ids"].shape[1]
        if seq_len != expected_len:
            raise ValueError(
                f"Tokenized length mismatch!\n"
                f"   Expected: {expected_len} tokens\n"
                f"   Got: {seq_len} tokens"
            )
        
        return encoded
    
    @torch.inference_mode()
    def _forward_core(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through NTv3 core
        
        Args:
            input_ids: (batch, seq_len)
            
        Returns:
            hidden_states: (batch, seq_len, hidden_dim)
        """
        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model(input_ids=input_ids)
        else:
            outputs = self.model(input_ids=input_ids)
        
        # Extract last hidden state (full resolution)
        hidden_states = outputs["hidden_states"][-1]  # (batch, seq_len, hidden_dim)
        
        return hidden_states.detach().cpu()
    
    def get_mut_embeddings_from_df(
        self,
        seq_df: pd.DataFrame,
        seq_col: str,
        idx_col: str,
        out_col: str = "mut_emb_max",
    ) -> pd.DataFrame:
        """
        Extract mutation embeddings from dataframe
        
        Args:
            seq_df: DataFrame with sequences and mutation indices
            seq_col: column name for sequences (must be exactly expected_seq_length)
            idx_col: column name for mutation indices
            out_col: output column name for embeddings
            
        Returns:
            DataFrame with mutation embeddings
        """
        if seq_col not in seq_df.columns:
            raise ValueError(f"Missing required column: {seq_col}")
        if idx_col not in seq_df.columns:
            raise ValueError(f"Missing required column: {idx_col}")
        
        # Validate first sequence
        first_seq = seq_df[seq_col].iloc[0]
        actual_len = len(first_seq)
        
        if actual_len != self.expected_seq_length:
            raise ValueError(
                f"Data length mismatch!\n"
                f"   Model expects: {self.expected_seq_length}bp\n"
                f"   Data has: {actual_len}bp\n"
                f"   Please preprocess:\n"
                f"     df['{seq_col}'] = df['{seq_col}'].str.ljust({self.expected_seq_length}, 'N')"
            )
        
        out_rows = []
        n = len(seq_df)
        
        with torch.inference_mode():
            for start in tqdm(range(0, n, self.batch_size), 
                            desc="NTv3 mutation pooling (max, character-level)"):
                end = min(start + self.batch_size, n)
                batch = seq_df.iloc[start:end]
                
                seqs = batch[seq_col].astype(str).tolist()
                mut_lists = batch[idx_col].tolist()
                
                # Encode (with validation, no padding)
                encoded = self._encode_sequences(seqs)
                input_ids = encoded["input_ids"].to(self.device)
                
                # Forward
                hidden_states = self._forward_core(input_ids)  # (B, L, H)
                B, L, H = hidden_states.shape
                
                # Extract mutation positions for each sequence
                for b in range(B):
                    seq_emb = hidden_states[b]  # (L, H)
                    
                    # Get valid mutation indices
                    mut_idx = ensure_int_list(mut_lists[b])
                    mut_idx = [j for j in mut_idx if 0 <= int(j) < L]
                    
                    if len(mut_idx) == 0:
                        out_rows.append({out_col: np.full((self.dim,), np.nan, dtype=np.float32)})
                        continue
                    
                    # Remove duplicates and sort
                    mut_idx = sorted(set(mut_idx))
                    
                    # Extract embeddings at mutation positions (direct indexing!)
                    mut_embs = seq_emb[mut_idx, :].numpy()  # (K, H)
                    
                    # Max pooling over mutation positions
                    pooled = mut_embs.max(axis=0)  # (H,)
                    
                    out_rows.append({out_col: pooled.astype(np.float32)})
                
                # Cleanup
                del encoded, input_ids, hidden_states
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        
        return pd.DataFrame(out_rows)


# ============================================================
# Argument Parser
# ============================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", type=int, default=500,
                    help="Sequence length in bp (must be multiple of 128)")
    
    ap.add_argument("--variant_list_path", type=str, required=True)
    ap.add_argument("--dnv_path", type=str, required=True)
    
    ap.add_argument("--base_ckpt", type=str, 
                    default="InstaDeepAI/NTv3_650M_post")
    ap.add_argument("--lora_adapter_dir", type=str, default="")
    ap.add_argument("--species", type=str, default="human")
    
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--merge_lora", action="store_true")
    ap.add_argument("--use_amp", action="store_true")
    
    ap.add_argument("--data_split", type=int, default=None)
    ap.add_argument("--num_splits", type=int, default=1)
    
    ap.add_argument("--out_path", type=str, required=True)
    
    return ap.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    
    bp = int(args.bp)
    seq_col = f"var_seq_{bp}bp"
    idx_col = f"mut_idx_{bp}bp"
    
    print("[INFO] Loading variant_list:", args.variant_list_path)
    variant_list = pd.read_feather(args.variant_list_path)
    
    print("[INFO] Loading dnv        :", args.dnv_path)
    dnv = pd.read_feather(args.dnv_path)
    
    if seq_col not in dnv.columns or idx_col not in dnv.columns:
        raise ValueError(f"dnv must contain {seq_col} and {idx_col}")
    
    # Data splitting
    if args.data_split is not None and int(args.num_splits) > 1:
        ds = int(args.data_split)
        ns = int(args.num_splits)
        if ds < 0 or ds >= ns:
            raise ValueError(f"--data_split must be in [0, {ns-1}]. Got {ds}")
        
        total_rows = len(dnv)
        split_size = total_rows // ns
        start_idx = ds * split_size
        end_idx = total_rows if ds == ns - 1 else start_idx + split_size
        
        print(f"[DATA SPLIT] split {ds}/{ns} | rows {start_idx}:{end_idx} (total={total_rows})")
        
        variant_list = variant_list.iloc[start_idx:end_idx].reset_index(drop=True)
        dnv = dnv.iloc[start_idx:end_idx].reset_index(drop=True)
    
    # Determine expected sequence length (from data, rounded to 128)
    actual_max_len = int(dnv[seq_col].astype(str).str.len().max())
    expected_seq_length = ((actual_max_len + 127) // 128) * 128
    
    print(f"[INFO] bp={bp} | seq_col={seq_col} | idx_col={idx_col}")
    print(f"[INFO] Actual max length: {actual_max_len}bp")
    print(f"[INFO] Expected seq length: {expected_seq_length}bp (rounded to 128 multiple)")
    
    # Initialize embedder
    embedder = NTv3MutPoolFineTuned(
        base_ckpt=args.base_ckpt,
        expected_seq_length=expected_seq_length,
        species_str=args.species,
        device=args.device,
        batch_size=args.batch_size,
        lora_adapter_dir=args.lora_adapter_dir if args.lora_adapter_dir.strip() else None,
        merge_lora=args.merge_lora,
        use_amp=args.use_amp,
    )
    
    # Extract mutation embeddings
    out_df = embedder.get_mut_embeddings_from_df(
        seq_df=dnv,
        seq_col=seq_col,
        idx_col=idx_col,
        out_col="mut_emb_max",
    )
    
    # Merge with variant list
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
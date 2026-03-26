"""
Convert verl FSDP checkpoint to HuggingFace format for vLLM inference.

Usage:
    python scripts/eval/convert_fsdp_to_hf.py \
        --ckpt_dir checkpoints/grpo_3b/global_step_1200 \
        --output_dir checkpoints/grpo_3b_merged
"""

import argparse
import os
import torch
from pathlib import Path


def _detensor(state_dict: dict) -> tuple:
    """Convert any DTensor values to regular torch.Tensor.

    Uses to_local() which works without distributed init.
    Falls back to full_tensor() then ._local_tensor if needed.

    Returns (state_dict, had_dtensors) - had_dtensors indicates FSDP sharding
    where to_local() returns only the local shard (needs concatenation).
    """
    try:
        from torch.distributed._tensor import DTensor
    except ImportError:
        return state_dict, False

    out = {}
    n_converted = 0
    for k, v in state_dict.items():
        if isinstance(v, DTensor):
            try:
                out[k] = v.to_local()
            except Exception:
                try:
                    out[k] = v.full_tensor()
                except Exception:
                    out[k] = v._local_tensor
            n_converted += 1
        else:
            out[k] = v
    if n_converted:
        print(f"  Converted {n_converted}/{len(state_dict)} DTensor -> Tensor")
    return out, n_converted > 0


def merge_fsdp_to_hf(ckpt_dir: str, output_dir: str):
    ckpt_path = Path(ckpt_dir)

    # Auto-detect structure: GRPO (actor/) vs SFT (flat)
    actor_dir = ckpt_path / "actor"
    if actor_dir.exists() and (actor_dir / "huggingface").exists():
        hf_dir = actor_dir / "huggingface"
        shard_dir = actor_dir
        print("Detected GRPO checkpoint structure (actor/)")
    elif (ckpt_path / "huggingface").exists():
        hf_dir = ckpt_path / "huggingface"
        shard_dir = ckpt_path
        print("Detected SFT checkpoint structure (flat)")
    else:
        print(f"ERROR: No huggingface/ dir found in {ckpt_dir} or {ckpt_dir}/actor/")
        return

    # Find model shards
    shards = sorted(shard_dir.glob("model_world_size_*_rank_*.pt"))
    world_size = len(shards)
    print(f"Found {world_size} model shards")

    # Load all shards
    states = []
    is_dtensor_ckpt = False
    for s in shards:
        size_gb = s.stat().st_size / 1e9
        print(f"Loading {s.name} ({size_gb:.2f} GB)...")
        state = torch.load(s, map_location="cpu", weights_only=False)
        state, had_dt = _detensor(state)
        if had_dt:
            is_dtensor_ckpt = True
        states.append(state)
        print(f"  Keys: {len(state)}")

    # Determine format and merge
    keys_per_rank = [set(s.keys()) for s in states]

    if is_dtensor_ckpt and world_size > 1 and all(k == keys_per_rank[0] for k in keys_per_rank):
        # DTensor checkpoint: to_local() returns local FSDP shard, must concatenate
        print("Format: FSDP DTensor SHARDING (concatenating local shards)")
        merged = {}
        for key in states[0].keys():
            tensors = [s[key] for s in states]
            if tensors[0].dim() == 0:
                merged[key] = tensors[0]  # scalar
            else:
                merged[key] = torch.cat(tensors, dim=0)
    elif all(k == keys_per_rank[0] for k in keys_per_rank):
        # All ranks have the same keys
        sample_key = list(states[0].keys())[0]
        shapes = [s[sample_key].shape for s in states]

        if all(sh == shapes[0] for sh in shapes):
            # Same keys + same shapes = FULL_STATE_DICT (redundant copies)
            print("Format: FULL_STATE_DICT (using rank 0)")
            merged = states[0]
        else:
            # Same keys + different shapes = FSDP flat sharding
            print("Format: FSDP FLAT_SHARDING (concatenating tensors)")
            merged = {}
            for key in states[0].keys():
                tensors = [s[key] for s in states]
                if tensors[0].dim() == 0:
                    merged[key] = tensors[0]  # scalar
                else:
                    merged[key] = torch.cat(tensors, dim=0)
    else:
        # Different keys = parameter-level sharding
        print("Format: PARAMETER_SHARDING (merging by key)")
        merged = {}
        for s in states:
            merged.update(s)

    del states  # free memory

    # Show sample keys
    print(f"\nMerged state dict: {len(merged)} keys")
    for i, (k, v) in enumerate(merged.items()):
        if i < 5:
            print(f"  {k}: {v.shape} {v.dtype}")
        elif i == 5:
            print(f"  ... ({len(merged) - 5} more)")

    # Detect LoRA checkpoint (PEFT format)
    is_lora = any("lora_A" in k for k in merged.keys())
    lora_meta_path = shard_dir / "lora_train_meta.json"

    if is_lora:
        import json
        print("\nDetected LoRA checkpoint (PEFT format)")

        # Read LoRA config (try multiple locations)
        adapter_config_path = shard_dir / "huggingface" / "adapter_config.json"
        if lora_meta_path.exists():
            with open(lora_meta_path) as f:
                lora_meta = json.load(f)
            lora_r = lora_meta.get("r", 64)
            lora_alpha = lora_meta.get("lora_alpha", 128)
            print(f"  Config source: {lora_meta_path}")
        elif adapter_config_path.exists():
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            lora_r = adapter_cfg.get("r", 64)
            lora_alpha = adapter_cfg.get("lora_alpha", 128)
            print(f"  Config source: {adapter_config_path}")
        else:
            lora_r = 64
            lora_alpha = 128
            print(f"  WARNING: No LoRA config found, using defaults r={lora_r}, alpha={lora_alpha}")
        scaling = lora_alpha / lora_r
        print(f"  LoRA r={lora_r}, alpha={lora_alpha}, scaling={scaling}")

        # Merge LoRA weights into base weights
        # Key pattern: base_model.model.model.X.base_layer.weight + lora_A + lora_B
        base_keys = {}
        lora_a_keys = {}
        lora_b_keys = {}
        other_keys = {}

        for k, v in merged.items():
            if ".lora_A.default.weight" in k:
                # Extract module path: base_model.model.model.X.q_proj.lora_A.default.weight -> X.q_proj
                module = k.replace("base_model.model.model.", "").replace(".lora_A.default.weight", "")
                lora_a_keys[module] = v
            elif ".lora_B.default.weight" in k:
                module = k.replace("base_model.model.model.", "").replace(".lora_B.default.weight", "")
                lora_b_keys[module] = v
            elif ".base_layer." in k:
                # base_model.model.model.X.q_proj.base_layer.weight -> model.X.q_proj.weight
                new_k = k.replace("base_model.model.", "").replace(".base_layer", "")
                base_keys[new_k] = v
            else:
                # Non-LoRA keys: base_model.model.model.X -> model.X
                new_k = k.replace("base_model.model.", "")
                other_keys[new_k] = v

        # Merge: W_merged = W_base + (B @ A) * scaling
        n_merged = 0
        for module in lora_a_keys:
            hf_key = "model." + module + ".weight"
            if hf_key in base_keys and module in lora_b_keys:
                A = lora_a_keys[module].float()
                B = lora_b_keys[module].float()
                base_keys[hf_key] = base_keys[hf_key].float() + (B @ A) * scaling
                n_merged += 1

        print(f"  Merged {n_merged}/{len(lora_a_keys)} LoRA layers into base weights")
        if n_merged != len(lora_a_keys):
            unmerged = set(lora_a_keys.keys()) - {m for m in lora_a_keys if "model." + m + ".weight" in base_keys and m in lora_b_keys}
            print(f"  WARNING: {len(unmerged)} LoRA layers NOT merged: {list(unmerged)[:5]}")

        # Combine all keys
        remapped = {}
        for k, v in base_keys.items():
            remapped[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
        for k, v in other_keys.items():
            remapped[k] = v.to(torch.bfloat16) if v.is_floating_point() else v

        del merged
    else:
        # Non-LoRA: original logic
        from transformers import AutoConfig
        ckpt_keys = set(merged.keys())

        # Detect prefix
        prefix_to_remove = ""
        if all(k.startswith("model.") for k in ckpt_keys):
            prefix_to_remove = "model."

        remapped = {}
        for k, v in merged.items():
            new_k = k[len(prefix_to_remove):] if prefix_to_remove and k.startswith(prefix_to_remove) else k
            remapped[new_k] = v.to(torch.bfloat16) if v.is_floating_point() else v

        del merged

    # Load HuggingFace model from config
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print(f"\nLoading config from {hf_dir}...")
    config = AutoConfig.from_pretrained(str(hf_dir), trust_remote_code=True)

    print("Creating empty model from config...")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    model_keys = set(model.state_dict().keys())
    remapped_keys = set(remapped.keys())

    missing = model_keys - remapped_keys
    unexpected = remapped_keys - model_keys
    if missing:
        print(f"  Missing keys (first 5): {list(missing)[:5]}")
    if unexpected:
        print(f"  Unexpected keys (first 5): {list(unexpected)[:5]}")

    try:
        model.load_state_dict(remapped, strict=True)
        print("State dict loaded (strict=True)")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        result = model.load_state_dict(remapped, strict=False)
        print(f"Loaded non-strict: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")

    del remapped

    # Save
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving merged model to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)

    # Copy tokenizer
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # Verify
    files = list(Path(output_dir).glob("*.safetensors"))
    total_size = sum(f.stat().st_size for f in files)
    print(f"\nDone! Saved {len(files)} safetensors files ({total_size / 1e9:.2f} GB)")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert verl FSDP checkpoint to HuggingFace format")
    parser.add_argument("--ckpt_dir", default="checkpoints/grpo_3b/global_step_1200",
                        help="Path to verl checkpoint directory")
    parser.add_argument("--output_dir", default="checkpoints/grpo_3b_merged",
                        help="Output directory for HuggingFace model")
    args = parser.parse_args()

    merge_fsdp_to_hf(args.ckpt_dir, args.output_dir)

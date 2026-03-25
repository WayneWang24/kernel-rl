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
    actor_dir = Path(ckpt_dir) / "actor"
    hf_dir = actor_dir / "huggingface"

    if not hf_dir.exists():
        print(f"ERROR: {hf_dir} not found (need config.json + tokenizer)")
        return

    # Find model shards
    shards = sorted(actor_dir.glob("model_world_size_*_rank_*.pt"))
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

    # Load HuggingFace model from config
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print(f"\nLoading config from {hf_dir}...")
    config = AutoConfig.from_pretrained(str(hf_dir), trust_remote_code=True)

    print("Creating empty model from config...")
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    # Try loading state dict with various key mappings
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(merged.keys())

    # Check if keys match directly
    if model_keys == ckpt_keys:
        print("Keys match directly")
        prefix_to_remove = ""
    elif all(k.startswith("model.") for k in ckpt_keys):
        # verl sometimes saves with "model." prefix
        stripped = {k[len("model."):] for k in ckpt_keys}
        if model_keys == stripped:
            print("Keys match after removing 'model.' prefix")
            prefix_to_remove = "model."
        else:
            prefix_to_remove = ""
    else:
        prefix_to_remove = ""

    # Remap keys and convert dtype
    remapped = {}
    for k, v in merged.items():
        new_k = k[len(prefix_to_remove):] if prefix_to_remove and k.startswith(prefix_to_remove) else k
        remapped[new_k] = v.to(torch.bfloat16) if v.is_floating_point() else v

    del merged  # free memory

    try:
        model.load_state_dict(remapped, strict=True)
        print("State dict loaded (strict=True)")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        result = model.load_state_dict(remapped, strict=False)
        print(f"Loaded non-strict: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")
        if result.missing_keys:
            print(f"  Missing (first 5): {result.missing_keys[:5]}")
        if result.unexpected_keys:
            print(f"  Unexpected (first 5): {result.unexpected_keys[:5]}")

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

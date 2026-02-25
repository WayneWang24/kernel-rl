"""
合并 LoRA 权重到基础模型。

SFT 训练使用 LoRA，RL 训练前需要先合并权重。

用法:
    python scripts/train/merge_lora.py \
        --base_model Qwen/Qwen2.5-Coder-7B-Instruct \
        --lora_path checkpoints/sft/epoch_3 \
        --output_dir checkpoints/sft_merged
"""

import argparse
import os

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights")
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Base model path or name",
    )
    parser.add_argument(
        "--lora_path",
        required=True,
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints/sft_merged",
        help="Output directory for merged model",
    )
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA weights: {args.lora_path}")
    model = PeftModel.from_pretrained(base_model, args.lora_path)

    print("Merging weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir)

    # 也复制 tokenizer
    print("Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()

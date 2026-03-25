"""
Generate ModelNew solutions using transformers (no vLLM needed).

Usage:
    python scripts/eval/generate_with_transformers.py \
        --model_path checkpoints/grpo_3b_merged \
        --run_name grpo-3b-step1200 \
        --levels 1 2 \
        --kernelbench_dir ~/workspace/my-kernel-bench

Then run Phase B (compile+verify) with:
    python scripts/eval/eval_with_compile.py \
        --run_name grpo-3b-step1200 --levels 1 2 \
        --kernelbench_dir ~/workspace/my-kernel-bench \
        --output_dir results --eval_only --num_gpus 2
"""

import argparse
import sys
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.reward.kernel_reward import extract_code_block
from src.prompts.cuda_prompt import CUDA_PROMPT_TEMPLATE


def collect_tasks(kernelbench_data_dir: Path, levels: list) -> list:
    tasks = []
    for level in levels:
        level_dir = kernelbench_data_dir / f"level{level}"
        if not level_dir.exists():
            print(f"  WARNING: {level_dir} not found, skipping")
            continue
        py_files = sorted(
            level_dir.glob("*.py"),
            key=lambda x: int(x.name.split("_")[0]),
        )
        for f in py_files:
            content = f.read_text(encoding="utf-8")
            tasks.append({
                "task_id": f.stem,
                "level": level,
                "model_code": content,
            })
        print(f"  Level {level}: {len(py_files)} tasks")
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--kernelbench_dir", default="~/workspace/my-kernel-bench")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    args = parser.parse_args()

    kernelbench_dir = Path(args.kernelbench_dir).expanduser()
    output_dir = Path(args.output_dir) / args.run_name

    # Collect tasks
    print("Collecting tasks...")
    tasks = collect_tasks(kernelbench_dir / "data", args.levels)
    print(f"Total: {len(tasks)} tasks\n")

    # Load model
    print(f"Loading model from {args.model_path}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.\n")

    # Generate
    ok_count = 0
    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        level = task["level"]
        print(f"[{i + 1}/{len(tasks)}] Level {level} / {task_id}...", end=" ", flush=True)

        prompt = CUDA_PROMPT_TEMPLATE.format(model_code=task["model_code"].strip())
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=1.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        code = extract_code_block(response)

        if code is None:
            print("no code extracted")
            continue

        # Save in my-kernel-bench expected format
        kernel_dir = output_dir / f"level{level}" / task_id
        kernel_dir.mkdir(parents=True, exist_ok=True)
        (kernel_dir / "generated_kernels.py").write_text(code, encoding="utf-8")
        ok_count += 1
        print(f"saved ({len(code)} chars)")

    print(f"\nDone! {ok_count}/{len(tasks)} solutions generated.")
    print(f"Output: {output_dir}")
    print(f"\nNext step - run compile+verify:")
    print(f"  python scripts/eval/eval_with_compile.py \\")
    print(f"      --run_name {args.run_name} --levels {' '.join(map(str, args.levels))} \\")
    print(f"      --kernelbench_dir {args.kernelbench_dir} \\")
    print(f"      --output_dir {args.output_dir} --eval_only --num_gpus 2")


if __name__ == "__main__":
    main()

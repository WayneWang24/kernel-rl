"""
用 my-kernel-bench 的 compile+execute 管线做真实评测。

两阶段评测（避免 GPU 冲突）：
  阶段 A: 启动 vLLM → 生成 ModelNew 代码 → 保存为 generated_kernels.py → 关闭 vLLM
  阶段 B: 导入 my-kernel-bench 的 evaluate_one_level() → 编译 + 验证 → 输出指标

用法:
    # 完整评测（生成 + 编译验证）
    python scripts/eval/eval_with_compile.py \
        --model_path checkpoints/grpo/epoch_20 \
        --run_name grpo_epoch20 \
        --levels 1 2 3 4

    # 仅生成（不做编译验证）
    python scripts/eval/eval_with_compile.py \
        --model_path checkpoints/grpo/epoch_20 \
        --run_name grpo_epoch20 \
        --generate_only

    # 仅评测（已有生成结果）
    python scripts/eval/eval_with_compile.py \
        --run_name grpo_epoch20 \
        --eval_only \
        --levels 1 2 3 4
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

# 添加项目根目录
sys.path.insert(0, str(PROJECT_DIR))
from src.reward.kernel_reward import extract_code_block


# 与训练一致的 prompt 模板
PROMPT_TEMPLATE = """You are an expert Performance Engineer specializing in Triton and PyTorch internals.

### TASK
Optimize the provided architecture named `Model` by replacing standard PyTorch operators with custom Triton kernels.

### RULES
1. Name the optimized output architecture `ModelNew`.
2. Preserve `__init__` structure (nn.Module definitions) for state_dict compatibility.
3. In `forward`, access underlying parameters (e.g., self.conv.weight) and pass them to your custom Triton kernels. Do NOT call module objects directly.
4. Use @triton.jit for kernel functions, include proper wrapper functions.
5. Generate REAL, compilable code with all imports. Output ONLY the code block.

### Input Architecture
```python
{model_code}
```

### Optimized Triton Implementation:"""


def collect_tasks(kernelbench_data_dir: Path, levels: list) -> list:
    """收集指定 level 的所有任务。"""
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
                "ref_filepath": str(f.resolve()),
                "model_code": content,
            })
        print(f"  Level {level}: {len(py_files)} tasks")

    return tasks


def generate_solutions(
    api_base: str,
    model_name: str,
    tasks: list,
    output_dir: Path,
    max_tokens: int = 8192,
    temperature: float = 0.0,
) -> dict:
    """阶段 A：通过 vLLM API 生成 ModelNew 代码，保存为 generated_kernels.py。"""
    results = {}

    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        level = task["level"]
        print(f"  [{i + 1}/{len(tasks)}] Level {level} / {task_id}...", end=" ", flush=True)

        prompt = PROMPT_TEMPLATE.format(model_code=task["model_code"].strip())

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": 1,
        }

        try:
            resp = requests.post(
                f"{api_base}/chat/completions",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            response_text = data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"API error: {e}")
            results[task_id] = {"status": "API_ERROR", "error": str(e)}
            continue

        # 提取代码块
        code = extract_code_block(response_text)
        if code is None:
            print("no code extracted")
            results[task_id] = {"status": "NO_CODE", "response": response_text[:200]}
            continue

        # 保存为 my-kernel-bench 期望的目录结构
        # results/{run_name}/level{N}/{task_stem}/generated_kernels.py
        kernel_dir = output_dir / f"level{level}" / task_id
        kernel_dir.mkdir(parents=True, exist_ok=True)
        kernel_file = kernel_dir / "generated_kernels.py"
        kernel_file.write_text(code, encoding="utf-8")

        print(f"saved ({len(code)} chars)")
        results[task_id] = {
            "status": "OK",
            "code_length": len(code),
            "kernel_file": str(kernel_file),
        }

    return results


def run_evaluation(
    kernelbench_data_dir: Path,
    output_dir: Path,
    run_name: str,
    levels: list,
    num_workers: int = 4,
    num_gpus: int = 2,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> dict:
    """阶段 B：用 my-kernel-bench 的 evaluate_one_level() 做编译 + 验证。"""
    # 动态导入 my-kernel-bench
    kernelbench_root = kernelbench_data_dir.parent
    sys.path.insert(0, str(kernelbench_root))

    try:
        from kernel_bench.evaluation.evaluate_kernels import evaluate_one_level
    except ImportError as e:
        print(f"ERROR: Cannot import my-kernel-bench: {e}")
        print(f"  Ensure my-kernel-bench is at: {kernelbench_root}")
        sys.exit(1)

    all_metrics = {}

    for level in levels:
        level_dir = output_dir / run_name / f"level{level}"
        if not level_dir.exists():
            print(f"  WARNING: No generated kernels for level {level}, skipping")
            continue

        num_kernels = len(list(level_dir.iterdir()))
        print(f"\n--- Evaluating Level {level} ({num_kernels} kernels) ---")

        evaluate_one_level(
            output_dir=output_dir,
            output_name=run_name,
            level=level,
            torch_ref_root=kernelbench_data_dir,
            num_workers=num_workers,
            num_gpus=num_gpus,
            rtol=rtol,
            atol=atol,
        )

        # 读取结果
        result_file = output_dir / run_name / f"level{level}.json"
        if result_file.exists():
            with open(result_file, "r") as f:
                level_results = json.load(f)

            # 计算指标
            total = len(level_results)
            exists = sum(1 for r in level_results.values() if r["is_exist"])
            compiled = sum(1 for r in level_results.values() if r["compilation"])
            verified = sum(1 for r in level_results.values() if r["verification"])

            metrics = {
                "total_tasks": total,
                "generated": exists,
                "compile_pass": compiled,
                "verify_pass": verified,
                "compile_pass_rate": f"{compiled / max(total, 1) * 100:.1f}%",
                "verify_pass_rate": f"{verified / max(total, 1) * 100:.1f}%",
            }

            all_metrics[f"level{level}"] = metrics

            print(f"  Total: {total}, Generated: {exists}, "
                  f"Compile: {compiled} ({compiled / max(total, 1) * 100:.1f}%), "
                  f"Verify: {verified} ({verified / max(total, 1) * 100:.1f}%)")

            # 逐题状态
            for task_name, r in level_results.items():
                status = "PASS" if r["verification"] else (
                    "COMPILE_OK" if r["compilation"] else (
                        "NO_GEN" if not r["is_exist"] else "COMPILE_FAIL"
                    )
                )
                if status != "PASS":
                    print(f"    {task_name}: {status}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate kernel generation with compile+execute verification"
    )
    parser.add_argument("--model_path", help="Model path for vLLM generation")
    parser.add_argument("--run_name", required=True, help="Name for this evaluation run")
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[1, 2, 3, 4],
        help="KernelBench levels to evaluate",
    )
    parser.add_argument(
        "--kernelbench_dir",
        default=os.path.expanduser("~/Code/my-kernel-bench"),
        help="Path to my-kernel-bench repo",
    )
    parser.add_argument("--output_dir", default="results", help="Output root directory")

    # 生成参数
    parser.add_argument("--api_base", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", default=None, help="Model name for API (defaults to model_path)")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)

    # 评测参数
    parser.add_argument("--num_workers", type=int, default=4, help="Compile parallelism")
    parser.add_argument("--num_gpus", type=int, default=2, help="GPU parallelism for verification")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)

    # 模式选择
    parser.add_argument("--generate_only", action="store_true", help="Only generate, skip evaluation")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate existing results")

    args = parser.parse_args()

    kernelbench_data_dir = Path(args.kernelbench_dir) / "data"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_name or args.model_path

    # ======== 阶段 A: 生成 ========
    if not args.eval_only:
        if not args.model_path:
            print("ERROR: --model_path required for generation phase")
            sys.exit(1)

        print(f"=== Phase A: Generation ===")
        print(f"  Model: {model_name}")
        print(f"  Output: {output_dir / args.run_name}")

        tasks = collect_tasks(kernelbench_data_dir, args.levels)
        print(f"  Total tasks: {len(tasks)}\n")

        gen_results = generate_solutions(
            api_base=args.api_base,
            model_name=model_name,
            tasks=tasks,
            output_dir=output_dir / args.run_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # 保存生成摘要
        gen_summary = {
            "model": model_name,
            "total": len(tasks),
            "ok": sum(1 for r in gen_results.values() if r["status"] == "OK"),
            "no_code": sum(1 for r in gen_results.values() if r["status"] == "NO_CODE"),
            "api_error": sum(1 for r in gen_results.values() if r["status"] == "API_ERROR"),
        }
        summary_path = output_dir / args.run_name / "generation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(gen_summary, f, indent=2)
        print(f"\nGeneration summary: {gen_summary}")

    if args.generate_only:
        print("\n--generate_only mode, skipping evaluation.")
        return

    # ======== 阶段 B: 编译验证评测 ========
    print(f"\n=== Phase B: Compile + Verify Evaluation ===")
    all_metrics = run_evaluation(
        kernelbench_data_dir=kernelbench_data_dir,
        output_dir=output_dir,
        run_name=args.run_name,
        levels=args.levels,
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        rtol=args.rtol,
        atol=args.atol,
    )

    # 保存汇总指标
    metrics_path = output_dir / args.run_name / "compile_eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n=== Final Metrics ===")
    for level_name, metrics in all_metrics.items():
        print(f"  {level_name}: compile={metrics['compile_pass_rate']}, verify={metrics['verify_pass_rate']}")

    print(f"\nResults saved to {output_dir / args.run_name}/")


if __name__ == "__main__":
    main()

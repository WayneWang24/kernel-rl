"""
KernelBench 评测执行器。

对指定 level 的所有任务生成 Triton kernel，然后验证正确性和性能。

用法:
    python scripts/eval/eval_kernelbench.py \
        --kernelbench_dir /tmp/KernelBench \
        --level 1 \
        --api_base http://localhost:8000/v1 \
        --model_name Qwen/Qwen2.5-Coder-7B-Instruct \
        --output_dir results/baseline
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import Optional

import requests

# 添加项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.reward.kernel_reward import compute_score, extract_code_block


TRITON_PROMPT_TEMPLATE = """You are an expert GPU programmer specializing in Triton kernel development. Your task is to convert the following PyTorch code into an optimized Triton kernel implementation.

## Requirements:
1. The Triton kernel must be functionally equivalent to the PyTorch code
2. Use @triton.jit decorator for kernel functions
3. Include a Python wrapper function that calls the Triton kernel
4. Optimize for GPU performance (memory coalescing, optimal block sizing, etc.)
5. Include proper imports (triton, triton.language as tl, torch, etc.)

## PyTorch Code:
```python
{python_code}
```

## Optimized Triton Implementation:"""


def load_problems(kernelbench_dir: str, level: int) -> list:
    """加载 KernelBench 指定 level 的所有问题。"""
    level_dir = os.path.join(kernelbench_dir, "KernelBench", "level" + str(level))
    if not os.path.isdir(level_dir):
        # 尝试另一种目录结构
        level_dir = os.path.join(kernelbench_dir, "data", f"level{level}")

    if not os.path.isdir(level_dir):
        print(f"WARNING: Level directory not found: {level_dir}")
        return []

    problems = []
    py_files = sorted(glob.glob(os.path.join(level_dir, "*.py")))
    for filepath in py_files:
        filename = os.path.basename(filepath)
        match = re.match(r"(\d+)_", filename)
        problem_id = match.group(1) if match else filename

        with open(filepath, "r") as f:
            code = f.read()

        problems.append({
            "level": level,
            "problem_id": problem_id,
            "filename": filename,
            "filepath": filepath,
            "python_code": code,
        })

    return problems


def generate_solution(
    api_base: str,
    model_name: str,
    python_code: str,
    n_samples: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> list:
    """通过 OpenAI-compatible API 生成解决方案。"""
    prompt = TRITON_PROMPT_TEMPLATE.format(python_code=python_code)

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n_samples,
    }

    try:
        resp = requests.post(
            f"{api_base}/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        solutions = []
        for choice in data["choices"]:
            solutions.append(choice["message"]["content"])
        return solutions

    except Exception as e:
        print(f"  API error: {e}")
        return []


def evaluate_problem(
    problem: dict,
    solutions: list,
) -> dict:
    """评估单个问题的所有解决方案。"""
    results = {
        "problem_id": problem["problem_id"],
        "level": problem["level"],
        "filename": problem["filename"],
        "solutions": [],
        "best_score": 0.0,
    }

    for i, solution in enumerate(solutions):
        score = compute_score(
            data_source="kernelbench",
            solution_str=solution,
            ground_truth=problem["python_code"],
        )
        results["solutions"].append({
            "index": i,
            "score": score,
            "code_extracted": extract_code_block(solution) is not None,
            "response_length": len(solution),
        })
        results["best_score"] = max(results["best_score"], score)

    return results


def compute_metrics(all_results: list) -> dict:
    """计算汇总指标。"""
    total = len(all_results)
    if total == 0:
        return {"total": 0}

    # 基于静态 score 的指标
    scores = [r["best_score"] for r in all_results]
    has_code = sum(1 for s in scores if s > 0.0)
    has_triton = sum(1 for s in scores if s >= 0.4)
    has_complete = sum(1 for s in scores if s >= 0.8)

    return {
        "total": total,
        "has_code_pct": f"{has_code / total * 100:.1f}%",
        "has_triton_pct": f"{has_triton / total * 100:.1f}%",
        "has_complete_pct": f"{has_complete / total * 100:.1f}%",
        "avg_score": f"{sum(scores) / total:.3f}",
        "score_distribution": {
            "0.0": sum(1 for s in scores if s == 0.0),
            "0.1": sum(1 for s in scores if s == 0.1),
            "0.2": sum(1 for s in scores if s == 0.2),
            "0.4": sum(1 for s in scores if s == 0.4),
            "0.6": sum(1 for s in scores if s == 0.6),
            "0.8+": sum(1 for s in scores if s >= 0.8),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate on KernelBench")
    parser.add_argument("--kernelbench_dir", required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--api_base", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="eval")
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载问题
    problems = load_problems(args.kernelbench_dir, args.level)
    print(f"Loaded {len(problems)} problems from level {args.level}")

    if not problems:
        print("No problems found. Check KernelBench directory structure.")
        return

    # 逐题评测
    all_results = []
    for i, problem in enumerate(problems):
        print(f"  [{i + 1}/{len(problems)}] Problem {problem['problem_id']}...", end=" ")

        solutions = generate_solution(
            args.api_base,
            args.model_name,
            problem["python_code"],
            args.n_samples,
            args.temperature,
            args.max_tokens,
        )

        if solutions:
            result = evaluate_problem(problem, solutions)
            all_results.append(result)
            print(f"score={result['best_score']:.1f}")
        else:
            print("FAILED (no solution)")
            all_results.append({
                "problem_id": problem["problem_id"],
                "level": problem["level"],
                "filename": problem["filename"],
                "solutions": [],
                "best_score": 0.0,
            })

    # 计算指标
    metrics = compute_metrics(all_results)
    print(f"\n=== Level {args.level} Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 保存结果
    output = {
        "run_name": args.run_name,
        "model": args.model_name,
        "level": args.level,
        "metrics": metrics,
        "results": all_results,
    }

    output_path = os.path.join(args.output_dir, f"level{args.level}_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

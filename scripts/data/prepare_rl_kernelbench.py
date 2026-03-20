"""
将 KernelBench 270 个任务转为 verl RL parquet 格式。

读取 my-kernel-bench/data/level{1,2,3,4}/*.py，用 Triton 版 prompt 模板，
生成 ModelNew 格式的 RL 训练数据。

拆分策略：
  - level 1+2 → train (~200 条)
  - level 3   → val   (~50 条)
  - level 4   → test  (~20 条)

输出：data/rl_kernelbench/{train,val,test}.parquet

用法:
    python scripts/data/prepare_rl_kernelbench.py \
        --kernelbench_dir /path/to/my-kernel-bench \
        --output_dir data/rl_kernelbench
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

# 添加项目根目录以 import 集中 prompt 模板
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)

from src.prompts.cuda_prompt import CUDA_PROMPT_TEMPLATE, TRITON_PROMPT_TEMPLATE

# 默认使用 CUDA prompt；通过 --backend triton 切换
PROMPT_TEMPLATES = {
    "cuda": CUDA_PROMPT_TEMPLATE,
    "triton": TRITON_PROMPT_TEMPLATE,
}


def parse_task_file(filepath: Path) -> dict:
    """解析 KernelBench 任务文件，提取 Model 类代码和元数据。"""
    content = filepath.read_text(encoding="utf-8")

    # 提取文件名作为 task_id（如 "19_ReLU"）
    task_id = filepath.stem

    # 提取 level（从父目录名）
    level = int(filepath.parent.name.replace("level", ""))

    return {
        "task_id": task_id,
        "level": level,
        "ref_filepath": str(filepath.resolve()),
        "model_code": content,
    }


def build_rl_record(task: dict, backend: str = "cuda") -> dict:
    """将单个任务转为 verl RL 格式。"""
    prompt_template = PROMPT_TEMPLATES[backend]
    prompt_text = prompt_template.format(model_code=task["model_code"].strip())

    ground_truth = {
        "task_id": task["task_id"],
        "level": task["level"],
        "ref_filepath": task["ref_filepath"],
        "model_code": task["model_code"],
        "backend": backend,
    }

    return {
        "data_source": "kernelbench",
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": f"{backend}_kernel_modelnew",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "task_id": task["task_id"],
            "level": task["level"],
            "backend": backend,
        },
    }


def collect_tasks(kernelbench_dir: Path) -> list:
    """收集所有 level 的任务。"""
    tasks = []
    for level in [1, 2, 3, 4]:
        level_dir = kernelbench_dir / "data" / f"level{level}"
        if not level_dir.exists():
            print(f"  WARNING: {level_dir} not found, skipping")
            continue

        py_files = sorted(
            level_dir.glob("*.py"),
            key=lambda x: int(x.name.split("_")[0]),
        )
        for f in py_files:
            task = parse_task_file(f)
            tasks.append(task)
        print(f"  Level {level}: {len(py_files)} tasks")

    return tasks


def split_by_level(tasks: list) -> dict:
    """按 level 拆分：level1+2 → train, level3 → val, level4 → test。"""
    train = [t for t in tasks if t["level"] in (1, 2)]
    val = [t for t in tasks if t["level"] == 3]
    test = [t for t in tasks if t["level"] == 4]
    return {"train": train, "val": val, "test": test}


def main():
    parser = argparse.ArgumentParser(
        description="Prepare KernelBench tasks as verl RL data"
    )
    parser.add_argument(
        "--kernelbench_dir",
        default=os.path.expanduser("~/Code/my-kernel-bench"),
        help="Path to my-kernel-bench repo",
    )
    parser.add_argument(
        "--output_dir",
        default="data/rl_kernelbench",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--backend",
        choices=["cuda", "triton"],
        default="cuda",
        help="Backend for prompt template (default: cuda)",
    )
    args = parser.parse_args()

    kernelbench_dir = Path(args.kernelbench_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting tasks from {kernelbench_dir}/data/...")
    tasks = collect_tasks(kernelbench_dir)
    print(f"  Total: {len(tasks)} tasks\n")

    # 按 level 拆分
    splits = split_by_level(tasks)
    for split_name, split_tasks in splits.items():
        print(f"  {split_name}: {len(split_tasks)} tasks (levels: {sorted(set(t['level'] for t in split_tasks))})")

    # 转换并保存
    print("\nConverting to RL format...")
    for split_name, split_tasks in splits.items():
        records = [build_rl_record(t, backend=args.backend) for t in split_tasks]
        df = pd.DataFrame(records)
        output_path = output_dir / f"{split_name}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"  Saved {len(df)} records to {output_path}")

    # 验证
    print("\n=== Verification ===")
    check = pd.read_parquet(output_dir / "train.parquet")
    sample = check.iloc[0]
    print(f"  Columns: {list(check.columns)}")
    print(f"  data_source: {sample['data_source']}")
    print(f"  prompt type: {type(sample['prompt'])}")
    prompt_text = sample["prompt"][0]["content"]
    print(f"  prompt length: {len(prompt_text)} chars")
    print(f"  prompt preview (first 300 chars):\n    {prompt_text[:300]}...")
    print(f"  ability: {sample['ability']}")
    gt = sample["reward_model"]["ground_truth"]
    print(f"  ground_truth keys: {list(gt.keys())}")
    print(f"  ground_truth.task_id: {gt['task_id']}")
    print(f"  ground_truth.level: {gt['level']}")
    print(f"  extra_info: {sample['extra_info']}")

    # 检查 5 条样本
    print("\n=== Sample Tasks ===")
    for i in range(min(5, len(check))):
        row = check.iloc[i]
        gt = row["reward_model"]["ground_truth"]
        print(f"  [{i}] task_id={gt['task_id']}, level={gt['level']}, "
              f"prompt_len={len(row['prompt'][0]['content'])}")


if __name__ == "__main__":
    main()

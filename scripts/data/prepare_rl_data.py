"""
将清洗后的 KernelBook 数据转换为 verl RL 训练格式。

输出格式（verl parquet）：
{
    "data_source": "kernelbook",
    "prompt": [{"role": "user", "content": "..."}],
    "ability": "triton_kernel",
    "reward_model": {"style": "rule", "ground_truth": "<python_code>"},
    "extra_info": {"entry_point": "...", "repo_name": "...", "module_name": "..."}
}

用法:
    python scripts/data/prepare_rl_data.py \
        --input data/cleaned/kernelbook_clean.parquet \
        --output_dir data/rl
"""

import argparse
import os
import random
import sys
from collections import defaultdict
from typing import Tuple

import pandas as pd

# 复用 SFT 脚本的 prompt 模板和 split 函数
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


PROMPT_TEMPLATE = """You are an expert GPU programmer specializing in Triton kernel development. Your task is to convert the following PyTorch code into an optimized Triton kernel implementation.

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


def split_by_repo(
    df: pd.DataFrame,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按 repo_name 分组划分（与 SFT 使用相同 seed 保持一致）。"""
    random.seed(seed)

    if "repo_name" not in df.columns:
        indices = list(range(len(df)))
        random.shuffle(indices)
        n_train = int(len(df) * train_ratio)
        n_val = int(len(df) * val_ratio)
        return (
            df.iloc[indices[:n_train]],
            df.iloc[indices[n_train : n_train + n_val]],
            df.iloc[indices[n_train + n_val :]],
        )

    repo_groups = defaultdict(list)
    for idx, row in df.iterrows():
        repo_groups[row["repo_name"]].append(idx)

    repos = list(repo_groups.keys())
    random.shuffle(repos)

    total = len(df)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    train_indices, val_indices, test_indices = [], [], []
    cumulative = 0

    for repo in repos:
        indices = repo_groups[repo]
        if cumulative < n_train:
            train_indices.extend(indices)
        elif cumulative < n_train + n_val:
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)
        cumulative += len(indices)

    if not val_indices and train_indices:
        val_indices = train_indices[-max(1, len(train_indices) // 20) :]
        train_indices = train_indices[: -len(val_indices)]
    if not test_indices and train_indices:
        test_indices = train_indices[-max(1, len(train_indices) // 20) :]
        train_indices = train_indices[: -len(test_indices)]

    return (
        df.loc[train_indices],
        df.loc[val_indices],
        df.loc[test_indices],
    )


def convert_to_rl_format(df: pd.DataFrame) -> pd.DataFrame:
    """将原始数据转换为 verl RL 格式。"""
    records = []
    for _, row in df.iterrows():
        prompt_text = PROMPT_TEMPLATE.format(python_code=row["python_code"].strip())

        extra_info = {}
        for key in ["entry_point", "repo_name", "module_name"]:
            if key in row and pd.notna(row[key]):
                extra_info[key] = row[key]

        record = {
            "data_source": "kernelbook",
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "triton_kernel",
            "reward_model": {
                "style": "rule",
                "ground_truth": row["python_code"].strip(),
            },
            "extra_info": extra_info,
        }
        records.append(record)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Prepare RL data from cleaned KernelBook")
    parser.add_argument(
        "--input",
        default="data/cleaned/kernelbook_clean.parquet",
        help="Input cleaned parquet",
    )
    parser.add_argument("--output_dir", default="data/rl", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading cleaned data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df)} records")

    # 划分（使用相同 seed 保持与 SFT 一致的划分）
    train_df, val_df, test_df = split_by_repo(
        df, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"  Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    # 转换格式并保存
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rl_df = convert_to_rl_format(split_df)
        output_path = os.path.join(args.output_dir, f"{split_name}.parquet")
        rl_df.to_parquet(output_path, index=False)
        print(f"  Saved {len(rl_df)} records to {output_path}")

    # 验证
    print("\n=== Verification ===")
    check = pd.read_parquet(os.path.join(args.output_dir, "train.parquet"))
    sample = check.iloc[0]
    print(f"  Columns: {list(check.columns)}")
    print(f"  data_source: {sample['data_source']}")
    print(f"  prompt type: {type(sample['prompt'])}")
    print(f"  prompt length: {len(str(sample['prompt']))} chars")
    print(f"  ability: {sample['ability']}")
    print(f"  reward_model keys: {list(sample['reward_model'].keys())}")
    print(f"  extra_info: {sample['extra_info']}")


if __name__ == "__main__":
    main()

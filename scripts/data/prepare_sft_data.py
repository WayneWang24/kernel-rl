"""
将清洗后的 KernelBook 数据转换为 verl SFT 格式。

输出格式（多轮对话 parquet）：
{
    "messages": [
        {"role": "user", "content": "<prompt with python_code>"},
        {"role": "assistant", "content": "<triton_code>"}
    ]
}

用法:
    python scripts/data/prepare_sft_data.py \
        --input data/cleaned/kernelbook_clean.parquet \
        --output_dir data/sft \
        --train_ratio 0.9 \
        --val_ratio 0.05
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import List, Tuple

import pandas as pd


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


def build_sft_messages(python_code: str, triton_code: str) -> list:
    """构造 SFT 消息格式。"""
    prompt = PROMPT_TEMPLATE.format(python_code=python_code.strip())
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": triton_code.strip()},
    ]


def split_by_repo(
    df: pd.DataFrame,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按 repo_name 分组划分数据集。
    同一 repo 的数据不会跨 train/val/test。
    """
    random.seed(seed)

    if "repo_name" not in df.columns:
        # 回退到随机划分
        indices = list(range(len(df)))
        random.shuffle(indices)
        n_train = int(len(df) * train_ratio)
        n_val = int(len(df) * val_ratio)
        return (
            df.iloc[indices[:n_train]],
            df.iloc[indices[n_train : n_train + n_val]],
            df.iloc[indices[n_train + n_val :]],
        )

    # 按 repo 分组
    repo_groups = defaultdict(list)
    for idx, row in df.iterrows():
        repo_groups[row["repo_name"]].append(idx)

    # 打乱 repo 顺序
    repos = list(repo_groups.keys())
    random.shuffle(repos)

    # 按累积样本数分割
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

    # 确保 val 和 test 不为空
    if not val_indices and train_indices:
        # 从 train 末尾取一些
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


def convert_to_sft_format(df: pd.DataFrame) -> pd.DataFrame:
    """将原始数据转换为 SFT 格式。"""
    records = []
    for _, row in df.iterrows():
        messages = build_sft_messages(row["python_code"], row["triton_code"])
        record = {"messages": messages}
        # 保留元数据（可选）
        if "entry_point" in row:
            record["entry_point"] = row["entry_point"]
        if "repo_name" in row:
            record["repo_name"] = row["repo_name"]
        records.append(record)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data from cleaned KernelBook")
    parser.add_argument(
        "--input",
        default="data/cleaned/kernelbook_clean.parquet",
        help="Input cleaned parquet",
    )
    parser.add_argument("--output_dir", default="data/sft", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载清洗数据
    print(f"Loading cleaned data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df)} records")

    # 划分
    train_df, val_df, test_df = split_by_repo(
        df, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"  Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    # 转换格式
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        sft_df = convert_to_sft_format(split_df)
        output_path = os.path.join(args.output_dir, f"{split_name}.parquet")
        sft_df.to_parquet(output_path, index=False)
        print(f"  Saved {len(sft_df)} records to {output_path}")

    # 验证
    print("\n=== Verification ===")
    check = pd.read_parquet(os.path.join(args.output_dir, "train.parquet"))
    sample = check.iloc[0]
    messages = sample["messages"]
    print(f"  Columns: {list(check.columns)}")
    print(f"  First sample messages count: {len(messages)}")
    print(f"  User prompt length: {len(messages[0]['content'])} chars")
    print(f"  Assistant response length: {len(messages[1]['content'])} chars")
    print(f"  User prompt preview: {messages[0]['content'][:200]}...")


if __name__ == "__main__":
    main()

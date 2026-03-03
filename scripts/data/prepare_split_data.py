"""
统一数据划分脚本：将 KernelBook 原始数据按方案 C 划分为 SFT + RL。

策略：SFT 学格式（高质量 + 全复杂度覆盖），RL 学质量（高复杂度 + 有优化）。
两者都使用 ModelNew 格式，按 repo 分组防止数据泄露。
评测使用 KernelBench Level 1-4（完全 held-out）。

用法:
    python scripts/data/prepare_split_data.py \
        --input data/raw/kernelbook_raw.parquet \
        --output_dir data/split

输出:
    data/split/sft/train.parquet    SFT 训练集
    data/split/sft/val.parquet      SFT 验证集
    data/split/rl/train.parquet     RL 训练集
    data/split/rl/val.parquet       RL 验证集
    data/split/split_stats.json     划分统计
"""

import argparse
import ast
import json
import os
import random
import re
import sys
import textwrap
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# 复用已有的 AST 工具
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)

from scripts.data.prepare_sft_modelnew import (
    MODELNEW_PROMPT_TEMPLATE,
    ORIGINAL_PROMPT_TEMPLATE,
    try_convert_to_modelnew,
)


# ============================================================
# 特征计算
# ============================================================

def estimate_tokens_simple(text: str) -> int:
    """简单 token 估计（字符数 / 4）。"""
    return len(text) // 4


def count_triton_kernels(code: str) -> int:
    """统计 @triton.jit / @triton.autotune 装饰的函数数量。"""
    return len(re.findall(r"@triton\.(?:jit|autotune)", code))


def has_nn_module(code: str) -> bool:
    """检查是否包含 nn.Module 子类。"""
    return bool(re.search(r"class\s+\w+\s*\(.*?(?:nn\.Module|torch\.nn\.Module)", code))


def has_autotune(code: str) -> bool:
    return bool(re.search(r"@triton\.autotune", code))


def has_block_size(code: str) -> bool:
    return bool(re.search(r"BLOCK_SIZE|tl\.constexpr|num_warps|num_stages", code))


def has_wrapper(code: str) -> bool:
    """检查是否有非 @triton.jit 的 wrapper 函数。"""
    defs = set(re.findall(r"def\s+(\w+)", code))
    jit_funcs = set(re.findall(r"@triton\.(?:jit|autotune)\s*(?:\([^)]*\))?\s*\ndef\s+(\w+)", code))
    return len(defs - jit_funcs) > 0


def is_non_trivial(code: str, ref: str) -> bool:
    """Jaccard 相似度 < 0.8 则视为非简单复制。"""
    code_lines = set(line.strip() for line in code.split("\n") if line.strip())
    ref_lines = set(line.strip() for line in ref.split("\n") if line.strip())
    if not code_lines:
        return False
    overlap = len(code_lines & ref_lines)
    jaccard = overlap / max(len(code_lines | ref_lines), 1)
    return jaccard < 0.8


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """为每条数据计算特征。"""
    df = df.copy()

    # 基础特征
    df["prompt_tokens"] = df["python_code"].apply(estimate_tokens_simple)
    df["response_tokens"] = df["triton_code"].apply(estimate_tokens_simple)
    df["n_kernels"] = df["triton_code"].apply(count_triton_kernels)
    df["has_nn_module"] = df["python_code"].apply(has_nn_module)
    df["has_autotune"] = df["triton_code"].apply(has_autotune)
    df["has_block_size"] = df["triton_code"].apply(has_block_size)
    df["has_wrapper"] = df["triton_code"].apply(has_wrapper)
    df["is_non_trivial"] = df.apply(
        lambda r: is_non_trivial(r["triton_code"], r["python_code"]), axis=1
    )

    # ModelNew 可转换性
    def check_convertible(row):
        try:
            return try_convert_to_modelnew(row["python_code"], row["triton_code"]) is not None
        except Exception:
            return False

    df["modelnew_convertible"] = df.apply(check_convertible, axis=1)

    # 复合分数
    # complexity: 归一化 prompt token 数 + kernel 数 + nn.Module
    prompt_max = max(df["prompt_tokens"].max(), 1)
    df["complexity_score"] = (
        df["prompt_tokens"] / prompt_max
        + df["n_kernels"] * 0.3
        + df["has_nn_module"].astype(float) * 0.2
    )

    # quality: autotune + block_size + wrapper + non_trivial
    df["quality_score"] = (
        df["has_autotune"].astype(float) * 0.3
        + df["has_block_size"].astype(float) * 0.2
        + df["has_wrapper"].astype(float) * 0.3
        + df["is_non_trivial"].astype(float) * 0.2
    )

    return df


# ============================================================
# 数据划分
# ============================================================

def split_by_repo_scheme_c(
    df: pd.DataFrame,
    sft_ratio: float = 0.65,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> dict:
    """方案 C 划分：SFT 学格式 + RL 学质量。

    按 repo 分组，根据 repo 内数据的特征决定分配到 SFT 还是 RL。
    - ModelNew 可转换率高 + stars 高的 repo → SFT
    - complexity/quality 分数高的 repo → RL
    """
    random.seed(seed)
    np.random.seed(seed)

    # 按 repo 汇总特征
    if "repo_name" not in df.columns:
        # 没有 repo 信息，退化为随机分割
        indices = list(range(len(df)))
        random.shuffle(indices)
        n_sft = int(len(df) * sft_ratio)
        sft_idx = indices[:n_sft]
        rl_idx = indices[n_sft:]
        return {"sft": sft_idx, "rl": rl_idx}

    repo_stats = df.groupby("repo_name").agg(
        count=("python_code", "len"),
        modelnew_rate=("modelnew_convertible", "mean"),
        mean_complexity=("complexity_score", "mean"),
        mean_quality=("quality_score", "mean"),
        mean_stars=("stars", lambda x: x.mean() if "stars" in df.columns else 0),
    ).reset_index()

    # 给 repo 打分：SFT 优先度 vs RL 优先度
    # SFT 优先：高 ModelNew 转换率 + 高 stars
    repo_stats["sft_priority"] = (
        repo_stats["modelnew_rate"] * 0.5
        + (repo_stats["mean_stars"] / max(repo_stats["mean_stars"].max(), 1)) * 0.3
        + (1 - repo_stats["mean_complexity"] / max(repo_stats["mean_complexity"].max(), 1)) * 0.2
    )

    # RL 优先：高复杂度 + 高质量
    repo_stats["rl_priority"] = (
        repo_stats["mean_complexity"] / max(repo_stats["mean_complexity"].max(), 1) * 0.5
        + repo_stats["mean_quality"] / max(repo_stats["mean_quality"].max(), 1) * 0.5
    )

    # 按 SFT 优先度排序，前 65% 的样本量分给 SFT
    repo_stats = repo_stats.sort_values("sft_priority", ascending=False)

    total = len(df)
    target_sft = int(total * sft_ratio)

    sft_repos = set()
    rl_repos = set()
    cumulative = 0

    for _, row in repo_stats.iterrows():
        if cumulative < target_sft:
            sft_repos.add(row["repo_name"])
        else:
            rl_repos.add(row["repo_name"])
        cumulative += row["count"]

    sft_idx = df[df["repo_name"].isin(sft_repos)].index.tolist()
    rl_idx = df[df["repo_name"].isin(rl_repos)].index.tolist()

    return {"sft": sft_idx, "rl": rl_idx}


# ============================================================
# 格式转换
# ============================================================

def convert_to_sft_format(row: pd.Series) -> dict:
    """转为 SFT 格式（尝试 ModelNew，失败回退原始格式）。"""
    result = try_convert_to_modelnew(row["python_code"], row["triton_code"])

    if result is not None:
        model_code, modelnew_code = result
        prompt = MODELNEW_PROMPT_TEMPLATE.format(model_code=model_code.strip())
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": modelnew_code.strip()},
            ],
            "format": "modelnew",
        }
    else:
        prompt = ORIGINAL_PROMPT_TEMPLATE.format(python_code=row["python_code"].strip())
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": row["triton_code"].strip()},
            ],
            "format": "original",
        }


def convert_to_rl_format(row: pd.Series) -> Optional[dict]:
    """转为 RL 格式（只有 prompt，无 reference answer）。

    使用 ModelNew 格式 prompt，ground_truth 包含原始 python_code 用于 reward 计算。
    """
    # 尝试转为 ModelNew 格式的 prompt
    result = try_convert_to_modelnew(row["python_code"], row["triton_code"])

    if result is not None:
        model_code, _ = result
        prompt_text = MODELNEW_PROMPT_TEMPLATE.format(model_code=model_code.strip())
        ground_truth = {
            "python_code": row["python_code"],
            "triton_code": row["triton_code"],
            "model_code": model_code.strip(),
            "format": "modelnew",
        }
    else:
        prompt_text = ORIGINAL_PROMPT_TEMPLATE.format(python_code=row["python_code"].strip())
        ground_truth = {
            "python_code": row["python_code"],
            "triton_code": row["triton_code"],
            "format": "original",
        }

    return {
        "data_source": "kernelbook",
        "prompt": [{"role": "user", "content": prompt_text}],
        "ability": "triton_kernel",
        "reward_model": {"ground_truth": ground_truth},
        "extra_info": {
            "repo_name": row.get("repo_name", ""),
            "entry_point": row.get("entry_point", ""),
            "format": ground_truth["format"],
        },
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Split KernelBook into SFT + RL (Scheme C)"
    )
    parser.add_argument(
        "--input",
        default="data/raw/kernelbook_raw.parquet",
        help="Raw KernelBook parquet",
    )
    parser.add_argument("--output_dir", default="data/split", help="Output directory")
    parser.add_argument("--sft_ratio", type=float, default=0.65)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 加载原始数据
    print(f"Loading raw data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df)} records")
    print(f"  Columns: {list(df.columns)}")

    # 只去空值
    before = len(df)
    df = df[
        df["python_code"].notna()
        & df["triton_code"].notna()
        & (df["python_code"].str.strip() != "")
        & (df["triton_code"].str.strip() != "")
    ].copy()
    df = df.reset_index(drop=True)
    print(f"  After removing empty: {len(df)} (removed {before - len(df)})")

    # 计算特征
    print("\nComputing features...")
    df = compute_features(df)
    print(f"  complexity_score: mean={df['complexity_score'].mean():.3f}, "
          f"median={df['complexity_score'].median():.3f}")
    print(f"  quality_score: mean={df['quality_score'].mean():.3f}, "
          f"median={df['quality_score'].median():.3f}")
    print(f"  ModelNew convertible: {df['modelnew_convertible'].sum()} "
          f"({df['modelnew_convertible'].mean() * 100:.1f}%)")

    # 划分
    print(f"\nSplitting (SFT={args.sft_ratio:.0%}, RL={1 - args.sft_ratio:.0%})...")
    split = split_by_repo_scheme_c(df, args.sft_ratio, args.val_ratio, args.seed)

    sft_df = df.loc[split["sft"]].copy()
    rl_df = df.loc[split["rl"]].copy()

    print(f"  SFT: {len(sft_df)} records ({len(sft_df) / len(df) * 100:.1f}%)")
    print(f"    ModelNew convertible: {sft_df['modelnew_convertible'].sum()} "
          f"({sft_df['modelnew_convertible'].mean() * 100:.1f}%)")
    print(f"    Mean complexity: {sft_df['complexity_score'].mean():.3f}")
    print(f"    Mean quality: {sft_df['quality_score'].mean():.3f}")

    print(f"  RL: {len(rl_df)} records ({len(rl_df) / len(df) * 100:.1f}%)")
    print(f"    ModelNew convertible: {rl_df['modelnew_convertible'].sum()} "
          f"({rl_df['modelnew_convertible'].mean() * 100:.1f}%)")
    print(f"    Mean complexity: {rl_df['complexity_score'].mean():.3f}")
    print(f"    Mean quality: {rl_df['quality_score'].mean():.3f}")

    # SFT 内部划分 train/val
    sft_indices = list(range(len(sft_df)))
    random.seed(args.seed)
    random.shuffle(sft_indices)
    n_sft_val = max(1, int(len(sft_df) * args.val_ratio))
    sft_train = sft_df.iloc[sft_indices[n_sft_val:]]
    sft_val = sft_df.iloc[sft_indices[:n_sft_val]]

    # RL 内部划分 train/val
    rl_indices = list(range(len(rl_df)))
    random.shuffle(rl_indices)
    n_rl_val = max(1, int(len(rl_df) * args.val_ratio))
    rl_train = rl_df.iloc[rl_indices[n_rl_val:]]
    rl_val = rl_df.iloc[rl_indices[:n_rl_val]]

    # 转换 SFT 格式
    print("\nConverting SFT data...")
    sft_train_records = [convert_to_sft_format(row) for _, row in sft_train.iterrows()]
    sft_val_records = [convert_to_sft_format(row) for _, row in sft_val.iterrows()]

    sft_train_df = pd.DataFrame(sft_train_records)
    sft_val_df = pd.DataFrame(sft_val_records)

    modelnew_count = sum(1 for r in sft_train_records if r["format"] == "modelnew")
    print(f"  Train: {len(sft_train_df)} (ModelNew: {modelnew_count}, "
          f"Original: {len(sft_train_df) - modelnew_count})")
    print(f"  Val: {len(sft_val_df)}")

    # 转换 RL 格式
    print("\nConverting RL data...")
    rl_train_records = [convert_to_rl_format(row) for _, row in rl_train.iterrows()]
    rl_val_records = [convert_to_rl_format(row) for _, row in rl_val.iterrows()]

    # 过滤 None
    rl_train_records = [r for r in rl_train_records if r is not None]
    rl_val_records = [r for r in rl_val_records if r is not None]

    rl_train_df = pd.DataFrame(rl_train_records)
    rl_val_df = pd.DataFrame(rl_val_records)

    rl_modelnew = sum(1 for r in rl_train_records if r["extra_info"]["format"] == "modelnew")
    print(f"  Train: {len(rl_train_df)} (ModelNew prompt: {rl_modelnew}, "
          f"Original prompt: {len(rl_train_df) - rl_modelnew})")
    print(f"  Val: {len(rl_val_df)}")

    # 保存
    sft_dir = os.path.join(args.output_dir, "sft")
    rl_dir = os.path.join(args.output_dir, "rl")
    os.makedirs(sft_dir, exist_ok=True)
    os.makedirs(rl_dir, exist_ok=True)

    sft_train_df.to_parquet(os.path.join(sft_dir, "train.parquet"), index=False)
    sft_val_df.to_parquet(os.path.join(sft_dir, "val.parquet"), index=False)
    rl_train_df.to_parquet(os.path.join(rl_dir, "train.parquet"), index=False)
    rl_val_df.to_parquet(os.path.join(rl_dir, "val.parquet"), index=False)

    print(f"\nSaved to {args.output_dir}/")
    print(f"  sft/train.parquet: {len(sft_train_df)}")
    print(f"  sft/val.parquet: {len(sft_val_df)}")
    print(f"  rl/train.parquet: {len(rl_train_df)}")
    print(f"  rl/val.parquet: {len(rl_val_df)}")

    # 保存统计
    stats = {
        "raw_count": before,
        "after_empty_filter": len(df),
        "sft_total": len(sft_df),
        "sft_train": len(sft_train_df),
        "sft_val": len(sft_val_df),
        "sft_modelnew_rate": f"{modelnew_count / max(len(sft_train_df), 1) * 100:.1f}%",
        "rl_total": len(rl_df),
        "rl_train": len(rl_train_df),
        "rl_val": len(rl_val_df),
        "rl_modelnew_rate": f"{rl_modelnew / max(len(rl_train_df), 1) * 100:.1f}%",
        "sft_mean_complexity": round(sft_df["complexity_score"].mean(), 3),
        "rl_mean_complexity": round(rl_df["complexity_score"].mean(), 3),
        "sft_mean_quality": round(sft_df["quality_score"].mean(), 3),
        "rl_mean_quality": round(rl_df["quality_score"].mean(), 3),
    }

    stats_path = os.path.join(args.output_dir, "split_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()

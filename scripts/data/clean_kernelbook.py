"""
KernelBook 数据清洗主脚本。

5 步清洗流程：
1. 基础过滤（空值、长度、结构）
2. 质量过滤（语法、Triton 标记、合成样本、repo 集中度）
3. 去重（精确 + 模糊）
4. 统计报告
5. 保存清洗结果

用法:
    python scripts/data/clean_kernelbook.py \
        --input data/raw/kernelbook_raw.parquet \
        --output data/cleaned/kernelbook_clean.parquet \
        --stats data/cleaned/stats.json

依赖:
    pip install pandas pyarrow tqdm
    pip install datasketch  # 可选，用于模糊去重
    pip install tiktoken    # 可选，用于精确 token 计算
"""

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter
from typing import Optional

import pandas as pd
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.dedup import exact_dedup, minhash_dedup
from src.utils.tokenizer_utils import estimate_tokens


# ============================================================
# Step 1: 基础过滤
# ============================================================

def filter_empty(df: pd.DataFrame) -> pd.DataFrame:
    """去除 python_code 或 triton_code 为空的记录。"""
    mask = (
        df["python_code"].notna()
        & df["triton_code"].notna()
        & (df["python_code"].str.strip() != "")
        & (df["triton_code"].str.strip() != "")
    )
    return df[mask].copy()


def filter_length(
    df: pd.DataFrame,
    max_prompt_tokens: int = 4096,
    max_response_tokens: int = 8192,
    min_response_tokens: int = 50,
    prompt_template_overhead: int = 200,
) -> pd.DataFrame:
    """按 token 长度过滤。

    prompt_template_overhead: SFT/RL prompt 模板额外占用的 token 数。
    实际训练时 prompt = 模板指令 + python_code，所以 python_code 的
    有效上限 = max_prompt_tokens - prompt_template_overhead。
    """
    effective_max_prompt = max_prompt_tokens - prompt_template_overhead
    print(f"  Estimating token lengths (prompt effective max: {effective_max_prompt})...")
    df = df.copy()
    df["_prompt_tokens"] = df["python_code"].apply(estimate_tokens)
    df["_response_tokens"] = df["triton_code"].apply(estimate_tokens)

    mask = (
        (df["_prompt_tokens"] <= effective_max_prompt)
        & (df["_response_tokens"] <= max_response_tokens)
        & (df["_response_tokens"] >= min_response_tokens)
    )
    result = df[mask].copy()
    result.drop(columns=["_prompt_tokens", "_response_tokens"], inplace=True)
    return result


def filter_structure(df: pd.DataFrame) -> pd.DataFrame:
    """去除无有效 Python class/function 结构的样本。"""
    def has_valid_structure(code: str) -> bool:
        # 检查是否有 class 或 def 定义
        return bool(
            re.search(r"\bclass\s+\w+", code)
            or re.search(r"\bdef\s+\w+", code)
        )

    mask = df["python_code"].apply(has_valid_structure)
    return df[mask].copy()


# ============================================================
# Step 2: 质量过滤
# ============================================================

def filter_syntax(df: pd.DataFrame) -> pd.DataFrame:
    """Python 语法检查（同时检查 python_code 和 triton_code）。"""
    def check_python_syntax(code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    mask_python = df["python_code"].apply(check_python_syntax)
    mask_triton = df["triton_code"].apply(check_python_syntax)
    return df[mask_python & mask_triton].copy()


def filter_triton_markers(df: pd.DataFrame) -> pd.DataFrame:
    """确保 triton_code 包含 Triton 相关标记。"""
    triton_patterns = [
        r"@triton\.jit",
        r"triton\.jit",
        r"tl\.\w+",         # triton.language 操作
        r"triton\.language",
    ]

    def has_triton_marker(code: str) -> bool:
        return any(re.search(p, code) for p in triton_patterns)

    mask = df["triton_code"].apply(has_triton_marker)
    return df[mask].copy()


def filter_synthetic_quality(df: pd.DataFrame) -> pd.DataFrame:
    """去除低质量合成样本（synthetic=True 且 stars=0）。"""
    if "synthetic" not in df.columns or "stars" not in df.columns:
        return df

    # 保留：非合成的 OR 合成但 stars > 0 的
    mask = (~df["synthetic"]) | (df["stars"] > 0)
    return df[mask].copy()


def filter_repo_concentration(df: pd.DataFrame, max_per_repo: int = 50) -> pd.DataFrame:
    """去除同一 repo 贡献超过阈值的过度集中数据。"""
    if "repo_name" not in df.columns:
        return df

    repo_counts = df["repo_name"].value_counts()
    heavy_repos = repo_counts[repo_counts > max_per_repo].index.tolist()

    if not heavy_repos:
        return df

    # 对重复 repo 进行采样
    keep_indices = []
    for repo in heavy_repos:
        repo_df = df[df["repo_name"] == repo]
        # 按 stars 排序，保留前 max_per_repo 个
        if "stars" in df.columns:
            sampled = repo_df.nlargest(max_per_repo, "stars")
        else:
            sampled = repo_df.head(max_per_repo)
        keep_indices.extend(sampled.index.tolist())

    # 合并非重复 repo 的记录
    non_heavy = df[~df["repo_name"].isin(heavy_repos)]
    heavy_kept = df.loc[keep_indices]

    return pd.concat([non_heavy, heavy_kept]).sort_index()


# ============================================================
# Step 3: 去重
# ============================================================

def deduplicate(
    df: pd.DataFrame,
    method: str = "minhash",
    threshold: float = 0.85,
) -> pd.DataFrame:
    """双侧去重：拼接 python_code + triton_code 后去重。"""
    # 拼接输入侧和输出侧，避免只对单侧去重导致输出模式重复
    codes = (df["python_code"] + "\n### SEPARATOR ###\n" + df["triton_code"]).tolist()
    priorities = df["stars"].tolist() if "stars" in df.columns else None

    if method == "minhash":
        kept = minhash_dedup(codes, threshold=threshold, priorities=priorities)
    else:
        kept = exact_dedup(codes, priorities=priorities)

    return df.iloc[kept].copy()


# ============================================================
# 主流程
# ============================================================

def run_cleaning(
    input_path: str,
    output_path: str,
    stats_path: Optional[str] = None,
    max_prompt_tokens: int = 4096,
    max_response_tokens: int = 8192,
    min_response_tokens: int = 50,
    max_per_repo: int = 50,
    dedup_method: str = "minhash",
    dedup_threshold: float = 0.85,
) -> pd.DataFrame:
    """运行完整清洗流程。"""
    stats = {"steps": []}

    def log_step(name: str, df: pd.DataFrame, prev_count: int):
        removed = prev_count - len(df)
        stats["steps"].append({
            "step": name,
            "remaining": len(df),
            "removed": removed,
            "removed_pct": f"{removed / max(prev_count, 1) * 100:.1f}%",
        })
        print(f"  [{name}] {prev_count} → {len(df)} (removed {removed})")

    # 加载数据
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    stats["original_count"] = len(df)
    print(f"  Loaded {len(df)} records")
    print(f"  Columns: {list(df.columns)}")

    # Step 1: 基础过滤
    print("\n=== Step 1: Basic Filtering ===")
    prev = len(df)
    df = filter_empty(df)
    log_step("remove_empty", df, prev)

    prev = len(df)
    df = filter_length(df, max_prompt_tokens, max_response_tokens, min_response_tokens)
    log_step("filter_length", df, prev)

    prev = len(df)
    df = filter_structure(df)
    log_step("filter_structure", df, prev)

    # Step 2: 质量过滤
    print("\n=== Step 2: Quality Filtering ===")
    prev = len(df)
    df = filter_syntax(df)
    log_step("syntax_check", df, prev)

    prev = len(df)
    df = filter_triton_markers(df)
    log_step("triton_markers", df, prev)

    prev = len(df)
    df = filter_synthetic_quality(df)
    log_step("synthetic_quality", df, prev)

    prev = len(df)
    df = filter_repo_concentration(df, max_per_repo)
    log_step("repo_concentration", df, prev)

    # Step 3: 去重
    print("\n=== Step 3: Deduplication ===")
    prev = len(df)
    df = deduplicate(df, method=dedup_method, threshold=dedup_threshold)
    log_step(f"dedup_{dedup_method}", df, prev)

    # 统计信息
    stats["final_count"] = len(df)
    stats["retention_rate"] = f"{len(df) / stats['original_count'] * 100:.1f}%"

    # 长度统计
    prompt_tokens = df["python_code"].apply(estimate_tokens)
    response_tokens = df["triton_code"].apply(estimate_tokens)
    stats["token_stats"] = {
        "prompt": {
            "mean": int(prompt_tokens.mean()),
            "median": int(prompt_tokens.median()),
            "p95": int(prompt_tokens.quantile(0.95)),
            "max": int(prompt_tokens.max()),
        },
        "response": {
            "mean": int(response_tokens.mean()),
            "median": int(response_tokens.median()),
            "p95": int(response_tokens.quantile(0.95)),
            "max": int(response_tokens.max()),
        },
    }

    if "synthetic" in df.columns:
        stats["synthetic_count"] = int(df["synthetic"].sum())
        stats["real_count"] = int((~df["synthetic"]).sum())

    if "repo_name" in df.columns:
        stats["unique_repos"] = int(df["repo_name"].nunique())

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\n=== Results ===")
    print(f"Saved {len(df)} cleaned records to {output_path}")
    print(f"Retention rate: {stats['retention_rate']}")

    if stats_path:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Stats saved to {stats_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Clean KernelBook dataset")
    parser.add_argument(
        "--input",
        default="data/raw/kernelbook_raw.parquet",
        help="Input parquet file",
    )
    parser.add_argument(
        "--output",
        default="data/cleaned/kernelbook_clean.parquet",
        help="Output cleaned parquet file",
    )
    parser.add_argument(
        "--stats",
        default="data/cleaned/stats.json",
        help="Output stats JSON file",
    )
    parser.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=4096,
        help="Max tokens for python_code (default: 4096)",
    )
    parser.add_argument(
        "--max_response_tokens",
        type=int,
        default=8192,
        help="Max tokens for triton_code (default: 8192)",
    )
    parser.add_argument(
        "--min_response_tokens",
        type=int,
        default=50,
        help="Min tokens for triton_code (default: 50)",
    )
    parser.add_argument(
        "--max_per_repo",
        type=int,
        default=50,
        help="Max samples per repo (default: 50)",
    )
    parser.add_argument(
        "--dedup_method",
        choices=["exact", "minhash"],
        default="minhash",
        help="Deduplication method (default: minhash)",
    )
    parser.add_argument(
        "--dedup_threshold",
        type=float,
        default=0.85,
        help="MinHash similarity threshold (default: 0.85)",
    )
    args = parser.parse_args()

    run_cleaning(
        input_path=args.input,
        output_path=args.output,
        stats_path=args.stats,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
        min_response_tokens=args.min_response_tokens,
        max_per_repo=args.max_per_repo,
        dedup_method=args.dedup_method,
        dedup_threshold=args.dedup_threshold,
    )


if __name__ == "__main__":
    main()

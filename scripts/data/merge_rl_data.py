"""
合并 KernelBook RL + KernelBench CUDA 数据为混合 RL 训练集。

KernelBook 样本 → compute_score（静态+编译）
KernelBench 样本 → compute_score_cuda（compile+run）
路由由 compute_score_auto 自动完成。

用法:
    python scripts/data/merge_rl_data.py \
        --kernelbook_dir data/rl \
        --kernelbench_dir data/rl_kernelbench_cuda \
        --output_dir data/rl_mixed
"""

import argparse
import os

import pandas as pd


def normalize_reward_model(rm):
    """统一 reward_model 格式：ground_truth 全部转为 dict。

    KernelBook 原始格式: {"ground_truth": "<python_code_string>"}
    KernelBench 格式:    {"ground_truth": {"task_id": ..., "backend": "cuda", ...}}

    统一后 KernelBook:   {"ground_truth": {"python_code": "...", "format": "original"}}
    """
    gt = rm.get("ground_truth")
    if isinstance(gt, str):
        return {
            "style": rm.get("style", "rule"),
            "ground_truth": {
                "python_code": gt,
                "format": "original",
            },
        }
    return rm


def main():
    parser = argparse.ArgumentParser(description="Merge KernelBook + KernelBench RL data")
    parser.add_argument("--kernelbook_dir", default="data/rl", help="KernelBook RL data dir")
    parser.add_argument("--kernelbench_dir", default="data/rl_kernelbench_cuda", help="KernelBench CUDA data dir")
    parser.add_argument("--output_dir", default="data/rl_mixed", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "val"]:
        dfs = []

        # KernelBook
        kb_path = os.path.join(args.kernelbook_dir, f"{split}.parquet")
        if os.path.exists(kb_path):
            df = pd.read_parquet(kb_path)
            # 统一 ground_truth 为 dict 格式
            df["reward_model"] = df["reward_model"].apply(normalize_reward_model)
            print(f"KernelBook {split}: {len(df)} rows")
            dfs.append(df)
        else:
            print(f"KernelBook {split}: not found at {kb_path}")

        # KernelBench CUDA
        bench_path = os.path.join(args.kernelbench_dir, f"{split}.parquet")
        if os.path.exists(bench_path):
            df = pd.read_parquet(bench_path)
            print(f"KernelBench {split}: {len(df)} rows")
            dfs.append(df)
        else:
            print(f"KernelBench {split}: not found at {bench_path}")

        if not dfs:
            print(f"WARNING: no data for {split}, skipping")
            continue

        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        output_path = os.path.join(args.output_dir, f"{split}.parquet")
        merged.to_parquet(output_path, index=False)
        print(f"Merged {split}: {len(merged)} rows -> {output_path}\n")

    # 验证
    check_path = os.path.join(args.output_dir, "train.parquet")
    if os.path.exists(check_path):
        df = pd.read_parquet(check_path)
        print("=== Verification ===")
        print(f"Total train rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")

        sources = df["data_source"].value_counts()
        for src, cnt in sources.items():
            print(f"  {src}: {cnt} ({cnt / len(df) * 100:.1f}%)")

        sample_kb = df[df["data_source"] == "kernelbook"].iloc[0] if "kernelbook" in sources.index else None
        sample_bench = df[df["data_source"] == "kernelbench"].iloc[0] if "kernelbench" in sources.index else None

        if sample_kb is not None:
            gt = sample_kb["reward_model"]["ground_truth"]
            print(f"\nKernelBook sample: ground_truth type={type(gt).__name__}, keys={list(gt.keys())}")
        if sample_bench is not None:
            gt = sample_bench["reward_model"]["ground_truth"]
            print(f"KernelBench sample: ground_truth type={type(gt).__name__}, keys={list(gt.keys())}")


if __name__ == "__main__":
    main()

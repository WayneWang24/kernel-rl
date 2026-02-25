"""
下载 GPUMODE/KernelBook 数据集到本地。

用法:
    python scripts/data/download_kernelbook.py [--output_dir data/raw]
"""

import argparse
import os

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download KernelBook dataset")
    parser.add_argument(
        "--output_dir",
        default="data/raw",
        help="Output directory for raw data (default: data/raw)",
    )
    parser.add_argument(
        "--dataset_name",
        default="GPUMODE/KernelBook",
        help="HuggingFace dataset name",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading {args.dataset_name}...")
    ds = load_dataset(args.dataset_name)

    # 保存为 parquet
    train_data = ds["train"]
    output_path = os.path.join(args.output_dir, "kernelbook_raw.parquet")
    train_data.to_parquet(output_path)

    print(f"Saved {len(train_data)} records to {output_path}")
    print(f"Columns: {train_data.column_names}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

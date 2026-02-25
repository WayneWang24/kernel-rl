"""
结果汇总对比脚本。

读取 results/ 目录下各实验的评测结果，生成对比表格。

用法:
    python scripts/eval/compare_results.py --results_dir results/
"""

import argparse
import glob
import json
import os
from typing import Optional


def load_results(results_dir: str) -> dict:
    """加载所有实验结果。"""
    experiments = {}

    for run_dir in sorted(glob.glob(os.path.join(results_dir, "*"))):
        if not os.path.isdir(run_dir):
            continue

        run_name = os.path.basename(run_dir)
        experiments[run_name] = {}

        for result_file in sorted(glob.glob(os.path.join(run_dir, "level*_results.json"))):
            with open(result_file, "r") as f:
                data = json.load(f)
            level = data.get("level", "unknown")
            experiments[run_name][f"level{level}"] = data.get("metrics", {})

    return experiments


def print_comparison_table(experiments: dict):
    """打印对比表格。"""
    if not experiments:
        print("No results found.")
        return

    # 收集所有 level
    all_levels = sorted(set(
        level
        for exp in experiments.values()
        for level in exp.keys()
    ))

    # 表头
    header = f"{'Experiment':<30}"
    for level in all_levels:
        header += f" | {'Score':>8} {'Triton%':>8} {'Complete%':>10}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # 各实验
    for run_name, levels in experiments.items():
        row = f"{run_name:<30}"
        for level in all_levels:
            metrics = levels.get(level, {})
            avg_score = metrics.get("avg_score", "N/A")
            triton_pct = metrics.get("has_triton_pct", "N/A")
            complete_pct = metrics.get("has_complete_pct", "N/A")
            row += f" | {avg_score:>8} {triton_pct:>8} {complete_pct:>10}"
        print(row)

    print("=" * len(header))

    # 图例
    print("\nMetrics:")
    print("  Score     = Average static reward score (0.0 ~ 1.0)")
    print("  Triton%   = Percentage of solutions with valid Triton kernel")
    print("  Complete% = Percentage of solutions with complete kernel + wrapper")


def print_detailed_comparison(experiments: dict):
    """打印详细对比（包含 score 分布）。"""
    for run_name, levels in experiments.items():
        print(f"\n{'=' * 50}")
        print(f"  {run_name}")
        print(f"{'=' * 50}")
        for level, metrics in sorted(levels.items()):
            print(f"\n  {level}:")
            for k, v in metrics.items():
                if isinstance(v, dict):
                    print(f"    {k}:")
                    for sk, sv in v.items():
                        print(f"      {sk}: {sv}")
                else:
                    print(f"    {k}: {v}")


def export_csv(experiments: dict, output_path: str):
    """导出为 CSV 文件。"""
    all_levels = sorted(set(
        level
        for exp in experiments.values()
        for level in exp.keys()
    ))

    with open(output_path, "w") as f:
        # 表头
        header = ["experiment"]
        for level in all_levels:
            header.extend([
                f"{level}_avg_score",
                f"{level}_triton_pct",
                f"{level}_complete_pct",
                f"{level}_total",
            ])
        f.write(",".join(header) + "\n")

        # 数据行
        for run_name, levels in experiments.items():
            row = [run_name]
            for level in all_levels:
                metrics = levels.get(level, {})
                row.extend([
                    metrics.get("avg_score", ""),
                    metrics.get("has_triton_pct", "").replace("%", ""),
                    metrics.get("has_complete_pct", "").replace("%", ""),
                    str(metrics.get("total", "")),
                ])
            f.write(",".join(row) + "\n")

    print(f"\nCSV exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument("--results_dir", default="results/", help="Results directory")
    parser.add_argument("--detailed", action="store_true", help="Show detailed results")
    parser.add_argument("--csv", default=None, help="Export to CSV file")
    args = parser.parse_args()

    experiments = load_results(args.results_dir)

    if not experiments:
        print(f"No results found in {args.results_dir}")
        print("Expected structure: results/<run_name>/level*_results.json")
        return

    print(f"\nFound {len(experiments)} experiments: {list(experiments.keys())}")
    print()

    print_comparison_table(experiments)

    if args.detailed:
        print_detailed_comparison(experiments)

    if args.csv:
        export_csv(experiments, args.csv)


if __name__ == "__main__":
    main()

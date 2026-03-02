"""
将 KernelBook 数据转为 ModelNew 格式的 SFT 数据。

转换策略（AST 分析）：
1. 分析 python_code：找到主函数/类、参数签名
2. 如果是裸函数（如 def relu(x)）→ 包装成 class Model(nn.Module) + 合成 get_inputs()
3. 如果已是 nn.Module → 直接复用
4. 分析 triton_code：找到 @triton.jit kernel 和 wrapper 函数
5. 包装成 class ModelNew(nn.Module)，forward() 调用 wrapper
6. 转换失败的样本 → 保留原格式（维持 Triton 基础能力）

输出：data/sft_modelnew/{train,val,test}.parquet
- 混合格式：~30-45% ModelNew 格式 + 55-70% 原始格式
- messages 格式：user=Triton 版 prompt，assistant=ModelNew 代码

用法:
    python scripts/data/prepare_sft_modelnew.py \
        --input data/cleaned/kernelbook_clean.parquet \
        --output_dir data/sft_modelnew
"""

import argparse
import ast
import os
import random
import re
import sys
import textwrap
from collections import defaultdict
from typing import List, Optional, Tuple

import pandas as pd


# ModelNew 格式的 prompt 模板（与 RL 训练一致）
MODELNEW_PROMPT_TEMPLATE = """You are an expert Performance Engineer specializing in Triton and PyTorch internals.

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


# 原始格式 prompt（保留用于混合训练）
ORIGINAL_PROMPT_TEMPLATE = """You are an expert GPU programmer specializing in Triton kernel development. Your task is to convert the following PyTorch code into an optimized Triton kernel implementation.

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


# ============================================================
# AST 分析工具
# ============================================================

def find_main_function(code: str) -> Optional[dict]:
    """找到代码中的主函数（非 helper/utility 函数）。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    functions = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            args = []
            for arg in node.args.args:
                args.append(arg.arg)
            functions.append({
                "name": node.name,
                "args": args,
                "lineno": node.lineno,
                "end_lineno": node.end_lineno,
            })

    if not functions:
        return None

    # 优先选：不以 _ 开头的、有参数的函数
    candidates = [f for f in functions if not f["name"].startswith("_") and f["args"]]
    if candidates:
        return candidates[0]
    return functions[0]


def find_nn_module_class(code: str) -> Optional[dict]:
    """找到代码中的 nn.Module 子类。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_str = ast.dump(base)
                if "Module" in base_str:
                    return {
                        "name": node.name,
                        "lineno": node.lineno,
                        "end_lineno": node.end_lineno,
                    }
    return None


def find_triton_kernels(code: str) -> list:
    """找到所有 @triton.jit 装饰的函数。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    kernels = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                dec_str = ast.dump(dec)
                if "triton" in dec_str and ("jit" in dec_str or "autotune" in dec_str):
                    kernels.append({
                        "name": node.name,
                        "args": [a.arg for a in node.args.args],
                    })
                    break
    return kernels


def find_wrapper_functions(code: str, kernel_names: list) -> list:
    """找到调用 triton kernel 的 wrapper 函数。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    wrappers = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            # 排除 triton kernel 自身
            is_kernel = False
            for dec in node.decorator_list:
                if "triton" in ast.dump(dec):
                    is_kernel = True
                    break
            if is_kernel:
                continue

            # 检查函数体是否调用了 kernel
            func_code = ast.get_source_segment(code, node) or ""
            for kname in kernel_names:
                if re.search(rf"\b{kname}\s*\[", func_code):
                    wrappers.append({
                        "name": node.name,
                        "args": [a.arg for a in node.args.args],
                    })
                    break

    return wrappers


def wrap_function_as_model(python_code: str, func_info: dict) -> Optional[str]:
    """将裸函数包装为 class Model(nn.Module)。"""
    func_name = func_info["name"]
    func_args = func_info["args"]

    if not func_args:
        return None

    # forward 接收所有函数参数
    forward_params = ", ".join(f"{arg}" for arg in func_args)
    call_args = ", ".join(func_args)

    model_code = f"""import torch
import torch.nn as nn

{python_code}

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, {forward_params}):
        return {func_name}({call_args})
"""
    return model_code


def wrap_triton_as_modelnew(
    triton_code: str,
    kernel_names: list,
    wrapper_names: list,
    wrapper_infos: list,
    has_nn_module_in_python: bool,
) -> Optional[str]:
    """将 Triton 代码包装为 class ModelNew(nn.Module)。

    如果原始 Python 有 nn.Module，保留其 __init__，forward 改用 wrapper。
    如果是裸函数，创建简单的 ModelNew。
    """
    if not kernel_names:
        return None

    # 选择主 wrapper 函数
    if not wrapper_names:
        return None

    main_wrapper = wrapper_names[0]

    # 从 wrapper 参数推断 forward 的参数
    if wrapper_infos:
        wrapper_args = wrapper_infos[0].get("args", ["x"])
    else:
        wrapper_args = ["x"]

    forward_params = ", ".join(wrapper_args)
    call_args = ", ".join(wrapper_args)

    # 构建 ModelNew 代码
    modelnew_code = f"""{triton_code}

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, {forward_params}):
        return {main_wrapper}({call_args})
"""
    return modelnew_code


def try_convert_to_modelnew(
    python_code: str,
    triton_code: str,
) -> Optional[Tuple[str, str]]:
    """尝试将一条 KernelBook 样本转为 (Model prompt_code, ModelNew response_code)。

    Returns:
        (model_code, modelnew_code) 如果成功，None 如果失败
    """
    # 1. 分析 python_code
    nn_module = find_nn_module_class(python_code)
    main_func = find_main_function(python_code)

    # 2. 确定 Model 代码
    if nn_module:
        model_code = python_code
    elif main_func:
        model_code = wrap_function_as_model(python_code, main_func)
        if model_code is None:
            return None
    else:
        return None

    # 3. 分析 triton_code
    kernels = find_triton_kernels(triton_code)
    kernel_names = [k["name"] for k in kernels]
    wrappers = find_wrapper_functions(triton_code, kernel_names)
    wrapper_names = [w["name"] for w in wrappers]

    if not kernels:
        return None

    # 4. 构建 ModelNew 代码
    if nn_module and wrappers:
        # 原始是 nn.Module，保留 __init__，改 forward
        # 从 python_code 提取 __init__ 部分
        try:
            tree = ast.parse(python_code)
        except SyntaxError:
            return None

        # 找到原始类的 __init__
        init_body = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == nn_module["name"]:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        init_body = ast.get_source_segment(python_code, item)
                        break

        if init_body:
            # 提取 forward 参数
            forward_args = "self, x"
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == nn_module["name"]:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "forward":
                            forward_args = ", ".join(
                                a.arg for a in item.args.args
                            )
                            break

            main_wrapper = wrapper_names[0]
            # 构建 wrapper 调用参数
            wrapper_call_args = ", ".join(
                a for a in forward_args.split(", ") if a != "self"
            )

            modelnew_code = f"""{triton_code}

class ModelNew(nn.Module):
{textwrap.indent(init_body, '    ')}

    def forward({forward_args}):
        return {main_wrapper}({wrapper_call_args})
"""
        else:
            modelnew_code = wrap_triton_as_modelnew(
                triton_code, kernel_names, wrapper_names, wrappers, True
            )
    else:
        modelnew_code = wrap_triton_as_modelnew(
            triton_code, kernel_names, wrapper_names, wrappers, False
        )

    if modelnew_code is None:
        return None

    # 5. 验证生成代码的语法
    try:
        ast.parse(model_code)
        ast.parse(modelnew_code)
    except SyntaxError:
        return None

    return (model_code, modelnew_code)


# ============================================================
# 数据划分
# ============================================================

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


# ============================================================
# 格式转换
# ============================================================

def convert_row_modelnew(row: pd.Series) -> Optional[dict]:
    """尝试将一行转为 ModelNew SFT 格式。"""
    result = try_convert_to_modelnew(
        row["python_code"],
        row["triton_code"],
    )
    if result is None:
        return None

    model_code, modelnew_code = result
    prompt = MODELNEW_PROMPT_TEMPLATE.format(model_code=model_code.strip())

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": modelnew_code.strip()},
        ],
        "format": "modelnew",
    }


def convert_row_original(row: pd.Series) -> dict:
    """转为原始格式 SFT 数据。"""
    prompt = ORIGINAL_PROMPT_TEMPLATE.format(python_code=row["python_code"].strip())

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": row["triton_code"].strip()},
        ],
        "format": "original",
    }


def convert_split(df: pd.DataFrame) -> pd.DataFrame:
    """转换一个 split 的数据，混合 ModelNew + 原始格式。"""
    records = []
    modelnew_count = 0

    for _, row in df.iterrows():
        # 先尝试 ModelNew 格式
        modelnew_record = convert_row_modelnew(row)
        if modelnew_record is not None:
            records.append(modelnew_record)
            modelnew_count += 1
        else:
            # 回退到原始格式
            records.append(convert_row_original(row))

    print(f"    ModelNew: {modelnew_count}/{len(df)} ({modelnew_count / max(len(df), 1) * 100:.1f}%), "
          f"Original: {len(df) - modelnew_count}/{len(df)}")

    return pd.DataFrame(records)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare ModelNew format SFT data from KernelBook"
    )
    parser.add_argument(
        "--input",
        default="data/cleaned/kernelbook_clean.parquet",
        help="Input cleaned parquet",
    )
    parser.add_argument("--output_dir", default="data/sft_modelnew", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading cleaned data from {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df)} records")
    print(f"  Columns: {list(df.columns)}")

    # 划分
    train_df, val_df, test_df = split_by_repo(
        df, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"  Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    # 转换格式
    print("\nConverting to mixed ModelNew + original format...")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n  {split_name}:")
        sft_df = convert_split(split_df)
        output_path = os.path.join(args.output_dir, f"{split_name}.parquet")
        sft_df.to_parquet(output_path, index=False)
        print(f"    Saved {len(sft_df)} records to {output_path}")

    # 验证
    print("\n=== Verification ===")
    check = pd.read_parquet(os.path.join(args.output_dir, "train.parquet"))
    total = len(check)
    modelnew_count = sum(1 for _, r in check.iterrows() if r.get("format") == "modelnew")
    original_count = total - modelnew_count

    print(f"  Total: {total}")
    print(f"  ModelNew format: {modelnew_count} ({modelnew_count / max(total, 1) * 100:.1f}%)")
    print(f"  Original format: {original_count} ({original_count / max(total, 1) * 100:.1f}%)")

    # 打印一条 ModelNew 样本
    modelnew_samples = check[check["format"] == "modelnew"]
    if len(modelnew_samples) > 0:
        sample = modelnew_samples.iloc[0]
        messages = sample["messages"]
        print(f"\n  ModelNew sample:")
        print(f"    User prompt (first 200 chars): {messages[0]['content'][:200]}...")
        print(f"    Assistant response (first 200 chars): {messages[1]['content'][:200]}...")


if __name__ == "__main__":
    main()

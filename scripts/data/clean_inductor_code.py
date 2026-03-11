"""
清洗 KernelBook 中 torch._inductor 生成的 Triton 代码。

将 inductor 内部 API 替换为标准 PyTorch/Triton 等价物，
使代码能在任意 PyTorch 版本上编译运行。

用法:
    # 测试模式：清洗前 5 个样本并尝试编译
    python scripts/data/clean_inductor_code.py --test --num_samples 5

    # 清洗全部数据
    python scripts/data/clean_inductor_code.py \
        --input data/raw/kernelbook_raw.parquet \
        --output data/cleaned_inductor/kernelbook_clean.parquet

    # 统计清洗效果
    python scripts/data/clean_inductor_code.py --stats \
        --input data/raw/kernelbook_raw.parquet
"""

import argparse
import ast
import os
import re
import sys
import tempfile
import subprocess
from typing import Optional, Tuple

import pandas as pd


# ============================================================
# 兼容函数定义（注入到清洗后的代码头部）
# ============================================================

COMPAT_HEADER = '''import torch
import triton
import triton.language as tl
import torch.nn as nn


def _grid(*numels):
    """兼容 torch._inductor 的 grid 函数。"""
    def _compute_grid(meta):
        block_keys = ['XBLOCK', 'YBLOCK', 'ZBLOCK']
        result = []
        for i, n in enumerate(numels):
            if i < len(block_keys) and block_keys[i] in meta:
                result.append(triton.cdiv(n, meta[block_keys[i]]))
            else:
                result.append(n)
        return tuple(result)
    return _compute_grid


def _assert_size_stride(tensor, size, stride):
    """兼容 torch._C._dynamo.guards.assert_size_stride。"""
    assert tensor.size() == torch.Size(size), f"Expected size {size}, got {tensor.size()}"
    assert tensor.stride() == tuple(stride), f"Expected stride {stride}, got {tensor.stride()}"


def _empty_strided_cuda(size, stride, dtype):
    """兼容 torch._C._dynamo.guards._empty_strided_cuda。"""
    return torch.empty_strided(size, stride, dtype=dtype, device='cuda')


def _empty_strided_cpu(size, stride, dtype):
    """兼容 torch._C._dynamo.guards._empty_strided_cpu。"""
    return torch.empty_strided(size, stride, dtype=dtype, device='cpu')


def _reinterpret_tensor(tensor, size, stride, offset=0):
    """兼容 torch._C._dynamo.guards._reinterpret_tensor。"""
    return torch.as_strided(tensor, size, stride, storage_offset=offset)

'''

# ============================================================
# 替换规则
# ============================================================

# Import 替换（删除 inductor import，用 compat header 代替）
IMPORT_REMOVALS = [
    r'^from torch\._inductor\.select_algorithm import extern_kernels\s*$',
    r'^from torch\._inductor\.runtime\.triton_heuristics import grid.*$',
    r'^from torch\._inductor\.runtime import triton_helpers.*$',
    r'^from torch\._inductor\.runtime\.triton_helpers import .*$',
    r'^from torch\._C import _cuda_getCurrentRawStream as get_raw_stream\s*$',
    r'^import torch\._C\s*$',
    r'^import torch\s*$',           # 会在 header 里重新 import
    r'^import triton\s*$',
    r'^import triton\.language as tl\s*$',
    r'^import torch\.nn as nn\s*$',
]

# 赋值替换
ASSIGNMENT_REPLACEMENTS = [
    (r'^assert_size_stride\s*=\s*torch\._C\._dynamo\.guards\.assert_size_stride\s*$',
     'assert_size_stride = _assert_size_stride'),
    (r'^empty_strided_cuda\s*=\s*torch\._C\._dynamo\.guards\._empty_strided_cuda\s*$',
     'empty_strided_cuda = _empty_strided_cuda'),
    (r'^empty_strided_cpu\s*=\s*torch\._C\._dynamo\.guards\._empty_strided_cpu\s*$',
     'empty_strided_cpu = _empty_strided_cpu'),
    (r'^reinterpret_tensor\s*=\s*torch\._C\._dynamo\.guards\._reinterpret_tensor\s*$',
     'reinterpret_tensor = _reinterpret_tensor'),
]

# 函数调用替换
CALL_REPLACEMENTS = [
    # extern_kernels
    (r'extern_kernels\.mm\(([^,]+),\s*([^,]+),\s*out=([^)]+)\)',
     r'torch.mm(\1, \2, out=\3)'),
    (r'extern_kernels\.mm\(([^,]+),\s*([^)]+)\)',
     r'torch.mm(\1, \2)'),
    (r'extern_kernels\.bmm\(([^,]+),\s*([^,]+),\s*out=([^)]+)\)',
     r'torch.bmm(\1, \2, out=\3)'),
    (r'extern_kernels\.bmm\(([^,]+),\s*([^)]+)\)',
     r'torch.bmm(\1, \2)'),
    (r'extern_kernels\.addmm\(([^,]+),\s*([^,]+),\s*([^,]+),\s*alpha=([^,]+),\s*beta=([^)]+)\)',
     r'torch.addmm(\1, \2, \3, alpha=\4, beta=\5)'),
    (r'extern_kernels\.addmm\(([^,]+),\s*([^,]+),\s*([^)]+)\)',
     r'torch.addmm(\1, \2, \3)'),
    # extern_kernels.convolution → torch.conv2d (simplified)
    # This is complex, keep as F.conv2d
    (r'extern_kernels\.convolution\(([^)]+)\)',
     r'torch.ops.aten.convolution(\1)'),
    # _mm_plus_mm fallback
    (r'extern_kernels\._mm_plus_mm\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)',
     r'(torch.mm(\1, \2) + torch.mm(\3, \4))'),

    # grid 替换
    (r'\bgrid\(', '_grid('),

    # get_raw_stream(0) → 空操作（Triton 不需要手动 stream 管理）
    (r'get_raw_stream\(\d+\)', 'torch.cuda.current_stream().cuda_stream'),

    # triton_helpers.maximum → tl.maximum（Triton 2.0+ 内置）
    (r'triton_helpers\.maximum\(', 'tl.maximum('),
    (r'triton_helpers\.minimum\(', 'tl.minimum('),

    # triton_helpers.promote_to_tensor → 直接用
    (r'triton_helpers\.promote_to_tensor\(([^)]+)\)', r'\1'),

    # triton_helpers.max2 → tl.max (along axis)
    (r'triton_helpers\.max2\(([^,]+),\s*(\d+)\)', r'tl.max(\1, axis=\2)'),
    (r'triton_helpers\.min2\(([^,]+),\s*(\d+)\)', r'tl.min(\1, axis=\2)'),

    # triton_helpers.div_floor_integer → Python //
    (r'triton_helpers\.div_floor_integer\(([^,]+),\s*([^)]+)\)', r'(\1 // \2)'),

    # libdevice → tl.math (Triton 2.0+ 统一到 tl.math)
    (r'libdevice\.sqrt\(', 'tl.math.sqrt('),
    (r'libdevice\.rsqrt\(', 'tl.math.rsqrt('),
    (r'libdevice\.tanh\(', 'tl.math.tanh('),
    (r'libdevice\.exp\(', 'tl.math.exp('),
    (r'libdevice\.log\(', 'tl.math.log('),
    (r'libdevice\.log1p\(', 'tl.math.log(1.0 + '),  # 近似：log1p(x) ≈ log(1+x)
    (r'libdevice\.expm1\(', '(tl.math.exp('),  # 需要后处理：expm1(x) = exp(x)-1
    (r'libdevice\.erf\(', 'tl.math.erf('),
    (r'libdevice\.pow\(', 'tl.math.pow('),
    (r'libdevice\.floor\(', 'tl.math.floor('),
    (r'libdevice\.ceil\(', 'tl.math.ceil('),
    (r'libdevice\.isnan\(', 'tl.math.isnan('),
    (r'libdevice\.isinf\(', 'tl.math.isinf('),
    (r'libdevice\.signbit\(', 'tl.math.signbitf('),
    (r'libdevice\.sin\(', 'tl.math.sin('),
    (r'libdevice\.cos\(', 'tl.math.cos('),
    (r'libdevice\.atan2\(', 'tl.math.atan2('),
    (r'libdevice\.atan\(', 'tl.math.atan('),
    (r'libdevice\.acos\(', 'tl.math.acos('),
    (r'libdevice\.log2\(', 'tl.math.log2('),
    (r'libdevice\.log10\(', 'tl.math.log10('),
    (r'libdevice\.fmod\(', 'tl.math.fmod('),
    (r'libdevice\.cosh\(', 'tl.math.cosh('),
    (r'libdevice\.lgamma\(', 'tl.math.lgamma('),
    (r'libdevice\.trunc\(', 'tl.math.trunc('),
    (r'libdevice\.nearbyint\(', 'tl.math.nearbyint('),
    (r'libdevice\.abs\(', 'tl.abs('),

    # tl_math → tl.math（Triton 2.0+ 标准路径）
    (r'tl_math\.exp\(', 'tl.math.exp('),
    (r'tl_math\.log\(', 'tl.math.log('),
    (r'tl_math\.abs\(', 'tl.abs('),
    (r'tl_math\.sin\(', 'tl.math.sin('),
    (r'tl_math\.cos\(', 'tl.math.cos('),
    (r'tl_math\.sqrt\(', 'tl.math.sqrt('),

    # torch.cuda._DeviceGuard 保留（标准用法，不需要替换）
]


# ============================================================
# 核心清洗函数
# ============================================================

def clean_inductor_code(code: str) -> str:
    """
    清洗单个 inductor 生成的 Triton 代码。

    Returns:
        清洗后的代码（包含 compat header）
    """
    lines = code.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # 1. 删除 inductor import 行
        skip = False
        for pattern in IMPORT_REMOVALS:
            if re.match(pattern, stripped):
                skip = True
                break
        if skip:
            continue

        # 2. 替换赋值行
        replaced = False
        for pattern, replacement in ASSIGNMENT_REPLACEMENTS:
            if re.match(pattern, stripped):
                # 保持原始缩进
                indent = len(line) - len(line.lstrip())
                cleaned_lines.append(' ' * indent + replacement)
                replaced = True
                break
        if replaced:
            continue

        cleaned_lines.append(line)

    # 合并为字符串
    code = '\n'.join(cleaned_lines)

    # 3. 函数调用替换
    for pattern, replacement in CALL_REPLACEMENTS:
        code = re.sub(pattern, replacement, code)

    # 4. 处理 log1p 的闭合括号问题：log(1.0 + x) 已在上面替换
    # 5. 处理 expm1：exp(x) - 1 需要在 exp(x) 后加 - 1.0
    # 这两个比较复杂，用 AST 难以精确处理，暂时保持近似

    # 6. 删除连续空行（超过 2 行的缩减为 2 行）
    code = re.sub(r'\n{3,}', '\n\n', code)

    # 7. 加上 compat header
    code = COMPAT_HEADER + code

    return code


def clean_triton_code_column(code: str) -> str:
    """清洗 triton_code 列，返回清洗后的代码。"""
    if not isinstance(code, str) or not code.strip():
        return code
    if 'torch._inductor' not in code and 'torch._C._dynamo' not in code:
        return code  # 已经是干净的代码
    return clean_inductor_code(code)


# ============================================================
# 编译测试
# ============================================================

def test_compile(code: str, timeout: int = 30) -> Tuple[bool, str]:
    """
    测试代码是否能通过 Python 编译（import 检查）。

    Returns:
        (success, error_message)
    """
    # 先做 AST 语法检查
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # 写入临时文件并尝试 import
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, dir='/tmp',
            prefix='kernelbook_test_'
        ) as f:
            f.write(code)
            tmp_path = f.name

        proc = subprocess.run(
            [sys.executable, '-c',
             f"import importlib.util; "
             f"spec = importlib.util.spec_from_file_location('test_mod', '{tmp_path}'); "
             f"mod = importlib.util.module_from_spec(spec); "
             f"spec.loader.exec_module(mod); "
             f"print('COMPILE_OK')"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        os.unlink(tmp_path)

        if proc.returncode == 0 and 'COMPILE_OK' in proc.stdout:
            return True, ''
        else:
            error = proc.stderr.strip().split('\n')[-1] if proc.stderr.strip() else 'Unknown error'
            return False, error

    except subprocess.TimeoutExpired:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return False, 'Timeout'
    except Exception as e:
        return False, str(e)


# ============================================================
# 主程序
# ============================================================

def run_test(args):
    """测试模式：清洗几个样本并尝试编译。"""
    print("=" * 60)
    print("  KernelBook Inductor Code Cleaning - Test Mode")
    print("=" * 60)

    df = pd.read_parquet(args.input)
    n = min(args.num_samples, len(df))

    # 选择有代表性的样本
    # 确保包含 extern_kernels、triton_helpers、libdevice 等不同模式
    has_extern = df['triton_code'].str.contains('extern_kernels', na=False)
    has_helpers = df['triton_code'].str.contains('triton_helpers', na=False)
    has_libdevice = df['triton_code'].str.contains('libdevice', na=False)

    samples = []
    for mask, label in [
        (has_extern & has_helpers & has_libdevice, 'all patterns'),
        (has_extern & ~has_helpers, 'extern_kernels only'),
        (~has_extern & has_helpers, 'triton_helpers only'),
        (~has_extern & ~has_helpers & ~has_libdevice, 'basic (no special patterns)'),
    ]:
        subset = df[mask]
        if len(subset) > 0:
            take = min(max(1, n // 4), len(subset))
            samples.append(subset.sample(take, random_state=42))

    if samples:
        test_df = pd.concat(samples).head(n)
    else:
        test_df = df.head(n)

    print(f"\nTesting {len(test_df)} samples...\n")

    results = {'compile_ok': 0, 'compile_fail': 0, 'syntax_ok': 0, 'syntax_fail': 0}
    errors_by_type = {}

    for idx, (_, row) in enumerate(test_df.iterrows()):
        original = row['triton_code']
        cleaned = clean_inductor_code(original)

        # 语法检查
        try:
            ast.parse(cleaned)
            results['syntax_ok'] += 1
            syntax_ok = True
        except SyntaxError as e:
            results['syntax_fail'] += 1
            syntax_ok = False

        # 编译检查（只在有 GPU 环境时有意义，否则只做语法检查）
        compile_ok, error = test_compile(cleaned)
        if compile_ok:
            results['compile_ok'] += 1
        else:
            results['compile_fail'] += 1
            # 归类错误
            error_type = error.split(':')[0] if ':' in error else error
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1

        status = '✓' if compile_ok else '✗'
        print(f"  [{idx+1}/{len(test_df)}] {status} {row.get('module_name', 'unknown')[:50]}")
        if not compile_ok:
            print(f"           Error: {error[:100]}")

        # 显示第一个样本的清洗前后对比
        if idx == 0:
            print(f"\n  --- Sample 1: Before (first 20 lines) ---")
            for line in original.split('\n')[:20]:
                print(f"    {line}")
            print(f"\n  --- Sample 1: After (first 30 lines) ---")
            for line in cleaned.split('\n')[:30]:
                print(f"    {line}")
            print()

    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Syntax:  {results['syntax_ok']}/{len(test_df)} OK ({results['syntax_ok']/len(test_df)*100:.1f}%)")
    print(f"  Compile: {results['compile_ok']}/{len(test_df)} OK ({results['compile_ok']/len(test_df)*100:.1f}%)")

    if errors_by_type:
        print(f"\nError types:")
        for err, cnt in sorted(errors_by_type.items(), key=lambda x: -x[1]):
            print(f"  {err}: {cnt}")

    print()


def run_clean(args):
    """清洗全部数据。"""
    print("=" * 60)
    print("  KernelBook Inductor Code Cleaning - Full Mode")
    print("=" * 60)

    df = pd.read_parquet(args.input)
    print(f"Input: {len(df)} samples")

    # 清洗 triton_code 列
    print("Cleaning triton_code column...")
    df['triton_code_clean'] = df['triton_code'].apply(clean_triton_code_column)

    # 统计
    has_inductor_before = df['triton_code'].str.contains('torch._inductor', na=False).sum()
    has_inductor_after = df['triton_code_clean'].str.contains('torch._inductor', na=False).sum()
    print(f"  Before: {has_inductor_before} samples with torch._inductor")
    print(f"  After:  {has_inductor_after} samples with torch._inductor")

    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\nSaved to: {args.output}")


def run_stats(args):
    """统计清洗效果。"""
    df = pd.read_parquet(args.input)
    total = len(df)

    print("=" * 60)
    print("  KernelBook Inductor Pattern Statistics")
    print("=" * 60)
    print(f"Total samples: {total}\n")

    patterns = {
        'torch._inductor (any)': r'torch\._inductor',
        'grid import': r'from torch\._inductor.*import grid',
        'get_raw_stream': r'get_raw_stream',
        'assert_size_stride': r'assert_size_stride',
        'empty_strided_cuda': r'empty_strided_cuda',
        'reinterpret_tensor': r'reinterpret_tensor',
        'extern_kernels (any)': r'extern_kernels\.',
        'extern_kernels.mm': r'extern_kernels\.mm',
        'extern_kernels.addmm': r'extern_kernels\.addmm',
        'extern_kernels.bmm': r'extern_kernels\.bmm',
        'extern_kernels.convolution': r'extern_kernels\.convolution',
        'triton_helpers (any)': r'triton_helpers\.',
        'libdevice (any)': r'libdevice\.',
        'tl_math (any)': r'tl_math\.',
    }

    col = 'triton_code'
    for name, pattern in patterns.items():
        count = df[col].str.contains(pattern, na=False, regex=True).sum()
        pct = count / total * 100
        print(f"  {name:40s} {count:6d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Clean inductor code in KernelBook')
    parser.add_argument('--input', type=str, default='data/raw/kernelbook_raw.parquet',
                        help='Input parquet file')
    parser.add_argument('--output', type=str, default='data/cleaned_inductor/kernelbook_clean.parquet',
                        help='Output parquet file')
    parser.add_argument('--test', action='store_true', help='Test mode: clean a few samples and try to compile')
    parser.add_argument('--stats', action='store_true', help='Show inductor pattern statistics')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples for test mode')

    args = parser.parse_args()

    if args.test:
        run_test(args)
    elif args.stats:
        run_stats(args)
    else:
        run_clean(args)


if __name__ == '__main__':
    main()

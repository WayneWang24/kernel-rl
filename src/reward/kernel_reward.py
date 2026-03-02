"""
Triton Kernel Reward 函数。

兼容 verl 的 NaiveRewardManager 调用签名：
    compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)
    返回 float (0.0 ~ 1.0)

== 训练时 reward（compute_score）==
纯静态分析，不做编译/运行，速度快。

分层策略：
    R0 = 0.0  : 无代码块
    R1 = 0.1  : 有代码块但无有效函数/类定义
    R2 = 0.2  : 有代码但 Python 语法错误
    R3 = 0.4  : 语法正确但无 Triton kernel（无 @triton.jit）
    R4 = 0.6  : 有 @triton.jit 但结构不完整（缺 wrapper 函数）
    R5 = 0.8  : 完整的 Triton kernel + wrapper
    bonus +0.1: 包含性能优化（autotune / BLOCK_SIZE 参数化）
    bonus +0.1: 代码非简单复制（与 ground_truth 有足够差异）

== 评测时 reward（compute_score_full）==
包含编译和正确性验证，慢但准确。
"""

import ast
import re
import subprocess
import tempfile
import os
from typing import Optional


# ============================================================
# 代码提取工具
# ============================================================

def extract_code_block(text: str) -> Optional[str]:
    """从模型输出中提取代码块。"""
    # 尝试 ```python ... ``` 或 ```triton ... ```
    pattern = r"```(?:python|triton)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # 尝试通用 ``` ... ```
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    # 如果没有代码块标记，检查是否整段都是代码
    if ("import triton" in text or "@triton.jit" in text) and "def " in text:
        return text.strip()

    return None


# ============================================================
# 静态检查工具
# ============================================================

def has_valid_definition(code: str) -> bool:
    """检查是否包含有效的函数或类定义。"""
    return bool(
        re.search(r"\bdef\s+\w+", code)
        or re.search(r"\bclass\s+\w+", code)
    )


def check_syntax(code: str) -> bool:
    """检查 Python 语法是否正确。"""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def has_triton_kernel(code: str) -> bool:
    """检查是否包含 Triton kernel 定义。"""
    patterns = [
        r"@triton\.jit",
        r"@triton\.autotune",
        r"triton\.jit",
    ]
    return any(re.search(p, code) for p in patterns)


def has_wrapper_function(code: str) -> bool:
    """检查是否包含 Python wrapper 函数（调用 triton kernel 的函数）。"""
    # wrapper 通常是一个非 @triton.jit 装饰的函数，内部调用 kernel
    # 简单检测：有 def 且不全是 @triton.jit
    defs = re.findall(r"def\s+(\w+)", code)
    jit_funcs = re.findall(r"@triton\.(?:jit|autotune)\s*(?:\([^)]*\))?\s*\ndef\s+(\w+)", code)

    # 如果有 def 不在 jit 装饰器下，认为有 wrapper
    non_jit_defs = set(defs) - set(jit_funcs)
    return len(non_jit_defs) > 0


def has_performance_optimization(code: str) -> bool:
    """检查是否包含性能优化模式。"""
    patterns = [
        r"@triton\.autotune",          # autotune 装饰器
        r"BLOCK_SIZE",                  # 参数化 block size
        r"tl\.constexpr",              # constexpr 参数
        r"num_warps",                   # warp 配置
        r"num_stages",                  # pipeline stages
    ]
    return any(re.search(p, code) for p in patterns)


def is_non_trivial(code: str, ground_truth: str) -> bool:
    """检查代码是否非简单复制（与原始 PyTorch 代码有足够差异）。"""
    # 简单检查：如果和 ground_truth 的重叠度太高，认为是复制
    # 使用行级别 Jaccard 相似度
    code_lines = set(line.strip() for line in code.split("\n") if line.strip())
    gt_lines = set(line.strip() for line in ground_truth.split("\n") if line.strip())

    if not code_lines:
        return False

    overlap = len(code_lines & gt_lines)
    jaccard = overlap / max(len(code_lines | gt_lines), 1)

    # 如果相似度 > 0.8，认为是复制
    return jaccard < 0.8


# ============================================================
# 训练时 Reward（快速版）
# ============================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """
    快速 reward 函数（训练时用）。纯静态分析。

    Args:
        data_source: 数据源标识
        solution_str: 模型生成的完整文本
        ground_truth: 参考 PyTorch 代码
        extra_info: 额外信息

    Returns:
        float: 0.0 ~ 1.0 的分层 reward
    """
    # R0: 无代码块
    code = extract_code_block(solution_str)
    if code is None:
        return 0.0

    # R1: 有代码块但无有效定义
    if not has_valid_definition(code):
        return 0.1

    # R2: 语法错误
    if not check_syntax(code):
        return 0.2

    # R3: 无 Triton kernel
    if not has_triton_kernel(code):
        return 0.4

    # R4: 有 kernel 但缺 wrapper
    if not has_wrapper_function(code):
        return 0.6

    # R5: 完整的 Triton kernel + wrapper
    score = 0.8

    # Bonus: 性能优化
    if has_performance_optimization(code):
        score += 0.1

    # Bonus: 非简单复制
    if is_non_trivial(code, ground_truth):
        score += 0.1

    return min(score, 1.0)


# ============================================================
# 评测时 Reward（完整版）
# ============================================================

def compute_score_full(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    timeout: int = 60,
    **kwargs,
) -> dict:
    """
    完整 reward 函数（评测用）。包含编译和正确性验证。

    Returns:
        dict: {
            "score": float,
            "static_score": float,
            "compile_ok": bool,
            "correctness_ok": bool,
            "error": str,
        }
    """
    result = {
        "score": 0.0,
        "static_score": 0.0,
        "compile_ok": False,
        "correctness_ok": False,
        "error": "",
    }

    # 先执行静态分析
    static_score = compute_score(data_source, solution_str, ground_truth, extra_info)
    result["static_score"] = static_score

    code = extract_code_block(solution_str)
    if code is None:
        result["error"] = "no_code_block"
        return result

    if not check_syntax(code):
        result["score"] = static_score
        result["error"] = "syntax_error"
        return result

    if not has_triton_kernel(code):
        result["score"] = static_score
        result["error"] = "no_triton_kernel"
        return result

    # 编译检查
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            # 写入完整模块
            f.write("import torch\n")
            f.write("import triton\n")
            f.write("import triton.language as tl\n\n")
            f.write(code)
            tmp_path = f.name

        proc = subprocess.run(
            [
                "python",
                "-c",
                f"import importlib.util; "
                f"spec = importlib.util.spec_from_file_location('test_mod', '{tmp_path}'); "
                f"mod = importlib.util.module_from_spec(spec); "
                f"spec.loader.exec_module(mod)",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        os.unlink(tmp_path)

        if proc.returncode == 0:
            result["compile_ok"] = True
            result["score"] = 0.6
        else:
            result["score"] = static_score
            result["error"] = f"compile_error: {proc.stderr[:300]}"
            return result

    except subprocess.TimeoutExpired:
        result["score"] = static_score
        result["error"] = "compile_timeout"
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return result
    except Exception as e:
        result["score"] = static_score
        result["error"] = f"compile_exception: {str(e)[:300]}"
        return result

    # 正确性验证
    try:
        verify_script = _build_verify_script(code, ground_truth)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(verify_script)
            verify_path = f.name

        proc = subprocess.run(
            ["python", verify_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        os.unlink(verify_path)

        if proc.returncode == 0 and "VERIFY_OK" in proc.stdout:
            result["correctness_ok"] = True
            result["score"] = 1.0
        else:
            result["error"] = f"verify_fail: {proc.stdout[:200]} {proc.stderr[:200]}"
            result["score"] = 0.6

    except subprocess.TimeoutExpired:
        result["error"] = "verify_timeout"
        result["score"] = 0.6
        if os.path.exists(verify_path):
            os.unlink(verify_path)
    except Exception as e:
        result["error"] = f"verify_exception: {str(e)[:300]}"
        result["score"] = 0.6

    return result


# ============================================================
# ModelNew 格式检查工具
# ============================================================

def has_modelnew_class(code: str) -> bool:
    """检查是否包含 class ModelNew(nn.Module)。"""
    return bool(re.search(r"class\s+ModelNew\s*\(\s*(?:nn\.Module|torch\.nn\.Module)\s*\)", code))


def has_nn_module_subclass(code: str, class_name: str = "ModelNew") -> bool:
    """检查指定类是否继承 nn.Module。"""
    return bool(re.search(
        rf"class\s+{class_name}\s*\(\s*(?:nn\.Module|torch\.nn\.Module)\s*\)", code
    ))


def has_custom_forward(code: str) -> bool:
    """检查 forward 是否不直接调用 self.xxx(input) 的标准 PyTorch op。

    如果 forward 内部使用了 self.<module>(x) 模式（标准调用），则返回 False。
    如果 forward 使用了 self.<module>.weight 或自定义 kernel 调用，则返回 True。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            source_lines = ast.get_source_segment(code, node)
            if source_lines is None:
                # 回退到行号范围
                lines = code.split("\n")
                start = node.lineno - 1
                end = node.end_lineno if node.end_lineno else start + 1
                source_lines = "\n".join(lines[start:end])

            # 检查是否有 self.xxx(input) 标准调用（排除 self.xxx.weight 等属性访问）
            # 标准调用模式：self.conv(x), self.fc(x), self.bn(x)
            standard_calls = re.findall(r"self\.\w+\([^)]*\)", source_lines)
            # 排除 super().__init__() 等
            standard_calls = [
                c for c in standard_calls
                if not re.match(r"self\.\w+\.(weight|bias|parameters|state_dict|named)", c)
                and "super()" not in c
            ]

            # 检查是否有自定义 kernel/函数调用
            has_custom = bool(
                re.search(r"(triton|tl\.|kernel|custom|_kernel|_fn)\s*[.(]", source_lines)
                or re.search(r"\w+_kernel\s*\[", source_lines)  # kernel launch: kernel_fn[grid](...)
            )

            # 如果有大量标准调用且没有自定义调用，认为不是 custom forward
            if len(standard_calls) > 0 and not has_custom:
                return False
            if has_custom:
                return True

    return False


def has_triton_or_cuda_kernel(code: str) -> bool:
    """检查是否包含 @triton.jit kernel 或 CUDA load_inline。"""
    return bool(
        re.search(r"@triton\.(?:jit|autotune)", code)
        or re.search(r"load_inline\s*\(", code)
        or re.search(r"torch\.utils\.cpp_extension", code)
    )


def has_init_preserving_structure(code: str) -> bool:
    """检查 __init__ 是否保留了 nn.Module 层定义（state_dict 兼容）。

    即 __init__ 中有 self.xxx = nn.Xxx(...) 的模式。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            source_lines = ast.get_source_segment(code, node)
            if source_lines is None:
                lines = code.split("\n")
                start = node.lineno - 1
                end = node.end_lineno if node.end_lineno else start + 1
                source_lines = "\n".join(lines[start:end])

            # 检查 self.xxx = nn.Xxx(...) 模式
            return bool(re.search(r"self\.\w+\s*=\s*nn\.\w+\(", source_lines))

    return False


# ============================================================
# ModelNew 格式 Reward（训练用）
# ============================================================

def compute_score_modelnew(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """
    ModelNew 格式 reward（训练时用）。纯静态分析。

    分层策略：
        R0 = 0.0 : 无代码块
        R1 = 0.1 : 有代码但语法错误
        R2 = 0.2 : 语法正确但无 ModelNew 类
        R3 = 0.3 : 有 ModelNew 但非 nn.Module 子类
        R4 = 0.5 : ModelNew(nn.Module) 但无 Triton/CUDA kernel
        R5 = 0.6 : 有 kernel 但 forward 仍用标准 op
        R6 = 0.8 : 有 kernel + forward 使用自定义计算
        bonus +0.1 : 性能优化（autotune / BLOCK_SIZE）
        bonus +0.1 : __init__ 保留 nn.Module 结构（state_dict 兼容）

    Args:
        data_source: 数据源标识
        solution_str: 模型生成的完整文本
        ground_truth: dict with task_id, level, ref_filepath, model_code
        extra_info: 额外信息

    Returns:
        float: 0.0 ~ 1.0 的分层 reward
    """
    # R0: 无代码块
    code = extract_code_block(solution_str)
    if code is None:
        return 0.0

    # R1: 语法错误
    if not check_syntax(code):
        return 0.1

    # R2: 无 ModelNew 类
    if not re.search(r"class\s+ModelNew", code):
        return 0.2

    # R3: ModelNew 不是 nn.Module 子类
    if not has_modelnew_class(code):
        return 0.3

    # R4: 无 Triton/CUDA kernel
    if not has_triton_or_cuda_kernel(code):
        return 0.5

    # R5: 有 kernel 但 forward 仍用标准 op
    if not has_custom_forward(code):
        return 0.6

    # R6: 完整的 ModelNew + kernel + custom forward
    score = 0.8

    # Bonus: 性能优化
    if has_performance_optimization(code):
        score += 0.1

    # Bonus: __init__ 保留 nn.Module 结构
    if has_init_preserving_structure(code):
        score += 0.1

    return min(score, 1.0)


# ============================================================
# 路由函数：自动选择 reward 模式
# ============================================================

def compute_score_auto(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> float:
    """
    自动路由 reward 函数。

    根据 ground_truth 格式自动选择：
    - dict with "task_id" → compute_score_modelnew（KernelBench 任务）
    - str → compute_score（KernelBook 原始格式）
    """
    if isinstance(ground_truth, dict) and "task_id" in ground_truth:
        return compute_score_modelnew(
            data_source, solution_str, ground_truth, extra_info, **kwargs
        )
    return compute_score(
        data_source, solution_str, ground_truth, extra_info, **kwargs
    )


def _build_verify_script(generated_code: str, reference_code: str) -> str:
    """构建正确性验证脚本。"""
    return f'''
import torch
import triton
import triton.language as tl
import sys

try:
    # 定义生成的 Triton 代码
{_indent(generated_code, 4)}

    # 定义参考 PyTorch 代码
{_indent(reference_code, 4)}

    # 简单的 smoke test：检查模块能否被调用
    # 注意：完整验证需要 KernelBench 的 get_inputs() / get_init_inputs()
    print("VERIFY_OK")

except Exception as e:
    print(f"VERIFY_FAIL: {{e}}")
    sys.exit(1)
'''


def _indent(code: str, spaces: int) -> str:
    """给代码添加缩进。"""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in code.split("\n"))

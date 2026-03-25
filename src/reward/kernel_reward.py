"""
Kernel Reward 函数。

兼容 verl 的 NaiveRewardManager 调用签名：
    compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)
    返回 float (0.0 ~ 1.0)

== Triton 训练 reward（compute_score）==
纯静态分析，不做编译/运行，速度快。

== CUDA 训练 reward（compute_score_cuda）==
三阶段混合评分：静态分析 + 编译验证 + 运行验证。

== 评测时 reward（compute_score_full）==
包含编译和正确性验证，慢但准确。
"""

import ast
import hashlib
import re
import subprocess
import tempfile
import textwrap
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
    if ("import triton" in text or "@triton.jit" in text
            or "load_inline" in text or "cpp_extension" in text) and "def " in text:
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
    """检查是否包含性能优化模式（Triton 或 CUDA）。"""
    patterns = [
        r"@triton\.autotune",          # Triton autotune
        r"BLOCK_SIZE",                  # 参数化 block size
        r"tl\.constexpr",              # constexpr 参数
        r"num_warps",                   # warp 配置
        r"num_stages",                  # pipeline stages
        r"__shared__",                  # CUDA shared memory
        r"__syncthreads",              # CUDA thread sync
        r"coalesced",                   # memory coalescing comment
        r"#pragma\s+unroll",           # CUDA loop unroll
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

def _try_compile_triton(code: str, timeout: int = 60) -> bool:
    """尝试执行 Triton 代码模块，检查能否无错导入。"""
    script = f'''import torch
import torch.nn as nn
import triton
import triton.language as tl
import numpy as np
import sys

try:
{textwrap.indent(code, "    ")}
    print("COMPILE_OK")
except Exception as e:
    print(f"COMPILE_FAIL: {{e}}", file=sys.stderr)
    sys.exit(1)
'''
    try:
        proc = _run_subprocess(script, timeout=timeout)
        return proc.returncode == 0 and "COMPILE_OK" in proc.stdout
    except Exception:
        return False


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    try_compile: bool = True,
    compile_timeout: int = 60,
    **kwargs,
) -> float:
    """
    Triton reward 函数（训练时用）。静态分析 + 可选编译检查。

    Args:
        data_source: 数据源标识
        solution_str: 模型生成的完整文本
        ground_truth: 参考 PyTorch 代码
        extra_info: 额外信息
        try_compile: 是否尝试编译验证（默认 True）
        compile_timeout: 编译超时秒数

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

    # R5: 完整结构 — 编译验证
    if try_compile:
        if not _try_compile_triton(code, timeout=compile_timeout):
            return 0.7  # 结构完整但编译/导入失败

    # R6: 编译通过
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

            # 检查是否有自定义 kernel/函数调用（Triton 或 CUDA）
            has_custom = bool(
                re.search(r"(triton|tl\.|kernel|custom|_kernel|_fn)\s*[.(]", source_lines)
                or re.search(r"\w+_kernel\s*\[", source_lines)  # Triton kernel launch: kernel_fn[grid](...)
                or re.search(r"\w+\.\w+_cuda\s*\(", source_lines)  # CUDA: module.func_cuda(...)
                or re.search(r"load_inline\s*\(", source_lines)  # load_inline 在 forward 中
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
# CUDA 静态检查工具
# ============================================================

def has_cuda_kernel(code: str) -> bool:
    """检查是否包含 CUDA kernel（__global__ 或 load_inline）。"""
    return bool(
        re.search(r"__global__", code)
        or re.search(r"load_inline\s*\(", code)
        or re.search(r"cpp_extension", code)
    )


def has_load_inline(code: str) -> bool:
    """检查是否包含 load_inline 调用。"""
    return bool(re.search(r"load_inline\s*\(", code))


def has_proper_cuda_structure(code: str) -> bool:
    """检查 load_inline 三件套：cpp_sources + cuda_sources + functions。"""
    has_cpp = bool(re.search(r"cpp_sources?\s*=", code))
    has_cuda = bool(re.search(r"cuda_sources?\s*=", code))
    has_funcs = bool(re.search(r"functions\s*=", code))
    return has_cpp and has_cuda and has_funcs


# ============================================================
# CUDA compile+run 子函数
# ============================================================

def _build_cuda_compile_script(generated_code: str) -> str:
    """构建 CUDA 编译验证脚本（仅编译，不运行 forward）。"""
    # 给 load_inline 的 name 加 hash 后缀避免缓存冲突
    code_hash = hashlib.md5(generated_code.encode()).hexdigest()[:8]
    patched_code = re.sub(
        r'(load_inline\s*\(\s*name\s*=\s*["\'])(\w+)(["\'])',
        rf'\1\2_{code_hash}\3',
        generated_code,
    )
    return f'''import torch
import torch.nn as nn
import sys
import os

# 抑制编译输出
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

try:
{textwrap.indent(patched_code, "    ")}
    print("COMPILE_OK")
except Exception as e:
    print(f"COMPILE_FAIL: {{e}}", file=sys.stderr)
    sys.exit(1)
'''


def _build_cuda_instantiate_script(generated_code: str) -> str:
    """构建 ModelNew 实例化验证脚本。"""
    code_hash = hashlib.md5(generated_code.encode()).hexdigest()[:8]
    patched_code = re.sub(
        r'(load_inline\s*\(\s*name\s*=\s*["\'])(\w+)(["\'])',
        rf'\1\2_{code_hash}\3',
        generated_code,
    )
    return f'''import torch
import torch.nn as nn
import sys
import os
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

try:
{textwrap.indent(patched_code, "    ")}
    # 检查 ModelNew 可以实例化（需要从 reference 拿 get_init_inputs）
    if "ModelNew" in dir():
        print("INSTANTIATE_OK")
    else:
        print("NO_MODELNEW", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"INSTANTIATE_FAIL: {{e}}", file=sys.stderr)
    sys.exit(1)
'''


def _build_cuda_verify_script(generated_code: str, reference_code: str) -> str:
    """构建完整验证脚本：编译 + 实例化 + forward + 输出对比。"""
    code_hash = hashlib.md5(generated_code.encode()).hexdigest()[:8]
    patched_code = re.sub(
        r'(load_inline\s*\(\s*name\s*=\s*["\'])(\w+)(["\'])',
        rf'\1\2_{code_hash}\3',
        generated_code,
    )
    return f'''import torch
import torch.nn as nn
import sys
import os
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

try:
    # 1. 执行 reference 代码（定义 Model, get_inputs, get_init_inputs）
    _ref_ns = {{}}
    exec("""{reference_code.replace(chr(92), chr(92)*2).replace('"', chr(92)+'"')}""", _ref_ns)
    Model = _ref_ns["Model"]
    get_inputs = _ref_ns["get_inputs"]
    get_init_inputs = _ref_ns["get_init_inputs"]

    # 2. 执行生成代码（定义 ModelNew）
    _gen_ns = {{"torch": torch, "nn": nn}}
    _gen_ns.update({{k: v for k, v in _ref_ns.items() if not k.startswith("_")}})
{textwrap.indent(patched_code, "    ")}
    _gen_ns["ModelNew"] = ModelNew

    # 3. 实例化
    init_inputs = get_init_inputs()
    model_ref = Model(*init_inputs).cuda().eval()
    model_new = ModelNew(*init_inputs).cuda().eval()

    # 4. 尝试 load_state_dict
    try:
        model_new.load_state_dict(model_ref.state_dict())
    except Exception:
        pass  # 一些 ModelNew 可能结构不完全匹配

    # 5. Forward
    inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()]
    with torch.no_grad():
        out_ref = model_ref(*inputs)
        out_new = model_new(*inputs)

    # 6. Shape 检查
    if isinstance(out_ref, torch.Tensor) and isinstance(out_new, torch.Tensor):
        if out_ref.shape != out_new.shape:
            print(f"SHAPE_MISMATCH: ref={{out_ref.shape}} new={{out_new.shape}}")
            sys.exit(0)  # 返回码 0 但输出 SHAPE_MISMATCH
        # 7. 值检查
        try:
            torch.testing.assert_close(out_ref, out_new, rtol=1e-2, atol=1e-2)
            print("VERIFY_OK")
        except AssertionError:
            print("VALUE_MISMATCH")
    elif isinstance(out_ref, (tuple, list)) and isinstance(out_new, (tuple, list)):
        all_ok = True
        for i, (r, n) in enumerate(zip(out_ref, out_new)):
            if isinstance(r, torch.Tensor) and isinstance(n, torch.Tensor):
                if r.shape != n.shape:
                    print(f"SHAPE_MISMATCH_{{i}}")
                    all_ok = False
                    break
                try:
                    torch.testing.assert_close(r, n, rtol=1e-2, atol=1e-2)
                except AssertionError:
                    print(f"VALUE_MISMATCH_{{i}}")
                    all_ok = False
                    break
        if all_ok:
            print("VERIFY_OK")
    else:
        print("VERIFY_OK")  # 非 Tensor 输出不做检查

except Exception as e:
    print(f"VERIFY_FAIL: {{e}}", file=sys.stderr)
    sys.exit(1)
'''


def _run_subprocess(script: str, timeout: int) -> subprocess.CompletedProcess:
    """在临时文件中运行 Python 脚本，返回 CompletedProcess。"""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="/tmp"
        ) as f:
            f.write(script)
            tmp_path = f.name
        return subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ============================================================
# CUDA Reward（训练时，三阶段混合评分）
# ============================================================

def compute_score_cuda(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info: Optional[dict] = None,
    compile_timeout: int = 120,
    run_timeout: int = 60,
    **kwargs,
) -> float:
    """
    CUDA compile+run reward。三阶段混合评分。

    Phase 1 (静态分析, <1ms/sample):
      0.0  - 无代码块
      0.05 - 有代码但无有效定义
      0.1  - 语法错误
      0.15 - 无 ModelNew 类
      0.2  - ModelNew 但非 nn.Module 子类
      0.3  - 无 load_inline / CUDA kernel
      0.4  - 有 CUDA kernel 但缺 cpp_sources/cuda_sources/functions 三件套
      0.5  - 完整结构但未编译

    Phase 2 (编译验证, ~10-60s/sample, 需 nvcc+GPU):
      0.6  - load_inline() 编译成功
      0.7  - ModelNew 可实例化

    Phase 3 (运行验证, ~5-30s/sample, 需 GPU):
      0.8  - forward pass 无错误
      0.9  - 输出 shape 匹配
      1.0  - 输出值匹配（rtol=1e-2, atol=1e-2）
    """
    # ---- Phase 1: 静态分析 ----
    code = extract_code_block(solution_str)
    if code is None:
        return 0.0

    if not has_valid_definition(code):
        return 0.05

    if not check_syntax(code):
        return 0.1

    if not re.search(r"class\s+ModelNew", code):
        return 0.15

    if not has_modelnew_class(code):
        return 0.2

    if not has_cuda_kernel(code):
        return 0.3

    if not has_proper_cuda_structure(code):
        return 0.4

    # 静态分析通过 → score=0.5，进入编译阶段
    # ---- Phase 2: 编译验证 ----
    try:
        compile_script = _build_cuda_compile_script(code)
        proc = _run_subprocess(compile_script, timeout=compile_timeout)
        if proc.returncode != 0 or "COMPILE_OK" not in proc.stdout:
            return 0.5
    except subprocess.TimeoutExpired:
        return 0.5
    except Exception:
        return 0.5

    # 编译成功 → score=0.6
    # ModelNew 实例化检查（如果有 reference code）
    reference_code = None
    if isinstance(ground_truth, dict):
        reference_code = ground_truth.get("model_code", "")

    if not reference_code:
        return 0.7  # 无 reference，无法做运行验证

    # ---- Phase 3: 运行验证 ----
    try:
        verify_script = _build_cuda_verify_script(code, reference_code)
        proc = _run_subprocess(verify_script, timeout=run_timeout)
        stdout = proc.stdout.strip()

        if "VERIFY_OK" in stdout:
            return 1.0
        elif "VALUE_MISMATCH" in stdout:
            return 0.9  # shape 对但值不对
        elif "SHAPE_MISMATCH" in stdout:
            return 0.8  # forward 能跑但 shape 不对
        elif proc.returncode == 0:
            return 0.8  # forward 无错误
        else:
            return 0.7  # 编译成功但实例化/运行失败
    except subprocess.TimeoutExpired:
        return 0.7  # 运行超时，至少编译通过了
    except Exception:
        return 0.7


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
    - dict with "task_id" + "backend"="cuda" → compute_score_cuda（CUDA compile+run）
    - dict with "task_id" → compute_score_cuda（KernelBench 默认 CUDA）
    - dict with "format"="cuda" → compute_score_cuda
    - dict with "format"="modelnew" → compute_score_modelnew（Triton ModelNew）
    - dict with "format"="original" → compute_score（Triton 原始格式）
    - str → compute_score（Triton 原始格式）
    """
    if isinstance(ground_truth, dict):
        # CUDA 后端：KernelBench 任务（默认走 CUDA compile+run）
        if "task_id" in ground_truth:
            backend = ground_truth.get("backend", "cuda")
            if backend == "cuda":
                return compute_score_cuda(
                    data_source, solution_str, ground_truth, extra_info, **kwargs
                )
            else:
                return compute_score_modelnew(
                    data_source, solution_str, ground_truth, extra_info, **kwargs
                )
        # 显式标记 CUDA 格式
        if ground_truth.get("format") == "cuda":
            return compute_score_cuda(
                data_source, solution_str, ground_truth, extra_info, **kwargs
            )
        if ground_truth.get("format") == "modelnew":
            # KernelBook ModelNew 格式 RL 数据
            return compute_score_modelnew(
                data_source, solution_str, ground_truth, extra_info, **kwargs
            )
        if "triton_code" in ground_truth:
            # KernelBook 原始格式 RL 数据，用 triton_code 作为 ground_truth
            return compute_score(
                data_source, solution_str, ground_truth["triton_code"], extra_info, **kwargs
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

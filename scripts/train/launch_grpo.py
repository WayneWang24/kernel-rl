#!/usr/bin/env python3
"""
GRPO 训练启动器。

用法：
    python scripts/train/launch_grpo.py [额外 Hydra override ...]

硬件：2× A100 80GB
"""
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REWARD_FN_PATH = os.path.join(PROJECT_DIR, "src", "reward", "kernel_reward.py")

# 确保不设置 expandable_segments（与 SGLang/vLLM memory pool 不兼容）
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)


def _find_module_file(module_name):
    """找模块文件路径，完全不触发 import（避免 PyTorch/verl 版本不兼容）。"""
    import importlib.util
    # 只 find_spec 顶层包 "verl"，然后手动拼路径
    verl_spec = importlib.util.find_spec("verl")
    if verl_spec is None or verl_spec.origin is None:
        raise ImportError("Cannot find verl package")
    verl_root = os.path.dirname(verl_spec.origin)  # verl/ 目录
    # "verl.utils.reward_score" → "utils/reward_score"
    parts = module_name.split(".")[1:]  # 去掉 "verl"
    candidate = os.path.join(verl_root, *parts) + ".py"
    if os.path.isfile(candidate):
        return candidate
    # 可能是包（目录/__init__.py）
    candidate_init = os.path.join(verl_root, *parts, "__init__.py")
    if os.path.isfile(candidate_init):
        return candidate_init
    raise ImportError(f"Cannot find file for module: {module_name}")


# ============================================================
# Step 0: 补丁 verl 的 default_compute_score 以支持 kernelbook
#
# verl 0.7.0 的 custom_reward_function 配置不生效，
# 所以我们直接在 default_compute_score 里注入 kernelbook 分支。
# 这个补丁会修改 pip 安装的 verl 包文件，
# Ray worker 进程也能读到（因为它们 import 同一个包）。
# ============================================================
def patch_verl_reward():
    """给 verl 的 default_compute_score 添加 kernelbook 处理。"""
    fpath = _find_module_file("verl.utils.reward_score")
    with open(fpath) as f:
        content = f.read()

    if "kernelbook" in content:
        print("[patch] verl reward_score already patched for kernelbook")
        return

    # 构建补丁代码（注意：用 4 空格 elif + 8 空格 body，匹配 verl 源码风格）
    reward_path_escaped = REWARD_FN_PATH.replace("\\", "\\\\")
    patch_block = "\n".join([
        '    elif "kernelbook" in str(data_source):',
        '        import importlib.util',
        f'        _spec = importlib.util.spec_from_file_location("kernel_reward", "{reward_path_escaped}")',
        '        _mod = importlib.util.module_from_spec(_spec)',
        '        _spec.loader.exec_module(_mod)',
        '        res = _mod.compute_score_auto(str(data_source), solution_str, ground_truth, extra_info)',
        '',
    ])

    # 在 else: raise NotImplementedError 之前插入
    marker = "    else:\n        raise NotImplementedError"
    if marker not in content:
        print(f"[patch] WARNING: cannot find insertion point in {fpath}")
        print("[patch] Reward function for kernelbook may not work!")
        return

    patched = content.replace(marker, patch_block + "\n" + marker)
    with open(fpath, "w") as f:
        f.write(patched)
    print(f"[patch] Patched verl reward_score: {fpath}")


# 先恢复再补丁（处理之前坏掉的补丁）
def ensure_clean_verl_reward():
    """确保 verl reward_score 文件干净，再打补丁。"""
    fpath = _find_module_file("verl.utils.reward_score")
    with open(fpath) as f:
        lines = f.readlines()

    # 如果文件有语法错误（之前的坏补丁），清除 kernelbook 相关行
    has_kernelbook = any("kernelbook" in line for line in lines)
    if has_kernelbook:
        clean_lines = []
        skip = False
        for line in lines:
            # 跳过整个 kernelbook elif 块
            if "kernelbook" in line and "elif" in line:
                skip = True
                continue
            if skip:
                # 遇到下一个同级 elif/else 时停止跳过
                stripped = line.lstrip()
                if stripped.startswith(("elif ", "else:")):
                    skip = False
                    clean_lines.append(line)
                elif stripped and not line.startswith("        "):
                    # 缩进回到更外层，停止跳过
                    skip = False
                    clean_lines.append(line)
                # 否则继续跳过（属于 kernelbook 块的 body）
                continue
            clean_lines.append(line)

        with open(fpath, "w") as f:
            f.writelines(clean_lines)
        print(f"[patch] Cleaned previous kernelbook patch from {fpath}")

    # 验证清理后文件语法正确
    try:
        with open(fpath) as f:
            compile(f.read(), fpath, "exec")
    except SyntaxError as e:
        print(f"[patch] ERROR: verl reward_score has syntax error after cleanup: {e}")
        print("[patch] Please run: pip install verl==0.7.0 --force-reinstall --no-deps")
        sys.exit(1)


# ============================================================
# Step 0b: 补丁 verl 的 fsdp_workers.py
#
# verl 0.7.0 在 rollout_mode() 中没有在 vLLM resume 前调用
# torch.cuda.empty_cache()，导致训练后 PyTorch 缓存分配器
# 仍占用大量 GPU 内存，vLLM 的 cumem 分配器无法回收。
# 新版 verl 已修复（aggressive_empty_cache），我们给 0.7.0 补上。
# ============================================================
EMPTY_CACHE_PATCH_MARKER = "# [kernel-rl-empty-cache-patch]"


def clean_verl_empty_cache():
    """移除之前的 empty_cache 补丁。"""
    fpath = _find_module_file("verl.workers.fsdp_workers")
    with open(fpath) as f:
        lines = f.readlines()
    if not any(EMPTY_CACHE_PATCH_MARKER in l for l in lines):
        return
    clean = [l for l in lines if EMPTY_CACHE_PATCH_MARKER not in l]
    with open(fpath, "w") as f:
        f.writelines(clean)
    print(f"[patch] Cleaned empty_cache patch from {fpath}")


def patch_verl_empty_cache():
    """在 fsdp_workers.py 的每个 rollout.resume() 前注入 empty_cache。"""
    fpath = _find_module_file("verl.workers.fsdp_workers")
    with open(fpath) as f:
        lines = f.readlines()

    if any(EMPTY_CACHE_PATCH_MARKER in l for l in lines):
        print("[patch] fsdp_workers empty_cache already patched")
        return

    if not any("self.rollout.resume" in l for l in lines):
        print(f"[patch] WARNING: no rollout.resume in {fpath}, skipping")
        return

    new_lines = []
    patched_count = 0
    for line in lines:
        if "self.rollout.resume" in line and "await" in line:
            indent = len(line) - len(line.lstrip())
            s = " " * indent
            new_lines.append(
                f"{s}import gc; gc.collect(); import torch; torch.cuda.empty_cache()  "
                f"{EMPTY_CACHE_PATCH_MARKER}\n"
            )
            patched_count += 1
        new_lines.append(line)

    with open(fpath, "w") as f:
        f.writelines(new_lines)
    print(f"[patch] Patched fsdp_workers.py: {patched_count} empty_cache calls added ({fpath})")


# ============================================================
# Step 0c: 补丁 checkpoint 加载，容忍缺失/损坏的 optimizer state
#
# 如果 checkpoint 的 optimizer state 因磁盘满等原因损坏，
# verl 0.7.0 会直接 crash。我们给 fsdp_checkpoint_manager.py
# 的 torch.load(optim) 加上 try-except，让它跳过损坏的文件。
# ============================================================
OPTIM_PATCH_MARKER = "# [kernel-rl-optim-tolerant-patch]"


def clean_verl_optim_patch():
    """移除之前的 optimizer 容错补丁。"""
    fpath = _find_module_file("verl.utils.checkpoint.fsdp_checkpoint_manager")
    with open(fpath) as f:
        lines = f.readlines()
    if not any(OPTIM_PATCH_MARKER in l for l in lines):
        return
    clean = [l for l in lines if OPTIM_PATCH_MARKER not in l]
    with open(fpath, "w") as f:
        f.writelines(clean)
    print(f"[patch] Cleaned optim-tolerant patch from {fpath}")


def patch_verl_optim_tolerant():
    """给 optimizer state 加载加上 try-except 容错。"""
    fpath = _find_module_file("verl.utils.checkpoint.fsdp_checkpoint_manager")
    with open(fpath) as f:
        content = f.read()

    if OPTIM_PATCH_MARKER in content:
        print("[patch] fsdp_checkpoint_manager optim-tolerant already patched")
        return

    # 找 optimizer_state_dict = torch.load(...) 这行
    # 在它前面加 try，在 self.optimizer.load_state_dict 后面加 except
    if "optimizer_state_dict = torch.load" not in content:
        print(f"[patch] WARNING: cannot find optimizer torch.load in {fpath}")
        return

    lines = content.split("\n")
    new_lines = []
    i = 0
    patched = False
    while i < len(lines):
        line = lines[i]
        if not patched and "optimizer_state_dict = torch.load" in line:
            indent = len(line) - len(line.lstrip())
            s = " " * indent
            # 插入 try
            new_lines.append(f"{s}try:  {OPTIM_PATCH_MARKER}")
            # 缩进后续 optimizer 相关行
            new_lines.append(f"  {line}")
            i += 1
            # 继续缩进直到遇到 load_state_dict
            while i < len(lines):
                line2 = lines[i]
                new_lines.append(f"  {line2}")
                if "load_state_dict" in line2 and "optimizer" in line2.lower():
                    i += 1
                    break
                i += 1
            # 插入 except
            new_lines.append(
                f'{s}except Exception as _e:  {OPTIM_PATCH_MARKER}'
            )
            new_lines.append(
                f'{s}    print(f"[kernel-rl] WARNING: Failed to load optimizer state: {{_e}}")  '
                f"{OPTIM_PATCH_MARKER}"
            )
            new_lines.append(
                f'{s}    print("[kernel-rl] Continuing with fresh optimizer state")  '
                f"{OPTIM_PATCH_MARKER}"
            )
            patched = True
        else:
            new_lines.append(line)
            i += 1

    if patched:
        with open(fpath, "w") as f:
            f.write("\n".join(new_lines))
        print(f"[patch] Patched fsdp_checkpoint_manager.py with optim-tolerant loading ({fpath})")
    else:
        print(f"[patch] WARNING: could not patch optimizer loading in {fpath}")


# ============================================================
# Step 0d: 准备 checkpoint resume
#
# 如果存在部分保存的 checkpoint（模型权重有但 optimizer 损坏），
# 清理损坏文件并写入 tracker 文件让 verl 的 auto-resume 能找到它。
# ============================================================
def prepare_checkpoint_resume(ckpt_dir):
    """检查并修复部分保存的 checkpoint。"""
    import glob

    if not os.path.isdir(ckpt_dir):
        return

    # 找到最新的 global_step_* 目录
    step_dirs = sorted(glob.glob(os.path.join(ckpt_dir, "global_step_*")))
    if not step_dirs:
        return

    latest = step_dirs[-1]
    step_num = latest.split("global_step_")[-1]
    actor_dir = os.path.join(latest, "actor")

    if not os.path.isdir(actor_dir):
        return

    # 检查模型权重是否存在
    model_files = glob.glob(os.path.join(actor_dir, "model_world_size_*.pt"))
    if not model_files:
        print(f"[resume] No model weights in {latest}, skipping")
        return

    # 删除可能损坏的 optimizer 和 extra_state 文件（大小异常小 = 损坏）
    for pattern in ["optim_world_size_*.pt", "extra_state_world_size_*.pt"]:
        for f in glob.glob(os.path.join(actor_dir, pattern)):
            try:
                size_mb = os.path.getsize(f) / (1024 * 1024)
                # optimizer state 应该 > 100MB，否则可能损坏
                if size_mb < 10:
                    print(f"[resume] Removing likely corrupted file: {f} ({size_mb:.1f}MB)")
                    os.remove(f)
            except OSError:
                pass

    # 写入 tracker 文件让 auto-resume 能找到
    tracker = os.path.join(ckpt_dir, "latest_checkpointed_iteration.txt")
    with open(tracker, "w") as f:
        f.write(step_num)
    print(f"[resume] Will resume from global_step_{step_num} (model weights only)")


# ============================================================
# Step 0e: 补丁 DTensor import（PyTorch 2.4 兼容）
#
# verl 0.7.0（新版）使用 from torch.distributed.tensor import DTensor
# 但 PyTorch 2.4 中 DTensor 在 torch.distributed._tensor
# 替换为 try/except 兼容写法
# ============================================================
DTENSOR_PATCH_MARKER = "# [kernel-rl-dtensor-compat]"


def patch_verl_dtensor_compat():
    """修复 verl 中所有 DTensor import 以兼容 PyTorch 2.4。"""
    # 检查是否需要补丁（PyTorch 2.5+ 不需要）
    try:
        from torch.distributed.tensor import DTensor  # noqa: F401
        print("[patch] DTensor import OK, no compat patch needed")
        return
    except ImportError:
        pass

    import importlib.util
    verl_spec = importlib.util.find_spec("verl")
    if verl_spec is None or verl_spec.origin is None:
        return
    verl_root = os.path.dirname(verl_spec.origin)

    old_import = "from torch.distributed.tensor import DTensor"
    new_import = (
        "try:  " + DTENSOR_PATCH_MARKER + "\n"
        "    from torch.distributed.tensor import DTensor  " + DTENSOR_PATCH_MARKER + "\n"
        "except ImportError:  " + DTENSOR_PATCH_MARKER + "\n"
        "    from torch.distributed._tensor import DTensor  " + DTENSOR_PATCH_MARKER
    )

    patched_files = 0
    for root, dirs, files in os.walk(verl_root):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath) as f:
                content = f.read()
            if old_import in content and DTENSOR_PATCH_MARKER not in content:
                content = content.replace(old_import, new_import)
                with open(fpath, "w") as f:
                    f.write(content)
                patched_files += 1

    if patched_files:
        print(f"[patch] Fixed DTensor import in {patched_files} verl files (PyTorch 2.4 compat)")
    else:
        print("[patch] No DTensor import fixes needed")


# ============================================================
# Step 0f: 补丁 vllm_async_server.py 的 import
#
# verl 0.7.0 的 vllm_async_server.py 引用了
# vllm.v1.engine.utils.CoreEngineProcManager，
# 但该模块仅在 vLLM 0.8+ 存在。
# CityU HPC 的 CUDA 驱动(535)最高支持 CUDA 12.2，
# 无法安装 vLLM 0.8+（需要 PyTorch 2.6+ / CUDA 12.4+）。
# 我们把该 import 包在 try/except 里，因为正常训练
# 走的是同步 rollout 路径，不会用到 async server。
# ============================================================
ASYNC_IMPORT_PATCH_MARKER = "# [kernel-rl-async-import-compat]"


def patch_verl_async_import():
    """让 vllm_async_server.py 在 vLLM 0.7.x 下整体优雅降级。

    该文件有大量 vLLM 0.8+ 专属 import，逐个补丁不可行。
    直接把整个文件内容包在 try/except 里，import 失败时
    只留一个 vLLMReplica = None 占位符。正常训练走同步 rollout，
    不会实际使用 vLLMReplica。
    """
    try:
        fpath = _find_module_file("verl.workers.rollout.vllm_rollout.vllm_async_server")
    except ImportError:
        print("[patch] vllm_async_server.py not found, skipping")
        return

    with open(fpath) as f:
        content = f.read()

    if ASYNC_IMPORT_PATCH_MARKER in content:
        print("[patch] vllm_async_server already patched for import compat")
        return

    # 把整个文件替换为 try/except 包裹的版本
    indented = "\n".join("    " + line if line.strip() else "" for line in content.split("\n"))
    new_content = (
        f"# This file has been wrapped for vLLM 0.7.x compatibility  {ASYNC_IMPORT_PATCH_MARKER}\n"
        f"try:  {ASYNC_IMPORT_PATCH_MARKER}\n"
        f"    _ASYNC_SERVER_AVAILABLE = True  {ASYNC_IMPORT_PATCH_MARKER}\n"
        f"{indented}\n"
        f"except (ImportError, Exception) as _e:  {ASYNC_IMPORT_PATCH_MARKER}\n"
        f"    import warnings  {ASYNC_IMPORT_PATCH_MARKER}\n"
        f"    warnings.warn(f'vllm_async_server unavailable (vLLM 0.7.x compat): {{_e}}')  {ASYNC_IMPORT_PATCH_MARKER}\n"
        f"    vLLMReplica = None  {ASYNC_IMPORT_PATCH_MARKER}\n"
        f"    _ASYNC_SERVER_AVAILABLE = False  {ASYNC_IMPORT_PATCH_MARKER}\n"
    )

    with open(fpath, "w") as f:
        f.write(new_content)
    print(f"[patch] Wrapped vllm_async_server.py for vLLM 0.7.x compat ({fpath})")


# ============================================================
# Step 0g: 补丁 ray_trainer.py 跳过 AgentLoopManager
#
# AgentLoopManager 在 init_workers() 中无条件初始化，
# 但它依赖 vLLM 0.8+ 的 async server。
# 我们让它在初始化失败时优雅跳过。
# ============================================================
AGENT_LOOP_PATCH_MARKER = "# [kernel-rl-agent-loop-skip]"


def patch_verl_skip_agent_loop():
    """让 ray_trainer.py 在 AgentLoopManager 初始化失败时跳过。"""
    fpath = _find_module_file("verl.trainer.ppo.ray_trainer")

    with open(fpath) as f:
        content = f.read()

    if AGENT_LOOP_PATCH_MARKER in content:
        print("[patch] ray_trainer already patched to skip AgentLoopManager")
        return

    # 找到 self.async_rollout_manager = AgentLoopManager( 这行
    target = "self.async_rollout_manager = AgentLoopManager("
    if target not in content:
        print("[patch] ray_trainer: AgentLoopManager init not found, skipping")
        return

    lines = content.split("\n")
    new_lines = []
    i = 0
    patched = False
    while i < len(lines):
        line = lines[i]
        if not patched and target in line:
            indent = len(line) - len(line.lstrip())
            s = " " * indent
            # 插入 try 在 AgentLoopManager import 之前
            # 先找到 from verl.experimental.agent_loop import 那行（在前面几行）
            # 从当前位置向前找 import 行
            import_idx = None
            for j in range(len(new_lines) - 1, max(len(new_lines) - 15, -1), -1):
                if "from verl.experimental.agent_loop import" in new_lines[j]:
                    import_idx = j
                    break

            if import_idx is not None:
                # 在 import 行前插入 try
                new_lines.insert(import_idx, f"{s}try:  {AGENT_LOOP_PATCH_MARKER}")
                # import 行和后续行都需要额外缩进
                for k in range(import_idx + 1, len(new_lines)):
                    new_lines[k] = "    " + new_lines[k]

            # 当前行（AgentLoopManager 初始化）也需要缩进
            new_lines.append("    " + line)
            i += 1
            # 继续缩进直到 AgentLoopManager(...) 结束（找到匹配的右括号）
            paren_depth = line.count("(") - line.count(")")
            while i < len(lines) and paren_depth > 0:
                line2 = lines[i]
                paren_depth += line2.count("(") - line2.count(")")
                new_lines.append("    " + line2)
                i += 1

            # 插入 except
            new_lines.append(f"{s}except (ImportError, TypeError, Exception) as _e:  {AGENT_LOOP_PATCH_MARKER}")
            new_lines.append(f"{s}    import warnings  {AGENT_LOOP_PATCH_MARKER}")
            new_lines.append(f'{s}    warnings.warn(f"AgentLoopManager unavailable, skipping async rollout: {{_e}}")  {AGENT_LOOP_PATCH_MARKER}')
            new_lines.append(f"{s}    self.async_rollout_manager = None  {AGENT_LOOP_PATCH_MARKER}")
            patched = True
        else:
            new_lines.append(line)
            i += 1

    if patched:
        with open(fpath, "w") as f:
            f.write("\n".join(new_lines))
        print(f"[patch] Patched ray_trainer.py to skip AgentLoopManager on failure ({fpath})")
    else:
        print("[patch] WARNING: could not patch AgentLoopManager in ray_trainer.py")


# ============================================================
# Step 0h: 补丁 worker.py 的 ROCR_VISIBLE_DEVICES 检查
#
# SLURM 会给每个进程注入 ROCR_VISIBLE_DEVICES（AMD ROCm 变量），
# 当同时存在 CUDA_VISIBLE_DEVICES 时，verl 会直接 raise ValueError。
# 在纯 NVIDIA 环境下，ROCR 变量无意义，直接在检查前 pop 掉。
# ============================================================
ROCR_PATCH_MARKER = "# [kernel-rl-rocr-fix]"


def patch_verl_rocr_fix():
    """在 worker.py 的 _setup_env_cuda_visible_devices 开头清除 ROCR_VISIBLE_DEVICES。"""
    fpath = _find_module_file("verl.single_controller.base.worker")

    with open(fpath) as f:
        content = f.read()

    if ROCR_PATCH_MARKER in content:
        print("[patch] worker.py ROCR fix already applied")
        return

    target = "def _setup_env_cuda_visible_devices(self):"
    if target not in content:
        print("[patch] worker.py: _setup_env_cuda_visible_devices not found, skipping")
        return

    # 在函数定义后插入 pop ROCR
    replacement = (
        f"{target}\n"
        f"        os.environ.pop('ROCR_VISIBLE_DEVICES', None)  {ROCR_PATCH_MARKER}"
    )
    content = content.replace(target, replacement)

    with open(fpath, "w") as f:
        f.write(content)
    print(f"[patch] Patched worker.py to remove ROCR_VISIBLE_DEVICES ({fpath})")


# ============================================================
# Step 7: 补丁 fsdp2_clip_grad_norm_（PyTorch 2.4 兼容）
#
# verl 0.7.0 的 fsdp2_clip_grad_norm_ 依赖 PyTorch 2.5+ 的
# _clip_grads_with_norm_ 和 _get_total_norm。
# PyTorch 2.4 没有这两个函数，需要用兼容实现替换整个函数。
# ============================================================
def patch_verl_fsdp_clip_grad():
    """为 PyTorch 2.4 提供 fsdp2_clip_grad_norm_ 兼容实现。"""
    # 先检查是否需要补丁
    try:
        from torch.nn.utils.clip_grad import _clip_grads_with_norm_
        print("[patch_fsdp_clip_grad] PyTorch has _clip_grads_with_norm_, no patch needed")
        return
    except ImportError:
        pass

    fpath = _find_module_file("verl.utils.fsdp_utils")
    with open(fpath, "r") as f:
        content = f.read()

    if "# PATCHED: fsdp2_clip_grad_norm_ compat" in content:
        print("[patch_fsdp_clip_grad] Already patched")
        return

    # 替换整个 fsdp2_clip_grad_norm_ 函数
    old_func = '''def fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """torch.nn.utils.clip_grad_norm_ cann't run on cpu parameter DTensor"""
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = total_norm.to(get_device_id(), non_blocking=True)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm'''

    new_func = '''def fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    # PATCHED: fsdp2_clip_grad_norm_ compat for PyTorch 2.4
    """torch.nn.utils.clip_grad_norm_ - compatible with PyTorch 2.4+"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor(0.0)
    if norm_type == float('inf'):
        norms = [g.detach().abs().max() for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(f"total_norm is {total_norm}")
    total_norm = total_norm.to(get_device_id(), non_blocking=True)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        if p.grad is not None:
            p.grad.detach().mul_(clip_coef_clamped)
    return total_norm'''

    if old_func not in content:
        print("[patch_fsdp_clip_grad] WARNING: original function not found, skipping")
        return

    content = content.replace(old_func, new_func)
    with open(fpath, "w") as f:
        f.write(content)
    print(f"[patch_fsdp_clip_grad] Patched {fpath}")


# ============================================================
# 公共辅助：应用所有补丁
# ============================================================
def apply_all_patches(project_dir=PROJECT_DIR, optim_tolerant=False):
    """应用所有 verl 补丁。可被 SLURM 脚本直接调用。

    Args:
        optim_tolerant: 是否打 optimizer 容错补丁（仅在从损坏 checkpoint 恢复时需要）
    """
    patch_verl_fsdp_clip_grad()
    patch_verl_dtensor_compat()
    patch_verl_async_import()
    patch_verl_skip_agent_loop()
    patch_verl_rocr_fix()
    ensure_clean_verl_reward()
    patch_verl_reward()
    clean_verl_empty_cache()
    patch_verl_empty_cache()
    if optim_tolerant:
        clean_verl_optim_patch()
        patch_verl_optim_tolerant()
    prepare_checkpoint_resume(os.path.join(project_dir, "checkpoints", "grpo_cuda"))
    print("[patches] All patches applied successfully")


# ============================================================
# Main: 数据探测 + 补丁 + 启动训练
# ============================================================
def main():
    # Step 1: 数据路径（优先 CUDA 数据）
    if os.path.isfile(f"{PROJECT_DIR}/data/rl_kernelbench_cuda/train.parquet"):
        train_path = f"{PROJECT_DIR}/data/rl_kernelbench_cuda/train.parquet"
        val_path = f"{PROJECT_DIR}/data/rl_kernelbench_cuda/val.parquet"
        reward_fn_name = "compute_score_auto"
        print("Using KernelBench CUDA RL data (compile+run reward)")
    elif os.path.isfile(f"{PROJECT_DIR}/data/split/rl/train.parquet"):
        train_path = f"{PROJECT_DIR}/data/split/rl/train.parquet"
        val_path = f"{PROJECT_DIR}/data/split/rl/val.parquet"
        reward_fn_name = "compute_score_auto"
        print("Using KernelBook split RL data (ModelNew format)")
    elif os.path.isfile(f"{PROJECT_DIR}/data/rl_kernelbench/train.parquet"):
        train_path = f"{PROJECT_DIR}/data/rl_kernelbench/train.parquet"
        val_path = f"{PROJECT_DIR}/data/rl_kernelbench/val.parquet"
        reward_fn_name = "compute_score_auto"
        print("Using KernelBench RL data (Triton format)")
    else:
        train_path = f"{PROJECT_DIR}/data/rl/train.parquet"
        val_path = f"{PROJECT_DIR}/data/rl/val.parquet"
        reward_fn_name = "compute_score"
        print("Using KernelBook RL data (original format)")

    # Step 2: 模型路径
    model_path = None
    for ckpt in ["checkpoints/sft_modelnew_merged", "checkpoints/sft_merged"]:
        full = os.path.join(PROJECT_DIR, ckpt)
        if os.path.isdir(full):
            model_path = full
            print(f"Using SFT checkpoint: {model_path}")
            break
    if model_path is None:
        model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-Coder-3B-Instruct")
        print(f"WARNING: SFT checkpoint not found, using base model: {model_path}")

    # Step 3: 检查数据
    if not os.path.isfile(train_path):
        print(f"ERROR: RL training data not found at {train_path}")
        sys.exit(1)

    # Step 4: 补丁
    apply_all_patches()

    # Step 5: 构建 Hydra overrides 并启动
    overrides = [
        "algorithm.adv_estimator=grpo",
        f"data.train_files={train_path}",
        f"data.val_files={val_path}",
        "data.train_batch_size=8",
        "data.max_prompt_length=2048",
        "data.max_response_length=4096",
        "data.filter_overlong_prompts=true",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.model.use_remove_padding=false",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.actor.fsdp_config.param_offload=false",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
        # 禁用 Adam foreach 优化，避免 _foreach_sqrt 产生 ~14GB 临时内存峰值
        "+actor_rollout_ref.actor.optim.override_optimizer_config.foreach=false",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.30",
        "actor_rollout_ref.rollout.free_cache_engine=true",
        "actor_rollout_ref.rollout.enforce_eager=false",
        "actor_rollout_ref.rollout.max_num_seqs=8",
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer",
        "+actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs=8",
        "actor_rollout_ref.rollout.n=3",
        # ref model 已禁用（use_kl_loss=false + use_kl_in_reward=false）
        "algorithm.use_kl_in_reward=false",
        f"+reward.custom_reward_function.path={REWARD_FN_PATH}",
        f"+reward.custom_reward_function.name={reward_fn_name}",
        "trainer.critic_warmup=0",
        'trainer.logger=["console"]',
        "trainer.project_name=kernel_rl",
        "trainer.experiment_name=grpo_cuda_sglang_2gpu",
        f"trainer.default_local_dir={PROJECT_DIR}/checkpoints/grpo_cuda",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        "trainer.save_freq=200",
        "trainer.test_freq=200",
        "trainer.max_actor_ckpt_to_keep=1",
        "trainer.total_epochs=3",
    ]

    # 追加用户额外参数
    overrides.extend(sys.argv[1:])

    # 启动
    cmd = [sys.executable, "-m", "verl.trainer.main_ppo"] + overrides
    print(f"\n=== Launching GRPO Training ===")
    print(f"Overrides: {len(overrides)} params")
    print()

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()

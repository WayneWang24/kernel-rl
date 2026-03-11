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
    import verl.utils.reward_score as reward_module

    fpath = reward_module.__file__
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
    import verl.utils.reward_score as reward_module

    fpath = reward_module.__file__
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

    # 验证清理后文件能正常 import
    try:
        import importlib
        importlib.reload(reward_module)
    except SyntaxError as e:
        print(f"[patch] ERROR: verl reward_score has syntax error after cleanup: {e}")
        print("[patch] Please run: pip install verl==0.7.0 --force-reinstall --no-deps")
        sys.exit(1)


# ============================================================
# Step 1: 数据路径（自动探测）
# ============================================================
if os.path.isfile(f"{PROJECT_DIR}/data/split/rl/train.parquet"):
    train_path = f"{PROJECT_DIR}/data/split/rl/train.parquet"
    val_path = f"{PROJECT_DIR}/data/split/rl/val.parquet"
    reward_fn_name = "compute_score_auto"
    print("Using KernelBook split RL data (ModelNew format)")
elif os.path.isfile(f"{PROJECT_DIR}/data/rl_kernelbench/train.parquet"):
    train_path = f"{PROJECT_DIR}/data/rl_kernelbench/train.parquet"
    val_path = f"{PROJECT_DIR}/data/rl_kernelbench/val.parquet"
    reward_fn_name = "compute_score_auto"
    print("Using KernelBench RL data (ModelNew format)")
else:
    train_path = f"{PROJECT_DIR}/data/rl/train.parquet"
    val_path = f"{PROJECT_DIR}/data/rl/val.parquet"
    reward_fn_name = "compute_score"
    print("Using KernelBook RL data (original format)")

# ============================================================
# Step 2: 模型路径（自动探测）
# ============================================================
model_path = None
for ckpt in ["checkpoints/sft_modelnew_merged", "checkpoints/sft_merged"]:
    full = os.path.join(PROJECT_DIR, ckpt)
    if os.path.isdir(full):
        model_path = full
        print(f"Using SFT checkpoint: {model_path}")
        break
if model_path is None:
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-Coder-7B-Instruct")
    print(f"WARNING: SFT checkpoint not found, using base model: {model_path}")

# ============================================================
# Step 3: 检查数据
# ============================================================
if not os.path.isfile(train_path):
    print(f"ERROR: RL training data not found at {train_path}")
    sys.exit(1)

# ============================================================
# Step 4: 补丁 verl reward
# ============================================================
ensure_clean_verl_reward()
patch_verl_reward()

# ============================================================
# Step 5: 构建 Hydra overrides 并启动
# ============================================================
overrides = [
    "algorithm.adv_estimator=grpo",
    f"data.train_files={train_path}",
    f"data.val_files={val_path}",
    "data.train_batch_size=16",
    "data.max_prompt_length=4096",
    "data.max_response_length=8192",
    "data.filter_overlong_prompts=true",
    "data.truncation=error",
    f"actor_rollout_ref.model.path={model_path}",
    "actor_rollout_ref.actor.optim.lr=5e-7",
    "actor_rollout_ref.model.use_remove_padding=false",
    "actor_rollout_ref.actor.ppo_mini_batch_size=16",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.actor.use_kl_loss=true",
    "actor_rollout_ref.actor.kl_loss_coef=0.005",
    "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
    "actor_rollout_ref.actor.entropy_coeff=0",
    "actor_rollout_ref.model.enable_gradient_checkpointing=true",
    "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
    "actor_rollout_ref.actor.fsdp_config.param_offload=false",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
    "actor_rollout_ref.rollout.name=vllm",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",
    "actor_rollout_ref.rollout.n=5",
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
    "actor_rollout_ref.ref.fsdp_config.param_offload=true",
    "algorithm.use_kl_in_reward=false",
    f"+reward.custom_reward_function.path={REWARD_FN_PATH}",
    f"+reward.custom_reward_function.name={reward_fn_name}",
    "trainer.critic_warmup=0",
    'trainer.logger=["console"]',
    "trainer.project_name=kernel_rl",
    "trainer.experiment_name=grpo_qwen25_coder_7b_modelnew",
    f"trainer.default_local_dir={PROJECT_DIR}/checkpoints/grpo",
    "trainer.n_gpus_per_node=2",
    "trainer.nnodes=1",
    "trainer.save_freq=10",
    "trainer.test_freq=5",
    "trainer.total_epochs=20",
]

# 追加用户额外参数
overrides.extend(sys.argv[1:])

# 启动
cmd = [sys.executable, "-m", "verl.trainer.main_ppo"] + overrides
print(f"\n=== Launching GRPO Training ===")
print(f"Overrides: {len(overrides)} params")
print()

os.execvp(cmd[0], cmd)
